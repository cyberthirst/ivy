from typing import Optional, Type
from contextlib import contextmanager

import vyper.ast.nodes as ast
from vyper.semantics.analysis.base import StateMutability, Modifiability
from vyper.semantics.types import (
    VyperType,
    TYPE_T,
    InterfaceT,
    StructT,
    MemberFunctionT,
    SelfT,
)
from vyper.semantics.types.module import ModuleT
from vyper.semantics.types.function import (
    ContractFunctionT,
)
from vyper.builtins._signatures import BuiltinFunctionT
from vyper.codegen.core import calculate_type_for_external_return
from vyper.semantics.analysis.base import VarInfo

from ivy.expr import ExprVisitor
from ivy.stmt import ReturnException, StmtVisitor
from ivy.evaluator import VyperEvaluator
from ivy.builtins.builtin_registry import BuiltinRegistry
from ivy.utils import compute_call_abi_data
from ivy.abi import abi_decode, abi_encode
from ivy.exceptions import (
    StaticCallViolation,
    AccessViolation,
    GasReference,
    FunctionNotFound,
)
from ivy.types import Address, Struct, StaticArray, DynamicArray, Map
from ivy.allocator import Allocator
from ivy.evm.evm_callbacks import EVMCallbacks
from ivy.evm.evm_core import EVMCore
from ivy.evm.evm_state import StateAccess


class VyperInterpreter(ExprVisitor, StmtVisitor, EVMCallbacks):
    evaluator: Type[VyperEvaluator]

    def __init__(self):
        self.evaluator = VyperEvaluator
        self.evm = EVMCore(callbacks=self)
        self.state: StateAccess = self.evm.state
        self.builtins = BuiltinRegistry(self.evm, self.state)

    def execute(self, *args, **kwargs):
        return self.evm.execute_tx(*args, **kwargs)

    @property
    def deployer(self):
        from ivy.frontend.vyper_contract import VyperDeployer

        return VyperDeployer

    @property
    def fun_ctx(self):
        return self.current_context.current_fun_context()

    @property
    def globals(self):
        return self.current_context.globals

    @property
    def msg(self):
        return self.current_context.msg

    @property
    def current_address(self):
        return self.current_context.msg.to

    @property
    def memory(self):
        # function variables are located in memory
        # fun ctx has set/get methods which are used to access the variables from scopes
        return self.fun_ctx

    @property
    def current_context(self):
        return self.evm.state.current_context

    def _push_fun_ctx(self, func_t):
        self.current_context.push_fun_context(func_t)

    def _pop_fun_ctx(self):
        self.current_context.pop_fun_context()

    def _push_scope(self):
        self.current_context.push_scope()

    def _pop_scope(self):
        self.current_context.pop_scope()

    def allocate_variables(self, module_t: ModuleT):
        allocator = Allocator()
        # separeate address allocation from variable creation
        # the allocator rewrites the varinfo.position
        nonreentrant, globals = allocator.allocate_addresses(module_t)

        for var in globals:
            get_location = self.storage_getter_from_varinfo(var)

            if var.is_constant:
                value = self.visit(var.decl_node.value)  # the value of the constant
                self.globals.new_variable(var, get_location, value)
                continue

            self.globals.new_variable(var, get_location)

        self.globals.allocate_reentrant_key(
            nonreentrant, lambda: self.current_context.transient
        )

    def lock(self, mutability):
        lock = self.globals.get_reentrant_key()
        if lock:
            raise AccessViolation("Reentrancy violation")

        # for view functions we can't write the lock, so we only check it's not locked
        if mutability == StateMutability.VIEW:
            return

        self.globals.set_reentrant_key(True)

    def unlock(self, mutability):
        lock = self.globals.get_reentrant_key()
        if mutability == StateMutability.VIEW:
            assert not lock
            return

        assert lock
        self.globals.set_reentrant_key(False)

    def _prologue(self, func_t, args):
        self._push_fun_ctx(func_t)
        self._push_scope()

        if func_t.nonreentrant:
            self.lock(func_t.mutability)

        func_args = func_t.arguments
        for arg, param in zip(args, func_args):
            self._new_local(param.name, param.typ)
            self.set_variable(param.name, arg)

        # check if we need to assign default values
        if len(args) < len(func_args):
            for param in func_args[len(args) :]:
                self._new_local(param.name, param.typ)
                default_value = self.visit(param.default_value)
                self.set_variable(param.name, default_value)

    def _epilogue(self, func_t):
        if func_t.nonreentrant:
            self.unlock(func_t.mutability)
        self._pop_scope()
        self._pop_fun_ctx()

    def execute_init_function(self, func_t):
        _, calldata_args_t = compute_call_abi_data(func_t, 0)
        min_size = calldata_args_t.abi_type.static_size()
        self._min_calldata_size_check(min_size)
        # msg.data doesn't contain bytecode in ivy's case
        args = abi_decode(calldata_args_t, self.msg.data)

        return self._execute_function(func_t, args)

    def _min_calldata_size_check(self, min_size):
        if len(self.msg.data) < min_size:
            raise BufferError(f"Provided calldata is too small, min_size is {min_size}")

    def dispatch(self):
        try:
            if len(self.msg.data) < 4:
                raise FunctionNotFound()

            selector = self.msg.data[:4]

            entry_points = self.current_context.entry_points

            if selector not in entry_points:
                raise FunctionNotFound()
            else:
                entry_point = entry_points[selector]

            if entry_point.function.is_payable:
                if self.msg.value != 0:
                    # TODO raise and rollback
                    pass

            self._min_calldata_size_check(entry_point.calldata_min_size)

            func_t = entry_point.function
            args = abi_decode(entry_point.calldata_args_t, self.msg.data[4:])

        except FunctionNotFound as e:
            if self.current_context.contract.fallback:
                func_t = self.current_context.contract.fallback
                args = ()
            else:
                raise e

        self._execute_external_function(func_t, args)

    def _execute_external_function(self, func_t, args):
        ret = self._execute_function(func_t, args)

        # an escape hatch for the minimal proxy where we want plain pass-through
        if self._inside_minimal_proxy():
            self.current_context.execution_output.output = ret
            return

        # abi-encode output
        typ = func_t.return_type
        if typ is None:
            assert ret is None
            return None
        typ = calculate_type_for_external_return(typ)
        # from https://github.com/vyperlang/vyper/blob/a1af967e675b72051cf236f75e1104378fd83030/vyper/codegen/core.py#L694
        output = (ret,) if (not isinstance(ret, tuple) or len(ret) <= 1) else ret
        self.current_context.execution_output.output = abi_encode(typ, output)

    def _execute_function(self, func_t, args):
        # TODO: rewrite this using a ctx manager?
        self._prologue(func_t, args)
        ret = None

        for stmt in func_t.decl_node.body:
            try:
                self.visit(stmt)
            except ReturnException as e:
                ret = e.value
                break

        self._epilogue(func_t)

        if func_t.is_deploy:
            return self.current_context.contract
        return ret

    @contextmanager
    def modifiable_context(self, target):
        if self.msg.is_static:
            raise StaticCallViolation(f"Cannot modify {target} in a static context")
        try:
            yield
        finally:
            pass

    def _is_global_var(self, varinfo: Optional[VarInfo]):
        if varinfo is None:
            return False
        return (
            varinfo.is_state_variable()
            or varinfo.modifiability == Modifiability.CONSTANT
        )

    def _resolve_variable_info(
        self, node: Optional[ast.VyperNode] = None
    ) -> tuple[Optional[VarInfo], bool]:
        # in some scenarios (eg auxiliary helper variables) we don't have a node
        varinfo = node._expr_info.var_info if node else None
        is_global = self._is_global_var(varinfo)
        return varinfo, is_global

    def get_variable(self, name: str, node: Optional[ast.VyperNode] = None):
        varinfo, is_global = self._resolve_variable_info(node)
        if is_global:
            res = self.globals[varinfo].value
        else:
            res = self.memory[name]
        assert res is not None
        return res

    def set_variable(self, name: str, value, node: Optional[ast.VyperNode] = None):
        varinfo, is_global = self._resolve_variable_info(node)
        if is_global:
            with self.modifiable_context(varinfo):
                var = self.globals[varinfo]
                var.value = value
        else:
            self.memory[name] = value

    # we want to decouple variables from the references to the actual storage location
    # it makes implementing certain features easier. thus the variables
    # will get the location based on current execution context on demand
    def storage_getter_from_varinfo(self, varinfo: VarInfo):
        if varinfo.is_immutable:
            return lambda: self.current_context.immutables
        elif varinfo.is_constant:
            return lambda: self.current_context.constants
        elif varinfo.is_transient:
            return lambda: self.current_context.transient
        else:
            assert varinfo.is_storage
            return lambda: self.current_context.storage

    def _new_local(self, identifier: str, typ: VyperType):
        self.memory.new_variable(identifier, typ)

    def _assign_target(self, target, value):
        if isinstance(target, ast.Name):
            self.set_variable(target.id, value, target)
        elif isinstance(target, ast.Tuple):
            if not isinstance(value, tuple):
                raise TypeError("Cannot unpack non-iterable to tuple")
            if len(target.elements) != len(value):
                raise ValueError("Mismatch in number of items to unpack")
            for t, v in zip(target.elements, value):
                self._assign_target(t, v)
        elif isinstance(target, ast.Subscript):
            container = self.visit(target.value)
            index = self.visit(target.slice)
            loc = target._expr_info.location
            if isinstance(container, (Map, StaticArray, DynamicArray)):
                container.__setitem__(index, value, loc)
            else:
                # TODO should we create a custom tuple too?
                # it's not necessary for journaling though
                assert isinstance(container, tuple)
                container[index] = value
        elif isinstance(target, ast.Attribute):
            typ = target.value._metadata["type"]
            if isinstance(typ, (SelfT, ModuleT)):
                self.set_variable(target.attr, value, target)
            else:
                assert isinstance(typ, StructT)
                obj = self.visit(target.value)
                loc = target._expr_info.location
                assert isinstance(obj, Struct)
                obj.__setitem__(target.attr, value, loc)
        else:
            raise NotImplementedError(f"Assignment to {type(target)} not implemented")

    def _handle_address_variable(self, node: ast.Attribute):
        # x.address
        if node.attr == "address":
            return self.visit(node.value)
        # x.balance: balance of address x
        if node.attr == "balance":
            addr = self.visit(node.value)
            return self.state.get_balance(addr)
        # x.codesize: codesize of address x
        elif node.attr == "codesize" or node.attr == "is_contract":
            addr = self.visit(node.value)
            if node.attr == "codesize":
                raise NotImplementedError("codesize")
            else:
                return self.state.get_code(addr) is not None
        # x.codehash: keccak of address x
        elif node.attr == "codehash":
            raise NotImplementedError("codehash")
        # x.code: codecopy/extcodecopy of address x
        elif node.attr == "code":
            raise NotImplementedError("code")

    def _handle_env_variable(self, node: ast.Attribute):
        key = f"{node.value.id}.{node.attr}"
        if key == "msg.sender":
            return self.msg.caller
        elif key == "msg.data":
            return self.msg.data
        elif (
            key == "msg.value"
        ):  # TODO check payble (context and self.context.is_payable:)
            return self.msg.value
        elif key in ("msg.gas", "msg.mana"):
            raise GasReference()
        elif key == "block.prevrandao":
            raise NotImplementedError("block.prevrandao")
        elif key == "block.difficulty":
            raise NotImplementedError("block.prevrandao")
        elif key == "block.timestamp":
            return self.env.time
        elif key == "block.coinbase":
            return self.env.coinbase
        elif key == "block.number":
            return self.env.number
        elif key == "block.gaslimit":
            raise GasReference()
        elif key == "block.basefee":
            raise GasReference()
        elif key == "block.blobbasefee":
            raise NotImplementedError("block.blobbasefee")
        elif key == "block.prevhash":
            raise NotImplementedError("block.prevhash")
        elif key == "tx.origin":
            return self.env.origin
        elif key == "tx.gasprice":
            raise GasReference()
        elif key == "chain.id":
            return self.env.chain_id

    def _inside_minimal_proxy(self):
        decl_node = self.current_context.contract.module_t.decl_node
        return "is_minimal_proxy" in decl_node._metadata

    def generic_call_handler(
        self,
        call: ast.Call,
        args,
        kws,
        typs,
        target: Optional[Address] = None,
        is_static: Optional[bool] = None,
    ):
        # `None` is a special case: range is not assigned `type` in Vyper's frontend
        func_t = call.func._metadata.get("type", None)

        if func_t is None or isinstance(func_t, BuiltinFunctionT):
            id = call.func.id
            # we don't tag runtime values with a corresponding vyper type, thus we have to
            # collect the types separately and for certain builtins inject them
            if id in ("abi_encode", "_abi_encode", "convert"):
                args = (typs, args)
            return self.builtins.get(id)(*args, **kws)

        if isinstance(func_t, TYPE_T):
            # struct & interface constructors
            typedef = func_t.typedef
            if isinstance(typedef, InterfaceT):
                # TODO should we return an address here? or an interface object wrapping the address?
                # we will likely need the attrs of the interface..
                assert len(args) == 1
                return args[0]
            else:
                assert isinstance(typedef, StructT) and len(args) == 0
                return Struct(typedef, kws)

        if isinstance(func_t, MemberFunctionT):
            # the function is an attribute of the array
            darray = self.visit(call.func.value)
            assert isinstance(darray, DynamicArray)
            if func_t.name == "append":
                assert len(args) == 1
                darray.append(args[0])
                return None
            else:
                assert func_t.name == "pop" and len(args) == 0
                return darray.pop()

        assert isinstance(func_t, ContractFunctionT)

        if func_t.is_external:
            assert target is not None and isinstance(func_t, ContractFunctionT)
            return self.external_function_call(func_t, args, kws, is_static, target)

        assert func_t.is_internal
        return self._execute_function(func_t, args)

    def external_function_call(
        self, func_t: ContractFunctionT, args, kwargs, is_static: bool, target: Address
    ):
        num_kwargs = len(args) - func_t.n_positional_args

        selector, calldata_args_t = compute_call_abi_data(func_t, num_kwargs)

        data = selector
        data += abi_encode(calldata_args_t, args)

        output = self.evm.do_message_call(
            target, kwargs.get("value", 0), data, is_static=is_static, is_delegate=False
        )

        returndata = self.current_context.returndata

        if output.error:
            # TODO do we forward the returndata on failure
            self.state.current_output.output = returndata
            raise output.error

        if len(returndata) == 0 and "default_return_value" in kwargs:
            return kwargs["default_return_value"]

        typ = func_t.return_type

        if typ is None:
            return None

        typ = calculate_type_for_external_return(typ)
        abi_typ = typ.abi_type

        max_return_size = abi_typ.size_bound()

        actual_output_size = min(max_return_size, len(returndata))
        to_decode = returndata[:actual_output_size]

        # NOTE: abi_decode implicitly checks minimum return size
        decoded = abi_decode(typ, to_decode)
        assert len(decoded) == 1
        # unwrap the tuple
        return decoded[0]

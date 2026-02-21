import time
from typing import Optional, Any

import vyper.ast.nodes as ast
from vyper.semantics.analysis.base import StateMutability, Modifiability
from vyper.semantics.types import (
    VyperType,
    TupleT,
    TYPE_T,
    InterfaceT,
    StructT,
    MemberFunctionT,
    SelfT,
    EventT,
    _BytestringT,
    BytesT,
)
from vyper.semantics.types.module import ModuleT
from vyper.semantics.types.function import (
    ContractFunctionT,
)
from vyper.builtins._signatures import BuiltinFunctionT
from vyper.codegen.core import (
    calculate_type_for_external_return,
    needs_external_call_wrap,
)
from vyper.semantics.analysis.base import VarInfo
from vyper.exceptions import TypeMismatch
from vyper.semantics.types.shortcuts import UINT256_T
from vyper.utils import keccak256

from ivy.expr.expr import ExprVisitor
from ivy.expr.clamper import box_value, box_value_from_node
from ivy.stmt import ReturnException, StmtVisitor
from ivy.builtins.builtin_registry import BuiltinRegistry
from ivy.utils import compute_call_abi_data
from ivy.abi import abi_decode, abi_encode
from ivy.exceptions import (
    AccessViolation,
    CallTimeout,
    GasReference,
    FunctionNotFound,
    PayabilityViolation,
    StaticCallViolation,
    UnsupportedFeature,
)
from ivy.tracer import Tracer
from ivy.types import (
    Address,
    Struct,
    StaticArray,
    DynamicArray,
    Map,
    Tuple as IvyTuple,
    boxed,
)
from ivy.allocator import Allocator
from ivy.evm.evm_callbacks import EVMCallbacks
from ivy.evm.evm_core import EVMCore
from ivy.evm.evm_state import StateAccess
from ivy.evm.evm_structures import ContractData, Log, Message, Environment
from ivy.exceptions import Revert


_EDGE_NODE_TYPES = (
    ast.AnnAssign,
    ast.Assign,
    ast.AugAssign,
    ast.Expr,
    ast.Log,
    ast.If,
    ast.Assert,
    ast.Raise,
    ast.For,
    ast.Return,
    ast.Break,
    ast.Continue,
    ast.Pass,
)


class VyperInterpreter(ExprVisitor, StmtVisitor, EVMCallbacks):
    _TIMEOUT_CHECK_MASK = 0xFF  # check every 256 visited nodes

    def __init__(self, env: Environment):
        self.evm = EVMCore(callbacks=self, env=env)
        self.state: StateAccess = self.evm.state
        self.builtins = BuiltinRegistry(self.evm, self.state)
        self.tracers: list[Tracer] = []

        # Edge-lite coverage needs per-message-call state: edges must not span
        # across separate ABI calls or across nested external calls/contracts.
        self._edge_stack: list[Optional[int]] = []
        self._prev_edge_node_id: Optional[int] = None
        self._timeout_deadline_ns: Optional[int] = None
        self._timeout_tick: int = 0

    def set_call_timeout_seconds(self, timeout_seconds: float) -> None:
        if timeout_seconds <= 0:
            self.clear_call_timeout()
            return
        self._timeout_deadline_ns = time.perf_counter_ns() + int(
            timeout_seconds * 1_000_000_000
        )
        self._timeout_tick = 0

    def clear_call_timeout(self) -> None:
        self._timeout_deadline_ns = None
        self._timeout_tick = 0

    def _check_call_timeout(self) -> None:
        deadline_ns = self._timeout_deadline_ns
        if deadline_ns is None:
            return
        self._timeout_tick += 1
        if self._timeout_tick & self._TIMEOUT_CHECK_MASK:
            return
        if time.perf_counter_ns() >= deadline_ns:
            raise CallTimeout("Call exceeded time budget")

    def check_call_timeout(self) -> None:
        self._check_call_timeout()

    def on_state_committed(self) -> None:
        for tracer in self.tracers:
            tracer.on_state_modified()

    def _commit_top_level_call(self, execution_output):
        if execution_output.error is None:
            self.evm._pending_accounts_to_delete.update(
                execution_output.accounts_to_delete
            )
        if self.evm.journal.pop_state_committed():
            self.on_state_committed()

    def deploy(self, sender, compiler_data, value, calldata):
        contract_address = self.evm.generate_create_address(sender)
        code = ContractData(compiler_data)

        self.state.env.caller = sender
        self.state.env.origin = sender

        message = Message(
            caller=sender,
            to=b"",
            create_address=contract_address,
            value=value,
            data=calldata,
            code_address=b"",
            code=code,
            depth=0,
            is_static=False,
        )

        execution_output = self.evm.process_create_message(message)
        self._commit_top_level_call(execution_output)
        return contract_address, execution_output

    def execute_code(self, to, sender, value, calldata, is_static):
        code = self.state.get_code(to)

        self.state.env.caller = sender
        self.state.env.origin = sender

        message = Message(
            caller=sender,
            to=to,
            create_address=None,
            value=value,
            data=calldata,
            code_address=to,
            code=code,
            depth=0,
            is_static=is_static,
        )

        execution_output = self.evm.process_message(message)
        self._commit_top_level_call(execution_output)
        return execution_output

    @property
    def deployer(self):
        from ivy.frontend.vyper_contract import VyperDeployer

        return VyperDeployer

    @property
    def fun_ctx(self):
        return self.current_context.current_fun_context()

    @property
    def globals(self):
        assert self.current_context.global_vars is not None
        return self.current_context.global_vars

    @property
    def env(self):
        return self.evm.state.env

    @property
    def msg(self):
        return self.current_context.msg

    @property
    def current_address(self):
        if self.current_context.msg.to == b"":
            return self.current_context.msg.create_address

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
        # separate address allocation from variable creation
        nonreentrant, globals, positions = allocator.allocate_addresses(module_t)

        for var in globals:
            get_location = self.storage_getter_from_varinfo(var)
            name = var.decl_node.target.id
            position = positions[var]
            if var.is_constant:
                value = self.visit(var.decl_node.value)  # the value of the constant
                self.globals.new_variable(
                    var, get_location, position, initial_value=value, name=name
                )
                continue

            self.globals.new_variable(var, get_location, position, name=name)

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
        try:
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
        except Exception:
            # Ensure interpreter state is unwound if prologue setup fails.
            self._pop_scope()
            self._pop_fun_ctx()
            raise

    def _epilogue(self, func_t):
        if func_t.nonreentrant:
            self.unlock(func_t.mutability)
        self._pop_scope()
        self._pop_fun_ctx()

    def execute_init_function(self, func_t):
        self._push_edge_state()
        try:
            _, calldata_args_t = compute_call_abi_data(func_t, 0)
            min_size = calldata_args_t.abi_type.static_size()
            self._min_calldata_size_check(min_size)
            # msg.data doesn't contain bytecode in ivy's case
            args = abi_decode(calldata_args_t, self.msg.data, from_calldata=True)

            self._check_payability(func_t)
            return self._execute_function(func_t, args)
        finally:
            self._pop_edge_state()

    def _min_calldata_size_check(self, min_size):
        if len(self.msg.data) < min_size:
            raise BufferError(f"Provided calldata is too small, min_size is {min_size}")

    def _check_payability(self, func_t):
        if not func_t.is_payable and self.msg.value != 0:
            raise PayabilityViolation(f"Function {func_t.name} is not payable")

    def _push_edge_state(self) -> None:
        self._edge_stack.append(self._prev_edge_node_id)
        self._prev_edge_node_id = None

    def _pop_edge_state(self) -> None:
        self._prev_edge_node_id = self._edge_stack.pop()

    def _on_node(self, node) -> None:
        self._check_call_timeout()

        tracers = self.tracers
        if not tracers:
            return
        if not isinstance(node, _EDGE_NODE_TYPES):
            return

        addr = self.current_address
        node_id = getattr(node, "node_id", None)
        if node_id is None:
            return

        prev_node_id = self._prev_edge_node_id
        if prev_node_id is not None:
            for tracer in tracers:
                tracer.on_edge(addr, prev_node_id, node_id)

        self._prev_edge_node_id = node_id

        for tracer in tracers:
            tracer.on_node(addr, node)

    def _on_branch(self, node, taken: bool) -> None:
        tracers = self.tracers
        if not tracers:
            return
        if getattr(node, "node_id", None) is None:
            return
        addr = self.current_address
        for tracer in tracers:
            tracer.on_branch(addr, node, taken)

    def _on_boolop(self, node, op: str, evaluated_count: int, result: bool) -> None:
        tracers = self.tracers
        if not tracers:
            return
        if getattr(node, "node_id", None) is None:
            return
        addr = self.current_address
        for tracer in tracers:
            tracer.on_boolop(addr, node, op, evaluated_count, result)

    def _on_loop(self, node, iteration_count: int) -> None:
        tracers = self.tracers
        if not tracers:
            return
        if getattr(node, "node_id", None) is None:
            return
        addr = self.current_address
        for tracer in tracers:
            tracer.on_loop(addr, node, iteration_count)

    def dispatch(self):
        assert self.current_context.entry_points is not None
        assert self.current_context.contract is not None

        self._push_edge_state()
        try:
            try:
                if len(self.msg.data) < 4:
                    raise FunctionNotFound()

                selector = self.msg.data[:4]

                entry_points = self.current_context.entry_points

                if selector not in entry_points:
                    raise FunctionNotFound()
                else:
                    entry_point = entry_points[selector]

                self._min_calldata_size_check(entry_point.calldata_min_size)

                func_t = entry_point.function
                args = abi_decode(
                    entry_point.calldata_args_t, self.msg.data[4:], from_calldata=True
                )

            except FunctionNotFound as e:
                if self.current_context.contract.fallback:
                    func_t = self.current_context.contract.fallback
                    args = ()
                else:
                    raise e

            self._check_payability(func_t)
            self._execute_external_function(func_t, args)
        finally:
            self._pop_edge_state()

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

        if func_t.do_raw_return:
            assert isinstance(typ, BytesT)
            assert isinstance(ret, bytes)
            self.current_context.execution_output.output = ret
            return None

        typ = calculate_type_for_external_return(typ)
        # from https://github.com/vyperlang/vyper/blob/a1af967e675b72051cf236f75e1104378fd83030/vyper/codegen/core.py#L694
        is_tuple_like = isinstance(ret, (tuple, IvyTuple))
        output = (ret,) if (not is_tuple_like or len(ret) <= 1) else ret
        self.current_context.execution_output.output = abi_encode(typ, output)

    def _execute_function(self, func_t, args):
        # TODO: rewrite this using a ctx manager?
        self._prologue(func_t, args)
        ret = None
        success = False

        try:
            for stmt in func_t.decl_node.body:
                try:
                    self.visit(stmt)
                except ReturnException as e:
                    ret = e.value
                    break
            success = True
        finally:
            if success:
                self._epilogue(func_t)
            else:
                # On failure, unwind interpreter scopes without touching EVM state.
                self._pop_scope()
                self._pop_fun_ctx()

        if func_t.is_deploy:
            return self.current_context.contract
        if getattr(func_t, "do_raw_return", False):
            assert isinstance(ret, bytes)
            return ret
        return boxed(ret)

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
        varinfo = node._expr_info.var_info if node else None  # type: ignore[attr-defined]
        is_global = self._is_global_var(varinfo)
        return varinfo, is_global

    def get_variable(self, name: str, node: Optional[ast.VyperNode] = None):
        varinfo, is_global = self._resolve_variable_info(node)
        if is_global:
            assert varinfo is not None
            res = self.globals[varinfo].value
        else:
            res = self.memory[name]
        assert res is not None
        return res

    def set_variable(self, name: str, value, node: Optional[ast.VyperNode] = None):
        boxed(value)
        varinfo, is_global = self._resolve_variable_info(node)
        if is_global:
            assert varinfo is not None
            value = box_value(value, varinfo.typ)
            var = self.globals[varinfo]
            var.value = value
        else:
            if name in self.memory:
                value = box_value(value, self.memory[name].typ)
            elif node is not None:
                value = box_value(value, node._expr_info.typ)  # type: ignore[attr-defined]
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

    def _assign_target(self, target, value, _container=None, _index=None, _obj=None):
        """
        Assign value to target. For augmented assignment, pass pre-evaluated
        _container/_index (for Subscript) or _obj (for Attribute) to avoid
        re-evaluating expressions with side effects.
        """
        if isinstance(target, ast.Tuple):
            if not isinstance(value, (tuple, IvyTuple)):
                raise TypeError("Cannot unpack non-iterable to tuple")
            seq = tuple(value) if isinstance(value, IvyTuple) else value
            if len(target.elements) != len(seq):
                raise ValueError("Mismatch in number of items to unpack")
            for t, v in zip(target.elements, seq):
                self._assign_target(t, v)
        elif isinstance(target, ast.Name):
            self.set_variable(target.id, value, target)
        elif isinstance(target, ast.Subscript):
            if _container is not None and _index is not None:
                container, index = _container, _index
            else:
                container = self.visit(target.value)
                index = self.visit(target.slice)
            loc = target._expr_info.location
            if isinstance(container, (Map, StaticArray, DynamicArray)):
                container.__setitem__(index, value, loc)
            else:
                assert isinstance(container, IvyTuple)
                container.__setitem__(index, value, loc)
        elif isinstance(target, ast.Attribute):
            typ = target.value._metadata["type"]
            if isinstance(typ, (SelfT, ModuleT)):
                self.set_variable(target.attr, value, target)
            else:
                assert isinstance(typ, StructT)
                if _obj is not None:
                    obj = _obj
                else:
                    obj = self.visit(target.value)
                loc = target._expr_info.location
                assert isinstance(obj, Struct)
                obj.__setitem__(target.attr, value, loc)
        else:
            raise NotImplementedError(f"Assignment to {type(target)} not implemented")

    def _handle_address_variable(self, node: ast.Attribute):
        if node.attr in ("code", "codesize", "codehash"):
            raise UnsupportedFeature(
                "Address code access (code/codesize/codehash) is unsupported in Ivy: "
                "https://github.com/cyberthirst/ivy/issues/21"
            )
        # x.address
        if node.attr == "address":
            return self.visit(node.value)
        # x.balance: balance of address x
        if node.attr == "balance":
            addr = self.visit(node.value)
            return box_value_from_node(node, self.state.get_balance(addr))
        # x.codesize: codesize of address x
        elif node.attr == "codesize" or node.attr == "is_contract":
            addr = self.visit(node.value)
            contract_data = self.state.get_code(addr)
            if node.attr == "codesize":
                if contract_data is None:
                    return box_value_from_node(node, 0)
                return box_value_from_node(
                    node, len(contract_data.compiler_data.bytecode_runtime)
                )
            else:
                return box_value_from_node(node, contract_data is not None)
        # x.codehash: keccak of address x
        elif node.attr == "codehash":
            addr = self.visit(node.value)
            contract_data = self.state.get_code(addr)
            if contract_data is None:
                # Check if account exists (has balance, nonce, or was touched)
                if not self.state.has_account(addr):
                    # Non-existent account returns 0 per EIP-1052
                    return box_value_from_node(node, b"\x00" * 32)
                # EOA (existing account with no code) returns keccak of empty bytes
                return box_value_from_node(node, keccak256(b""))
            return box_value_from_node(
                node, keccak256(contract_data.compiler_data.bytecode_runtime)
            )
        # x.code: codecopy/extcodecopy of address x
        elif node.attr == "code":
            # code has special semantics - type annotation is Bytes[0] placeholder
            # but actual value can be any length, so don't box it
            addr = self.visit(node.value)
            contract_data = self.state.get_code(addr)
            if contract_data is None:
                return b""
            return contract_data.compiler_data.bytecode_runtime
        else:
            assert False, "unreachable"

    def _handle_env_variable(self, node: ast.Attribute):
        key = f"{node.value.id}.{node.attr}"  # type: ignore[attr-defined]
        if key == "msg.sender":
            return self.msg.caller
        elif key == "msg.data":
            # msg.data has special semantics - type annotation is Bytes[0] placeholder
            # but actual value can be any length, so don't box it
            return self.msg.data
        elif (
            key == "msg.value"
        ):  # TODO check payble (context and self.context.is_payable:)
            return box_value_from_node(node, self.msg.value)
        elif key in ("msg.gas", "msg.mana"):
            raise GasReference()
        elif key == "block.prevrandao":
            return box_value_from_node(node, self.env.prev_randao)
        elif key == "block.difficulty":
            return box_value_from_node(node, int.from_bytes(self.env.prev_randao, "big"))
        elif key == "block.timestamp":
            return box_value_from_node(node, self.env.time)
        elif key == "block.coinbase":
            return self.env.coinbase
        elif key == "block.number":
            return box_value_from_node(node, self.env.block_number)
        elif key == "block.gaslimit":
            raise GasReference()
        elif key == "block.basefee":
            raise GasReference()
        elif key == "block.blobbasefee":
            raise GasReference()
        elif key == "block.prevhash":
            # Return the hash of the previous block (block_hashes[-1])
            # If no previous block exists (block_number == 0 or empty block_hashes), return 0
            if self.env.block_number == 0 or len(self.env.block_hashes) == 0:
                return box_value_from_node(node, b"\x00" * 32)
            return box_value_from_node(node, self.env.block_hashes[-1])
        elif key == "tx.origin":
            return self.env.origin
        elif key == "tx.gasprice":
            raise GasReference()
        elif key == "chain.id":
            return box_value_from_node(node, self.env.chain_id)
        else:
            assert False, "unreachable"

    def _inside_minimal_proxy(self):
        assert self.current_context.contract is not None
        decl_node = self.current_context.contract.module_t.decl_node
        assert decl_node is not None
        return "is_minimal_proxy" in decl_node._metadata

    def _encode_log_topics(self, event: EventT, arg_nodes: list[tuple[Any, VyperType]]):
        event_id = abi_encode(UINT256_T, event.event_id)
        topics = [event_id]

        for arg, typ in arg_nodes:
            if typ._is_prim_word:
                value = abi_encode(typ, arg)
            elif isinstance(typ, _BytestringT):
                if isinstance(arg, str):
                    arg = arg.encode("utf-8")
                value = keccak256(arg)
            else:
                # this check is done in vyper's codegen so we need to replicate it
                # TODO block at higher level
                raise TypeMismatch("Event indexes may only be value types", event)
            topics.append(value)

        return topics

    def _log(self, event: EventT, args):
        topic_nodes: list[tuple[Any, VyperType]] = []
        data_nodes: list[tuple[Any, VyperType]] = []
        typs = event.members.values()
        assert len(args) == len(event.indexed) and len(typs) == len(args)
        for arg, typ, is_indexed in zip(args, typs, event.indexed):
            if is_indexed:
                topic_nodes.append((arg, typ))
            else:
                data_nodes.append((arg, typ))

        topics = self._encode_log_topics(event, topic_nodes)

        data_values = tuple(arg for arg, _ in data_nodes)
        data_typs = TupleT(tuple(typ for _, typ in data_nodes))

        encoded_data = abi_encode(data_typs, data_values)

        assert len(topics) <= 4, "too many topics"  # sanity check

        if self.msg.is_static:
            raise StaticCallViolation("LOG in static context")

        address = self.current_address
        self.state.current_output.logs.append(
            Log(address.canonical_address, topics, encoded_data)
        )

    def generic_call_handler(
        self,
        call: ast.Call,
        args,
        kws,
        target: Optional[Address] = None,
        is_static: bool = False,
    ):
        # `None` is a special case: range is not assigned `type` in Vyper's frontend
        func_t = call.func._metadata.get("type", None)

        if func_t is None or isinstance(func_t, BuiltinFunctionT):
            id = call.func.id  # type: ignore[attr-defined]
            result = self.builtins.get(id).execute(*args, **kws)
            if id == "range":
                return result  # Special case: untyped iterator
            return boxed(box_value_from_node(call, result))

        if isinstance(func_t, TYPE_T):
            # struct & interface constructors
            typedef = func_t.typedef
            if isinstance(typedef, InterfaceT):
                # TODO should we return an address here? or an interface object wrapping the address?
                # we will likely need the attrs of the interface..
                assert len(args) == 1
                return boxed(args[0])
            elif isinstance(typedef, EventT):
                _args = args if len(args) > 0 else kws.values()
                self._log(typedef, _args)
                return None
            else:
                assert isinstance(typedef, StructT) and len(args) == 0
                return boxed(Struct(typedef, kws))

        if isinstance(func_t, MemberFunctionT):
            if func_t.name == "__at__":
                assert len(args) == 1
                return boxed(args[0])
            darray = self.visit(call.func.value)  # type: ignore[attr-defined]
            assert isinstance(darray, DynamicArray)
            loc = call.func.value._expr_info.location  # type: ignore[attr-defined]
            if func_t.name == "append":
                assert len(args) == 1
                darray.append(args[0], loc)
                return None
            else:
                assert func_t.name == "pop" and len(args) == 0
                return boxed(darray.pop(loc))

        assert isinstance(func_t, ContractFunctionT)

        if func_t.is_external:
            assert target is not None and isinstance(func_t, ContractFunctionT)
            return boxed(
                self._external_function_call(func_t, args, kws, is_static, target)
            )

        assert func_t.is_internal or func_t.is_deploy
        if func_t.is_deploy:
            return self._execute_function(func_t, args)
        return boxed(self._execute_function(func_t, args))

    def _external_function_call(
        self, func_t: ContractFunctionT, args, kwargs, is_static: bool, target: Address
    ):
        if "gas" in kwargs:
            raise GasReference()

        skip_contract_check = kwargs.get("skip_contract_check", False)
        if not skip_contract_check:
            code = self.state.get_code(target)
            if code is None:
                raise Revert(message=f"Account at {target} does not have code")

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
            to_eval = kwargs["default_return_value"]
            return self.deep_copy_visit(to_eval)

        typ = func_t.return_type

        # TODO maybe this return is premature and we should validate the return size
        if typ is None:
            return None

        typ = calculate_type_for_external_return(typ)
        abi_typ = typ.abi_type

        max_return_size = abi_typ.size_bound()

        actual_output_size = min(max_return_size, len(returndata))
        to_decode = returndata[:actual_output_size]

        # NOTE: abi_decode validates return size
        decoded = abi_decode(typ, to_decode)
        if not needs_external_call_wrap(func_t.return_type):
            return decoded
        # external call wrap means result is wrapped in a 1-tuple
        assert isinstance(decoded, IvyTuple)
        assert len(decoded) == 1
        return decoded[0]

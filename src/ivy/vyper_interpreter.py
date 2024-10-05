from typing import Optional, Type
import inspect


from eth._utils.address import generate_contract_address

import vyper.ast.nodes as ast
from vyper.semantics.types import VyperType
from vyper.semantics.types.module import ModuleT
from vyper.semantics.types.function import (
    ContractFunctionT,
)
from vyper.builtins._signatures import BuiltinFunctionT
from vyper.codegen.core import calculate_type_for_external_return

from ivy.expr import ExprVisitor
from titanoboa.boa.util.abi import Address

from ivy.evm_structures import Account, Environment, Message, ContractData
from ivy.stmt import ReturnException, StmtVisitor
from ivy.evaluator import VyperEvaluator
from ivy.variable import GlobalVariable
from ivy.context import ExecutionContext
import ivy.builtins as vyper_builtins
from ivy.utils import compute_call_abi_data
from ivy.abi import abi_decode, abi_encode


class EVMException(Exception):
    pass


class VyperInterpreter(ExprVisitor, StmtVisitor):
    execution_ctxs: list[ExecutionContext]
    env: Optional[Environment]
    contract: Optional[ContractData]
    evaluator: Type[VyperEvaluator]

    def __init__(self):
        self.state = {}
        self.returndata = None
        self.evaluator = VyperEvaluator
        self.execution_ctxs = []
        self.builtins = {}
        self._collect_builtins()
        self.env = None

    def _collect_builtins(self):
        for name, func in inspect.getmembers(vyper_builtins, inspect.isfunction):
            if name.startswith("builtin_"):
                builtin_name = name[8:]  # Remove 'builtin_' prefix
                self.builtins[builtin_name] = func

    @property
    def deployer(self):
        from ivy.frontend.vyper_contract import VyperDeployer

        return VyperDeployer

    @property
    def fun_ctx(self):
        return self.execution_ctxs[-1].current_fun_context()

    @property
    def globals(self):
        return self.exec_ctx.contract.global_vars

    @property
    def msg(self):
        return self.execution_ctxs[-1].msg

    @property
    def current_address(self):
        return self.exec_ctx.msg.to

    @property
    def memory(self):
        # function variables are located in memory
        # fun ctx has set/get methods which are used to access the variables from scopes
        return self.fun_ctx

    @property
    def exec_ctx(self):
        return self.execution_ctxs[-1]

    def get_nonce(self, address):
        if address not in self.state:
            self.state[address] = Account(0, 0, {}, {}, None)
        return self.state[address].nonce

    def increment_nonce(self, address):
        assert address in self.state
        self.state[address].nonce += 1

    def _push_fun_ctx(self, func_t):
        self.execution_ctxs[-1].push_fun_context(func_t)

    def _pop_fun_ctx(self):
        self.execution_ctxs[-1].pop_fun_context()

    def _push_scope(self):
        self.execution_ctxs[-1].push_scope()

    def _pop_scope(self):
        self.execution_ctxs[-1].pop_scope()

    def generate_create_address(self, sender):
        nonce = self.get_nonce(sender.canonical_address)
        self.increment_nonce(sender.canonical_address)
        return Address(generate_contract_address(sender.canonical_address, nonce))

    def deploy(
        self,
        sender: Address,
        to: Address,
        module: ast.Module,
        value: int,
        calldata=None,  # abi-encoded constructor args
    ):
        module_t = module._metadata["type"]
        assert isinstance(module_t, ModuleT)

        message = Message(
            caller=sender,
            to=b"",
            create_address=to,
            value=value,
            data=calldata,
            code_address=to,
            code=module,
            depth=0,
            is_static=False,
        )

        env = Environment(
            caller=sender,
            block_hashes=[],
            origin=to,
            coinbase=sender,
            number=0,
            time=0,
            prev_randao=b"",
            chain_id=0,
        )

        error = self.process_create_message(message, env)

        if error:
            raise error

    def execute_code(
        self,
        sender: Address,
        to: Address,
        code_address: Address,
        value: int,
        calldata: bytes,
        is_static: bool = False,
    ):
        code = self.get_code(to)

        message = Message(
            caller=sender,
            to=to,
            create_address=to,
            value=value,
            data=calldata,
            code_address=code_address,
            code=code,
            depth=0,
            is_static=is_static,
        )

        env = Environment(
            caller=sender,
            block_hashes=[],
            origin=sender,
            coinbase=sender,
            number=0,
            time=0,
            prev_randao=b"",
            chain_id=0,
        )

        output, error = self.process_message(message, env)

        if error:
            raise error

        return output

    def process_message(
        self, message: Message, env: Environment
    ) -> tuple[Optional[bytes], Optional[Exception]]:
        account = self.state.get(message.to, Account(0, 0, {}, {}, None))
        exec_ctx = ExecutionContext(
            account,
            message,
            account.contract_data.module if account.contract_data else None,
        )
        self.execution_ctxs.append(exec_ctx)

        output = None
        error = None

        try:
            if message.value > 0:
                if self.state[message.caller].balance < message.value:
                    raise EVMException("Insufficient balance for transfer")
                self.state[message.caller].balance -= message.value
                self.state[message.to].balance += message.value

            if message.code:
                self._extcall()

            output = self.exec_ctx.output

        except Exception as e:
            # TODO rollback the journal
            error = e

        finally:
            self.execution_ctxs.pop()

        return output, error

    def process_create_message(
        self, message: Message, env: Environment
    ) -> Optional[Exception]:
        if message.create_address in self.state:
            raise EVMException("Address already taken")

        new_account = Account(0, message.value, {}, {}, None)
        self.state[message.create_address] = new_account

        module_t = message.code._metadata["type"]
        assert isinstance(module_t, ModuleT)
        new_account.contract_data = ContractData(module_t)
        exec_ctx = ExecutionContext(new_account, message, module_t)
        self.execution_ctxs.append(exec_ctx)

        error = None

        try:
            if message.value > 0:
                if self.state[message.caller].balance < message.value:
                    raise EVMException("Insufficient balance for contract creation")
                self.state[message.caller].balance -= message.value

            for decl in module_t.variable_decls:
                name = decl.target.id
                typ = decl._metadata["type"]
                loc = self.get_location_from_decl(decl)
                self._new_variable(name, typ, loc)

            if module_t.init_function is not None:
                self._execute_function(module_t.init_function, message.data)

        except Exception as e:
            # TODO rollback the journal
            error = e
            del self.state[message.create_address]

        finally:
            self.execution_ctxs.pop()

        return error

    def get_code(self, address):
        return self.state[address].contract_data.module

    def _dispatch(self, selector):
        entry_points = self.exec_ctx.contract.entry_points

        if selector not in entry_points:
            # TODO check fallback
            # TODO rollback the journal
            raise Exception(f"function {selector} not found")
        else:
            entry_point = entry_points[selector]

        if entry_point.function.is_payable:
            if self.msg.value != 0:
                # TODO raise and rollback
                pass

        # check args

        # check decorators

        return entry_point

    def _prologue(self, func_t, args):
        self._push_fun_ctx(func_t)
        self._push_scope()

        # TODO handle reentrancy lock
        func_args = func_t.arguments
        for arg, param in zip(args, func_args):
            self._new_variable(param.name, param.typ)
            self.set_variable(param.name, arg)

        # check if we need to assign default values
        if len(args) < len(func_args):
            for param in func_args[len(args) :]:
                self._new_variable(param.name, param.typ)
                default_value = self.visit(param.default_value)
                self.set_variable(param.name, default_value)

    def _epilogue(self, *args):
        # TODO handle reentrancy lock
        # TODO handle return value
        self._pop_scope()
        self._pop_fun_ctx()
        pass

    def _exec_body(self):
        for stmt in self.fun_ctx.function.decl_node.body:
            try:
                self.visit(stmt)
            except ReturnException as e:
                # TODO handle the return value
                self.exec_ctx.output = e.value
                break

    def _execute_function(self, func_t, args):
        self._prologue(func_t, args)
        self._exec_body()
        self._epilogue()

        return self.exec_ctx.output

    def _extcall(self):
        if len(self.msg.data) < 4:
            # TODO goto fallback or revert
            pass

        selector = self.msg.data[:4]
        entry_point = self._dispatch(selector)

        if entry_point.calldata_min_size > len(self.msg.data):
            raise BufferError(
                f"Provided calldata is too small, min_size is {entry_point.calldata_min_size}"
            )

        func_t = entry_point.function
        args = abi_decode(entry_point.calldata_args_t, self.msg.data[4:])

        self.exec_ctx.function = func_t

        self._execute_function(func_t, args)

        # abi-encode output
        typ = self.exec_ctx.function.return_type
        if typ is None:
            assert self.exec_ctx.output is None
            return None
        typ = calculate_type_for_external_return(typ)
        output = self.exec_ctx.output
        # from https://github.com/vyperlang/vyper/blob/a1af967e675b72051cf236f75e1104378fd83030/vyper/codegen/core.py#L694
        output = (
            (output,) if (not isinstance(output, tuple) or len(output) <= 1) else output
        )
        self.exec_ctx.output = abi_encode(typ, output)

    def get_variable(self, name: str):
        print(self.memory)
        if name in self.globals:
            return self.globals[name].value
        else:
            return self.memory[name]

    def set_variable(self, name: str, value):
        print(f"assigning {name} = {value}")
        if name in self.globals:
            var = self.globals[name]
            var.value = value
        else:
            self.memory[name] = value

    def _assign_target(self, target, value):
        if isinstance(target, ast.Name):
            self.set_variable(target.id, value)
        elif isinstance(target, ast.Tuple):
            if not isinstance(value, tuple):
                raise TypeError("Cannot unpack non-iterable to tuple")
            if len(target.elts) != len(value):
                raise ValueError("Mismatch in number of items to unpack")
            for t, v in zip(target.elts, value):
                self._assign_target(t, v)
        elif isinstance(target, ast.Subscript):
            container = self.visit(target.value)
            index = self.visit(target.slice)
            container[index] = value
        elif isinstance(target, ast.Attribute):
            if isinstance(target.value, ast.Name) and target.value.id == "self":
                try:
                    self.set_variable(target.attr, value)
                except KeyError:
                    pass
            else:
                obj = self.visit(target.value)
                setattr(obj, target.attr, value)
        else:
            raise NotImplementedError(f"Assignment to {type(target)} not implemented")

    def get_location_from_decl(self, decl: ast.VariableDecl):
        if decl.is_immutable:
            return self.exec_ctx.contract.immutables
        elif decl.is_constant:
            return self.exec_ctx.contract.constants
        elif decl.is_transient:
            return self.exec_ctx.transient
        else:
            return self.exec_ctx.storage

    def _new_variable(self, name: str, typ: VyperType, global_loc: dict = None):
        if global_loc is not None:
            var = GlobalVariable(name, typ, global_loc)
            assert name not in self.globals
            self.globals[name] = var
        else:
            self.memory.new_variable(name, typ)

    def handle_call(
        self,
        call: ast.Call,
        args,
        kws,
        target: Optional[Address] = None,
        is_static: Optional[bool] = None,
    ):
        # `None` is a special case: range is not assigned `type` in Vyper's frontend
        func_t = call.func._metadata.get("type", None)

        if func_t is None or isinstance(func_t, BuiltinFunctionT):
            return self.builtins[call.func.id](*args, **kws)

        if func_t.is_external:
            assert target is not None
            assert isinstance(func_t, ContractFunctionT)
            return self.handle_external_call(func_t, args, kws, is_static, target)

        assert func_t.is_internal
        return self._execute_function(func_t, args)

    # TODO add support for delegatecall
    def handle_external_call(
        self, func_t: ContractFunctionT, args, kwargs, is_static: bool, target: Address
    ):
        num_kwargs = len(args) - func_t.n_positional_args

        selector, calldata_args_t = compute_call_abi_data(func_t, num_kwargs)

        data = selector
        data += abi_encode(calldata_args_t, args)

        code = self.get_code(target)

        msg = Message(
            caller=self.msg.to,
            to=target,
            create_address=None,
            value=kwargs.get("value", 0),
            data=data,
            code_address=target,
            code=code,
            depth=self.exec_ctx.msg.depth + 1,
            is_static=is_static,
        )

        output, error = self.process_message(msg, self.env)

        # TODO: for raw_call and revert_on_failure=False this doesn't hold
        if error:
            raise error

        self.exec_ctx.returndata = output if output is not None else b""

        if len(self.exec_ctx.returndata) == 0 and "default_return_value" in kwargs:
            return kwargs["default_return_value"]

        typ = func_t.return_type
        typ = calculate_type_for_external_return(typ)
        abi_typ = typ.abi_type

        max_return_size = abi_typ.size_bound()

        actual_output_size = min(max_return_size, len(self.exec_ctx.returndata))
        to_decode = self.exec_ctx.returndata[:actual_output_size]

        # NOTE: abi_decode implicitly checks minimum return size
        decoded = abi_decode(typ, to_decode)
        assert len(decoded) == 1
        # unwrap the tuple
        return decoded[0]

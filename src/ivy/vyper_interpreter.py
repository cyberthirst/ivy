from typing import Any, Optional, Union, Tuple
import inspect

import vyper.ast.nodes as ast
from vyper.semantics.types.module import ModuleT
from vyper.semantics.types.function import (
    ContractFunctionT,
    _FunctionArg,
)
from vyper.semantics.data_locations import DataLocation
from vyper.builtins._signatures import BuiltinFunctionT
from vyper.codegen.core import calculate_type_for_external_return

from titanoboa.boa.util.abi import Address

from ivy.base_interpreter import BaseInterpreter
from ivy.evm_structures import Account, Environment, Message, ContractData
from ivy.stmt import ReturnException
from ivy.evaluator import VyperEvaluator
from ivy.context import ExecutionContext, Variable
import ivy.builtins as vyper_builtins
from ivy.utils import compute_call_abi_data
from ivy.abi import abi_decode, abi_encode


class EVMException(Exception):
    pass


class VyperInterpreter(BaseInterpreter):
    contract: Optional[ContractData]
    evaluator: VyperEvaluator
    vars: dict[str, Any]

    def __init__(self):
        super().__init__()
        self.returndata = None
        self.evaluator = VyperEvaluator()
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

    def get_nonce(self, address):
        if address not in self.state:
            self.state[address] = Account(0, 0, {}, {}, None)
        return self.state[address].nonce

    def increment_nonce(self, address):
        assert address in self.state
        self.state[address].nonce += 1

    @property
    def fun_ctx(self):
        return self.execution_ctxs[-1].current_fun_context()

    @property
    def msg(self):
        return self.execution_ctxs[-1].msg

    @property
    def current_address(self):
        return self.exec_ctx.msg.to

    def process_message(
        self, message: Message, env: Environment
    ) -> Tuple[Optional[bytes], Optional[Exception]]:
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
                self._new_variable(decl)

            if module_t.init_function is not None:
                self._execute_function(module_t.init_function, message.data)

        except Exception as error:
            # TODO rollback the journal
            self.exec_ctx.error = error
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

        for arg, param in zip(args, func_t.arguments):
            self._new_variable(param)
            self.set_variable(param.name, arg)

    def _epilogue(self, *args):
        # TODO handle reentrancy lock
        # TODO handle return value
        self._pop_scope()
        self._pop_fun_ctx()
        # TODO register arguments as local variables
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
        typ = calculate_type_for_external_return(typ)
        output = self.exec_ctx.output
        output = (
            (output,) if (not isinstance(output, tuple) or len(output) <= 1) else output
        )
        self.exec_ctx.output = abi_encode(typ, output)

    @property
    def var_locations(self):
        return (
            self.fun_ctx,
            self.exec_ctx.storage,
            self.exec_ctx.transient,
            self.exec_ctx.immutables,
            self.exec_ctx.constants,
        )

    def get_variable(self, name: str):
        for loc in self.var_locations:
            if name in loc:
                return loc[name].value
        else:
            raise KeyError(f"Variable {name} not found")

    def set_variable(self, name: str, value):
        for loc in self.var_locations:
            if name in loc:
                loc[name].value = value
                break
        else:
            raise KeyError(f"Variable {name} not found")

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

    def _new_variable(self, target: Union[ast.Name, ast.VariableDecl, _FunctionArg]):
        if isinstance(target, ast.Name):
            info = target._expr_info
            var = Variable(
                self.evaluator.default_value(info.typ), info.typ, DataLocation.MEMORY
            )
            self.fun_ctx[target.id] = var
        elif isinstance(target, _FunctionArg):
            var = Variable(
                self.evaluator.default_value(target.typ),
                target.typ,
                DataLocation.MEMORY,
            )
            self.fun_ctx[target.name] = var
        elif isinstance(target, ast.VariableDecl):
            # TODO handle public variables
            id = target.target.id
            typ = target._metadata["type"]
            default_value = self.evaluator.default_value(typ)
            if target.is_immutable:
                self.exec_ctx.immutables[id] = Variable(
                    default_value, typ, DataLocation.CODE
                )
            elif target.is_constant:
                self.exec_ctx.contract.constants[id] = Variable(
                    self.visit(target.value), typ, DataLocation.CODE
                )
            elif target.is_transient:
                self.exec_ctx.transient[id] = Variable(
                    default_value, typ, DataLocation.TRANSIENT
                )
            else:  # storage
                self.exec_ctx.storage[id] = Variable(
                    default_value, typ, DataLocation.STORAGE
                )
        else:
            raise RuntimeError(f"Cannot create variable for {type(target)}")

    def _new_internal_variable(self, node):
        info = node._expr_info
        typ = info.typ
        var = Variable(self.evaluator.default_value(typ), typ, DataLocation.MEMORY)
        self.exec_ctx.new_variable(node.id, var)

    def handle_call(
        self,
        call: ast.Call,
        args,
        kws,
        target: Optional[Address] = None,
        is_static: Optional[bool] = None,
    ):
        # special case: range not assigned `type`
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

        if error:
            raise error

        if len(output) == 0:
            if "default_return_value" in kwargs:
                return kwargs["default_return_value"]
            return None

        typ = func_t.return_type
        typ = calculate_type_for_external_return(typ)
        decoded = abi_decode(typ, output)
        assert len(decoded) == 1
        # unwrap the tuple
        return decoded[0]

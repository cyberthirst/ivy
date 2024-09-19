from typing import Any, Optional, Union
from abc import abstractmethod
import inspect

import vyper.ast.nodes as ast
from vyper.semantics.types.module import ModuleT
from vyper.semantics.types.function import ContractFunctionT, _FunctionArg
from vyper.semantics.data_locations import DataLocation
from vyper.builtins._signatures import BuiltinFunctionT

from titanoboa.boa.util.abi import Address, abi_encode

import eth.constants as constants
from eth._utils.address import generate_contract_address

from ivy.evm_structures import Account, Environment, Message, ContractData
from ivy.expr import ExprVisitor
from ivy.stmt import StmtVisitor, ReturnException
from ivy.evaluator import VyperEvaluator
from ivy.context import ExecutionContext, Variable
import ivy.builtins as vyper_builtins


class BaseInterpreter(ExprVisitor, StmtVisitor):
    execution_ctxs: list[ExecutionContext]
    env: Optional[Environment]

    def __init__(self):
        self.state = {}

    def get_code(self, address):
        pass

    @property
    @abstractmethod
    def deployer(self):
        pass

    @abstractmethod
    def _extcall(self, func_name: str, raw_args: Optional[bytes], *args: Any):
        pass

    @abstractmethod
    def _execute_function(self, func_t, args):
        pass

    @abstractmethod
    def get_nonce(self, address):
        pass

    @abstractmethod
    def increment_nonce(self, address):
        pass

    @abstractmethod
    def _return(self):
        pass

    @abstractmethod
    def _init_execution(self, acc: Account, msg: Message, module: ModuleT = None):
        pass

    @property
    def exec_ctx(self):
        return self.execution_ctxs[-1]

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
        origin: Address,
        target_address: Address,
        module: ast.Module,
        value: int,
        *args: Any,
        raw_args=None,  # abi-encoded constructor args
    ):
        module_t = module._metadata["type"]
        assert isinstance(module_t, ModuleT)

        # TODO follow the evm semantics for nonce, value, storage etc..)
        self.state[target_address] = Account(1, 0, {}, {}, None)

        msg = Message(
            caller=sender,
            to=constants.CREATE_CONTRACT_ADDRESS,
            create_address=target_address,
            value=value,
            data=args,
            code_address=target_address,
            code=module_t.init_function,
            depth=0,
            is_static=False,
        )

        self._init_execution(self.state[target_address], msg, module_t)

        if module_t.init_function is not None:
            constructor = module_t.init_function

            # TODO this probably should return ContractData?
            self._execute_function(constructor, args)

        # module's immutables were fixed upon upon constructor execution
        contract = self.exec_ctx.contract

        self.state[target_address].contract_data = contract

        print("deployed contract!")

    def execute_code(
        self,
        sender: Address,
        to: Address,
        value: int,
        code: ContractFunctionT,
        func_name: str,
        *args: Any,
        raw_args: Optional[bytes],
        is_static: bool = False,
    ):
        print("executing code!")

        self.env = Environment(
            caller=sender,
            block_hashes=[],
            origin=sender,
            coinbase=sender,
            number=0,
            time=0,
            prev_randao=b"",
            chain_id=0,
        )

        msg = Message(
            caller=sender,
            to=to,
            create_address=to,
            value=value,
            data=args,
            code_address=to,
            code=code,
            depth=0,
            is_static=is_static,
        )

        self._init_execution(self.state[to], msg)

        # TODO return value from this call
        self._extcall(func_name, raw_args, args)

        # return abi_encode("(int256)", (42,))
        return self._return()


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
        from ivy.vyper_contract import VyperDeployer

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

    def _dispatch(self, function_name, *args):
        functions = self.execution_ctxs[-1].contract.ext_funs

        if function_name not in functions:
            # TODO check fallback
            # TODO rollback the evm journal
            raise Exception(f"function {function_name} not found")
        else:
            function = functions[function_name]

        if function.is_payable:
            if self.msg.value != 0:
                # TODO raise and rollback
                pass

        # check args

        # check decorators

        return function

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
                self.returndata = e.value
                break

    def _execute_function(self, func_t, args):
        self._prologue(func_t, args)
        self._exec_body()
        self._epilogue()

        return self.returndata

    def _extcall(self, func_name: str, raw_args: Optional[bytes], *args: Any):
        if raw_args:
            # TODO decode args and continue as normal
            # args = abi_decode(raw_args, schema based on func_name)
            pass

        func_t = self._dispatch(func_name, args)

        self._execute_function(func_t, *args)

    def _return(self):
        return abi_encode("(int256)", (self.returndata,))

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

    def _init_execution(self, acc: Account, msg: Message, module_t: ModuleT = None):
        self.execution_ctxs.append(ExecutionContext(acc, msg, module_t))

        # intialize variables to default values
        if module_t:
            decls = module_t.variable_decls
            for d in decls:
                self._new_variable(d)

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

    def handle_call(self, func: ast.Call, args, kws):
        print(f"Handling function call to {func} with arguments {args}")
        func_t = func.func._metadata.get("type")

        if func_t is not None:
            if isinstance(func_t, BuiltinFunctionT):
                return self.builtins[func_t._id](*args)
            elif isinstance(func_t, ContractFunctionT):
                assert func_t.is_internal
                return self._execute_function(func_t, args)
            raise NotImplementedError(f"Function type {func_t} not supported")
        else:  # range()
            return self.builtins[func.func.id](*args, **kws)

    def handle_external_call(self, node):
        print(f"Handling external call with node {node}")
        return None

    def handle_static_call(self, node):
        print(f"Handling static call with node {node}")
        return None

    def handle_unaryop(self, op, operand):
        print(f"Handling unary operation: {op} on operand {operand}")
        return None

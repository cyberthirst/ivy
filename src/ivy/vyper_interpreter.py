from typing import Any, Optional, Union
from abc import abstractmethod

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


class BaseInterpreter(ExprVisitor, StmtVisitor):
    execution_ctx: list[ExecutionContext]
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
    def get_nonce(self, address):
        pass

    @abstractmethod
    def increment_nonce(self, address):
        pass

    @abstractmethod
    def _return(self):
        pass

    @abstractmethod
    def _init_execution(
        self, acc: Account, msg: Message, fun: ContractFunctionT = None
    ):
        pass

    def _push_fun_ctx(self, func_t):
        self.execution_ctx[-1].push_fun_context(func_t)

    def _pop_fun_ctx(self):
        self.execution_ctx[-1].pop_fun_context()

    def _push_scope(self):
        self.execution_ctx[-1].push_scope()

    def _pop_scope(self):
        self.execution_ctx[-1].pop_scope()

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
        typ = module._metadata["type"]
        assert isinstance(typ, ModuleT)

        # TODO follow the evm semantics for nonce, value, storage etc..)
        self.state[target_address] = Account(1, 0, {}, {}, None)

        if typ.init_function is not None:
            constructor = typ.init_function
            msg = Message(
                caller=sender,
                to=constants.CREATE_CONTRACT_ADDRESS,
                create_address=target_address,
                value=value,
                data=args,
                code_address=target_address,
                code=constructor,
                depth=0,
                is_static=False,
            )
            self._init_execution(self.state[target_address], msg, constructor)

            # TODO this probably should return ContractData
            self._extcall(msg)

        # module's immutables were fixed up within the _process_message call
        contract = ContractData(typ)

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
        self.executor = None
        self.returndata = None
        self.evaluator = VyperEvaluator()
        self.execution_ctx = []
        self.builtin_funcs = {}
        self.env = None

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
    def ctx(self):
        return self.execution_ctx[-1].current_fun_context()

    @property
    def msg(self):
        return self.execution_ctx[-1].msg

    def _dispatch(self, function_name, *args):
        functions = self.execution_ctx[-1].contract.ext_funs

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
        for stmt in self.ctx.function.decl_node.body:
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

    def get_variable(self, name: ast.Name):
        if name.id in self.ctx:
            return self.ctx[name.id].value
        else:
            raise NotImplementedError(f"{name.id}'s location not yet supported")

    def _init_execution(
        self, acc: Account, msg: Message, fun: ContractFunctionT = None
    ):
        self.execution_ctx.append(ExecutionContext(acc, msg, fun))

    def _get_var_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, _FunctionArg):
            return node.name
        else:
            raise NotImplementedError(
                f"Getting variable name from {type(node)} not implemented"
            )

    # name: locals, immutables, constants
    # attribute: storage, transient
    def set_variable(self, name: str, value):
        if name in self.ctx:
            self.ctx[name].value = value
        else:
            raise NotImplementedError(f"{name}'s location not yet supported")

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
            obj = self.visit(target.value)
            setattr(obj, target.attr, value)
        else:
            raise NotImplementedError(f"Assignment to {type(target)} not implemented")

    def _get_target_value(self, target):
        if isinstance(target, ast.Name):
            return self.get_variable(target)
        elif isinstance(target, ast.Subscript):
            container = self.visit(target.value)
            index = self.visit(target.slice)
            return container[index]
        elif isinstance(target, ast.Attribute):
            obj = self.visit(target.value)
            return getattr(obj, target.attr)
        else:
            raise NotImplementedError(
                f"Getting value from {type(target)} not implemented"
            )

    def _default_type_value(self, typ):
        # TODO based on type return its default value
        return None

    def _new_variable(self, target: Union[ast.Name, _FunctionArg]):
        if isinstance(target, ast.Name):
            info = target._expr_info
            var = Variable(
                self._default_type_value(info.typ), info.typ, DataLocation.MEMORY
            )
            self.ctx[target.id] = var
        elif isinstance(target, _FunctionArg):
            var = Variable(
                self._default_type_value(target.typ), target.typ, DataLocation.MEMORY
            )
            self.ctx[target.name] = var
        else:
            raise RuntimeError(f"Cannot create variable for {type(target)}")

    def _new_internal_variable(self, node):
        info = node._expr_info
        typ = info.typ
        var = Variable(self._default_type_value(typ), typ, DataLocation.MEMORY)
        self.execution_ctx[-1].new_variable(node.id, var)

    def handle_call(self, func: ast.Call, args):
        func_t = func.func._metadata["type"]

        print(f"Handling function call to {func} with arguments {args}")

        if isinstance(func_t, BuiltinFunctionT):
            return self.builtin_funcs[func_t.name](*args)
        elif isinstance(func_t, ContractFunctionT):
            assert func_t.is_internal
            return self._execute_function(func_t, args)
        else:
            raise NotImplementedError(f"Function type {func_t} not supported")

    def handle_external_call(self, node):
        print(f"Handling external call with node {node}")
        return None

    def handle_static_call(self, node):
        print(f"Handling static call with node {node}")
        return None

    def handle_unaryop(self, op, operand):
        print(f"Handling unary operation: {op} on operand {operand}")
        return None

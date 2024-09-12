from typing import Any, Optional, Union
from abc import abstractmethod

import vyper.ast.nodes as ast
from vyper.semantics.types.module import ModuleT, ContractFunctionT
from vyper.semantics.data_locations import DataLocation

from titanoboa.boa.util.abi import Address, abi_encode

import eth.constants as constants
from eth._utils.address import generate_contract_address

from ivy.evm import EVM, Account, Environment, Message, ContractData
from ivy.expr import ExprVisitor
from ivy.stmt import StmtVisitor, ReturnException
from ivy.evaluator import VyperEvaluator
from ivy.context import ExecutionContext, Variable
from vyper.builtins._signatures import BuiltinFunctionT


class BaseInterpreter(ExprVisitor, StmtVisitor):
    evm: EVM
    execution_ctx: list[ExecutionContext]

    def __init__(self, evm: EVM):
        self.evm = evm

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
    def _return(self):
        pass

    @abstractmethod
    def _init_execution(self, acc: Account, fun: ContractFunctionT = None):
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
        nonce = self.evm.get_nonce(sender.canonical_address)
        self.evm.increment_nonce(sender.canonical_address)
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
        self.evm.state[target_address] = Account(1, 0, {}, {}, None)

        if typ.init_function is not None:
            constructor = typ.init_function
            self._init_execution(self.evm.state[target_address], constructor)
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

            # TODO this probably should return ContractData
            self._extcall(msg)

        # module's immutables were fixed up within the _process_message call
        contract = ContractData(typ)

        self.evm.state[target_address].contract_data = contract

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

        self.evm.msg = msg

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

        self.evm.env = env

        self._init_execution(self.evm.state[to])

        # TODO return value from this call
        self._extcall(func_name, raw_args, args)

        # return abi_encode("(int256)", (42,))
        return self._return()


class VyperInterpreter(BaseInterpreter):
    contract: Optional[ContractData]
    evaluator: VyperEvaluator
    vars: dict[str, Any]

    def __init__(self, evm: EVM):
        super().__init__(evm)
        self.executor = None
        self.returndata = None
        self.evaluator = VyperEvaluator()
        self.execution_ctx = []
        self.builtin_funcs = {}

    @property
    def deployer(self):
        from ivy.vyper_contract import VyperDeployer

        return VyperDeployer

    @property
    def ctx(self):
        return self.execution_ctx[-1].current_fun_context()

    def _dispatch(self, function_name, *args):
        functions = self.execution_ctx[-1].contract.ext_funs

        if function_name not in functions:
            # TODO check fallback
            # TODO rollback the evm journal
            raise Exception(f"function {function_name} not found")
        else:
            function = functions[function_name]

        if function.is_payable:
            if self.evm.msg.value != 0:
                # TODO raise and rollback
                pass

        # check args

        # check decorators

        return function

    def _prologue(self, func_t, *args):
        self._push_fun_ctx(func_t)
        # TODO handle reentrancy lock

        # TODO handle args

        pass

    def _epilogue(self, *args):
        # TODO handle reentrancy lock
        # TODO handle return value
        self._pop_fun_ctx()
        # TODO register arguments as local variables
        pass

    def _exec_body(self):
        # for stmt in self.execution_ctx[-1].function.decl_node.body:
        for stmt in self.ctx.function.decl_node.body:
            try:
                self.visit(stmt)
            except ReturnException as e:
                # TODO handle the return value
                self.returndata = e.value
                break

    def _execute_function(self, func_t, *args):
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

    def _init_execution(self, acc: Account, fun: ContractFunctionT = None):
        self.execution_ctx.append(ExecutionContext(acc, fun))

    # name: locals, immutables, constants
    # attribute: storage, transient
    def set_variable(self, name: Union[ast.Name, ast.Attribute], value):
        name = name.id if isinstance(name, ast.Name) else name.attr

        if name in self.ctx:
            self.ctx[name].value = value
        else:
            raise NotImplementedError(f"{name}'s location not yet supported")

    def _assign_target(self, target, value):
        if isinstance(target, ast.Name):
            self.set_variable(target, value)
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

    def _new_variable(self, target: ast.Name):
        assert isinstance(target, ast.Name)
        info = target._expr_info
        var = Variable(
            self._default_type_value(info.typ), info.typ, DataLocation.MEMORY
        )
        self.ctx[target.id] = var

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
            return self._execute_function(func_t, *args)
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

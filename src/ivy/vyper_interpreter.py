import ast
from typing import Any, Optional, Union
from abc import abstractmethod

from vyper import ast as vy_ast
from vyper.semantics.types.module import ModuleT, ContractFunctionT
from vyper.semantics.data_locations import DataLocation

from titanoboa.boa.util.abi import Address, abi_encode

import eth.constants as constants
from eth._utils.address import generate_contract_address

from ivy.evm import EVM, Account, Environment, Message, ContractData
from ivy.expr import ExprVisitor
from ivy.stmt import StmtVisitor, ReturnException
from ivy.evaluator import VyperEvaluator
from ivy.ctx import Ctx


class BaseInterpreter(ExprVisitor, StmtVisitor):
    evm: EVM
    contract: Optional[ContractData]
    function: Optional[ContractFunctionT]
    ctxs: list[Ctx]

    def __init__(self, evm: EVM):
        self.evm = evm
        # contract being executed
        self.contract = None
        # function being executed
        self.function = None
        self.ctxs = []

    def get_code(self, address):
        pass

    @property
    @abstractmethod
    def deployer(self):
        pass

    @abstractmethod
    def _call(self, func_name: str, raw_args: Optional[bytes], *args: Any):
        pass

    @abstractmethod
    def _return(self):
        pass

    def _push_ctx(self):
        self.ctxs.append(Ctx())

    def _pop_ctx(self):
        self.ctxs.pop()

    def generate_create_address(self, sender):
        nonce = self.evm.get_nonce(sender.canonical_address)
        self.evm.increment_nonce(sender.canonical_address)
        return Address(generate_contract_address(sender.canonical_address, nonce))

    def deploy(
        self,
        sender: Address,
        origin: Address,
        target_address: Address,
        module: vy_ast.Module,
        value: int,
        *args: Any,
        raw_args=None,  # abi-encoded constructor args
    ):
        typ = module._metadata["type"]
        assert isinstance(typ, ModuleT)

        # TODO follow the evm semantics for nonce, value, storage etc..)
        self.evm.state[target_address] = Account(1, 0, {}, None)

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

            # TODO this probably should return ContractData
            self._call(msg)

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

        self.contract = self.evm.state[to].contract_data

        # TODO return value from this call
        self._call(func_name, raw_args, args)

        # return abi_encode("(int256)", (42,))
        return self._return()


class VyperInterpreter(BaseInterpreter):
    contract: Optional[ContractData]
    returndata: bytes
    evaluator: VyperEvaluator

    def __init__(self, evm: EVM):
        super().__init__(evm)
        self.executor = None
        self.returndata = None
        self.evaluator = VyperEvaluator()

    @property
    def deployer(self):
        from ivy.vyper_contract import VyperDeployer

        return VyperDeployer

    def _dispatch(self, function_name, *args):
        functions = self.contract.ext_funs

        if function_name not in functions:
            # TODO check fallback
            # TODO rollback the evm journal
            raise Exception(f"function {function_name} not found")
        else:
            self.function = functions[function_name]

        if self.function.is_payable:
            if self.evm.msg.value != 0:
                # TODO raise and rollback
                pass

        # check args

        # check decorators
        pass

    def _prologue(self):
        # TODO handle reentrancy lock

        # TODO handle args
        self._push_ctx()

        pass

    def _epilogue(self):
        # TODO handle reentrancy lock
        # TODO handle return value
        self._pop_ctx()
        pass

    def _exec_body(self):
        for stmt in self.function.decl_node.body:
            try:
                self.visit(stmt)
            except ReturnException as e:
                # TODO handle the return value
                self.returndata = e.value
                break

    def _call(self, func_name: str, raw_args: Optional[bytes], *args: Any):
        if raw_args:
            # TODO decode args and continue as normal
            # args = abi_decode(raw_args, schema based on func_name)
            pass

        self._dispatch(func_name, args)

        self._prologue()

        self._exec_body()

        self._epilogue()

    def _return(self):
        return abi_encode("(int256)", (self.returndata,))

    def get_variable(self, name: ast.Name):
        info = name._expr_info
        loc = info.location
        if loc == DataLocation.MEMORY:
            return self.ctxs[-1][name.id]
        else:
            raise NotImplementedError(f"Data location {loc} not implemented")

    def handle_call(self, func, args):
        print(f"Handling function call to {func} with arguments {args}")
        return None

    def handle_external_call(self, node):
        print(f"Handling external call with node {node}")
        return None

    def handle_static_call(self, node):
        print(f"Handling static call with node {node}")
        return None

    def handle_unaryop(self, op, operand):
        print(f"Handling unary operation: {op} on operand {operand}")
        return None

    def set_variable(
        self, name: Union[vy_ast.Name, vy_ast.Subscript, vy_ast.Attribute], value
    ):
        info = name._expr_info
        loc = info.location
        if loc == DataLocation.MEMORY:
            self.ctxs[-1][name.id] = value
        else:
            raise NotImplementedError(f"Data location {loc} not implemented")

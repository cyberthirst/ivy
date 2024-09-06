from typing import Any, Optional, Dict
from abc import ABC, abstractmethod

from vyper import ast as vy_ast
from vyper.semantics.types.module import ModuleT, ContractFunctionT
from vyper.semantics.types.function import ContractFunctionT

from titanoboa.boa.util.abi import Address, abi_encode, abi_decode

import eth.constants as constants
from eth._utils.address import generate_contract_address

from ivy.evm import EVM, Account, Environment, Message, ContractData, VyperEVM
from ivy.expr import ExprVisitor
from ivy.stmt import StmtVisitor


class BaseInterpreter(ABC, ExprVisitor, StmtVisitor):
    evm: EVM
    contract: Optional[ContractData]
    function: Optional[ContractFunctionT]

    def __init__(self, evm: EVM):
        self.evm = evm
        # contract being executed
        self.contract = None
        # function being executed
        self.function = None


    def get_code(self, address):
        pass

    @property
    @abstractmethod
    def deployer(self):
        pass


    @abstractmethod
    def _call(self, func_name: str, raw_args: Optional[bytes], *args: Any):
        pass


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

        return abi_encode("(int256)", (42,))



class VyperInterpreter(BaseInterpreter):
    contract: Optional[ContractData]

    def __init__(self, evm: EVM):
        super().__init__(evm)
        self.executor = None

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

        pass

    def _epilogue(self):
        # TODO handle reentrancy lock
        # TODO handle return value
        pass

    def _exec_body(self):
        for stmt in self.function.decl_node.body:
            self.executor.eval(stmt)

    def _call(self, func_name: str, raw_args: Optional[bytes], *args: Any):
        if raw_args:
            # TODO decode args and continue as normal
            # args = abi_decode(raw_args, schema based on func_name)
            pass

        self._dispatch(func_name, args)

        self._prologue()

        self._exec_body()

        self._epilogue()

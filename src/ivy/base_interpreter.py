from abc import abstractmethod
from typing import Any, Optional

import eth.constants as constants
from eth._utils.address import generate_contract_address

from titanoboa.boa.util.abi import Address

from vyper.semantics.types.module import ModuleT
import vyper.ast.nodes as ast

from ivy.expr import ExprVisitor
from ivy.stmt import StmtVisitor
from ivy.evm_structures import Account, Message, Environment
from ivy.context import ExecutionContext


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
    def _extcall(self):
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
    def _init_execution(
        self, acc: Account, msg: Message, sender, module: ModuleT = None
    ):
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

        self._init_execution(self.state[target_address], msg, sender, module_t)

        if module_t.init_function is not None:
            constructor = module_t.init_function

            # TODO this probably should return ContractData?
            self._execute_function(constructor, args)

        # module's immutables were fixed upon upon constructor execution
        contract = self.exec_ctx.contract

        self.state[target_address].contract_data = contract

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

        msg = Message(
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

        self._init_execution(self.state[to], msg, sender)

        self._extcall()

        # return the value to the frontend, otherwise just incorporate
        # the return value into the parent evm
        if self.exec_ctx.msg.depth == 0:
            return self.exec_ctx.output
        else:
            self.execution_ctxs[-2].returndata = self.exec_ctx.output

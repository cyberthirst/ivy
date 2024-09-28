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

    @property
    @abstractmethod
    def deployer(self):
        pass

    @abstractmethod
    def get_code(self, address):
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

    @abstractmethod
    def process_message(self, message: Message, env: Environment):
        pass

    @abstractmethod
    def process_create_message(self, message: Message, env: Environment):
        pass

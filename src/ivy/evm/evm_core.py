from typing import Optional, Union

import vyper.ast.nodes as ast
from vyper.semantics.types.module import ModuleT

from ivy.evm.evm_callbacks import EVMCallbacks
from ivy.evm.evm_structures import (
    Environment,
    ContractData,
    Message,
)
from ivy.evm.evm_state import EVMState, StateAccessor
from ivy.journal import Journal
from ivy.types import Address
from ivy.exceptions import EVMException
from ivy.utils import compute_contract_address
from ivy.context import ExecutionContext, ExecutionOutput


class EVMCore:
    def __init__(self, callbacks: EVMCallbacks):
        # TODO use state accessor facade
        self._state = EVMState()
        self.state = StateAccessor(self._state)
        self.journal = Journal()
        self.callbacks = callbacks
        # TODO move this to state
        self.environment: Optional[Environment] = None

    def execute_tx(
        self,
        sender: Address,
        to: Union[Address, bytes],
        value: int,
        calldata: bytes = b"",
        is_static: bool = False,
        module: Optional[ast.Module] = None,
    ):
        is_deploy = to == b""
        create_address, code = None, None

        if is_deploy:
            module_t = module._metadata["type"]
            assert isinstance(module_t, ModuleT)
            create_address = self.generate_create_address(sender)
            code = ContractData(module_t)
        else:
            code = self.state.get_code(to)

        message = Message(
            caller=sender,
            to=b"" if is_deploy else to,
            create_address=create_address,
            value=value,
            data=calldata,
            code_address=to,
            code=code,
            depth=0,
            is_static=is_static,
        )

        self.env = Environment(
            caller=sender,
            block_hashes=[],
            origin=to,
            coinbase=sender,
            number=0,
            time=0,
            prev_randao=b"",
            chain_id=0,
        )

        self.journal.begin_call()
        output = (
            self.process_create_message(message)
            if is_deploy
            else self.process_message(message)
        )
        self.journal.finalize_call(output.is_error)

        self.state.clear_transient_storage()

        if output.is_error:
            raise output.error

        return create_address if is_deploy else output.bytes_output()

    def process_create_message(
        self, message: Message, is_runtime_copy: Optional[bool] = False
    ) -> ExecutionOutput:
        if self.state.has_account(message.create_address):
            raise EVMException("Address already taken")

        new_account = self.state[message.create_address]
        self.state.add_accessed_account(new_account)

        exec_ctx = ExecutionContext(new_account, message, message.code)
        self.state.push_context(exec_ctx)

        try:
            self._handle_value_transfer(message)

            module_t = message.code.module_t

            if not is_runtime_copy:
                self.callbacks.allocate_variables(module_t)

            new_contract_code = message.code

            # skip if we're doing a runtime code copy
            if module_t.init_function is not None and not is_runtime_copy:
                new_contract_code = self.callbacks.execute_init_function(
                    module_t.init_function
                )

            new_account.contract_data = new_contract_code

        except Exception as e:
            # TODO rollback the journal
            self.state.current_output.error = e
            del self.state[message.create_address]

        finally:
            ret = self.state.current_output
            self.state.pop_context()

        return ret

    def process_message(self, message: Message) -> ExecutionOutput:
        account = self.state[message.to]
        self.state.add_accessed_account(account)
        exec_ctx = ExecutionContext(
            account,
            message,
            account.contract_data.module_t if account.contract_data else None,
        )
        self.state.push_context(exec_ctx)

        try:
            self._handle_value_transfer(message)

            if message.code:
                self.callbacks.dispatch()

        except Exception as e:
            # TODO rollback the journal
            self.state.current_output.error = e

        finally:
            ret = self.state.current_output
            self.state.pop_context()

        return ret

    def do_message_call(
        self,
        target: Address,
        value: int,
        data: bytes,
        is_static: bool = False,
        is_delegate: bool = False,
    ) -> ExecutionOutput:
        code_address = target
        code = self.state.get_code(code_address)

        if is_delegate:
            target = self.state.current_context.msg.to

        msg = Message(
            caller=self.state.current_context.msg.to,
            to=target,
            create_address=None,
            value=value,
            data=data,
            code_address=code_address,
            code=code,
            depth=self.state.current_context.msg.depth + 1,
            is_static=is_static,
        )

        self.journal.begin_call()
        output = self.process_message(msg)
        self.journal.finalize_call(output.is_error)

        return output

    def do_create_message_call(
        self,
        value: int,
        data: bytes,
        code: ContractData,
        salt: Optional[bytes] = None,
        is_runtime_copy: Optional[bool] = False,
    ) -> tuple[ExecutionOutput, Address]:
        if salt is not None:
            raise NotImplemented(
                "Create2 depends on bytecode which isn't currently supported"
            )

        current_address = self.state.current_context.msg.to
        create_address = self.generate_create_address(current_address)

        # TODO move nonce_inc to process_create_message
        self.state.increment_nonce(current_address)

        msg = Message(
            caller=self.state.current_context.msg.to,
            to=b"",
            create_address=create_address,
            value=value,
            data=data,
            code_address=b"",
            code=code,
            depth=self.state.current_context.msg.depth + 1,
            is_static=False,
        )

        self.journal.begin_call()
        output = self.process_create_message(msg, is_runtime_copy=is_runtime_copy)
        self.journal.finalize_call(output.is_error)

        # TODO we probably shouldn't return this tuple
        return output, create_address

    def _handle_value_transfer(self, message: Message) -> None:
        if message.value > 0:
            if self.state[message.caller].balance < message.value:
                raise EVMException("Insufficient balance for transfer")
            self.state[message.caller].balance -= message.value
            self.state[message.to].balance += message.value

    def generate_create_address(self, sender):
        nonce = self.state.get_nonce(sender.canonical_address)
        self.state.increment_nonce(sender.canonical_address)
        return Address(compute_contract_address(sender.canonical_address, nonce))

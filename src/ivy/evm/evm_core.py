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
from ivy.exceptions import EVMException, Revert
from ivy.utils import compute_contract_address
from ivy.context import ExecutionContext, ExecutionOutput
from ivy.evm.precompiles import PRECOMPILE_REGISTRY


class EVMCore:
    def __init__(self, callbacks: EVMCallbacks):
        # TODO use state accessor facade
        self._state = EVMState()
        self.state = StateAccessor(self._state)
        self.journal = Journal()
        self.callbacks = callbacks

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

        # TODO merge this with do_create_message_call and do_message_call
        if is_deploy:
            module_t = module._metadata["type"]
            assert isinstance(module_t, ModuleT)
            # Compute address with current nonce (before incrementing)
            create_address = self.generate_create_address(sender)
            code = ContractData(module_t)
        else:
            code = self.state.get_code(to)

        self.state.increment_nonce(sender)

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

        self.state.env = Environment(
            caller=sender,
            block_hashes=[],
            origin=to,
            coinbase=sender,
            block_number=1,
            time=1750080732,
            prev_randao=b"",
            chain_id=0,
        )

        assert not self.journal.is_active

        output = (
            self.process_create_message(message)
            if is_deploy
            else self.process_message(message)
        )

        assert not self.journal.is_active

        self.state.clear_transient_storage()

        if is_deploy:
            return create_address, output

        return output

    def execute_message(
        self,
        sender: Address,
        to: Address,
        value: int,
        calldata: bytes = b"",
        is_static: bool = False,
    ) -> ExecutionOutput:
        """Execute a message call in the current transaction context.

        This is for Vyper test suite compatibility where multiple calls
        happen within the same transaction context. Unlike execute_tx:
        - Does not increment sender's nonce
        - Does not clear transient storage
        - Reuses existing environment if set
        - Maintains a single journal context across calls
        """
        if to == b"":
            raise ValueError("Message calls cannot deploy contracts")

        code = self.state.get_code(to)

        # Set up environment if not already set (first call in test)
        if self.state.env is None:
            self.state.env = Environment(
                caller=sender,
                block_hashes=[],
                origin=sender,  # For message calls, origin is the sender
                coinbase=sender,
                block_number=1,
                time=1750080732,
                prev_randao=b"",
                chain_id=0,
            )

        self.journal.begin_call(is_static=False)

        message = Message(
            caller=sender,
            to=to,
            create_address=None,
            value=value,
            data=calldata,
            code_address=to,
            code=code,
            depth=0,  # Top-level call
            is_static=is_static,
        )

        # Process the message without additional journal management
        output = self.process_message(message, manage_journal=False)

        # Finalize this message call's journal frame
        # If there was an error, this will rollback all changes
        self.journal.finalize_call(output.is_error)

        return output

    def process_create_message(
        self, message: Message, is_runtime_copy: Optional[bool] = False
    ) -> ExecutionOutput:
        self.journal.begin_call(message.is_static)

        if self.state.has_account(message.create_address):
            raise EVMException("Address already taken")

        new_account = self.state[message.create_address]
        self.state.add_accessed_account(new_account)

        exec_ctx = ExecutionContext(new_account, message)
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

        # TODO can we merge the exception handler from
        # process_message and process_create_message?
        except Exception as e:
            # TODO rollback the journal
            self.state.current_output.error = e
            del self.state[message.create_address]
            if isinstance(e, Revert):
                self.state.current_output.output = e.data

        finally:
            ret = self.state.current_output
            self.state.pop_context()

        self.journal.finalize_call(ret.is_error)
        return ret

    def _execute_precompile(self, message: Message):
        to = message.to
        data = message.data
        self.state.current_output.output = PRECOMPILE_REGISTRY[to](data)

    def process_message(
        self, message: Message, manage_journal: bool = True
    ) -> ExecutionOutput:
        if manage_journal:
            self.journal.begin_call(is_static=message.is_static)
        account = self.state[message.to]
        self.state.add_accessed_account(account)
        exec_ctx = ExecutionContext(account, message)
        self.state.push_context(exec_ctx)

        try:
            self._handle_value_transfer(message)

            if message.to in PRECOMPILE_REGISTRY:
                self._execute_precompile(message)

            elif message.code:
                self.callbacks.dispatch()

        except Exception as e:
            self.state.current_output.error = e
            if isinstance(e, Revert):
                self.state.current_output.output = e.data

        finally:
            ret = self.state.current_output
            self.state.pop_context()

        # TODO shouldn't this be in the finally block
        if manage_journal:
            self.journal.finalize_call(ret.is_error)
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

        caller = self.state.current_context.msg.to
        if is_delegate:
            target = self.state.current_context.msg.to
            assert value == 0
            value = self.state.current_context.msg.value
            caller = self.state.current_context.msg.caller

        is_static = is_static if is_static else self.state.current_context.msg.is_static

        msg = Message(
            caller=caller,
            to=target,
            create_address=None,
            value=value,
            data=data,
            code_address=code_address,
            code=code,
            depth=self.state.current_context.msg.depth + 1,
            is_static=is_static,
        )

        child_output = self.process_message(msg)
        current_context = self.state.current_context

        if child_output.error:
            # incorporate_child_on_error(evm, child_evm)
            current_context.returndata = child_output.output
            # push(evm.stack, U256(0))
        else:
            self.incorporate_child_on_success(child_output)
            current_context.returndata = child_output.output
            # push(evm.stack, U256(1))

        return child_output

    def do_create_message_call(
        self,
        value: int,
        data: bytes,
        code: ContractData,
        salt: Optional[bytes] = None,
        is_runtime_copy: Optional[bool] = False,
    ) -> tuple[ExecutionOutput, Address]:
        if salt is not None:
            raise NotImplementedError(
                "Create2 depends on bytecode which isn't currently supported"
            )

        current_address = self.state.current_context.msg.to
        # we're in a constructor
        if current_address == b"":
            current_address = self.state.current_context.msg.create_address

        # First compute address with current nonce
        create_address = self.generate_create_address(current_address)

        # Then increment nonce
        self.state.increment_nonce(current_address)

        if self.account_has_code_or_nonce(create_address):
            # Return empty output and address(0) if account already exists
            empty_output = ExecutionOutput()
            empty_output.error = EVMException("Address collision")
            return empty_output, Address(0)

        msg = Message(
            caller=self.state.current_context.msg.to,
            to=b"",
            create_address=create_address,
            value=value,
            data=data,
            code_address=b"",
            code=code,
            depth=self.state.current_context.msg.depth + 1,
            is_static=self.state.current_context.msg.is_static,
        )

        child_output = self.process_create_message(msg, is_runtime_copy=is_runtime_copy)
        current_output = self.state.current_output
        return_address = create_address

        if child_output.is_error:
            # TODO: eventually implemente this incorporate
            # self.incorporate_child_on_error(evm, child_evm)
            current_output.return_data = child_output.output
            return_address = Address(0)
        else:
            self.incorporate_child_on_success(child_output)
            current_output.return_data = b""

        # TODO we probably shouldn't return this tuple
        return child_output, return_address

    def _handle_value_transfer(self, message: Message) -> None:
        if message.value > 0:
            if self.state[message.caller].balance < message.value:
                raise EVMException("Insufficient balance for transfer")
            self.state[message.caller].balance -= message.value
            # For deployments, message.to is b"" but create_address holds the new contract address
            recipient = message.create_address if message.create_address else message.to
            self.state[recipient].balance += message.value

    def generate_create_address(self, sender):
        # Make sure we're using the sender as an Address object
        sender_addr = sender if isinstance(sender, Address) else Address(sender)
        nonce = self.state.get_nonce(sender_addr)
        return Address(compute_contract_address(sender_addr.canonical_address, nonce))

    def account_has_code_or_nonce(self, address: Address) -> bool:
        code = self.state.get_code(address)
        nonce = self.state.get_nonce(address)
        return code is not None or nonce > 0

    def incorporate_child_on_success(self, child_output: ExecutionOutput) -> None:
        output = self.state.current_output

        output.logs += child_output.logs
        output.refund_counter += child_output.refund_counter
        output.accounts_to_delete.update(child_output.accounts_to_delete)
        output.touched_accounts.update(child_output.touched_accounts)
        # TODO enable this
        # if account_exists_and_is_empty(
        #        evm.env.state, child_output.message.current_target
        # ):
        #    evm.touched_accounts.add(child_output.message.current_target)
        output.accessed_addresses.update(child_output.accessed_addresses)

    def finalize_transaction(self, is_error: bool = False) -> None:
        """Finalize the current transaction's journal.

        This should be called by the test framework after all message calls
        in a transaction context are complete.
        """
        if self.journal.is_active:
            self.journal.finalize_call(is_error)

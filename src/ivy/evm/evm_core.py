from typing import Optional, Union

from vyper.compiler import CompilerData

from ivy.evm.evm_callbacks import EVMCallbacks
from ivy.evm.evm_structures import (
    Environment,
    ContractData,
    Message,
)
from ivy.evm.evm_state import EVMState, StateAccessor
from ivy.journal import Journal
from ivy.types import Address
from ivy.exceptions import EVMException, Revert, SelfDestruct, StaticCallViolation
from ivy.utils import compute_contract_address, compute_create2_address
from ivy.context import ExecutionContext, ExecutionOutput
from ivy.evm.precompiles import PRECOMPILE_REGISTRY

# Maximum call depth limit as per EVM specification
STACK_DEPTH_LIMIT = 1024

# Maximum nonce value (uint64 max) - CREATE fails gracefully at this value
MAX_NONCE = 2**64 - 1


class EVMCore:
    def __init__(self, callbacks: EVMCallbacks, env: Environment):
        # TODO use state accessor facade
        self._state = EVMState()
        self._state._env = env
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
        compiler_data: Optional[CompilerData] = None,
    ):
        # EIP-6780: Clear created_accounts at transaction start
        # This ensures the set is transaction-scoped, not persisted across transactions
        self._state.created_accounts.clear()

        is_deploy = to == b""
        create_address, code = None, None

        # TODO merge this with do_create_message_call and do_message_call
        if is_deploy:
            assert compiler_data is not None
            # Compute address with current nonce (before incrementing)
            create_address = self.generate_create_address(sender)
            code = ContractData(compiler_data)
        else:
            assert isinstance(to, Address)
            code = self.state.get_code(to)

        # TODO should we inc if this is deployment tx?
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

        # Update per-call fields on existing env
        self.state.env.caller = sender
        self.state.env.origin = sender

        assert not self.journal.is_active

        output = (
            self.process_create_message(message)
            if is_deploy
            else self.process_message(message)
        )

        assert not self.journal.is_active

        if self.journal.pop_state_committed():
            self.callbacks.on_state_committed()

        # EIP-6780: Delete accounts marked for deletion at transaction end
        # Only if transaction succeeded (no error in output)
        if output.error is None:
            for address in output.accounts_to_delete:
                del self.state[address]

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

        # Update per-call fields on existing env
        self.state.env.caller = sender
        self.state.env.origin = sender

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

        if self.journal.pop_state_committed():
            self.callbacks.on_state_committed()

        # NOTE: Unlike execute_tx(), we do NOT delete accounts here.
        # This matches Boa's test environment behavior where account deletion
        # only happens at real transaction boundaries, not message calls.
        # The accounts_to_delete set is populated but deletion is deferred.

        return output

    def process_create_message(
        self, message: Message, is_runtime_copy: Optional[bool] = False
    ) -> ExecutionOutput:
        self.journal.begin_call(message.is_static)

        # Destroy any pre-existing storage at the target address.
        # This handles the edge case where CREATE collides with a previously
        # self-destructed address that still has storage.
        self.state.destroy_storage(message.create_address)

        if self.state.has_account(message.create_address):
            raise EVMException("Address already taken")

        new_account = self.state[message.create_address]
        self.state.add_accessed_account(new_account)

        # Track for EIP-6780 selfdestruct semantics
        self._state.created_accounts.add(message.create_address)

        # EIP-161: newly created contracts start with nonce=1, not nonce=0
        # This must happen before init code runs, so any CREATE inside __init__
        # uses the correct nonce for address computation
        self.state.increment_nonce(message.create_address)

        exec_ctx = ExecutionContext(new_account, message)
        self.state.push_context(exec_ctx)

        try:
            self._handle_value_transfer(message)

            new_contract_code = message.code

            if message.code is not None:
                module_t = message.code.module_t

                if not is_runtime_copy:
                    self.callbacks.allocate_variables(module_t)

                # skip if we're doing a runtime code copy
                if module_t.init_function is not None and not is_runtime_copy:
                    new_contract_code = self.callbacks.execute_init_function(
                        module_t.init_function
                    )

            self.state.set_code(message.create_address, new_contract_code)

        except SelfDestruct:
            # SelfDestruct in constructor - clean halt
            # The contract destroys itself during deployment
            # Account deletion already handled by builtin_selfdestruct
            pass

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
        code_address = message.code_address
        data = message.data
        self.state.current_output.output = PRECOMPILE_REGISTRY[code_address](data)

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

            if message.code_address in PRECOMPILE_REGISTRY:
                self._execute_precompile(message)

            elif message.code:
                self.callbacks.dispatch()

        except SelfDestruct:
            # SelfDestruct is a clean halt, NOT an error
            # State changes (balance transfer, account deletion) are already done
            # Don't set error - execution was successful
            pass

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
        # Check call depth limit before creating child message
        # Per EVM spec: if depth + 1 > STACK_DEPTH_LIMIT, return failure gracefully
        new_depth = self.state.current_context.msg.depth + 1
        if new_depth > STACK_DEPTH_LIMIT:
            output = ExecutionOutput()
            output.error = EVMException("Stack depth limit exceeded")
            self.state.current_context.returndata = b""
            return output

        code_address = target
        code = self.state.get_code(code_address)

        # Get current_target: use create_address if in constructor (msg.to == b"")
        current_target = self.state.current_context.msg.to
        if current_target == b"":
            current_target = self.state.current_context.msg.create_address

        caller = current_target
        if is_delegate:
            target = current_target
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
            should_transfer_value=not is_delegate,  # DELEGATECALL should not transfer value
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
        # Check call depth limit before creating child message
        # Per EVM spec: if depth + 1 > STACK_DEPTH_LIMIT, return failure gracefully
        # Note: nonce is NOT incremented if depth limit is exceeded
        new_depth = self.state.current_context.msg.depth + 1
        if new_depth > STACK_DEPTH_LIMIT:
            output = ExecutionOutput()
            output.error = EVMException("Stack depth limit exceeded")
            self.state.current_context.returndata = b""
            return output, Address(0)

        current_address = self.state.current_context.msg.to
        # we're in a constructor
        if current_address == b"":
            current_address = self.state.current_context.msg.create_address

        # Per Execution Spec: check balance and nonce BEFORE incrementing nonce
        # If any check fails, return early without modifying state
        sender_balance = self.state[current_address].balance
        sender_nonce = self.state.get_nonce(current_address)

        if sender_balance < value:
            # Insufficient balance - fail gracefully without incrementing nonce
            empty_output = ExecutionOutput()
            empty_output.error = EVMException("Insufficient balance for transfer")
            return empty_output, Address(0)

        if sender_nonce >= MAX_NONCE:
            # Nonce overflow - fail gracefully without incrementing nonce
            empty_output = ExecutionOutput()
            empty_output.error = EVMException("Nonce overflow")
            return empty_output, Address(0)

        # Compute address - CREATE2 if salt provided, CREATE otherwise
        if salt is not None:
            # CREATE2: address = keccak256(0xff ++ sender ++ salt ++ keccak256(init_code))[-20:]
            init_code = code.compiler_data.bytecode
            create_address = Address(
                compute_create2_address(
                    current_address.canonical_address, salt, init_code
                )
            )
        else:
            # CREATE: address derived from sender + nonce
            create_address = self.generate_create_address(current_address)

        # Increment nonce (only after all checks pass)
        # Note: nonce is incremented for both CREATE and CREATE2
        self.state.increment_nonce(current_address)

        if self.account_has_code_or_nonce(create_address):
            # Return empty output and address(0) if account already exists
            empty_output = ExecutionOutput()
            empty_output.error = EVMException("Address collision")
            return empty_output, Address(0)

        msg = Message(
            caller=current_address,
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
        current_context = self.state.current_context
        return_address = create_address

        if child_output.is_error:
            # TODO: eventually implement incorporate_child_on_error
            current_context.returndata = child_output.output
            return_address = Address(0)
        else:
            self.incorporate_child_on_success(child_output)
            current_context.returndata = b""

        # TODO we probably shouldn't return this tuple
        return child_output, return_address

    def _handle_value_transfer(self, message: Message) -> None:
        if message.is_static and message.value != 0:
            raise StaticCallViolation("Cannot transfer value in static context")
        # DELEGATECALL should not transfer value, only inherit msg.value for context
        if message.should_transfer_value and message.value > 0:
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

    def get_current_address(self) -> Address:
        """Get the address of the currently executing contract."""
        current_target = self.state.current_context.msg.to
        if current_target == b"":
            current_target = self.state.current_context.msg.create_address
        return current_target

    def selfdestruct(self, beneficiary: Address) -> None:
        """
        EVM SELFDESTRUCT opcode implementation (EIP-6780 semantics).

        Post-Cancun behavior:
        - Always transfers full balance to beneficiary
        - Only deletes account if it was created in the same transaction
        - If beneficiary == originator and account is deleted, ether is burnt
        - Raises SelfDestruct to halt execution
        """
        originator = self.get_current_address()
        originator_balance = self.state.get_balance(originator)

        # Transfer ALL ether from originator to beneficiary
        # When beneficiary == originator, this is a no-op
        if originator_balance > 0 and beneficiary != originator:
            self.state.set_balance(originator, 0)
            beneficiary_balance = self.state.get_balance(beneficiary)
            self.state.set_balance(
                beneficiary, beneficiary_balance + originator_balance
            )

        # EIP-6780: Only mark for deletion if created in the same transaction
        # Actual deletion happens at transaction end (in execute_tx/execute_message)
        if originator in self._state.created_accounts:
            self.state.current_output.accounts_to_delete.add(originator)
            # If beneficiary == originator, the ether is burnt
            # (balance is zeroed, and account will be deleted at tx end)
            self.state.set_balance(originator, 0)

        # Halt execution
        raise SelfDestruct()

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

        if self.journal.pop_state_committed():
            self.callbacks.on_state_committed()

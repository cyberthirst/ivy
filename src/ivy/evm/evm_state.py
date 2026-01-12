from collections import defaultdict
from typing import Optional, Protocol

from ivy.evm.evm_structures import Account, ContractData, Environment
from ivy.types import Address
from ivy.context import ExecutionContext, ExecutionOutput
from ivy.journal import Journal, JournalEntryType


class EVMState:
    def __init__(self):
        self.state = defaultdict(lambda: self._create_account(None))
        self.execution_contexts: list[ExecutionContext] = []
        self.accessed_accounts = set()
        self._env = None
        self._journal = Journal()

    def _create_account(self, address):
        """Create a new account and journal it if we're in an active transaction."""
        account = Account(0, 0, {}, {}, None)
        if address is not None and self._journal.is_active:
            self._journal.record(
                JournalEntryType.ACCOUNT_CREATION,
                self.state,  # obj is the state dict
                address,  # key is the address
                None,  # old value is None for new accounts
            )
        return account

    def __getitem__(self, key):
        # Reading a non-existent account returns an empty account but doesn't
        # modify state - account creation is only journaled when actually
        # modifying the account (balance transfer, code set, etc.)
        return self.state[key]

    def __delitem__(self, key: Address):
        if key in self.state:
            # Journal the account destruction
            if self._journal.is_active:
                self._journal.record(
                    JournalEntryType.ACCOUNT_DESTRUCTION,
                    self.state,
                    key,
                    self.state[key],  # Save the account for potential rollback
                )
            # TODO do we care about accessed accounts?
            # account = self.state[key]
            # if account in self.accessed_accounts:
            #    self.accessed_accounts.remove(account)
            del self.state[key]

    def has_account(self, address: Address) -> bool:
        # An account "exists" if it has non-zero balance, nonce, or code
        # Per EIP-161: empty accounts are those with no code, zero nonce, zero balance
        if address not in self.state:
            return False
        account = self.state[address]
        return (
            account.balance != 0
            or account.nonce != 0
            or account.contract_data is not None
        )

    def get_nonce(self, address: Address) -> int:
        if address not in self.state:
            return 0
        return self.state[address].nonce

    def increment_nonce(self, address: Address):
        account = self.state[address]
        if self._journal.is_active:
            self._journal.record(
                JournalEntryType.NONCE, account, "nonce", account.nonce
            )
        account.nonce += 1

    def get_balance(self, address: Address) -> int:
        if address not in self.state:
            return 0
        return self.state[address].balance

    def set_balance(self, address: Address, value: int):
        self.state[address].balance = value

    def get_code(self, address: Address) -> Optional[ContractData]:
        if address not in self.state:
            return None
        return self.state[address].contract_data

    def set_code(self, address: Address, code: Optional[ContractData]):
        account = self.state[address]
        if self._journal.is_active:
            self._journal.record(
                JournalEntryType.CODE, account, "contract_data", account.contract_data
            )
        account.contract_data = code

    def get_storage(self, address: Address) -> dict:
        return self.state[address].storage

    def destroy_storage(self, address: Address) -> None:
        """Destroy all storage at the given address (for CREATE address reuse edge case)."""
        if address not in self.state:
            return
        account = self.state[address]
        if not account.storage:
            return
        # Journal the old storage for potential rollback
        if self._journal.is_active:
            self._journal.record(
                JournalEntryType.STORAGE_DESTRUCTION,
                account,
                "storage",
                account.storage,
            )
        account.storage = {}

    def get_transient(self, address: Address) -> dict:
        account = self.state[address]
        self.accessed_accounts.add(account)
        return account.transient

    def add_accessed_account(self, acc):
        self.accessed_accounts.add(acc)

    def clear_transient_storage(self):
        # global_vars reference the storage, it's necessary to clear instead of assigning a new dict
        # NOTE: it might be better to refactor GlobalVariable to receive a function to retrieve storage
        # instaed of receiving the storage directly
        for account in self.accessed_accounts:
            account.transient.clear()
        self.accessed_accounts.clear()

    def get_account(self, address: Address) -> Account:
        account = self.state[address]
        self.accessed_accounts.add(account)
        return account

    @property
    def current_context(self) -> Optional[ExecutionContext]:
        return self.execution_contexts[-1] if self.execution_contexts else None

    def push_context(self, context: ExecutionContext):
        self.execution_contexts.append(context)

    def pop_context(self):
        self.execution_contexts.pop()

    @property
    def current_output(self) -> ExecutionOutput:
        return self.current_context.execution_output

    @property
    def env(self):
        return self._env


class StateAccess(Protocol):
    def __getitem__(self, address: Address) -> Account: ...

    def __delitem__(self, key: Address): ...

    def get_nonce(self, address: Address) -> int: ...

    def get_account(self, address: Address) -> Account: ...

    def increment_nonce(self, address: Address): ...

    def get_balance(self, address: Address) -> int: ...

    def set_balance(self, address: Address, value: int): ...

    def get_code(self, address: Address) -> Optional[ContractData]: ...

    def set_code(self, address: Address, code: Optional[ContractData]): ...

    def get_storage(self, address: Address) -> int: ...

    def destroy_storage(self, address: Address) -> None: ...

    def add_accessed_account(self, acc): ...

    def has_account(self, address) -> bool: ...

    def clear_transient_storage(self) -> None: ...

    @property
    def current_context(self) -> ExecutionContext: ...

    def push_context(self, context: ExecutionContext): ...

    def pop_context(self): ...

    @property
    def current_output(self) -> ExecutionOutput: ...


class StateAccessor(StateAccess):
    def __init__(self, state: EVMState):
        self._state = state

    def __getitem__(self, address: Address) -> Account:
        return self._state[address]

    def __delitem__(self, address: Address):
        del self._state[address]

    def get_nonce(self, address):
        return self._state.get_nonce(address)

    def increment_nonce(self, address):
        self._state.increment_nonce(address)

    def get_balance(self, address):
        return self._state.get_balance(address)

    def set_balance(self, address: Address, value: int):
        self._state.set_balance(address, value)

    def get_account(self, address: Address) -> Account:
        return self._state.get_account(address)

    def get_code(self, address):
        return self._state.get_code(address)

    def set_code(self, address: Address, code: Optional[ContractData]):
        self._state.set_code(address, code)

    def get_storage(self, address: Address) -> dict:
        return self._state.get_storage(address)

    def destroy_storage(self, address: Address) -> None:
        self._state.destroy_storage(address)

    def add_accessed_account(self, acc):
        self._state.add_accessed_account(acc)

    def has_account(self, address) -> bool:
        return self._state.has_account(address)

    def clear_transient_storage(self) -> None:
        self._state.clear_transient_storage()

    @property
    def current_context(self) -> ExecutionContext:
        return self._state.current_context

    def push_context(self, context: ExecutionContext):
        self._state.push_context(context)

    def pop_context(self):
        self._state.pop_context()

    @property
    def current_output(self) -> ExecutionOutput:
        return self._state.current_output

    @property
    def env(self) -> Environment:
        return self._state.env

    @env.setter
    def env(self, value):
        self._state._env = value

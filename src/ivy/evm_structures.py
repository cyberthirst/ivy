from typing import Any, Optional
from dataclasses import dataclass
from collections import defaultdict

from vyper.semantics.types.function import ContractFunctionT
from vyper.semantics.types.module import ModuleT
from vyper.semantics.types.subscriptable import TupleT

from ivy.utils import compute_call_abi_data
from ivy.variable import GlobalVariables
from ivy.types import Address


@dataclass
class EntryPointInfo:
    function: ContractFunctionT
    calldata_args_t: TupleT
    calldata_min_size: int


# dict for constant variables
# TODO: should we use it also for immutables?
class AssignOnceDict(dict):
    def __setitem__(self, key, value):
        if key in self:
            raise ValueError(
                f"Cannot reassign key '{key}': already has value '{self[key]}'"
            )
        super().__setitem__(key, value)


class ContractData:
    module_t: ModuleT
    ext_funs: dict[str, ContractFunctionT]
    internal_funs: dict[str, ContractFunctionT]
    immutables: dict[str, Any]
    constants: dict[str, Any]
    entry_points: dict[bytes, EntryPointInfo]
    global_vars: GlobalVariables
    fallback: Optional[ContractFunctionT]

    def __init__(self, module: ModuleT):
        self.module_t = module

        self.ext_funs: dict[str, ContractFunctionT] = {
            f.name: f for f in module.exposed_functions
        }
        self.internal_funs: dict[str, ContractFunctionT] = {
            f: f for f in module.functions if f not in self.ext_funs.values()
        }
        self.immutables = {}
        self.constants = AssignOnceDict()
        self.entry_points = {}
        self._generate_entry_points()
        self.global_vars = GlobalVariables()
        self.fallback = next(
            (f for f in module.exposed_functions if f.is_fallback), None
        )

    def _generate_entry_points(self):
        def process(func_t, calldata_kwargs):
            selector, calldata_args_t = compute_call_abi_data(
                func_t, len(calldata_kwargs)
            )

            args_abi_t = calldata_args_t.abi_type
            calldata_min_size = args_abi_t.static_size() + 4

            return selector, calldata_min_size, calldata_args_t

        for f in self.module_t.exposed_functions:
            if f.name == "__default__":
                continue

            keyword_args = f.keyword_args

            for i, _ in enumerate(keyword_args):
                calldata_kwargs = keyword_args[:i]

                selector, calldata_min_size, calldata_args_t = process(
                    f, calldata_kwargs
                )

                assert selector not in self.entry_points
                self.entry_points[selector] = EntryPointInfo(
                    f, calldata_args_t, calldata_min_size
                )

            selector, calldata_min_size, calldata_args_t = process(f, keyword_args)
            assert selector not in self.entry_points
            self.entry_points[selector] = EntryPointInfo(
                f, calldata_args_t, calldata_min_size
            )


class EVMOutput:
    data: bytes
    error: Exception

    def __init__(self, data: bytes = None, error: Exception = None):
        self.data = data
        self.error = error

    @property
    def is_error(self):
        return self.error is not None

    def bytes_output(self, safe=True):
        if safe and self.is_error:
            raise self.error

        if self.data is None:
            return b""
        return self.data


@dataclass
class Account:
    nonce: Any
    balance: Any
    storage: Any
    transient: Any
    contract_data: Optional[ContractData]

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return self is other


@dataclass
class Environment:  # env from execution specs
    caller: Any  # Address
    block_hashes: Any  # List[Hash32]
    origin: Any  # Address
    coinbase: Any  # Address
    number: Any  # Uint
    # base_fee_per_gas: Uint
    # gas_limit: Uint
    # gas_price: Uint
    time: Any  # U256
    prev_randao: Any  # Bytes32
    # state: Any#State
    chain_id: Any  # U64
    # traces: List[dict]


@dataclass
class Message:  # msg from execution specs
    caller: Any  # Address
    to: Any  # Union[Bytes0, Address]
    create_address: Any  # Address
    # gas: Uint
    value: Any  # U256
    data: Any  # Bytes
    code_address: Any  # Optional[Address]
    code: ContractData
    depth: Any  # Uint
    # should_transfer_value: bool
    is_static: bool
    # accessed_addresses: Set[Address]
    # accessed_storage_keys: Set[Tuple[Address, Bytes32]]
    # parent_evm: Optional["Evm"]


class EVMState:
    def __init__(self):
        self.state = defaultdict(lambda: Account(0, 0, {}, {}, None))
        self.accessed_accounts = set()

    def __getitem__(self, key):
        return self.state[key]

    def __delitem__(self, key: Address):
        if key in self.state:
            # TODO do we care about accessed accounts?
            # account = self.state[key]
            # if account in self.accessed_accounts:
            #    self.accessed_accounts.remove(account)
            del self.state[key]

    def has_account(self, address) -> bool:
        # TODO add detection for an empty account (+ maybe rename to smth like non-empty)
        return False

    def get_nonce(self, address: Address) -> int:
        return self.state[address].nonce

    def increment_nonce(self, address: Address):
        self.state[address].nonce += 1

    def get_balance(self, address: Address) -> int:
        return self.state[address].balance

    def set_balance(self, address: Address, value: int):
        self.state[address].balance = value

    def get_code(self, address: Address) -> Optional[ContractData]:
        return self.state[address].contract_data

    def set_code(self, address: Address, code: ContractData):
        self.state[address].contract_data = code

    def get_storage(self, address: Address) -> dict:
        return self.state[address].storage

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

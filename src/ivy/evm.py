from typing import Any, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass

from vyper.semantics.types.function import ContractFunctionT
from vyper.semantics.types.module import ModuleT

from titanoboa.boa.util.abi import Address


class ContractData:
    module: ModuleT
    ext_funs: Dict[str, ContractFunctionT]
    internal_funs: Dict[str, ContractFunctionT]
    immutables: Dict[str, Any]

    def __init__(self, module: ModuleT):
        self.module = module

        self.ext_funs: Dict[str, ContractFunctionT] = {
            f.name: f for f in module.exposed_functions
        }
        self.internal_funs: Dict[str, ContractFunctionT] = {
            f: f for f in module.functions if f not in self.ext_funs.values()
        }
        self.immutables = {}


@dataclass
class Account:
    nonce: Any
    balance: Any
    storage: Any
    contract_data: ContractData | None


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
    code: Any  # Bytes
    depth: Any  # Uint
    # should_transfer_value: bool
    is_static: bool
    # accessed_addresses: Set[Address]
    # accessed_storage_keys: Set[Tuple[Address, Bytes32]]
    # parent_evm: Optional["Evm"]


class EVM(ABC):
    state: Dict[Any, Account]
    msg: Message
    env: Environment

    @abstractmethod
    def set_slot(self, key, value):
        pass

    @abstractmethod
    def get_slot(self, key):
        pass

    @abstractmethod
    def get_nonce(self, address):
        pass

    @abstractmethod
    def increment_nonce(self, address):
        pass


class VyperEVM(EVM):
    def __init__(self):
        self.state = {}
        self.msg = None
        self.env = None

    def process_message(self, msg: Message):
        pass

    def get_nonce(self, address):
        if address not in self.state:
            self.state[address] = Account(0, 0, {}, None)
        return self.state[address].nonce

    def increment_nonce(self, address):
        assert address in self.state
        self.state[address].nonce += 1

    def set_slot(self, key, value):
        pass

    def get_slot(self, key):
        pass

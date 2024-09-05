from dataclasses import dataclass
from typing import Any

from vyper import ast as vy_ast
from vyper.semantics.types.module import ModuleT, ContractFunctionT

from titanoboa.boa.util.abi import Address, abi_encode, abi_decode

import eth.constants as constants
from eth._utils.address import generate_contract_address

class ContractData:
    def __init__(self, module):
        self.module = module
        self.storage = {}
        self.ext_funs = {}
        self.internal_funs = {}
        self.immutables = {}


@dataclass
class Account:
    nonce: Any
    balance: Any
    contract_data: ContractData | None
    storage: Any


@dataclass
class Environment: # env from execution specs
    caller: Any#Address
    block_hashes: Any#List[Hash32]
    origin: Any#Address
    coinbase: Any#Address
    number: Any#Uint
    # base_fee_per_gas: Uint
    # gas_limit: Uint
    # gas_price: Uint
    time: Any#U256
    prev_randao: Any#Bytes32
    state: Any#State
    chain_id: Any#U64
    # traces: List[dict]


@dataclass
class Message: # msg from execution specs
    caller: Any#Address
    to: Any#Union[Bytes0, Address]
    create_address: Any#Address
    # gas: Uint
    value: Any#U256
    data: Any#Bytes
    code_address: Any#Optional[Address]
    code: Any#Bytes
    depth: Any#Uint
    # should_transfer_value: bool
    is_static: bool
    # accessed_addresses: Set[Address]
    # accessed_storage_keys: Set[Tuple[Address, Bytes32]]
    # parent_evm: Optional["Evm"]


class EVM:
    def __init__(self):
        self.state = {}

    def process_message(self, msg: Message):
        pass

    def get_nonce(self, address):
        if address not in self.state:
            self.state[address] = Account(0, 0, None, {})
        return self.state[address].nonce

    def increment_nonce(self, address):
        assert address in self.state
        self.state[address].nonce += 1



class Interpreter:

    def __init__(self):
        # address -> Account
        self.evm = EVM()


    def deploy(
            self,
            sender: Address,
            origin: Address,
            target_address: Address,
            module: vy_ast.Module,
            value: int,
            args=None,  # abi-encoded constructor args
    ):
        typ = module._metadata["type"]
        assert isinstance(typ, ModuleT)

        # TODO follow the evm semantics for nonce, value, storage etc..)
        self.evm.state[target_address] = Account(1, 0, None, {})

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
            self._process_message(msg)

        # module's immutables were fixed up within the _process_message call
        contract = ContractData(module)

        self.evm.state[target_address].contract_data = contract

        print("deployed contract!")



    def execute_code(
            self,
            sender: Address,
            to: Address,
            value: int,
            code: ContractFunctionT,
            data: bytes,
            is_static: bool = False,
    ):
        print("executing code!")
        return abi_encode("(int256)", (42,))


    def get_code(self, address):
        pass

    def generate_create_address(self, sender):
        nonce = self.evm.get_nonce(sender.canonical_address)
        self.evm.increment_nonce(sender.canonical_address)
        return Address(generate_contract_address(sender.canonical_address, nonce))


    def _process_message(self, msg: Message):
        pass
from typing import Any, Optional
from dataclasses import dataclass

from vyper.semantics.types.function import ContractFunctionT
from vyper.semantics.types.module import ModuleT
from vyper.semantics.types.subscriptable import TupleT

from ivy.utils import compute_call_abi_data
from ivy.variable import GlobalVariable


@dataclass
class EntryPointInfo:
    function: ContractFunctionT
    calldata_args_t: TupleT
    calldata_min_size: int


class ContractData:
    module_t: ModuleT
    ext_funs: dict[str, ContractFunctionT]
    internal_funs: dict[str, ContractFunctionT]
    immutables: dict[str, Any]
    constants: dict[str, Any]
    entry_points: dict[bytes, EntryPointInfo]
    global_vars: dict[str, GlobalVariable]

    def __init__(self, module: ModuleT):
        self.module_t = module

        self.ext_funs: dict[str, ContractFunctionT] = {
            f.name: f for f in module.exposed_functions
        }
        self.internal_funs: dict[str, ContractFunctionT] = {
            f: f for f in module.functions if f not in self.ext_funs.values()
        }
        self.immutables = {}
        self.constants = {}
        self.entry_points = {}
        self._generate_entry_points()
        self.global_vars = {}

    def _generate_entry_points(self):
        def process(func_t, calldata_kwargs):
            selector, calldata_args_t = compute_call_abi_data(
                func_t, len(calldata_kwargs)
            )

            args_abi_t = calldata_args_t.abi_type
            calldata_min_size = args_abi_t.static_size() + 4

            return selector, calldata_min_size, calldata_args_t

        for f in self.module_t.exposed_functions:
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

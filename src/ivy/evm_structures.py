from typing import Any, Dict, Optional
from dataclasses import dataclass

from vyper.semantics.types.function import ContractFunctionT
from vyper.semantics.types.module import ModuleT
from vyper.semantics.types.subscriptable import TupleT
from vyper.utils import method_id


@dataclass
class EntryPointInfo:
    function: ContractFunctionT
    sig: str
    calldata_args_t: TupleT
    calldata_min_size: int


class ContractData:
    module: ModuleT
    ext_funs: Dict[str, ContractFunctionT]
    internal_funs: Dict[str, ContractFunctionT]
    immutables: Dict[str, Any]
    constants: Dict[str, Any]
    entry_points: Dict[bytes, EntryPointInfo]

    def __init__(self, module: ModuleT):
        self.module = module

        self.ext_funs: Dict[str, ContractFunctionT] = {
            f.name: f for f in module.exposed_functions
        }
        self.internal_funs: Dict[str, ContractFunctionT] = {
            f: f for f in module.functions if f not in self.ext_funs.values()
        }
        self.immutables = {}
        self.constants = {}
        self.entry_points = {}
        self._generate_entry_points()

    def _generate_entry_points(self):
        def process(func_t, calldata_kwargs, default_kwargs):
            calldata_args = func_t.positional_args + calldata_kwargs
            # create a fake type so that get_element_ptr works
            calldata_args_t = TupleT(list(arg.typ for arg in calldata_args))

            abi_sig = func_t.abi_signature_for_kwargs(calldata_kwargs)

            args_abi_t = calldata_args_t.abi_type
            calldata_min_size = args_abi_t.static_size() + 4

            return abi_sig, calldata_min_size, calldata_args_t

        for f in self.module.exposed_functions:
            keyword_args = f.keyword_args

            for i, _ in enumerate(keyword_args):
                calldata_kwargs = keyword_args[:i]
                default_kwargs = keyword_args[i:]

                sig, calldata_min_size, calldata_args_t = process(
                    f, calldata_kwargs, default_kwargs
                )

                selector = method_id(sig)

                assert selector not in self.entry_points
                self.entry_points[selector] = EntryPointInfo(
                    f, sig, calldata_args_t, calldata_min_size
                )

            sig, calldata_min_size, calldata_args_t = process(f, keyword_args, [])
            selector = method_id(sig)
            assert selector not in self.entry_points
            self.entry_points[selector] = EntryPointInfo(
                f, sig, calldata_args_t, calldata_min_size
            )


@dataclass
class Account:
    nonce: Any
    balance: Any
    storage: Any
    transient: Any
    contract_data: Optional[ContractData]


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

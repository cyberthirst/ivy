from typing import Any, Optional
from dataclasses import dataclass

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
    time: Any  # U256
    prev_randao: Any  # Bytes32
    chain_id: Any  # U64


@dataclass
class Message:  # msg from execution specs
    caller: Any  # Address
    to: Any  # Union[Bytes0, Address]
    create_address: Any  # Address
    value: Any  # U256
    data: Any  # Bytes
    code_address: Any  # Optional[Address]
    code: ContractData
    depth: Any  # Uint
    is_static: bool


@dataclass
class Log:
    address: Address
    topics: list[bytes]
    data: bytes

from enum import Enum
from typing import Callable, Any

from ivy.builtins.builtins import (
    builtin_len,
    builtin_slice,
    builtin_concat,
    builtin_max,
    builtin_min,
    builtin_uint2str,
    builtin_empty,
    builtin_max_value,
    builtin_min_value,
    builtin_range,
    builtin_convert,
    builtin_as_wei_value,
    builtin_abi_decode,
    builtin_abi_encode,
    builtin_method_id,
    builtin_print,
    builtin_raw_call,
    builtin_send,
    builtin_create_copy_of,
    builtin_create_from_blueprint,
    builtin_create_minimal_proxy_to,
)
from ivy.evm.evm_core import EVMCore
from ivy.evm.evm_state import StateAccess


class BuiltinType(Enum):
    PURE = "pure"  # Functions that don't need state/evm access
    STATE = "state"  # Functions that need state access
    EVM = "evm"  # Functions that need full EVM access


class PureBuiltin:
    def __init__(self, fn: Callable):
        self.fn = fn

    def execute(self, *args, **kwargs) -> Any:
        return self.fn(*args, **kwargs)


class StateBuiltin:
    def __init__(self, state: StateAccess, fn: Callable):
        self.state = state
        self.fn = fn

    def execute(self, *args, **kwargs) -> Any:
        return self.fn(self.state, *args, **kwargs)


class EVMBuiltin:
    def __init__(self, evm: EVMCore, fn: Callable):
        self.evm = evm
        self.fn = fn

    def execute(self, *args, **kwargs) -> Any:
        return self.fn(self.evm, *args, **kwargs)


class BuiltinRegistry:
    def __init__(self, evm_core: EVMCore, state: StateAccess):
        self.evm = evm_core
        self.state = state
        self.builtins = self._register_builtins()

    def _register_builtins(self):
        return {
            # Pure builtins
            "len": PureBuiltin(builtin_len),
            "slice": PureBuiltin(builtin_slice),
            "concat": PureBuiltin(builtin_concat),
            "max": PureBuiltin(builtin_max),
            "min": PureBuiltin(builtin_min),
            "uint2str": PureBuiltin(builtin_uint2str),
            "empty": PureBuiltin(builtin_empty),
            "max_value": PureBuiltin(builtin_max_value),
            "min_value": PureBuiltin(builtin_min_value),
            "range": PureBuiltin(builtin_range),
            "convert": PureBuiltin(builtin_convert),
            "as_wei_value": PureBuiltin(builtin_as_wei_value),
            "_abi_decode": PureBuiltin(builtin_abi_decode),
            "abi_decode": PureBuiltin(builtin_abi_decode),
            "abi_encode": PureBuiltin(builtin_abi_encode),
            "_abi_encode": PureBuiltin(builtin_abi_encode),
            "method_id": PureBuiltin(builtin_method_id),
            "print": PureBuiltin(builtin_print),
            # State builtins
            # EVM builtins
            "raw_call": EVMBuiltin(self.evm, builtin_raw_call),
            "send": EVMBuiltin(self.evm, builtin_send),
            "create_copy_of": EVMBuiltin(self.evm, builtin_create_copy_of),
            "create_from_blueprint": EVMBuiltin(
                self.evm, builtin_create_from_blueprint
            ),
            "create_minimal_proxy_to": EVMBuiltin(
                self.evm, builtin_create_minimal_proxy_to
            ),
        }

    def get(self, name: str) -> Callable:
        if name not in self.builtins:
            raise ValueError(f"Unknown builtin: {name}")
        return self.builtins[name].execute

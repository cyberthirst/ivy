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
    builtin_raw_revert,
    builtin_unsafe_add,
    builtin_unsafe_sub,
    builtin_unsafe_mul,
    builtin_unsafe_div,
    builtin_floor,
    builtin_epsilon,
    builtin_sqrt,
    builtin_isqrt,
    builtin_keccak256,
)


class BuiltinWrapper:
    def __init__(self, fn, context=None, needs_types=False):
        self.fn = fn
        self.context = context
        # we don't tag runtime values with a corresponding vyper type, thus we have to
        # collect the types separately and for certain builtins inject them
        self.needs_types = needs_types

    def execute(self, *args, typs=None, **kwargs):
        if self.needs_types:
            if typs is None:
                raise ValueError("Type information is required for this built-in")
            if self.context:
                return self.fn(self.context, typs, *args, **kwargs)
            return self.fn(typs, *args, **kwargs)
        if self.context:
            return self.fn(self.context, *args, **kwargs)
        return self.fn(*args, **kwargs)


class BuiltinRegistry:
    def __init__(self, evm_core, state):
        self.evm = evm_core
        self.state = state
        self.builtins = self._register_builtins()

    def _register_builtins(self):
        return {
            # Pure built-ins (no context, no types)
            "len": BuiltinWrapper(builtin_len),
            "slice": BuiltinWrapper(builtin_slice),
            "concat": BuiltinWrapper(builtin_concat),
            "max": BuiltinWrapper(builtin_max),
            "min": BuiltinWrapper(builtin_min),
            "uint2str": BuiltinWrapper(builtin_uint2str),
            "empty": BuiltinWrapper(builtin_empty),
            "max_value": BuiltinWrapper(builtin_max_value),
            "min_value": BuiltinWrapper(builtin_min_value),
            "range": BuiltinWrapper(builtin_range),
            "as_wei_value": BuiltinWrapper(builtin_as_wei_value),
            "_abi_decode": BuiltinWrapper(builtin_abi_decode),
            "abi_decode": BuiltinWrapper(builtin_abi_decode),
            "method_id": BuiltinWrapper(builtin_method_id),
            "print": BuiltinWrapper(builtin_print),
            "raw_revert": BuiltinWrapper(builtin_raw_revert),
            # Pure built-ins (no context, with types)
            "convert": BuiltinWrapper(builtin_convert, needs_types=True),
            "abi_encode": BuiltinWrapper(builtin_abi_encode, needs_types=True),
            "_abi_encode": BuiltinWrapper(builtin_abi_encode, needs_types=True),
            # EVM built-ins (with evm context, no types)
            "raw_call": BuiltinWrapper(builtin_raw_call, context=self.evm),
            "send": BuiltinWrapper(builtin_send, context=self.evm),
            "create_copy_of": BuiltinWrapper(builtin_create_copy_of, context=self.evm),
            "create_from_blueprint": BuiltinWrapper(
                builtin_create_from_blueprint, context=self.evm
            ),
            "create_minimal_proxy_to": BuiltinWrapper(
                builtin_create_minimal_proxy_to, context=self.evm
            ),
            "unsafe_add": BuiltinWrapper(builtin_unsafe_add, needs_types=True),
            "unsafe_sub": BuiltinWrapper(builtin_unsafe_sub, needs_types=True),
            "unsafe_mul": BuiltinWrapper(builtin_unsafe_mul, needs_types=True),
            "unsafe_div": BuiltinWrapper(builtin_unsafe_div, needs_types=True),
            "floor": BuiltinWrapper(builtin_floor),
            "epsilon": BuiltinWrapper(builtin_epsilon),
            "sqrt": BuiltinWrapper(builtin_sqrt),
            "isqrt": BuiltinWrapper(builtin_isqrt),
            "keccak256": BuiltinWrapper(builtin_keccak256),
        }

    def get(self, name):
        if name not in self.builtins:
            raise ValueError(f"Unknown builtin: {name}")
        return self.builtins[name]

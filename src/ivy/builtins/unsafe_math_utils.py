from vyper.semantics.types import IntegerT


# EVM div semantics as a python function
def evm_div(x, y):
    if y == 0:
        return 0
    sign = -1 if (x * y) < 0 else 1
    return sign * (abs(x) // abs(y))  # adapted from py-evm


def get_int_params(x, y):
    """Get bits and signed from boxed integer values."""
    typ = x.typ
    assert typ == y.typ, f"Type mismatch: {x.typ} != {y.typ}"
    assert isinstance(typ, IntegerT)
    return typ.bits, typ.is_signed


def wrap_value(value, bits, signed):
    mod = 2**bits
    result = value % mod  # Wrap to [0, 2^bits - 1]
    if signed and result > 2 ** (bits - 1) - 1:
        result -= mod  # Adjust to [-2^(bits-1), 2^(bits-1) - 1]
    return result

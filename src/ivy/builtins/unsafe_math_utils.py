from vyper.semantics.types import IntegerT


# EVM div semantics as a python function
def evm_div(x, y):
    if y == 0:
        return 0
    sign = -1 if (x * y) < 0 else 1
    return sign * (abs(x) // abs(y))  # adapted from py-evm


def validate_typs(typs):
    assert len(typs) == 2
    # TODO is this the right way to compare?
    assert typs[0] == typs[1]
    typ = typs[0]
    assert isinstance(typ, IntegerT)
    return typ.bits, typ.is_signed


def wrap_value(value, bits, signed):
    mod = 2**bits
    result = value % mod  # Wrap to [0, 2^bits - 1]
    if signed and result > 2 ** (bits - 1) - 1:
        result -= mod  # Adjust to [-2^(bits-1), 2^(bits-1) - 1]
    return result

# all the convert functionality is adapted from vyper's test suite
# - https://github.com/vyperlang/vyper/blob/c32b9b4c6f0d8b8cdb103d3017ff540faf56a305/tests/functional/builtins/codegen/test_convert.py#L301

import enum
import math

from vyper.semantics.types import (
    BytesT,
    StringT,
    BytesM_T,
    DecimalT,
    IntegerT,
    BoolT,
    AddressT,
)
from vyper.utils import unsigned_to_signed

from ivy.abi import abi_encode, abi_decode, DecodeError
from ivy.types import VyperDecimal


class _PadDirection(enum.Enum):
    Left = enum.auto()
    Right = enum.auto()


def _padding_direction(typ):
    if isinstance(typ, (BytesM_T, StringT, BytesT)):
        return _PadDirection.Right
    return _PadDirection.Left


def _convert_decimal_to_int(val: VyperDecimal, o_typ: IntegerT):
    lo, hi = o_typ.int_bounds
    if not lo <= val.value <= hi:
        raise ConvertError()

    return val.truncate()


def _convert_int_to_int(val, o_typ):
    lo, hi = o_typ.int_bounds
    if not lo <= val <= hi:
        raise ConvertError()
    return val


def _convert_int_to_decimal(val, o_typ):
    assert isinstance(o_typ, DecimalT)
    try:
        ret = VyperDecimal(val)
    except ValueError:
        raise ConvertError()

    return ret


def _to_bits(val, i_typ):
    # i_typ: the type to convert from
    if isinstance(i_typ, DecimalT):
        val = val * i_typ.divisor
        assert math.ceil(val) == math.floor(val)
        val = int(val)
    return abi_encode(i_typ, val)


def _bits_of_type(typ):
    if isinstance(typ, (IntegerT, DecimalT)):
        return typ.bits
    if isinstance(typ, BoolT):
        return 8
    if isinstance(typ, AddressT):
        return 160
    if isinstance(typ, BytesM_T):
        return typ.m_bits
    if isinstance(typ, BytesT):
        return typ.length * 8

    raise Exception(f"Unknown type {typ}")


def bytes_of_type(typ):
    ret = _bits_of_type(typ)
    assert ret % 8 == 0
    return ret // 8


# TODO this could be a function in vyper.builtins._convert
# which implements literal folding and also serves as a reference/spec
def _padconvert(val_bits, direction, n, padding_byte=None):
    """
    Takes the ABI representation of a value, and convert the padding if needed.
    If fill_zeroes is false, the two halves of the bytestring are just swapped
    and the dirty bytes remain dirty. If fill_zeroes is true, the padding
    bytes get set to 0
    """
    assert len(val_bits) == 32

    # convert left-padded to right-padded
    if direction == _PadDirection.Right:
        tail = val_bits[:-n]
        if padding_byte is not None:
            tail = padding_byte * len(tail)
        return val_bits[-n:] + tail

    # right- to left- padded
    if direction == _PadDirection.Left:
        head = val_bits[n:]
        if padding_byte is not None:
            head = padding_byte * len(head)
        return head + val_bits[:n]


def _signextend(val_bytes, bits):
    as_uint = int.from_bytes(val_bytes, byteorder="big")

    as_sint = unsigned_to_signed(as_uint, bits)

    return (as_sint % 2**256).to_bytes(32, byteorder="big")


def _from_bits(val_bits, o_typ):
    # o_typ: the type to convert to
    try:
        ret = abi_decode(o_typ, val_bits)
        return ret
    except DecodeError:
        raise ConvertError()


class ConvertError(Exception):
    pass

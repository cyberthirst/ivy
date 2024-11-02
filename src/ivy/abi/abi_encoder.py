from typing import Any, Callable, Union
from eth_utils import to_canonical_address
from vyper.abi_types import (
    ABI_Address,
    ABI_Bool,
    ABI_Bytes,
    ABI_BytesM,
    ABI_DynamicArray,
    ABI_GIntM,
    ABI_StaticArray,
    ABI_String,
    ABI_Tuple,
    ABIType,
)
from vyper.utils import int_bounds
from vyper.semantics.types import VyperType

from ivy.base_types import Struct, Flag


class EncodeError(Exception):
    pass


# spec: https://docs.soliditylang.org/en/latest/abi-spec.html
def abi_encode(typ: VyperType, value: Any) -> bytes:
    abi_t = typ.abi_type
    return _encode_r(abi_t, value)


def _encode_tuple(abi_t: ABI_Tuple, value: Union[tuple, Struct]) -> bytes:
    # TODO rethink whether not to represent structs as tuples
    if isinstance(value, Struct):
        value = tuple(value.values())
    if not isinstance(value, tuple):
        raise EncodeError(f"Expected tuple, got {type(value)}")
    if len(value) != len(abi_t.subtyps):
        raise EncodeError(
            f"Tuple length mismatch: expected {len(abi_t.subtyps)}, got {len(value)}"
        )

    head = b""
    tail = b""
    dynamic_head_size = sum(subtyp.embedded_static_size() for subtyp in abi_t.subtyps)

    for subtyp, subval in zip(abi_t.subtyps, value):
        if subtyp.is_dynamic():
            head += (dynamic_head_size + len(tail)).to_bytes(32, "big")
            encoded = _encode_r(subtyp, subval)
            tail += encoded
        else:
            encoded = _encode_r(subtyp, subval)
            head += encoded

    return head + tail


def _encode_static_array(abi_t: ABI_StaticArray, value: list) -> bytes:
    if not isinstance(value, list):
        raise EncodeError(f"Expected list, got {type(value)}")
    if len(value) != abi_t.m_elems:
        raise EncodeError(
            f"Static array length mismatch: expected {abi_t.m_elems}, got {len(value)}"
        )

    tuple_abi_t = ABI_Tuple(subtyps=[abi_t.subtyp for _ in range(abi_t.m_elems)])
    return _encode_tuple(tuple_abi_t, tuple(value))


def _encode_dynamic_array(abi_t: ABI_DynamicArray, value: list) -> bytes:
    if not isinstance(value, list):
        raise EncodeError(f"Expected list, got {type(value)}")

    length = len(value).to_bytes(32, "big")

    tuple_abi_t = ABI_Tuple(subtyps=[abi_t.subtyp for _ in range(len(value))])

    encoded_items = _encode_tuple(tuple_abi_t, tuple(value))

    return length + encoded_items


def _encode_bytes(_: ABI_Bytes, value: bytes) -> bytes:
    if not isinstance(value, bytes):
        raise EncodeError(f"Expected bytes, got {type(value)}")

    length = len(value).to_bytes(32, "big")
    padded_value = value.ljust((len(value) + 31) // 32 * 32, b"\x00")
    return length + padded_value


def _encode_string(_: ABI_String, value: str) -> bytes:
    if not isinstance(value, str):
        raise EncodeError(f"Expected str, got {type(value)}")

    encoded_str = value.encode("utf-8")
    # TODO is it correct to pass the len() here?
    return _encode_bytes(ABI_Bytes(len(encoded_str)), encoded_str)


def _encode_int(abi_t: ABI_GIntM, value: Union[int, Flag]) -> bytes:
    if isinstance(value, Flag):
        value = value.value
    if not isinstance(value, int):
        raise EncodeError(f"Expected int, got {type(value)}")

    lo, hi = int_bounds(signed=abi_t.signed, bits=abi_t.m_bits)
    if not (lo <= value <= hi):
        raise EncodeError(f"Value {value} out of bounds for {abi_t}")

    return value.to_bytes(32, "big", signed=abi_t.signed)


def _encode_bytesM(abi_t: ABI_BytesM, value: bytes) -> bytes:
    if not isinstance(value, bytes):
        raise EncodeError(f"Expected bytes, got {type(value)}")
    if len(value) != abi_t.m_bytes:
        raise EncodeError(
            f"BytesM length mismatch: expected {abi_t.m_bytes}, got {len(value)}"
        )

    return value.ljust(32, b"\x00")


def _encode_bool(_: ABI_Bool, value: bool) -> bytes:
    if not isinstance(value, bool):
        raise EncodeError(f"Expected bool, got {type(value)}")
    return (1 if value else 0).to_bytes(32, "big")


def _encode_address(_: ABI_Address, value: str) -> bytes:
    try:
        address_bytes = to_canonical_address(value)
        return address_bytes.rjust(32, b"\x00")
    except ValueError:
        raise EncodeError(f"Invalid address: {value}")


# NOTE: bc of the dict dispatch some of the args are not used and are there just to match the signature
ENCODE_FUNCTIONS: dict[type, Callable] = {
    ABI_Tuple: _encode_tuple,
    ABI_StaticArray: _encode_static_array,
    ABI_DynamicArray: _encode_dynamic_array,
    ABI_Bytes: _encode_bytes,
    ABI_String: _encode_string,
    ABI_GIntM: _encode_int,
    ABI_BytesM: _encode_bytesM,
    ABI_Bool: _encode_bool,
    ABI_Address: _encode_address,
}


def _encode_r(abi_t: ABIType, value: Any) -> bytes:
    encode_func = ENCODE_FUNCTIONS.get(type(abi_t))
    if encode_func is None:
        raise EncodeError(f"Unsupported type: {abi_t}")
    return encode_func(abi_t, value)

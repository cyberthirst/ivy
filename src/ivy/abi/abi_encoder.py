import decimal
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

from ivy.types import (
    Struct,
    Flag,
    DynamicArray,
    StaticArray,
    VyperDecimal,
    VyperBool,
    Tuple as IvyTuple,
)


class EncodeError(Exception):
    pass


# spec: https://docs.soliditylang.org/en/latest/abi-spec.html
def abi_encode(typ: VyperType, value: Any) -> bytes:
    abi_t = typ.abi_type
    return _encode_r(abi_t, value)


def _encode_tuple(
    abi_t: ABI_Tuple, value: Union[tuple, list, Struct, IvyTuple]
) -> bytes:
    # TODO rethink whether not to represent structs as tuples
    if isinstance(value, Struct):
        value = tuple(value.values())
    elif isinstance(value, IvyTuple):
        value = tuple(value)
    elif isinstance(value, list):
        # Convert list to tuple for struct types
        value = tuple(value)
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


def _encode_static_array(
    abi_t: ABI_StaticArray, value: Union[list, StaticArray]
) -> bytes:
    if not isinstance(value, (list, StaticArray)):
        raise EncodeError(f"Expected list, got {type(value)}")
    # see __len__ in StaticArray to understand why we can't use it
    value_len = len(value) if isinstance(value, list) else value.length
    if value_len != abi_t.m_elems:
        raise EncodeError(
            f"Static array length mismatch: expected {abi_t.m_elems}, got {value.length}"
        )

    tuple_abi_t = ABI_Tuple(subtyps=[abi_t.subtyp for _ in range(abi_t.m_elems)])
    # see __len__ in StaticArray to understand why we use iter
    return _encode_tuple(tuple_abi_t, tuple(iter(value)))


def _encode_dynamic_array(abi_t: ABI_DynamicArray, value: list) -> bytes:
    if not isinstance(value, (list, DynamicArray)):
        raise EncodeError(f"Expected DynamicArray or list, got {type(value)}")

    length = len(value).to_bytes(32, "big")

    tuple_abi_t = ABI_Tuple(subtyps=[abi_t.subtyp for _ in range(len(value))])

    encoded_items = _encode_tuple(tuple_abi_t, tuple(value))

    return length + encoded_items


def _encode_bytes(_: ABI_Bytes, value: Union[bytes, str]) -> bytes:
    if not isinstance(value, (bytes, str)):
        raise EncodeError(f"Expected bytes or str, got {type(value)}")

    if isinstance(value, str):
        # Handle hex string with 0x prefix
        if value.startswith("0x"):
            hex_value = value.removeprefix("0x")
        else:
            # Handle hex string without 0x prefix
            hex_value = value

        try:
            value = bytes.fromhex(hex_value)
        except ValueError:
            raise EncodeError(f"Invalid hex string: {value}")

    length = len(value).to_bytes(32, "big")
    padded_value = value.ljust((len(value) + 31) // 32 * 32, b"\x00")
    return length + padded_value


def _encode_string(_: ABI_String, value: Union[str, bytes]) -> bytes:
    if isinstance(value, bytes):
        encoded_str = value
    elif isinstance(value, str):
        encoded_str = value.encode("utf-8")
    else:
        raise EncodeError(f"Expected str or bytes, got {type(value)}")

    return _encode_bytes(ABI_Bytes(len(encoded_str)), encoded_str)


def _encode_int(abi_t: ABI_GIntM, value: Union[int, Flag]) -> bytes:
    if isinstance(value, Flag):
        value = value.value
    if isinstance(value, VyperDecimal):
        value = value.value
    if isinstance(value, decimal.Decimal):
        value = int(value * decimal.Decimal("10") ** 10)
    if not isinstance(value, int):
        raise EncodeError(f"Expected int, got {type(value)}")

    lo, hi = int_bounds(signed=abi_t.signed, bits=abi_t.m_bits)
    if not (lo <= value <= hi):
        raise EncodeError(f"Value {value} out of bounds for {abi_t}")

    return value.to_bytes(32, "big", signed=abi_t.signed)


def _encode_bytesM(abi_t: ABI_BytesM, value: Union[bytes, str]) -> bytes:
    if isinstance(value, str):
        if not value.startswith("0x"):
            raise EncodeError("Hex string must start with 0x")
        if value == "0x":
            raise EncodeError("Invalid hex string: empty hex value after '0x'")
        try:
            value = bytes.fromhex(value.removeprefix("0x"))
        except ValueError:
            raise EncodeError("Invalid hex string")

    if not isinstance(value, bytes):
        raise EncodeError(f"Expected bytes or hex string, got {type(value)}")

    if len(value) > 32:
        raise EncodeError("Input exceeds 32 bytes")

    m = abi_t.m_bytes
    if any(value[i] != 0 for i in range(m, len(value))):
        raise EncodeError(f"Value exceeds bytes{m} bounds")

    return value.ljust(32, b"\x00")


def _encode_bool(_: ABI_Bool, value: Union[bool, VyperBool]) -> bytes:
    if not isinstance(value, (bool, VyperBool)):
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

from typing import Any, Union, Optional
import math

from vyper.exceptions import UnimplementedException
from eth_utils import keccak
from vyper.semantics.types import (
    VyperType,
    TupleT,
    DecimalT,
    IntegerT,
    BoolT,
    AddressT,
    BytesT,
    BytesM_T,
    StringT,
)
from vyper.semantics.types.shortcuts import UINT256_T, BYTES32_T, INT256_T
from vyper.codegen.core import (
    calculate_type_for_external_return,
    needs_external_call_wrap,
)


from ivy.abi import abi_decode, abi_encode
import ivy.builtins.create_utils as create_utils
from ivy.evm.precompiles import precompile_ecrecover
from ivy.context import ExecutionOutput
from ivy.expr.default_values import get_default_value
from ivy.exceptions import Assert, GasReference, Revert, UnsupportedFeature
from ivy.types import Address, VyperDecimal, VyperBytes, VyperInt, VyperString
import ivy.builtins.convert_utils as convert_utils
from ivy.evm.evm_core import EVMCore
from ivy.builtins import unsafe_math_utils as unsafe_math


def builtin_range(*args, bound=None):
    if len(args) == 2:
        start, stop = args
    else:
        assert len(args) == 1
        start, stop = 0, args[0]

    if start > stop:
        raise ValueError(f"start({start}) is greater than stop({stop})")

    if bound:
        if stop > bound and len(args) == 1:
            raise RuntimeError(f"Stop value is greater than bound={bound} value")
        if stop - start > bound:
            raise RuntimeError(f"Range is greater than bound={bound} value")

    return range(start, stop)


_WEI_DENOMS = {
    "wei": 1,
    "femtoether": 10**3,
    "kwei": 10**3,
    "babbage": 10**3,
    "picoether": 10**6,
    "mwei": 10**6,
    "lovelace": 10**6,
    "nanoether": 10**9,
    "gwei": 10**9,
    "shannon": 10**9,
    "microether": 10**12,
    "szabo": 10**12,
    "milliether": 10**15,
    "finney": 10**15,
    "ether": 10**18,
    "kether": 10**21,
    "grand": 10**21,
}


def builtin_as_wei_value(value: Union[int, VyperDecimal], denom: VyperString):
    from ivy.exceptions import Assert

    denom_str = denom.decode("utf-8")

    if denom_str not in _WEI_DENOMS:
        raise ValueError(f"Unknown wei denomination: {denom_str}")
    multiplier = _WEI_DENOMS[denom_str]

    if isinstance(value, VyperDecimal):
        # Vyper asserts value >= 0 for decimals
        if value.value < 0:
            raise Assert(data=b"")
        # For decimals: (scaled_value * multiplier) // SCALING_FACTOR
        return (value.value * multiplier) // VyperDecimal.SCALING_FACTOR

    if value < 0:
        raise Assert(data=b"")

    result = value * multiplier

    # result must fit in uint256
    if result >= 2**256:
        raise Assert(data=b"")

    return result


def builtin_len(x):
    return len(x)


def builtin_print(*args):
    print(*args)


def builtin_abi_decode(data: VyperBytes, typ: VyperType, unwrap_tuple=True):
    assert isinstance(data, VyperBytes)
    assert isinstance(typ, VyperType)

    if unwrap_tuple:
        wrapped_typ = calculate_type_for_external_return(typ)
        result = abi_decode(wrapped_typ, data)
        # Unwrap if the type was wrapped (non-tuples and single-element tuples get wrapped)
        if needs_external_call_wrap(typ):
            return result[0]
        return result
    else:
        return abi_decode(typ, data)


def builtin_abi_encode(*values: tuple[Any], ensure_tuple=True, method_id=None):
    assert isinstance(values, tuple)

    # Get types from the boxed values
    typs = tuple(v.typ for v in values)

    if len(values) == 1 and not ensure_tuple:
        # unwrap tuple
        typ = typs[0]
        encode_input = values[0]
    else:
        typ = TupleT(list(typs))
        # values are already a tuple
        encode_input = values

    ret = abi_encode(typ, encode_input)

    if method_id is not None:
        ret = method_id + ret

    return ret


def builtin_empty(typ):
    return get_default_value(typ)


def _get_bound(typ, get_high: bool):
    if isinstance(typ, DecimalT):
        return VyperDecimal.max() if get_high else VyperDecimal.min()
    low, high = typ.int_bounds
    return high if get_high else low


def builtin_max_value(typ):
    return _get_bound(typ, get_high=True)


def builtin_min_value(typ):
    return _get_bound(typ, get_high=False)


def builtin_max(x, y):
    return max(x, y)


def builtin_min(x, y):
    return min(x, y)


def builtin_uint2str(x: int) -> str:
    assert x >= 0
    return str(x)


def builtin_method_id(method: VyperString, output_type: Optional[VyperType] = None):
    # output_type (bytes4 vs Bytes[4]) only affects ABI encoding, not the value
    if not isinstance(method, VyperString):
        raise TypeError(f"Expected VyperString, got {type(method)}")
    return keccak(bytes(method))[:4]


def builtin_send(evm: EVMCore, to, value, gas: int = 0) -> None:
    if gas != 0:
        raise GasReference()
    # Vyper's send() compiles to assert(call(...)), which discards returndata
    # and reverts with empty data on failure. We must NOT propagate the callee's
    # revert data.
    success = builtin_raw_call(evm, to, b"", value=value, revert_on_failure=False)
    if not success:
        from ivy.exceptions import Assert

        raise Assert(data=b"")


def builtin_raw_call(
    evm: EVMCore,
    to: Address,
    data: bytes,
    max_outsize: int = 0,
    gas: Optional[int] = None,
    value: int = 0,
    is_delegate_call: bool = False,
    is_static_call: bool = False,
    revert_on_failure: bool = True,
):
    if gas is not None:
        raise GasReference()

    assert not (is_static_call and is_delegate_call)
    assert not (value != 0 and (is_static_call or is_delegate_call))

    output: ExecutionOutput = evm.do_message_call(
        to, value, data, is_static_call, is_delegate_call
    )

    success = not output.is_error
    returndata = evm.state.current_context.returndata

    if not revert_on_failure:
        if max_outsize == 0:
            return success
        return success, returndata[:max_outsize]

    if not success:
        raise output.error

    if max_outsize == 0:
        return None  # void return
    return returndata[:max_outsize]


def builtin_slice(b: Union[bytes, str], start: int, length: int) -> Union[bytes, str]:
    assert start >= 0 and length >= 0

    if start + length > len(b):
        raise IndexError("Slice out of bounds")

    return b[start : start + length]


def builtin_concat(*args):
    if len(args) < 2:
        raise ValueError("concat() requires at least 2 arguments")

    is_string = isinstance(args[0], str)
    is_bytes = isinstance(args[0], bytes)

    if not (is_string or is_bytes):
        raise TypeError("concat() arguments must be either all strings or all bytes")

    for arg in args[1:]:
        if is_string and not isinstance(arg, str):
            raise TypeError(
                "If first argument is string, all arguments must be strings"
            )
        if is_bytes and not isinstance(arg, bytes):
            raise TypeError("If first argument is bytes, all arguments must be bytes")

    if is_string:
        return "".join(args)
    else:
        return b"".join(args)


# all the convert functionality is adapted from vyper's test suite
# - https://github.com/vyperlang/vyper/blob/c32b9b4c6f0d8b8cdb103d3017ff540faf56a305/tests/functional/builtins/codegen/test_convert.py#L301
def builtin_convert(val: Any, o_typ: VyperType):
    """
    Perform conversion on the Python representation of a Vyper value.
    Returns None if the conversion is invalid (i.e., would revert in Vyper)
    """
    # Get input type from the boxed value
    i_typ = val.typ

    if isinstance(i_typ, IntegerT) and isinstance(o_typ, IntegerT):
        return convert_utils._convert_int_to_int(val, o_typ)

    if isinstance(i_typ, DecimalT) and isinstance(o_typ, IntegerT):
        return convert_utils._convert_decimal_to_int(val, o_typ)

    if isinstance(i_typ, (BoolT, IntegerT)) and isinstance(o_typ, DecimalT):
        # Note: Decimal(True) == Decimal("1")
        return convert_utils._convert_int_to_decimal(val, o_typ)

    # Track actual bytes length for BytesT (needed for correct padding conversion).
    # StringT not tracked: String to int conversions are rejected by Vyper semantics.
    actual_bytes_len = None
    if isinstance(i_typ, BytesT):
        actual_bytes_len = len(val)

    val_bits = convert_utils._to_bits(val, i_typ)

    if isinstance(i_typ, (BytesT, StringT)) and not isinstance(
        o_typ, (BytesT, StringT)
    ):
        val_bits = val_bits[32:]
        # Right-pad to 32 bytes (needed for empty/short bytes and BytesM_T conversion)
        val_bits = val_bits.ljust(32, b"\x00")

    if convert_utils._padding_direction(i_typ) != convert_utils._padding_direction(
        o_typ
    ):
        # subtle! the padding conversion follows the bytes argument
        if isinstance(i_typ, (BytesM_T, BytesT)):
            # For BytesT, use actual value length, not type's max length
            if isinstance(i_typ, BytesT):
                n = actual_bytes_len
            else:
                n = convert_utils.bytes_of_type(i_typ)
            padding_byte = None
        else:
            # output type is bytes
            n = convert_utils.bytes_of_type(o_typ)
            padding_byte = b"\x00"

        val_bits = convert_utils._padconvert(
            val_bits, convert_utils._padding_direction(o_typ), n, padding_byte
        )

    # Sign extension for signed output types from BytesM_T or BytesT
    if getattr(o_typ, "is_signed", False) and isinstance(i_typ, (BytesM_T, BytesT)):
        if isinstance(i_typ, BytesT):
            n_bits = actual_bytes_len * 8
        else:
            n_bits = convert_utils._bits_of_type(i_typ)
        # Only sign-extend if there are bits to extend from (empty bytes = 0)
        if n_bits > 0:
            val_bits = convert_utils._signextend(val_bits, n_bits)

    try:
        if isinstance(o_typ, BoolT):
            return convert_utils._from_bits(val_bits, UINT256_T) != 0

        ret = convert_utils._from_bits(val_bits, o_typ)

        if isinstance(o_typ, AddressT):
            return Address(ret)
        return ret

    except convert_utils.ConvertError:
        raise ValueError(f"Cannot convert value {val} of typ {i_typ} to {o_typ}")


def builtin_create_copy_of(
    evm: EVMCore,
    target: Address,
    value: int = 0,
    revert_on_failure: bool = True,
    salt: Optional[bytes] = None,
) -> Address:
    if salt is not None:
        raise UnsupportedFeature(
            "create_copy_of with salt (CREATE2) is unsupported in Ivy: "
            "https://github.com/cyberthirst/ivy/issues/21"
        )
    code = create_utils.deepcopy_code(evm.state, target)
    if code is None:
        # Target has no code to copy - this is an extcodesize check failure
        # which always reverts regardless of revert_on_failure flag
        raise Revert(data=b"")
    return create_utils.create_builtin_shared(
        evm,
        code,
        data=b"",
        value=value,
        revert_on_failure=revert_on_failure,
        salt=salt,
        is_runtime_copy=True,
    )


def builtin_create_from_blueprint(
    evm: EVMCore,
    target: Address,
    *args,
    value: int = 0,
    raw_args: bool = False,
    code_offset: int = 3,
    revert_on_failure: bool = True,
    salt: Optional[bytes] = None,
) -> Address:
    if salt is not None:
        raise UnsupportedFeature(
            "create_from_blueprint with salt (CREATE2) is unsupported in Ivy: "
            "https://github.com/cyberthirst/ivy/issues/21"
        )
    # reset_global_vars=True because blueprint creation runs the constructor
    # which will allocate fresh variables
    code = create_utils.deepcopy_code(evm.state, target, reset_global_vars=True)
    if code is None:
        # Blueprint target has no code - this is an extcodesize check failure
        # which always reverts regardless of revert_on_failure flag
        raise Revert(data=b"")

    values = args  # remaining positional args are constructor values

    if not raw_args:
        # encode the arguments - types are read from the boxed values
        if values:
            encoded_args = builtin_abi_encode(*values)
        else:
            encoded_args = b""
    else:
        assert len(values) == 1
        encoded_args = values[0]

    if code_offset != 3:
        raise UnimplementedException("Code offset != 3 is not supported")

    return create_utils.create_builtin_shared(
        evm,
        code,
        data=encoded_args,
        value=value,
        revert_on_failure=revert_on_failure,
        salt=salt,
    )


def builtin_create_minimal_proxy_to(
    evm: EVMCore,
    target: Address,
    value: int = 0,
    revert_on_failure: bool = True,
    salt: Optional[bytes] = None,
) -> Address:
    # target is an Address which has .typ
    encoded_target = builtin_abi_encode(target)
    code = create_utils.MinimalProxyFactory.get_proxy_contract_data()
    return create_utils.create_builtin_shared(
        evm,
        code,
        data=encoded_target,
        value=value,
        revert_on_failure=revert_on_failure,
        salt=salt,
    )


def builtin_raw_revert(x: VyperBytes):
    assert isinstance(x, VyperBytes)
    raise Revert(data=x)


def builtin_unsafe_add(x, y):
    bits, signed = unsafe_math.get_int_params(x, y)
    return unsafe_math.wrap_value(x + y, bits, signed)


def builtin_unsafe_sub(x, y):
    bits, signed = unsafe_math.get_int_params(x, y)
    return unsafe_math.wrap_value(x - y, bits, signed)


def builtin_unsafe_mul(x, y):
    bits, signed = unsafe_math.get_int_params(x, y)
    return unsafe_math.wrap_value(x * y, bits, signed)


def builtin_unsafe_div(x, y):
    bits, signed = unsafe_math.get_int_params(x, y)
    return unsafe_math.wrap_value(unsafe_math.evm_div(x, y), bits, signed)


def builtin_floor(x):
    assert isinstance(x, VyperDecimal)
    return x.value // x.SCALING_FACTOR


def builtin_ceil(x):
    assert isinstance(x, VyperDecimal)
    # negate, floor divide (rounds toward negative infinity), negate (changing the floor to ceil)
    return -((-x.value) // x.SCALING_FACTOR)


def builtin_epsilon(typ):
    assert isinstance(typ, DecimalT)
    return VyperDecimal(1, scaled=True)


def builtin_abs(x: VyperInt):
    """Absolute value for int256."""
    assert isinstance(x, VyperInt)
    if x == INT256_T.int_bounds[0]:
        raise Assert(data=b"")
    return abs(x)


def builtin_sqrt(x: "VyperDecimal") -> "VyperDecimal":
    """
    The underlying value is stored as an integer N = floor(d * 10**10).
    We want to compute floor(sqrt(d) * 10**10) in one step.

    Let S = 10**10 and N = x.value = floor(d · 10**10).

    Then:
      N · S = d · 10**10 · 10**10 = d · 10**20
      sqrt(N · S) = sqrt(d · 10**10) = sqrt(d) · 10**10
    """
    assert isinstance(x, VyperDecimal)
    if x.value < 0:
        raise ValueError("Square root of negative number")
    if x.value == 0:
        return VyperDecimal(0, scaled=True)

    y_scaled: int = math.isqrt(x.value * VyperDecimal.SCALING_FACTOR)
    return VyperDecimal(y_scaled, scaled=True)


def builtin_isqrt(a: int) -> int:
    if a < 0:
        raise ValueError("Square root of negative number")
    return math.isqrt(a)


def builtin_uint256_addmod(a: int, b: int, c: int) -> int:
    """Return (a + b) % c. Reverts if c == 0.
    The intermediate (a + b) is not subject to 2^256 modulo (EVM ADDMOD semantics).
    """
    if c == 0:
        raise Revert(data=b"")
    return (a + b) % c


def builtin_uint256_mulmod(a: int, b: int, c: int) -> int:
    """Return (a * b) % c. Reverts if c == 0.
    The intermediate (a * b) is not subject to 2^256 modulo (EVM MULMOD semantics).
    """
    if c == 0:
        raise Revert(data=b"")
    return (a * b) % c


def builtin_shift(x: int, shift: int) -> int:
    """Shift x by shift bits. Positive = left shift, negative = right shift.
    For signed types, converts the result to signed representation when >= 2^255.
    """
    if shift >= 256:
        result = 0
    elif shift >= 0:
        result = (x << shift) % 2**256
    else:
        if -shift >= 256:
            result = -1 if x.typ.is_signed and int(x) < 0 else 0
        else:
            result = x >> (-shift)

    if x.typ.is_signed and result >= 2**255:
        result -= 2**256
    return result


def builtin_keccak256(value: Union[str, bytes]) -> bytes:
    if isinstance(value, str):
        # Convert string to bytes using UTF-8 encoding
        value = value.encode("utf-8")
    elif not isinstance(value, bytes):
        raise TypeError(f"keccak256 expects bytes or string, got {type(value)}")

    return keccak(value)


def builtin_extract32(
    b: VyperBytes, start: VyperInt, output_type: Optional[VyperType] = None
) -> Any:
    if output_type is None:
        output_type = BYTES32_T
    assert isinstance(b, VyperBytes)
    assert start >= 0

    # Revert if we can't extract 32 bytes from position start
    if start + 32 > len(b):
        raise Revert(data=b"")

    # Extract 32 bytes
    extracted = b[start : start + 32]

    # Convert based on output_type
    if isinstance(output_type, BytesM_T):
        # For bytesN, the first N bytes are the value, remaining bytes must be zero
        n = output_type.length
        if n < 32 and extracted[n:] != b"\x00" * (32 - n):
            raise Revert(data=b"")
        return extracted[:n]
    elif isinstance(output_type, AddressT):
        # Address: first 12 bytes must be zero (right-aligned in 32-byte word)
        if extracted[:12] != b"\x00" * 12:
            raise Revert(data=b"")
        addr_int = int.from_bytes(extracted[-20:], "big")
        return Address(addr_int)
    elif isinstance(output_type, IntegerT):
        # Integer: interpret as big-endian
        # For signed output types, interpret as signed 256-bit first
        value = int.from_bytes(extracted, "big", signed=False)
        if output_type.is_signed and value >= 2**255:
            value -= 2**256
        # Check bounds for target type
        low, high = output_type.int_bounds
        if not (low <= value <= high):
            raise Revert(data=b"")
        return value
    else:
        raise TypeError(f"extract32: unsupported output_type {output_type}")


def builtin_ecrecover(
    h: bytes, v: int, r: Union[int, bytes], s: Union[int, bytes]
) -> Address:
    # Normalize r and s to ints if they're bytes
    if isinstance(r, bytes):
        r = int.from_bytes(r, "big")
    if isinstance(s, bytes):
        s = int.from_bytes(s, "big")

    # Build precompile calldata: hash(32) + v(32) + r(32) + s(32)
    calldata = h + v.to_bytes(32, "big") + r.to_bytes(32, "big") + s.to_bytes(32, "big")

    # Call precompile
    result = precompile_ecrecover(calldata)

    # Return Address(0) on failure (empty result), else recovered address
    if len(result) != 32:
        return Address(0)
    return Address(result[-20:])


def builtin_selfdestruct(evm: EVMCore, beneficiary: Address) -> None:
    """Vyper selfdestruct builtin - delegates to EVM."""
    evm.selfdestruct(beneficiary)

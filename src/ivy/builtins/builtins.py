from typing import Any, Union, Optional

from vyper.exceptions import UnimplementedException
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
from vyper.semantics.types.shortcuts import UINT256_T
from vyper.codegen.core import calculate_type_for_external_return
from vyper.utils import method_id


from ivy.abi import abi_decode, abi_encode
import ivy.builtins.create_utils as create_utils
from ivy.context import ExecutionOutput
from ivy.evaluator import VyperEvaluator
from ivy.exceptions import GasReference, Revert
from ivy.types import Address, VyperDecimal
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


def builtin_as_wei_value(value: int, denom: str):
    if denom == "ether":
        return value * 10**18
    if denom == "wei":
        return value
    if denom == "gwei":
        return value * 10**9
    raise UnimplementedException()


def builtin_len(x):
    return len(x)


def builtin_print(*args):
    print(*args)


def builtin_abi_decode(data: bytes, typ: VyperType, unwrap_tuple=True):
    assert isinstance(data, bytes)
    assert isinstance(typ, VyperType)

    if unwrap_tuple is True:
        typ = calculate_type_for_external_return(typ)
        return abi_decode(typ, data)[0]
    else:
        return abi_decode(typ, data)


# we don't follow the api of vyper's abi_encode - this is because we also need the
# types of the arguments which in the case of the compiler are available as metadata
# of the arguments
# however, in ivy, we represent values as primitive python objects, and we don't associate them
# with the corresponding vyper type. thus, we have to retrieve the types from
# the ast nodes and pass them through
def builtin_abi_encode(
    typs: tuple[VyperType], *values: tuple[Any], ensure_tuple=True, method_id=None
):
    assert len(typs) == len(values)
    assert isinstance(values, tuple) and isinstance(typs, tuple)

    if len(values) == 1 and not ensure_tuple:
        # unwrap tuple
        typs = typs[0]
        encode_input = values[0]
    else:
        typs = TupleT(list(arg for arg in typs))
        # values are already a tuple
        encode_input = values

    ret = abi_encode(typs, encode_input)

    if method_id is not None:
        ret = method_id + ret

    return ret


def builtin_empty(typ):
    return VyperEvaluator.default_value(typ)


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


# TODO handle typ
def builtin_method_id(method: str, output_type: VyperType = None):
    return method_id(method)


def builtin_send(evm: EVMCore, to, value, gas: int = 0) -> None:
    if gas != 0:
        raise GasReference()
    builtin_raw_call(evm, to, b"", value=value)


def builtin_raw_call(
    evm: EVMCore,
    to: Address,
    data: bytes,
    max_outsize: int = 0,
    gas: int = None,
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
def builtin_convert(typs: tuple[VyperType], *values: tuple[Any, VyperType]):
    assert len(typs) == 2
    i_typ = typs[0]
    val, o_typ = values

    """
    Perform conversion on the Python representation of a Vyper value.
    Returns None if the conversion is invalid (i.e., would revert in Vyper)
    """
    if isinstance(i_typ, IntegerT) and isinstance(o_typ, IntegerT):
        return convert_utils._convert_int_to_int(val, o_typ)

    if isinstance(i_typ, DecimalT) and isinstance(o_typ, IntegerT):
        return convert_utils._convert_decimal_to_int(val, o_typ)

    if isinstance(i_typ, (BoolT, IntegerT)) and isinstance(o_typ, DecimalT):
        # Note: Decimal(True) == Decimal("1")
        return convert_utils._convert_int_to_decimal(val, o_typ)

    val_bits = convert_utils._to_bits(val, i_typ)

    if isinstance(i_typ, (BytesT, StringT)) and not isinstance(
        o_typ, (BytesT, StringT)
    ):
        val_bits = val_bits[32:]

    if convert_utils._padding_direction(i_typ) != convert_utils._padding_direction(
        o_typ
    ):
        # subtle! the padding conversion follows the bytes argument
        if isinstance(i_typ, (BytesM_T, BytesT)):
            n = convert_utils.bytes_of_type(i_typ)
            padding_byte = None
        else:
            # output type is bytes
            n = convert_utils.bytes_of_type(o_typ)
            padding_byte = b"\x00"

        val_bits = convert_utils._padconvert(
            val_bits, convert_utils._padding_direction(o_typ), n, padding_byte
        )

    if getattr(o_typ, "is_signed", False) and isinstance(i_typ, BytesM_T):
        n_bits = convert_utils._bits_of_type(i_typ)
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
    code = create_utils.deepcopy_code(evm.state, target)
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
    typs: tuple[VyperType],
    target: Address,
    values: tuple[Any],
    value: int = 0,
    raw_args: bool = False,
    code_offset: int = 3,
    revert_on_failure: bool = True,
    salt: Optional[bytes] = None,
) -> Address:
    code = create_utils.deepcopy_code(evm.state, target)

    if not raw_args:
        # encode the arguments
        encoded_args = builtin_abi_encode(typs, values)
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
    encoded_target = builtin_abi_encode((AddressT(),), *(target,))
    code = create_utils.MinimalProxyFactory.get_proxy_contract_data()
    return create_utils.create_builtin_shared(
        evm,
        code,
        data=encoded_target,
        value=value,
        revert_on_failure=revert_on_failure,
        salt=salt,
    )


def builtin_raw_revert(x):
    assert isinstance(x, bytes)
    raise Revert(data=x)


def builtin_unsafe_add(typs, x, y):
    bits, signed = unsafe_math.validate_typs(typs)
    return unsafe_math.wrap_value(x + y, bits, signed)


def builtin_unsafe_sub(typs, x, y):
    bits, signed = unsafe_math.validate_typs(typs)
    return unsafe_math.wrap_value(x - y, bits, signed)


def builtin_unsafe_mul(typs, x, y):
    bits, signed = unsafe_math.validate_typs(typs)
    return unsafe_math.wrap_value(x * y, bits, signed)


def builtin_unsafe_div(typs, x, y):
    bits, signed = unsafe_math.validate_typs(typs)
    return unsafe_math.wrap_value(unsafe_math.evm_div(x, y), bits, signed)


def builtin_floor(x):
    assert isinstance(x, VyperDecimal)
    return x.value // x.SCALING_FACTOR

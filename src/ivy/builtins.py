from typing import Callable, Any, Union

from vyper.semantics.types import VyperType, TupleT, DecimalT
from vyper.codegen.core import calculate_type_for_external_return
from vyper.utils import method_id


from ivy.abi import abi_decode, abi_encode
from ivy.evaluator import VyperEvaluator
from ivy.evm_structures import EVMOutput
from ivy.exceptions import GasReference
from ivy.types import Address, VyperDecimal


def builtin_range(*args, bound=None):
    if len(args) == 2:
        start, stop = args
    else:
        assert len(args) == 1
        start, stop = 0, args[0]

    if bound:
        if stop > bound and len(args) == 1:
            raise RuntimeError(f"Stop value is greater than bound={bound} value")
        if stop - start > bound:
            raise RuntimeError(f"Range is greater than bound={bound} value")

    return range(start, stop)


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


def builtin__abi_decode(data: bytes, typ: VyperType, unwrap_tuple=True):
    return builtin_abi_decode(data, typ, unwrap_tuple)


# we don't follow the api of vyper's abi_encode - this is because we also need the
# types of the arguments which in the case of the compiler are available as metadata
# of the arguments
# however, in ivy, we represent values as python objects, and we don't associate them
# with the corresponding vyper type. thus, we have to retrieve the types from
# the ast nodes and pass them through
def builtin_abi_encode(
    typs: tuple[VyperType], values: tuple[Any], ensure_tuple=True, method_id=None
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


def builtin__abi_encode(
    typs: tuple[VyperType], values: tuple[Any], ensure_tuple=True, method_id=None
):
    return builtin_abi_encode(typs, values, ensure_tuple, method_id)


def builtin_empty(typ):
    return VyperEvaluator.default_value(typ)


def get_bound(typ, get_high: bool):
    if isinstance(typ, DecimalT):
        return VyperDecimal.max() if get_high else VyperDecimal.min()
    low, high = typ.int_bounds
    return high if get_high else low


def builtin_max_value(typ):
    return get_bound(typ, get_high=True)


def builtin_min_value(typ):
    return get_bound(typ, get_high=False)


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


def builtin_send(mesage_call, to, value, gas: int = 0) -> None:
    if gas != 0:
        raise GasReference()
    builtin_raw_call(mesage_call, to, b"", value=value)


def builtin_raw_call(
    message_call: Callable,
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

    output: EVMOutput = message_call(to, value, data, is_static_call, is_delegate_call)

    success = not output.is_error

    if not revert_on_failure:
        return success, output.bytes_output(safe=False)[:max_outsize]

    if not success:
        raise output.error

    return output.bytes_output()[:max_outsize]


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

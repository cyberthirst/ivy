from typing import Callable

from vyper.semantics.types import VyperType
from vyper.codegen.core import calculate_type_for_external_return
from vyper.utils import method_id


from ivy.abi import abi_decode
from ivy.evaluator import VyperEvaluator
from titanoboa.boa.util.abi import Address


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


"""
*args: Arbitrary arguments

ensure_tuple: If set to True, ensures that even a single argument is encoded as a tuple.
In other words, bytes gets encoded as (bytes,), and (bytes,) gets encoded as ((bytes,),)
This is the calling convention for Vyper and Solidity functions.
 Except for very specific use cases, this should be set to True. Must be a literal.
"""


def builtin_abi_encode(typ: VyperType, value):
    pass


def builtin__abi_encode(*args, value):
    # return abi_encode(*args, value)
    pass


def builtin_empty(typ):
    return VyperEvaluator.default_value(typ)


# TODO handle typ
def builtin_method_id(method: str, output_type: VyperType = None):
    return method_id(method)


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
        raise NotImplementedError("Gas is not supported in AST interpreter!")

    assert not (is_static_call and is_delegate_call)

    output, error = message_call(to, value, data, is_static_call, is_delegate_call)

    success = error is None

    if not revert_on_failure:
        return success, output[:max_outsize]

    if not success:
        raise error

    return output[:max_outsize]

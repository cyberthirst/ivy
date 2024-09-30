from vyper.semantics.types import VyperType
from vyper.codegen.core import calculate_type_for_external_return


from ivy.abi import abi_decode


def builtin_range(*args, bound=None):
    if len(args) == 2:
        start, stop = args
    else:
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


def builtin_abi_decode(*args, unwrap_tuple=True):
    assert len(args) == 2
    typ = args[1]
    data = args[0]
    assert isinstance(data, bytes)
    assert isinstance(typ, VyperType)
    if unwrap_tuple is True:
        typ = calculate_type_for_external_return(typ)
        return abi_decode(typ, data)[0]
    else:
        return abi_decode(typ, data)


def builtin__abi_decode(args, unwrap_tuple=True):
    return builtin_abi_decode(args, unwrap_tuple)

from vyper.semantics.types.primitives import _PrimT

from ivy.types import StaticArray, DynamicArray, Map


def decode_ivy_object(v, typ):
    if isinstance(v, (StaticArray, DynamicArray)):
        # see __len__ in StaticArray to understand why we use iter
        v = list(iter(v))
        if typ_needs_decode(typ.value_type):
            v = [decode_ivy_object(x, typ.value_type) for x in v]
    if isinstance(v, Map):
        v = dict(v)
        if typ_needs_decode(typ.value_type):
            v = {k: decode_ivy_object(v, typ.value_type) for k, v in v.items()}
    # TODO add struct and flag
    return v


def typ_needs_decode(typ):
    if isinstance(typ, _PrimT):
        return False
    return True

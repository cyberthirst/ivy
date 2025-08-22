from vyper.semantics.types.primitives import _PrimT

from ivy.types import StaticArray, DynamicArray, Map, Address, Struct


def decode_ivy_object(v, typ):
    if isinstance(v, (StaticArray, DynamicArray)):
        # see __len__ in StaticArray to understand why we use iter
        v = list(iter(v))
        if typ_needs_decode(typ.value_type):
            v = [decode_ivy_object(x, typ.value_type) for x in v]
    elif isinstance(v, Map):
        v = dict(v)
        if typ_needs_decode(typ.value_type):
            v = {k: decode_ivy_object(v, typ.value_type) for k, v in v.items()}
    elif isinstance(v, Address):
        v = str(v)
    elif isinstance(v, Struct):
        # Convert struct to dict like boa does
        result = {}
        for key in v._typ.members.keys():
            value = v[key]
            member_typ = v._typ.members[key]
            if typ_needs_decode(member_typ):
                value = decode_ivy_object(value, member_typ)
            result[key] = value
        v = result
    elif isinstance(v, tuple):
        v = list(v)
        for i, member_typ in enumerate(typ.member_types):
            if typ_needs_decode(member_typ):
                v[i] = decode_ivy_object(v[i], member_typ)
        v = tuple(v)
    # TODO add flag
    return v


def typ_needs_decode(typ):
    if isinstance(typ, _PrimT):
        return False
    return True

from vyper.semantics.types.primitives import _PrimT

from ivy.types import (
    StaticArray,
    DynamicArray,
    Map,
    Address,
    Struct,
    Tuple as IvyTuple,
    VyperInt,
    VyperBool,
    VyperBytes,
    VyperBytesM,
    VyperString,
)


def decode_ivy_object(v, typ):
    if isinstance(v, (StaticArray, DynamicArray)):
        # see __len__ in StaticArray to understand why we use iter
        v = list(iter(v))
        if typ_needs_decode(typ.value_type):
            v = [decode_ivy_object(x, typ.value_type) for x in v]
    elif isinstance(v, Map):
        v = dict(v)
        key_typ = typ.key_type
        value_typ = typ.value_type
        if typ_needs_decode(key_typ) or typ_needs_decode(value_typ):
            decoded = {}
            for k, value in v.items():
                if typ_needs_decode(key_typ):
                    k = decode_ivy_object(k, key_typ)
                if typ_needs_decode(value_typ):
                    value = decode_ivy_object(value, value_typ)
                decoded[k] = value
            v = decoded
    elif isinstance(v, Address):
        v = str(v)
    elif isinstance(v, Struct):
        # Convert struct to dict like boa does
        result = {}
        for key in v.typ.members.keys():
            value = v[key]
            member_typ = v.typ.members[key]
            if typ_needs_decode(member_typ):
                value = decode_ivy_object(value, member_typ)
            result[key] = value
        v = result
    elif isinstance(v, (tuple, IvyTuple)):
        v = list(v)
        for i, member_typ in enumerate(typ.member_types):
            if typ_needs_decode(member_typ):
                v[i] = decode_ivy_object(v[i], member_typ)
        v = tuple(v)
    # Unbox primitives to their raw Python types
    elif isinstance(v, VyperInt):
        v = int(v)
    elif isinstance(v, VyperBool):
        v = bool(v)
    elif isinstance(v, (VyperBytes, VyperBytesM)):
        v = bytes(v)
    elif isinstance(v, VyperString):
        v = str(v)
    # TODO add flag
    return v


def typ_needs_decode(typ):
    if isinstance(typ, _PrimT):
        return False
    return True

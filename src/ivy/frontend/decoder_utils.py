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
        v = [decode_ivy_object(x, typ.value_type) for x in v]
    elif isinstance(v, Map):
        v = {
            decode_ivy_object(k, typ.key_type): decode_ivy_object(val, typ.value_type)
            for k, val in v.items()
        }
    elif isinstance(v, Address):
        v = str(v)
    elif isinstance(v, Struct):
        v = {
            key: decode_ivy_object(v[key], v.typ.members[key])
            for key in v.typ.members
        }
    elif isinstance(v, (tuple, IvyTuple)):
        v = tuple(
            decode_ivy_object(v[i], mt) for i, mt in enumerate(typ.member_types)
        )
    elif isinstance(v, VyperInt):
        v = int(v)
    elif isinstance(v, VyperBool):
        v = bool(v)
    elif isinstance(v, (VyperBytes, VyperBytesM)):
        v = bytes(v)
    elif isinstance(v, VyperString):
        v = str(v)
    return v

from ivy.types import VyperValue


def decode_ivy_object(v, typ):
    if isinstance(v, VyperValue):
        return v.unbox()
    if isinstance(v, tuple):
        return tuple(
            decode_ivy_object(v[i], mt) for i, mt in enumerate(typ.member_types)
        )
    return v

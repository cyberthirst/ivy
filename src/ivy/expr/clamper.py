from typing import Any

from vyper.semantics.types import (
    IntegerT,
    BytesT,
    BytesM_T,
    StringT,
    BoolT,
    SArrayT,
    DArrayT,
    TupleT,
    VyperType,
)
from vyper.semantics.types.base import _GenericTypeAcceptor

from ivy.types import (
    VyperInt,
    VyperBytes,
    VyperBytesM,
    VyperString,
    VyperBool,
    StaticArray,
    DynamicArray,
    Tuple as IvyTuple,
)


def _box_sequence_value(value: Any, typ: SArrayT | DArrayT) -> Any:
    if isinstance(typ, SArrayT) and isinstance(value, StaticArray) and value.typ == typ:
        return value
    if isinstance(typ, DArrayT) and isinstance(value, DynamicArray) and value.typ == typ:
        return value

    if isinstance(value, (StaticArray, DynamicArray)):
        assert typ.compare_type(value.typ) or value.typ.compare_type(typ), (
            f"Sequence type mismatch: {value.typ} not compatible with {typ}"
        )
        items = [value[i] for i in range(value.length)]
    else:
        assert isinstance(value, (list, tuple)), (
            f"Expected sequence for {typ}, got {type(value)}"
        )
        items = list(value)

    if isinstance(typ, SArrayT):
        if len(items) != typ.length:
            raise ValueError(
                f"Expected static array length {typ.length}, got {len(items)}"
            )
        return StaticArray(typ, {i: box_value(v, typ.value_type) for i, v in enumerate(items)})

    if len(items) > typ.length:
        raise ValueError(f"Cannot exceed maximum length {typ.length}")
    return DynamicArray(typ, {i: box_value(v, typ.value_type) for i, v in enumerate(items)})


def box_value(value: Any, typ: VyperType) -> Any:
    """
    Box a raw Python value with its Vyper type.

    For primitive types, this wraps the value in a type-aware container that
    validates bounds/length at construction time.

    For other types (decimal, arrays, structs, etc.) the value is
    returned as-is since they're either already boxed or can't be invalid.

    Raises Revert if the value is out of bounds for the given type.
    """
    if isinstance(typ, _GenericTypeAcceptor):
        if typ.type_ is StringT:
            if isinstance(value, bytes):
                byte_len = len(value)
            else:
                byte_len = len(value.encode("utf-8"))
            typ = StringT(byte_len)
        else:
            return value
    if isinstance(typ, IntegerT):
        if isinstance(value, VyperInt):
            if value.typ == typ:
                return value
            # Re-box with new type (e.g., after convert)
            return VyperInt(int(value), typ)
        return VyperInt(value, typ)

    if isinstance(typ, BytesT):
        if isinstance(value, VyperBytes):
            if value.typ == typ:
                return value
            return VyperBytes(bytes(value), typ)
        return VyperBytes(value, typ)

    if isinstance(typ, BytesM_T):
        if isinstance(value, VyperBytesM):
            if value.typ == typ:
                return value
            return VyperBytesM(bytes(value), typ)
        return VyperBytesM(value, typ)

    if isinstance(typ, StringT):
        if isinstance(value, VyperString):
            if value.typ == typ:
                return value
            # VyperString is bytes-based, preserve raw bytes when re-boxing
            return VyperString(bytes(value), typ)
        return VyperString(value, typ)

    if isinstance(typ, BoolT):
        if isinstance(value, VyperBool):
            return value
        return VyperBool(value)

    if isinstance(typ, (SArrayT, DArrayT)):
        return _box_sequence_value(value, typ)

    if isinstance(typ, TupleT):
        if isinstance(value, IvyTuple):
            return value
        # Convert Python tuple to IvyTuple, boxing each element
        assert isinstance(value, tuple), f"Expected tuple, got {type(value)}"
        member_types = typ.member_types
        assert len(value) == len(member_types)
        result = IvyTuple(typ)
        for i, (v, t) in enumerate(zip(value, member_types)):
            result[i] = box_value(v, t)
        return result

    # These types are either already validated or can't be invalid:
    # - DecimalT: VyperDecimal already validates
    # - StructT: Already boxed
    # - FlagT: Already boxed
    # - AddressT: Already boxed
    return value


def box_value_from_node(node: Any, value: Any) -> Any:
    """
    Box a value using the type from an AST node's metadata.

    This is a convenience wrapper around box_value that extracts
    the type from the node's _metadata["type"].
    """
    typ = node._metadata.get("type")
    if typ is None:
        assert value is None, f"Untyped node returned {type(value)}: {value}"
        return None
    return box_value(value, typ)

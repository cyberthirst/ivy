from typing import Any

from vyper.semantics.types import (
    IntegerT,
    BytesT,
    BytesM_T,
    StringT,
    VyperType,
)

from ivy.types import VyperInt, VyperBytes, VyperBytesM, VyperString


def box_value(value: Any, typ: VyperType) -> Any:
    """
    Box a raw Python value with its Vyper type.

    For integers, bytes (dynamic and fixed-size), and strings this wraps
    the value in a type-aware container that validates bounds/length at
    construction time.

    For other types (bool, decimal, arrays, structs, etc.) the value is
    returned as-is since they're either already boxed or can't be invalid.

    Raises Revert if the value is out of bounds for the given type.
    """
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
            return VyperString(str(value), typ)
        return VyperString(value, typ)

    # These types are either already validated or can't be invalid:
    # - BoolT: Python bool is always valid
    # - DecimalT: VyperDecimal already validates
    # - SArrayT, DArrayT: Already boxed with validation
    # - StructT: Already boxed
    # - FlagT: Already boxed
    # - AddressT: Already boxed
    # - TupleT: Already boxed
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

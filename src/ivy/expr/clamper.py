from typing import Any

from vyper.semantics.types import (
    IntegerT,
    BytesT,
    StringT,
    VyperType,
)

from ivy.types import VyperInt, VyperBytes, VyperString


def box_value(value: Any, typ: VyperType) -> Any:
    """
    Box a raw Python value with its Vyper type.

    For integers, bytes, and strings this wraps the value in a type-aware
    container that validates bounds/length at construction time.

    For other types (bool, decimal, arrays, structs, etc.) the value is
    returned as-is since they're either already boxed or can't be invalid.

    Raises Revert if the value is out of bounds for the given type.
    """
    if isinstance(typ, IntegerT):
        # Already boxed - no need to re-wrap
        if isinstance(value, VyperInt):
            return value
        return VyperInt(value, typ)

    if isinstance(typ, BytesT):
        if isinstance(value, VyperBytes):
            return value
        return VyperBytes(value, typ)

    if isinstance(typ, StringT):
        if isinstance(value, VyperString):
            return value
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
        return value
    return box_value(value, typ)

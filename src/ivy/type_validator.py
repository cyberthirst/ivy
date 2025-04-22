from typing import Any, Callable, Dict, Type

from vyper.semantics.types import (
    IntegerT,
    BoolT,
    BytesT,
    StringT,
    SArrayT,
    DArrayT,
    FlagT,
    DecimalT,
    StructT,
    HashMapT,
    InterfaceT,
)
from ivy.types import (
    StaticArray,
    DynamicArray,
    Flag,
    VyperDecimal,
)

VALIDATOR_REGISTRY: Dict[Type[Any], Callable[[Any, Any], None]] = {}


def register_validator(
    type_cls: Type[Any],
) -> Callable[[Callable[[Any, Any], None]], Callable[[Any, Any], None]]:
    """
    Decorator to register a validation function for a Vyper semantic type.

    Usage:
        @register_validator(IntegerT)
        def validate_integer(value, typ: IntegerT):
            ...
    """

    def decorator(fn: Callable[[Any, Any], None]) -> Callable[[Any, Any], None]:
        VALIDATOR_REGISTRY[type_cls] = fn
        return fn

    return decorator


def validate_value(node: Any, value: Any) -> None:
    typ = node._metadata["type"]
    validator = VALIDATOR_REGISTRY.get(type(typ))
    if validator is None:
        raise NotImplementedError(f"Missing validator for type {typ}")
    validator(value, typ)


@register_validator(IntegerT)
def validate_integer(value: int, typ: IntegerT) -> None:
    lo, hi = typ.ast_bounds
    if not (lo <= value <= hi):
        raise ValueError(f"Value {value} out of bounds for {typ}")


@register_validator(BoolT)
def validate_bool(value: bool, typ: BoolT) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"Expected a bool for {typ}, got {type(value).__name__}")


@register_validator(DecimalT)
def validate_decimal(value: VyperDecimal, typ: DecimalT) -> None:
    if not (VyperDecimal.min() <= value <= VyperDecimal.max()):
        raise ValueError(f"Value {value} out of bounds for {typ}")


def _validate_sequence_len(value: Any, length: int) -> None:
    actual = len(value)
    if actual != length:
        raise ValueError(f"Invalid length: expected {length}, got {actual}")


@register_validator(SArrayT)
def validate_static_array(value: StaticArray, typ: SArrayT) -> None:
    if value.length != typ.length:
        raise ValueError(
            f"Invalid length for {typ}: expected {typ.length}, got {value.length}"
        )
    for elem in value:
        validate_value(elem, elem)


@register_validator(DArrayT)
def validate_dynamic_array(value: DynamicArray, typ: DArrayT) -> None:
    if len(value) > typ.count:
        raise ValueError(
            f"Invalid length for {typ}: expected at most {typ.count}, got {len(value)}"
        )
    for elem in value:
        validate_value(elem, elem)


@register_validator(BytesT)
def validate_bytes(value: bytes, typ: BytesT) -> None:
    if len(value) > typ.length:
        raise ValueError(
            f"Invalid bytes length for {typ}: expected at most {typ.length}, got {len(value)}"
        )


@register_validator(StringT)
def validate_string(value: str, typ: StringT) -> None:
    if len(value) > typ.length:
        raise ValueError(
            f"Invalid string length for {typ}: expected at most {typ.length}, got {len(value)}"
        )


@register_validator(FlagT)
def validate_flag(value: Flag, typ: FlagT) -> None:
    if (value.value >> len(typ._flag_members)) != 0:
        raise ValueError(f"Invalid flag value {value} for {typ}")


@register_validator(StructT)
def validate_struct(value: Any, typ: StructT) -> None:
    for name, member_typ in typ.members.items():
        member_value = getattr(value, name)
        validate_value(member_value, member_value)


@register_validator(HashMapT)
def validate_map(value: Any, typ: HashMapT) -> None:
    for k, v in value.items():
        continue


@register_validator(InterfaceT)
def validate_interface(value: Any, typ: InterfaceT) -> None:
    return None

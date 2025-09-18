from typing import Any, Callable, Type

from vyper.semantics.types import (
    IntegerT,
    DArrayT,
    SArrayT,
    BytesT,
    BytesM_T,
    StringT,
    StructT,
    HashMapT,
    BoolT,
    AddressT,
    InterfaceT,
    FlagT,
    TupleT,
    DecimalT,
)
from ivy.types import (
    Address,
    Struct,
    Flag,
    StaticArray,
    DynamicArray,
    Map,
    VyperDecimal,
    Tuple as IvyTuple,
)

DEFAULT_FACTORY_REGISTRY: dict[Type[Any], Callable[[Any], Any]] = {}


def register_default(
    type_cls: Type[Any],
) -> Callable[[Callable[[Any], Any]], Callable[[Any], Any]]:
    """
    Decorator to register a default-value factory for a given Vyper semantic type.

    Usage:
        @register_default(IntegerT)
        def default_integer(typ: IntegerT) -> int:
            return 0
    """

    def decorator(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
        DEFAULT_FACTORY_REGISTRY[type_cls] = fn
        return fn

    return decorator


def get_default_value(typ: Any) -> Any:
    factory = DEFAULT_FACTORY_REGISTRY.get(type(typ))
    if factory is None:
        raise KeyError(f"No default factory registered for type {type(typ)}")
    return factory(typ)


@register_default(IntegerT)
def default_integer(_: IntegerT) -> int:
    return 0


@register_default(DecimalT)
def default_decimal(_: DecimalT) -> VyperDecimal:
    return VyperDecimal(0)


@register_default(BoolT)
def default_bool(_: BoolT) -> bool:
    return False


@register_default(AddressT)
@register_default(InterfaceT)
def default_address(_: Any) -> Address:
    return Address(0)


@register_default(FlagT)
def default_flag(typ: FlagT) -> Flag:
    return Flag(typ, 0)


@register_default(DArrayT)
def default_dynamic_array(typ: DArrayT) -> DynamicArray:
    return DynamicArray(typ)


@register_default(SArrayT)
def default_static_array(typ: SArrayT) -> StaticArray:
    return StaticArray(typ)


@register_default(BytesT)
def default_bytes(_: BytesT) -> bytes:
    # variable length byte array defaults to empty bytes
    return b""


@register_default(BytesM_T)
def default_fixed_bytes(typ: BytesM_T) -> bytes:
    # fixed-length bytes default to zero-filled
    return b"\x00" * typ.length


@register_default(StringT)
def default_string(_: StringT) -> str:
    return ""


@register_default(StructT)
def default_struct(typ: StructT) -> Struct:
    # build defaults for each member
    defaults = {
        name: get_default_value(member_typ) for name, member_typ in typ.members.items()
    }
    return Struct(typ, defaults)


@register_default(HashMapT)
def default_map(typ: HashMapT) -> Map:
    return Map(typ)


@register_default(TupleT)
def default_tuple(typ: TupleT) -> IvyTuple:
    return IvyTuple(typ)

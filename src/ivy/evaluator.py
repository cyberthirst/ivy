from typing import Any

from vyper.ast import nodes as ast
from vyper.semantics.types import (
    BoolT,
    InterfaceT,
    FlagT,
    IntegerT,
    BytesT,
    StringT,
    StructT,
    AddressT,
    BytesM_T,
    TupleT,
    HashMapT,
    DArrayT,
    SArrayT,
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
)
from ivy.visitor import BaseClassVisitor


class VyperValidator:
    @classmethod
    def validate_IntegerT(cls, value, typ):
        lo, hi = typ.ast_bounds
        if not lo <= value <= hi:
            raise ValueError(f"Value {value} out of bounds for {typ}")
        return True

    @classmethod
    def validate_BoolT(cls, value, typ):
        return True

    @classmethod
    def validate_sequence_len(cls, value, typ):
        if isinstance(value, StaticArray):
            assert isinstance(typ, SArrayT)
            assert value.length == typ.length
            return
        if len(value) > typ.length:
            raise ValueError(
                f"Invalid length for {typ}: expected at most {typ.count}, got {len(value)}"
            )

    @classmethod
    def validate_BytesT(cls, value, typ):
        cls.validate_sequence_len(value, typ)

    @classmethod
    def validate_StringT(cls, value, typ):
        cls.validate_sequence_len(value, typ)

    @classmethod
    def validate_SequenceT(cls, value, typ):
        cls.validate_sequence_len(value, typ)
        for item in value:
            cls.validate_value(item, typ.value_type)

    @classmethod
    def validate_FlagT(cls, value, typ):
        return value.value >> len(typ._flag_members) == 0

    @classmethod
    def validate_DecimalT(cls, value, typ):
        if not VyperDecimal.min() <= value <= VyperDecimal.max():
            raise ValueError(f"Value {value} out of bounds for {typ}")

    @classmethod
    def validate_StructT(cls, value, typ):
        pass

    @classmethod
    def validate_HashmapT(cls, value, typ):
        pass

    @classmethod
    def validate_InterfaceT(cls, value, typ):
        pass

    # TODO: create a proper generic visitor (duplicated code with BaseVisitor)
    # currently we don't inherit from BaseVisitor as it's instance based, also
    # we retrieve the type from metadata
    @classmethod
    def validate_value(cls, node, value):
        typ = node._metadata["type"]
        method_name = f"validate_{typ.__class__.__name__}"
        validator = getattr(cls, method_name)
        validator(value, typ)


# TODO maybe rethink the direct usage of operators and add explicit validation
# so we better mimic safe math operations
class VyperEvaluator(BaseClassVisitor, VyperValidator):
    pass

    # TODO: rewrite to smth like dict for const-time dispatch
    # NOTE: maybe should just lazily fetch the default value for state vars?
    @classmethod
    def default_value(cls, typ):
        if isinstance(typ, IntegerT):
            return 0
        if isinstance(typ, DArrayT):
            return DynamicArray(typ)
        if isinstance(typ, SArrayT):
            return StaticArray(typ)
        if isinstance(typ, BytesT):
            return b""
        if isinstance(typ, BytesM_T):
            return b"\x00" * typ.length
        if isinstance(typ, StringT):
            return ""
        if isinstance(typ, StructT):
            kws = {k: cls.default_value(v) for k, v in typ.members.items()}
            return Struct(typ, kws)
        if isinstance(typ, HashMapT):
            return Map(typ)
        if isinstance(typ, BoolT):
            return False
        if isinstance(typ, AddressT) or isinstance(typ, InterfaceT):
            return Address(0)
        if isinstance(typ, FlagT):
            return Flag(typ, 0)
        if isinstance(typ, TupleT):
            return tuple(cls.default_value(t) for t in typ.member_types)
        if isinstance(typ, DecimalT):
            return VyperDecimal(0)
        raise NotImplementedError(f"Default value for {typ} not implemented")

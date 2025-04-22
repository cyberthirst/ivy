from vyper.semantics.types import (
    SArrayT,
)

from ivy.types import (
    StaticArray,
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

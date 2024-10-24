from typing import Any
from collections import defaultdict
from operator import (
    add,
    sub,
    mul,
    truediv,
    floordiv,
    mod,
    pow,
    lshift,
    rshift,
    and_,
    or_,
    xor,
)

from vyper.ast import nodes as ast
from vyper.semantics.types import BoolT, InterfaceT
from vyper.semantics.types.primitives import IntegerT
from vyper.semantics.types.subscriptable import _SequenceT, HashMapT
from vyper.semantics.types.bytestrings import BytesT, StringT
from vyper.semantics.types.user import StructT

from ivy.types import Address, Struct
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
    @classmethod
    def visit_And(cls, _, values):
        return all(values)

    @classmethod
    def visit_Or(cls, _, values):
        return any(values)

    @classmethod
    def visit_Not(cls, _, operand):
        return not operand

    @classmethod
    def visit_USub(cls, _, operand):
        return -operand

    @classmethod
    def visit_Invert(cls, _, operand):
        return ~operand

    @classmethod
    def visit_Add(cls, _, left, right):
        return add(left, right)

    @classmethod
    def visit_Sub(cls, _, left, right):
        return sub(left, right)

    @classmethod
    def visit_Mult(cls, _, left, right):
        return mul(left, right)

    @classmethod
    def visit_Div(cls, _, left, right):
        return truediv(left, right)

    @classmethod
    def visit_FloorDiv(cls, _, left, right):
        if right == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        # python has different rounding semantics than vyper
        # vyper rounds towards zero, python towards negative infinity
        # thus we need to use a custom function
        sign = -1 if (left * right) < 0 else 1
        return sign * (abs(left) // abs(right))

    @classmethod
    def visit_Mod(cls, _, left, right):
        if right == 0:
            raise ZeroDivisionError("Cannot modulo by zero")
        sgn = -1 if left < 0 else 1
        return sgn * (abs(left) % abs(right))

    @classmethod
    def visit_Pow(cls, _, left, right):
        # exponentiation by negative number is not allowed as negative numbers
        # are represented via two's complement in EVM and thus the exponentiation
        # would lead to overflow for any base but 0, 1, -1
        # for consistency, Vyper disallows it for all bases and we follow this
        # so this is basically just a quick path out
        if right < 0:
            raise ValueError("Exponentiation by negative number")

        # optimization - calling `pow` with large numbers is computationally expensive
        # when we're sure the result won't fit in 256 bits, we raise an error early
        if left not in (0, 1, -1) and abs(right) > 256:
            raise ValueError(
                f"Exponentiation {left} ** {right} too large for the given type"
            )

        return pow(left, right)

    @classmethod
    def visit_LShift(cls, _, left, right):
        return lshift(left, right)

    @classmethod
    def visit_RShift(cls, _, left, right):
        return rshift(left, right)

    @classmethod
    def visit_BitOr(cls, _, left, right):
        return or_(left, right)

    @classmethod
    def visit_BitXor(cls, _, left, right):
        return xor(left, right)

    @classmethod
    def visit_BitAnd(cls, _, left, right):
        return and_(left, right)

    @classmethod
    def visit_Eq(cls, _, left, right):
        return left == right

    @classmethod
    def visit_NotEq(cls, _, left, right):
        return left != right

    @classmethod
    def visit_Lt(cls, _, left, right):
        return left < right

    @classmethod
    def visit_LtE(cls, _, left, right):
        return left <= right

    @classmethod
    def visit_Gt(cls, _, left, right):
        return left > right

    @classmethod
    def visit_GtE(cls, _, left, right):
        return left >= right

    @classmethod
    def eval_boolop(cls, op, values):
        res = cls.visit(op.op, values)
        cls.validate_value(op, res)
        return res

    @classmethod
    def eval_unaryop(cls, op, operand):
        res = cls.visit(op.op, operand)
        cls.validate_value(op, res)
        return res

    # aug_assign node is not annotated with a type, so we take the type from the target
    # alternatively we could fetch the type at the call site and have it passed in
    @classmethod
    def eval_binop(cls, op: ast.BinOp, left: Any, right: Any, aug_assign=False):
        res = cls.visit(op.op, left, right)
        if aug_assign:
            cls.validate_value(op.target, res)
        else:
            cls.validate_value(op, res)
        return res

    @classmethod
    def eval_compare(cls, op: ast.Compare, left, right):
        res = cls.visit(op.op, left, right)
        cls.validate_value(op, res)
        return res

    # rewrite to smth like dict for const-time dispatch
    @classmethod
    def default_value(cls, typ):
        if isinstance(typ, IntegerT):
            return 0
        if isinstance(typ, _SequenceT):
            return []
        if isinstance(typ, BytesT):
            return b""
        if isinstance(typ, StringT):
            return ""
        if isinstance(typ, StructT):
            kws = {k: cls.default_value(v) for k, v in typ.members.items()}
            return cls.construct_struct(typ.name, kws)
        if isinstance(typ, HashMapT):
            return defaultdict(lambda: cls.default_value(typ.value_type))
        if isinstance(typ, BoolT):
            return False
        if isinstance(typ, Address) or isinstance(typ, InterfaceT):
            return Address(0)
        return None

    @classmethod
    def construct_struct(cls, name, kws):
        return Struct(name, kws)

# ivy/evaluator/operators.py
from __future__ import annotations

from typing import Callable, TypeAlias
from vyper.ast import nodes as ast

from ivy.types import (
    DynamicArray,
    Flag,
    StaticArray,
    VyperBool,
    VyperBytesM,
    VyperDecimal,
    VyperInt,
    VyperValue,
)

OperatorResult: TypeAlias = VyperValue | bool | int
OperatorHandler: TypeAlias = Callable[..., OperatorResult]

NumericValue: TypeAlias = VyperInt | VyperDecimal
NumericResult: TypeAlias = int | VyperDecimal
BitwiseValue: TypeAlias = VyperInt | VyperBytesM | Flag
BitwiseResult: TypeAlias = int | VyperBytesM | Flag
ShiftableValue: TypeAlias = VyperInt | VyperBytesM
ShiftResult: TypeAlias = int | VyperBytesM
SequenceValue: TypeAlias = StaticArray | DynamicArray
MembershipTarget: TypeAlias = SequenceValue | Flag

OPERATOR_REGISTRY: dict[type[ast.VyperNode], OperatorHandler] = {}


def register_operator(
    op_cls: type[ast.VyperNode],
) -> Callable[[OperatorHandler], OperatorHandler]:
    """
    Register a handler for a given AST operator class.

    Usage:
        @register_operator(ast.Add)
        def add(left, right):
            return left + right
    """

    def decorator(fn: OperatorHandler) -> OperatorHandler:
        OPERATOR_REGISTRY[op_cls] = fn
        return fn

    return decorator


def get_operator_handler(op_node: ast.VyperNode) -> OperatorHandler:
    handler = OPERATOR_REGISTRY.get(type(op_node))
    if handler is None:
        raise NotImplementedError(f"{op_node} is not implemented")
    return handler


# ----------------------------------------------------------------------------
# Arithmetic operators
# ----------------------------------------------------------------------------


@register_operator(ast.Add)
def add_op(left: NumericValue, right: NumericValue) -> NumericResult:
    return left + right


@register_operator(ast.Sub)
def sub_op(left: NumericValue, right: NumericValue) -> NumericResult:
    return left - right


@register_operator(ast.Mult)
def mul_op(left: NumericValue, right: NumericValue) -> NumericResult:
    return left * right


@register_operator(ast.Div)
def div_op(left: VyperDecimal, right: VyperDecimal) -> VyperDecimal:
    return left / right


@register_operator(ast.FloorDiv)
def floor_div_op(left: VyperInt, right: VyperInt) -> int:
    return left // right


@register_operator(ast.Mod)
def mod_op(left: NumericValue, right: NumericValue) -> NumericResult:
    return left % right


@register_operator(ast.Pow)
def pow_op(left: VyperInt, right: VyperInt) -> int:
    return left ** right


# ----------------------------------------------------------------------------
# Bitwise operators
# ----------------------------------------------------------------------------


@register_operator(ast.LShift)
def lshift_op(left: ShiftableValue, right: VyperInt) -> ShiftResult:
    return left << right


@register_operator(ast.RShift)
def rshift_op(left: ShiftableValue, right: VyperInt) -> ShiftResult:
    return left >> right


@register_operator(ast.BitOr)
def bitor_op(left: BitwiseValue, right: BitwiseValue) -> BitwiseResult:
    return left | right


@register_operator(ast.BitXor)
def bitxor_op(left: BitwiseValue, right: BitwiseValue) -> BitwiseResult:
    return left ^ right


@register_operator(ast.BitAnd)
def bitand_op(left: BitwiseValue, right: BitwiseValue) -> BitwiseResult:
    return left & right


# ----------------------------------------------------------------------------
# Comparison operators
# ----------------------------------------------------------------------------


@register_operator(ast.Eq)
def eq_op(left: VyperValue, right: VyperValue) -> bool:
    return left == right


@register_operator(ast.NotEq)
def not_eq_op(left: VyperValue, right: VyperValue) -> bool:
    return left != right


@register_operator(ast.Lt)
def lt_op(left: NumericValue, right: NumericValue) -> bool:
    return left < right


@register_operator(ast.LtE)
def lte_op(left: NumericValue, right: NumericValue) -> bool:
    return left <= right


@register_operator(ast.Gt)
def gt_op(left: NumericValue, right: NumericValue) -> bool:
    return left > right


@register_operator(ast.GtE)
def gte_op(left: NumericValue, right: NumericValue) -> bool:
    return left >= right


@register_operator(ast.In)
def in_op(left: VyperValue, right: MembershipTarget) -> bool:
    return left in right


@register_operator(ast.NotIn)
def not_in_op(left: VyperValue, right: MembershipTarget) -> bool:
    return left not in right


# ----------------------------------------------------------------------------
# Unary operators
# ----------------------------------------------------------------------------


@register_operator(ast.USub)
def usub_op(operand: NumericValue) -> NumericResult:
    return -operand


@register_operator(ast.Invert)
def invert_op(operand: BitwiseValue) -> BitwiseResult:
    return ~operand


# ----------------------------------------------------------------------------
# Note: Boolean operators (And, Or) are n-ary and use short-circuit logic.
# These remain implemented in ExprVisitor.visit_BoolOp


@register_operator(ast.Not)
def not_op(operand: VyperBool) -> bool:
    return not operand

# ivy/evaluator/operators.py
from typing import Any, Callable, Optional, Type
from vyper.ast import nodes as ast
from vyper.utils import unsigned_to_signed
from ivy.types import VyperDecimal, Flag


OPERATOR_REGISTRY: dict[Type[ast.VyperNode], Callable[..., Any]] = {}


def register_operator(
    op_cls: Type[ast.VyperNode],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Register a handler for a given AST operator class.

    Usage:
        @register_operator(ast.Add)
        def add(left, right):
            return left + right
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        OPERATOR_REGISTRY[op_cls] = fn
        return fn

    return decorator


def get_operator_handler(op_node: ast.VyperNode) -> Optional[Callable[..., Any]]:
    handler = OPERATOR_REGISTRY.get(type(op_node))
    if handler is None:
        raise NotImplementedError(f"{op_node} is not implemented")
    return OPERATOR_REGISTRY.get(type(op_node))


# ----------------------------------------------------------------------------
# Arithmetic operators
# ----------------------------------------------------------------------------


@register_operator(ast.Add)
def add_op(left: Any, right: Any) -> Any:
    return left + right


@register_operator(ast.Sub)
def sub_op(left: Any, right: Any) -> Any:
    return left - right


@register_operator(ast.Mult)
def mul_op(left: Any, right: Any) -> Any:
    return left * right


@register_operator(ast.Div)
def div_op(left: Any, right: Any) -> Any:
    return left / right


@register_operator(ast.FloorDiv)
def floor_div_op(left: Any, right: Any) -> Any:
    if isinstance(left, VyperDecimal):
        assert isinstance(right, VyperDecimal)
        return left // right
    # ints: truncate toward zero
    if right == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    sign = -1 if (left * right) < 0 else 1
    return sign * (abs(left) // abs(right))


@register_operator(ast.Mod)
def mod_op(left: Any, right: Any) -> Any:
    if isinstance(left, VyperDecimal):
        assert isinstance(right, VyperDecimal)
        return left % right
    if right == 0:
        raise ValueError("Cannot modulo by 0")
    sgn = -1 if left < 0 else 1
    return sgn * (abs(left) % abs(right))


@register_operator(ast.Pow)
def pow_op(left: Any, right: Any) -> Any:
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


# ----------------------------------------------------------------------------
# Bitwise operators
# ----------------------------------------------------------------------------


@register_operator(ast.LShift)
def lshift_op(left: int, right: int, *, typ) -> int:
    bits = typ.bits
    result = (left << right) % 2**bits
    if typ.is_signed:
        result = unsigned_to_signed(result, bits)
    return result


@register_operator(ast.RShift)
def rshift_op(left: int, right: int) -> int:
    return left >> right


@register_operator(ast.BitOr)
def bitor_op(left: int, right: int) -> int:
    return left | right


@register_operator(ast.BitXor)
def bitxor_op(left: int, right: int) -> int:
    return left ^ right


@register_operator(ast.BitAnd)
def bitand_op(left: int, right: int) -> int:
    return left & right


# ----------------------------------------------------------------------------
# Comparison operators
# ----------------------------------------------------------------------------


@register_operator(ast.Eq)
def eq_op(left: Any, right: Any) -> bool:
    return left == right


@register_operator(ast.NotEq)
def not_eq_op(left: Any, right: Any) -> bool:
    return left != right


@register_operator(ast.Lt)
def lt_op(left: Any, right: Any) -> bool:
    return left < right


@register_operator(ast.LtE)
def lte_op(left: Any, right: Any) -> bool:
    return left <= right


@register_operator(ast.Gt)
def gt_op(left: Any, right: Any) -> bool:
    return left > right


@register_operator(ast.GtE)
def gte_op(left: Any, right: Any) -> bool:
    return left >= right


@register_operator(ast.In)
def in_op(left: Any, right: Any) -> bool:
    return left in right


@register_operator(ast.NotIn)
def not_in_op(left: Any, right: Any) -> bool:
    return left not in right


# ----------------------------------------------------------------------------
# Unary operators
# ----------------------------------------------------------------------------


@register_operator(ast.USub)
def usub_op(operand: Any) -> Any:
    return -operand


@register_operator(ast.Invert)
def invert_op(operand: Any, *, typ) -> Any:
    if isinstance(operand, Flag):
        return ~operand

    bits = typ.bits
    mask = (1 << bits) - 1
    return mask ^ operand


# ----------------------------------------------------------------------------
# Note: Boolean operators (And, Or) are n-ary and use short-circuit logic.
# These remain implemented in ExprVisitor.visit_BoolOp


@register_operator(ast.Not)
def not_op(operand: Any) -> Any:
    return not operand

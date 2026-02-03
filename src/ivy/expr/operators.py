# ivy/evaluator/operators.py
from typing import Any, Callable, Type
from vyper.ast import nodes as ast


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


def get_operator_handler(op_node: ast.VyperNode) -> Callable[..., Any]:
    handler = OPERATOR_REGISTRY.get(type(op_node))
    if handler is None:
        raise NotImplementedError(f"{op_node} is not implemented")
    return handler


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
    return left // right


@register_operator(ast.Mod)
def mod_op(left: Any, right: Any) -> Any:
    return left % right


@register_operator(ast.Pow)
def pow_op(left: Any, right: Any) -> Any:
    return left ** right


# ----------------------------------------------------------------------------
# Bitwise operators
# ----------------------------------------------------------------------------


@register_operator(ast.LShift)
def lshift_op(left: Any, right: int) -> Any:
    return left << right


@register_operator(ast.RShift)
def rshift_op(left: Any, right: int) -> Any:
    return left >> right


@register_operator(ast.BitOr)
def bitor_op(left: Any, right: Any) -> Any:
    return left | right


@register_operator(ast.BitXor)
def bitxor_op(left: Any, right: Any) -> Any:
    return left ^ right


@register_operator(ast.BitAnd)
def bitand_op(left: Any, right: Any) -> Any:
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
def invert_op(operand: Any) -> Any:
    return ~operand


# ----------------------------------------------------------------------------
# Note: Boolean operators (And, Or) are n-ary and use short-circuit logic.
# These remain implemented in ExprVisitor.visit_BoolOp


@register_operator(ast.Not)
def not_op(operand: Any) -> Any:
    return not operand

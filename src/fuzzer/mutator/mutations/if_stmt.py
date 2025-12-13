from __future__ import annotations

from vyper.ast import nodes as ast

from ..context import ScopeType
from ..strategy import Strategy, StrategyRegistry
from .base import MutationCtx


def register(registry: StrategyRegistry) -> None:
    registry.register(
        Strategy(
            name="if.negate_condition",
            type_classes=(ast.If,),
            tags=frozenset({"mutation", "if"}),
            is_applicable=lambda **_: True,
            weight=lambda **_: 1.0,
            run=_negate_condition,
        )
    )
    registry.register(
        Strategy(
            name="if.swap_branches",
            type_classes=(ast.If,),
            tags=frozenset({"mutation", "if"}),
            is_applicable=_has_both_branches,
            weight=lambda **_: 1.0,
            run=_swap_branches,
        )
    )
    registry.register(
        Strategy(
            name="if.inject_statement",
            type_classes=(ast.If,),
            tags=frozenset({"mutation", "if"}),
            is_applicable=lambda **_: True,
            weight=lambda **_: 1.0,
            run=_inject_statement,
        )
    )


def _has_both_branches(*, ctx: MutationCtx, **_) -> bool:
    return bool(ctx.node.body and ctx.node.orelse)


def _negate_expr(expr: ast.VyperNode) -> ast.VyperNode:
    """Return a logically negated version of the expression."""
    if isinstance(expr, ast.Compare):
        op_map = {
            ast.Lt: ast.GtE(),
            ast.LtE: ast.Gt(),
            ast.Gt: ast.LtE(),
            ast.GtE: ast.Lt(),
            ast.Eq: ast.NotEq(),
            ast.NotEq: ast.Eq(),
            ast.In: ast.NotIn(),
            ast.NotIn: ast.In(),
        }
        if type(expr.op) in op_map:
            expr.op = op_map[type(expr.op)]
            return expr

    return ast.UnaryOp(op=ast.Not(), operand=expr)


def _negate_condition(*, ctx: MutationCtx, **_) -> ast.If:
    ctx.node.test = _negate_expr(ctx.node.test)
    return ctx.node


def _swap_branches(*, ctx: MutationCtx, **_) -> ast.If:
    ctx.node.body, ctx.node.orelse = ctx.node.orelse, ctx.node.body
    ctx.node.test = _negate_expr(ctx.node.test)
    return ctx.node


def _inject_statement(*, ctx: MutationCtx, **_) -> ast.If:
    # Pick which branch to inject into
    if ctx.node.orelse and ctx.rng.random() < 0.5:
        target = ctx.node.orelse
    else:
        target = ctx.node.body

    with ctx.context.new_scope(ScopeType.IF):
        ctx.stmt_gen.inject_statements(
            target, ctx.context, ctx.node, depth=1, n_stmts=1
        )
    return ctx.node

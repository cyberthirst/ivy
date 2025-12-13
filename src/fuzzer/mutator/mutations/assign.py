from __future__ import annotations

from vyper.ast import nodes as ast

from ..strategy import Strategy, StrategyRegistry
from .base import MutationCtx


def register(registry: StrategyRegistry) -> None:
    registry.register(
        Strategy(
            name="assign.use_var_as_rhs",
            type_classes=(ast.Assign,),
            tags=frozenset({"mutation", "assign"}),
            is_applicable=_has_rhs_type,
            weight=lambda **_: 1.0,
            run=_use_var_as_rhs,
        )
    )
    registry.register(
        Strategy(
            name="assign.generate_new_expr",
            type_classes=(ast.Assign,),
            tags=frozenset({"mutation", "assign"}),
            is_applicable=_has_rhs_type,
            weight=lambda **_: 1.0,
            run=_generate_new_expr,
        )
    )


def _has_rhs_type(*, ctx: MutationCtx, **_) -> bool:
    return ctx.inferred_type is not None


def _use_var_as_rhs(*, ctx: MutationCtx, **_) -> ast.Assign:
    other_var = ctx.context.pick_var(ctx.rng, ctx.inferred_type)
    if other_var:
        ctx.node.value = other_var
    return ctx.node


def _generate_new_expr(*, ctx: MutationCtx, **_) -> ast.Assign:
    new_expr = ctx.expr_gen.generate(ctx.inferred_type, ctx.context, depth=2)
    ctx.node.value = new_expr
    return ctx.node

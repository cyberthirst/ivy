from __future__ import annotations

from vyper.ast import nodes as ast
from vyper.semantics.types.primitives import NumericT

from fuzzer.mutator.strategy import Strategy, StrategyRegistry
from fuzzer.mutator.mutations.base import MutationCtx


def register(registry: StrategyRegistry) -> None:
    registry.register(
        Strategy(
            name="compare.swap_numeric_op",
            type_classes=(ast.Compare,),
            tags=frozenset({"mutation", "compare"}),
            is_applicable=_is_numeric_compare,
            weight=lambda **_: 1.0,
            run=_swap_numeric_op,
        )
    )
    registry.register(
        Strategy(
            name="compare.toggle_eq",
            type_classes=(ast.Compare,),
            tags=frozenset({"mutation", "compare"}),
            is_applicable=_is_eq_or_neq,
            weight=lambda **_: 1.0,
            run=_toggle_eq,
        )
    )


def _is_numeric_compare(*, ctx: MutationCtx, **_) -> bool:
    return ctx.inferred_type is not None and isinstance(ctx.inferred_type, NumericT)


def _is_eq_or_neq(*, ctx: MutationCtx, **_) -> bool:
    return isinstance(ctx.node.op, (ast.Eq, ast.NotEq))


def _swap_numeric_op(*, ctx: MutationCtx, **_) -> ast.Compare:
    ops = [ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq]
    new_op_type = ctx.rng.choice(ops)
    ctx.node.op = new_op_type()
    return ctx.node


def _toggle_eq(*, ctx: MutationCtx, **_) -> ast.Compare:
    if isinstance(ctx.node.op, ast.Eq):
        ctx.node.op = ast.NotEq()
    else:
        ctx.node.op = ast.Eq()
    return ctx.node

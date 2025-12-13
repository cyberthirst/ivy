from __future__ import annotations

from vyper.ast import nodes as ast

from ..strategy import Strategy, StrategyRegistry
from .base import MutationCtx


def register(registry: StrategyRegistry) -> None:
    registry.register(
        Strategy(
            name="boolop.duplicate_operand",
            type_classes=(ast.BoolOp,),
            tags=frozenset({"mutation", "boolop"}),
            is_applicable=_has_multiple_values,
            weight=lambda **_: 1.0,
            run=_duplicate_operand,
        )
    )
    registry.register(
        Strategy(
            name="boolop.swap_and_or",
            type_classes=(ast.BoolOp,),
            tags=frozenset({"mutation", "boolop"}),
            is_applicable=lambda **_: True,
            weight=lambda **_: 1.0,
            run=_swap_and_or,
        )
    )


def _has_multiple_values(*, ctx: MutationCtx, **_) -> bool:
    return len(ctx.node.values) >= 2


def _duplicate_operand(*, ctx: MutationCtx, **_) -> ast.BoolOp:
    """Duplicate one operand (a and b → a and a)."""
    idx = ctx.rng.randint(0, len(ctx.node.values) - 1)
    other_idx = ctx.rng.choice([i for i in range(len(ctx.node.values)) if i != idx])
    ctx.node.values[other_idx] = ctx.node.values[idx]
    return ctx.node


def _swap_and_or(*, ctx: MutationCtx, **_) -> ast.BoolOp:
    """Swap between And and Or."""
    if isinstance(ctx.node.op, ast.And):
        ctx.node.op = ast.Or()
    else:
        ctx.node.op = ast.And()
    return ctx.node

from __future__ import annotations

from vyper.ast import nodes as ast

from ..strategy import Strategy, StrategyRegistry
from .base import MutationCtx


def register(registry: StrategyRegistry) -> None:
    registry.register(
        Strategy(
            name="for.inject_statement",
            type_classes=(ast.For,),
            tags=frozenset({"mutation", "for"}),
            is_applicable=lambda **_: True,
            weight=lambda **_: 1.0,
            run=_inject_statement,
        )
    )


def _inject_statement(*, ctx: MutationCtx, **_) -> ast.For:
    """Inject a statement into the for loop body."""
    # Scope is already created by the visitor before mutation
    ctx.stmt_gen.inject_statements(
        ctx.node.body, ctx.context, ctx.node, depth=1, n_stmts=1
    )
    return ctx.node

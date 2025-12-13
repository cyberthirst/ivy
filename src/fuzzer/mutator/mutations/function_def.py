from __future__ import annotations

from vyper.ast import nodes as ast

from ..strategy import Strategy, StrategyRegistry
from .base import MutationCtx


def register(registry: StrategyRegistry) -> None:
    registry.register(
        Strategy(
            name="function.inject_statement",
            type_classes=(ast.FunctionDef,),
            tags=frozenset({"mutation", "function"}),
            is_applicable=_has_body,
            weight=lambda **_: 1.0,
            run=_inject_statement,
        )
    )


def _has_body(*, ctx: MutationCtx, **_) -> bool:
    return len(ctx.node.body) > 0


def _inject_statement(*, ctx: MutationCtx, **_) -> ast.FunctionDef:
    ctx.stmt_gen.inject_statements(
        ctx.node.body, ctx.context, ctx.node, depth=0, n_stmts=1
    )
    return ctx.node

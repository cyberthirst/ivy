from __future__ import annotations

from vyper.ast import nodes as ast

from fuzzer.mutator.strategy import Strategy, StrategyRegistry
from fuzzer.mutator.mutations.base import MutationCtx


def register(registry: StrategyRegistry) -> None:
    registry.register(
        Strategy(
            name="module.inject_statement",
            type_classes=(ast.Module,),
            tags=frozenset({"mutation"}),
            is_applicable=_can_inject,
            weight=lambda **_: 1.0,
            run=_inject_statement,
        )
    )


def _can_inject(*, ctx: MutationCtx, **_) -> bool:
    return len(ctx.node.body) < 30


def _inject_statement(*, ctx: MutationCtx, **_) -> ast.Module:
    ctx.stmt_gen.inject_statements(
        ctx.node.body, ctx.context, ctx.node, depth=0, n_stmts=1
    )
    return ctx.node

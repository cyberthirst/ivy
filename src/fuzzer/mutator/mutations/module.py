from __future__ import annotations

from vyper.ast import nodes as ast

from fuzzer.mutator.strategy import strategy
from fuzzer.mutator.mutations.base import MutationCtx


def _can_inject(*, ctx: MutationCtx, **_) -> bool:
    return len(ctx.node.body) < 30


@strategy(
    name="module.inject_statement",
    type_classes=(ast.Module,),
    tags=frozenset({"mutation"}),
    is_applicable="_can_inject",
)
def _inject_statement(*, ctx: MutationCtx, **_) -> ast.Module:
    ctx.stmt_gen.inject_statements(
        ctx.node.body, ctx.context, ctx.node, depth=0, n_stmts=1
    )
    return ctx.node

from __future__ import annotations

from vyper.ast import nodes as ast

from fuzzer.mutator.strategy import strategy
from fuzzer.mutator.mutations.base import MutationCtx


def _has_body(*, ctx: MutationCtx, **_) -> bool:
    return len(ctx.node.body) > 0


@strategy(
    name="function.inject_statement",
    type_classes=(ast.FunctionDef,),
    tags=frozenset({"mutation", "function"}),
    is_applicable="_has_body",
)
def _inject_statement(*, ctx: MutationCtx, **_) -> ast.FunctionDef:
    ctx.stmt_gen.inject_statements(
        ctx.node.body,
        ctx.context,
        ctx.node,
        depth=0,
        min_stmts=1,
        max_stmts=1,
    )
    return ctx.node

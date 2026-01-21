from __future__ import annotations

from vyper.ast import nodes as ast

from fuzzer.mutator.strategy import strategy
from fuzzer.mutator.mutations.base import MutationCtx


@strategy(
    name="for.inject_statement",
    type_classes=(ast.For,),
    tags=frozenset({"mutation", "for"}),
)
def _inject_statement(*, ctx: MutationCtx, **_) -> ast.For:
    """Inject a statement into the for loop body."""
    # Scope is already created by the visitor before mutation
    ctx.stmt_gen.inject_statements(
        ctx.node.body,
        ctx.context,
        ctx.node,
        depth=ctx.stmt_gen.child_depth(ctx.stmt_gen.root_depth()),
        min_stmts=1,
        max_stmts=1,
    )
    return ctx.node

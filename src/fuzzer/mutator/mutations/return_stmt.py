from __future__ import annotations

from vyper.ast import nodes as ast

from fuzzer.mutator.strategy import strategy
from fuzzer.mutator.mutations.base import MutationCtx


def _has_return_type(*, ctx: MutationCtx, **_) -> bool:
    return ctx.inferred_type is not None


@strategy(
    name="return.generate_new_expr",
    type_classes=(ast.Return,),
    tags=frozenset({"mutation", "return"}),
    is_applicable="_has_return_type",
)
def _generate_new_expr(*, ctx: MutationCtx, **_) -> ast.Return:
    assert ctx.inferred_type is not None
    new_expr = ctx.expr_gen.generate(
        ctx.inferred_type,
        ctx.context,
        depth=ctx.expr_gen.root_depth(),
    )
    ctx.node.value = new_expr
    return ctx.node

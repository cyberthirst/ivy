from __future__ import annotations

from vyper.ast import nodes as ast
from vyper.semantics.types import TupleT, SArrayT, DArrayT

from fuzzer.mutator.strategy import strategy
from fuzzer.mutator.mutations.base import MutationCtx


def _base_type(ctx: MutationCtx):
    return (
        ctx.node.value._metadata.get("type")
        if hasattr(ctx.node.value, "_metadata")
        else None
    )


def _is_array_subscript(*, ctx: MutationCtx, **_) -> bool:
    return isinstance(_base_type(ctx), (SArrayT, DArrayT))


@strategy(
    name="subscript.mutate_index",
    type_classes=(ast.Subscript,),
    tags=frozenset({"mutation", "subscript"}),
    is_applicable="_is_array_subscript",
)
def _mutate_index(*, ctx: MutationCtx, **_) -> ast.Subscript:
    """Replace the index with a freshly generated bounds-aware expression."""
    base_t = _base_type(ctx)
    ctx.expr_gen._active_context = ctx.context
    ctx.node.slice = ctx.expr_gen._generate_index_for_sequence(
        ctx.node.value, base_t, ctx.context,
        depth=ctx.expr_gen.root_depth(),
    )
    return ctx.node

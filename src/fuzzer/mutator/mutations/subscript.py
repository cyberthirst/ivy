from __future__ import annotations

from vyper.ast import nodes as ast
from vyper.semantics.types import TupleT

from fuzzer.mutator.strategy import strategy
from fuzzer.mutator.mutations.base import MutationCtx


def _has_int_slice_non_tuple(*, ctx: MutationCtx, **_) -> bool:
    if not isinstance(ctx.node.slice, ast.Int):
        return False
    # Skip tuples to avoid changing element types
    base_type = (
        ctx.node.value._metadata.get("type")
        if hasattr(ctx.node.value, "_metadata")
        else None
    )
    return not isinstance(base_type, TupleT)


@strategy(
    name="subscript.mutate_index",
    type_classes=(ast.Subscript,),
    tags=frozenset({"mutation", "subscript"}),
    is_applicable="_has_int_slice_non_tuple",
)
def _mutate_index(*, ctx: MutationCtx, **_) -> ast.Subscript:
    """Mutate the index value."""
    current = ctx.node.slice.value
    ctx.node.slice.value = ctx.rng.choice([0, 1, current + 1, current - 1])
    return ctx.node

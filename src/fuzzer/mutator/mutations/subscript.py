from __future__ import annotations

from vyper.ast import nodes as ast
from vyper.semantics.types import BytesT, DArrayT, SArrayT, StringT, TupleT

from fuzzer.mutator.strategy import Strategy, StrategyRegistry
from fuzzer.mutator.mutations.base import MutationCtx


def register(registry: StrategyRegistry) -> None:
    registry.register(
        Strategy(
            name="subscript.mutate_index",
            type_classes=(ast.Subscript,),
            tags=frozenset({"mutation", "subscript"}),
            is_applicable=_has_int_slice_non_tuple,
            weight=lambda **_: 1.0,
            run=_mutate_index,
        )
    )


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


def _mutate_index(*, ctx: MutationCtx, **_) -> ast.Subscript:
    """Mutate the index value."""
    current = ctx.node.slice.value
    mutated = ctx.rng.choice([0, 1, current + 1, current - 1])

    base_type = (
        ctx.node.value._metadata.get("type")
        if hasattr(ctx.node.value, "_metadata")
        else None
    )
    if isinstance(base_type, (SArrayT, DArrayT, BytesT, StringT)):
        cap = base_type.length
        if cap <= 0:
            mutated = 0
        else:
            mutated = mutated % cap

    ctx.node.slice.value = mutated
    return ctx.node

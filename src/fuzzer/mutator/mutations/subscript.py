from __future__ import annotations

from vyper.ast import nodes as ast
from vyper.semantics.types import TupleT

from fuzzer.mutator.strategy import Strategy, StrategyRegistry
from fuzzer.mutator.mutations.base import MutationCtx
from fuzzer.xfail import XFailExpectation


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


def _check_subscript_index_validity(ctx: MutationCtx) -> None:
    """Check subscript index validity using our own semantic rules.

    Applies our rules:
    1. Negative indices are invalid in Vyper
    2. Out of bounds indices are invalid for static arrays

    Note: We use the node's .value directly, not .reduced(), because
    reduced() returns a cached folded value from original analysis
    that doesn't reflect our mutations.
    """
    slice_node = ctx.node.slice

    # Only validate if slice is a simple Int literal
    if not isinstance(slice_node, ast.Int):
        return

    value = slice_node.value

    # Rule 1: Negative indices are invalid
    if value < 0:
        ctx.context.compilation_xfails.append(
            XFailExpectation(kind="compilation", reason="negative index")
        )
        return

    # Rule 2: Out of bounds for static arrays
    base_type = (
        ctx.node.value._metadata.get("type")
        if hasattr(ctx.node.value, "_metadata")
        else None
    )
    if base_type and hasattr(base_type, "length"):
        if value >= base_type.length:
            ctx.context.compilation_xfails.append(
                XFailExpectation(kind="compilation", reason="index out of bounds")
            )


def _mutate_index(*, ctx: MutationCtx, **_) -> ast.Subscript:
    """Mutate the index value."""
    current = ctx.node.slice.value
    ctx.node.slice.value = ctx.rng.choice([0, 1, current + 1, current - 1])
    _check_subscript_index_validity(ctx)
    return ctx.node

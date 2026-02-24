from __future__ import annotations

from vyper.ast import nodes as ast
from vyper.semantics.types import IntegerT

from fuzzer.mutator.strategy import strategy
from fuzzer.mutator.mutations.base import MutationCtx


def _in_bounds(ctx: MutationCtx, value: int) -> bool:
    """Return True if *value* is within the inferred integer type bounds."""
    if ctx.inferred_type is None or not isinstance(ctx.inferred_type, IntegerT):
        return True  # no type info → optimistically allow
    lower, upper = ctx.inferred_type.ast_bounds
    return lower <= value <= upper


def _has_integer_type(*, ctx: MutationCtx, **_) -> bool:
    return ctx.inferred_type is not None and isinstance(ctx.inferred_type, IntegerT)


@strategy(
    name="int.add_one",
    type_classes=(ast.Int,),
    tags=frozenset({"mutation", "int"}),
)
def _add_one(*, ctx: MutationCtx, **_) -> ast.Int:
    new_value = ctx.node.value + 1
    if not _in_bounds(ctx, new_value):
        return ctx.node
    ctx.node.value = new_value
    return ctx.node


@strategy(
    name="int.subtract_one",
    type_classes=(ast.Int,),
    tags=frozenset({"mutation", "int"}),
)
def _subtract_one(*, ctx: MutationCtx, **_) -> ast.Int:
    new_value = ctx.node.value - 1
    if not _in_bounds(ctx, new_value):
        return ctx.node
    ctx.node.value = new_value
    return ctx.node


@strategy(
    name="int.set_zero",
    type_classes=(ast.Int,),
    tags=frozenset({"mutation", "int"}),
)
def _set_zero(*, ctx: MutationCtx, **_) -> ast.Int:
    ctx.node.value = 0
    return ctx.node


@strategy(
    name="int.type_aware",
    type_classes=(ast.Int,),
    tags=frozenset({"mutation", "int"}),
    is_applicable="_has_integer_type",
    weight=lambda **_: 2.0,
)
def _type_aware_mutate(*, ctx: MutationCtx, **_) -> ast.Int:
    ctx.node.value = ctx.value_mutator.mutate(ctx.node.value, ctx.inferred_type)
    return ctx.node

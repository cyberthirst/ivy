from __future__ import annotations

from vyper.ast import nodes as ast
from vyper.semantics.types import IntegerT

from fuzzer.mutator.strategy import Strategy, StrategyRegistry
from fuzzer.mutator.mutations.base import MutationCtx
from fuzzer.xfail import XFailExpectation


def _check_bounds_and_xfail(ctx: MutationCtx, new_value: int) -> None:
    """Check if new_value is within type bounds, add xfail if not."""
    if ctx.inferred_type is None or not isinstance(ctx.inferred_type, IntegerT):
        return
    lower, upper = ctx.inferred_type.ast_bounds
    if new_value < lower or new_value > upper:
        ctx.context.compilation_xfails.append(
            XFailExpectation(
                kind="compilation",
                reason=f"integer literal {new_value} out of bounds for {ctx.inferred_type}",
            )
        )


def register(registry: StrategyRegistry) -> None:
    registry.register(
        Strategy(
            name="int.add_one",
            type_classes=(ast.Int,),
            tags=frozenset({"mutation", "int"}),
            is_applicable=lambda **_: True,
            weight=lambda **_: 1.0,
            run=_add_one,
        )
    )
    registry.register(
        Strategy(
            name="int.subtract_one",
            type_classes=(ast.Int,),
            tags=frozenset({"mutation", "int"}),
            is_applicable=lambda **_: True,
            weight=lambda **_: 1.0,
            run=_subtract_one,
        )
    )
    registry.register(
        Strategy(
            name="int.set_zero",
            type_classes=(ast.Int,),
            tags=frozenset({"mutation", "int"}),
            is_applicable=lambda **_: True,
            weight=lambda **_: 1.0,
            run=_set_zero,
        )
    )
    registry.register(
        Strategy(
            name="int.type_aware",
            type_classes=(ast.Int,),
            tags=frozenset({"mutation", "int"}),
            is_applicable=_has_integer_type,
            weight=lambda **_: 2.0,
            run=_type_aware_mutate,
        )
    )


def _has_integer_type(*, ctx: MutationCtx, **_) -> bool:
    return ctx.inferred_type is not None and isinstance(ctx.inferred_type, IntegerT)


def _add_one(*, ctx: MutationCtx, **_) -> ast.Int:
    new_value = ctx.node.value + 1
    _check_bounds_and_xfail(ctx, new_value)
    ctx.node.value = new_value
    return ctx.node


def _subtract_one(*, ctx: MutationCtx, **_) -> ast.Int:
    new_value = ctx.node.value - 1
    _check_bounds_and_xfail(ctx, new_value)
    ctx.node.value = new_value
    return ctx.node


def _set_zero(*, ctx: MutationCtx, **_) -> ast.Int:
    ctx.node.value = 0
    return ctx.node


def _type_aware_mutate(*, ctx: MutationCtx, **_) -> ast.Int:
    ctx.node.value = ctx.value_mutator.mutate(ctx.node.value, ctx.inferred_type)
    return ctx.node

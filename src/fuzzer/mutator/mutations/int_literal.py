from __future__ import annotations

from vyper.ast import nodes as ast
from vyper.semantics.types import IntegerT

from fuzzer.mutator.strategy import Strategy, StrategyRegistry
from fuzzer.mutator.mutations.base import MutationCtx
from fuzzer.xfail import XFailExpectation


def _clamp_pow_exponent(ctx: MutationCtx, value: int) -> int:
    parent = ctx.node.get_ancestor() if hasattr(ctx.node, "get_ancestor") else None
    if (
        not isinstance(parent, ast.BinOp)
        or not isinstance(parent.op, ast.Pow)
        or parent.right is not ctx.node
    ):
        return value

    left_type = getattr(parent.left, "_metadata", {}).get("type")
    if not isinstance(left_type, IntegerT):
        return value

    max_exp = 255 if left_type.is_signed else 256
    if value < 0:
        value = abs(value)
    if value > max_exp:
        value = max_exp
    return value


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
    new_value = _clamp_pow_exponent(ctx, ctx.node.value + 1)
    _check_bounds_and_xfail(ctx, new_value)
    ctx.node.value = new_value
    return ctx.node


def _subtract_one(*, ctx: MutationCtx, **_) -> ast.Int:
    new_value = _clamp_pow_exponent(ctx, ctx.node.value - 1)
    _check_bounds_and_xfail(ctx, new_value)
    ctx.node.value = new_value
    return ctx.node


def _set_zero(*, ctx: MutationCtx, **_) -> ast.Int:
    ctx.node.value = _clamp_pow_exponent(ctx, 0)
    return ctx.node


def _type_aware_mutate(*, ctx: MutationCtx, **_) -> ast.Int:
    mutated = ctx.value_mutator.mutate(ctx.node.value, ctx.inferred_type)
    ctx.node.value = _clamp_pow_exponent(ctx, mutated)
    return ctx.node

from __future__ import annotations

from vyper.ast import nodes as ast
from vyper.semantics.types import BytesT, DArrayT, IntegerT, SArrayT, StringT, TupleT

from fuzzer.mutator.strategy import Strategy, StrategyRegistry
from fuzzer.mutator.mutations.base import MutationCtx


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

    if isinstance(parent.left, ast.Int):
        base_val = parent.left.value
        if base_val not in (-1, 0, 1):
            lo, hi = left_type.ast_bounds
            while value > 0:
                pow_val = pow(base_val, value)
                if lo <= pow_val <= hi:
                    break
                value -= 1
    return value


def _clamp_subscript_index(ctx: MutationCtx, value: int, original: int) -> int:
    parent = ctx.node.get_ancestor() if hasattr(ctx.node, "get_ancestor") else None
    if not isinstance(parent, ast.Subscript) or parent.slice is not ctx.node:
        return value

    base_type = getattr(parent.value, "_metadata", {}).get("type")
    if isinstance(base_type, TupleT):
        return original

    if isinstance(base_type, (SArrayT, DArrayT, BytesT, StringT)):
        cap = base_type.length
        if cap <= 0:
            return 0
        return value % cap

    return value


def _clamp_int_bounds(ctx: MutationCtx, value: int, original: int) -> int:
    if ctx.inferred_type is None or not isinstance(ctx.inferred_type, IntegerT):
        return value
    lower, upper = ctx.inferred_type.ast_bounds
    if lower <= value <= upper:
        return value

    candidate = ctx.value_mutator.mutate(original, ctx.inferred_type)
    if lower <= candidate <= upper and candidate != original:
        return candidate

    if original != upper:
        return upper
    if original != lower:
        return lower
    return original


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
    original = ctx.node.value
    new_value = _clamp_pow_exponent(ctx, original + 1)
    new_value = _clamp_subscript_index(ctx, new_value, original)
    new_value = _clamp_int_bounds(ctx, new_value, original)
    ctx.node.value = new_value
    return ctx.node


def _subtract_one(*, ctx: MutationCtx, **_) -> ast.Int:
    original = ctx.node.value
    new_value = _clamp_pow_exponent(ctx, original - 1)
    new_value = _clamp_subscript_index(ctx, new_value, original)
    new_value = _clamp_int_bounds(ctx, new_value, original)
    ctx.node.value = new_value
    return ctx.node


def _set_zero(*, ctx: MutationCtx, **_) -> ast.Int:
    original = ctx.node.value
    new_value = _clamp_pow_exponent(ctx, 0)
    new_value = _clamp_subscript_index(ctx, new_value, original)
    ctx.node.value = _clamp_int_bounds(ctx, new_value, original)
    return ctx.node


def _type_aware_mutate(*, ctx: MutationCtx, **_) -> ast.Int:
    original = ctx.node.value
    mutated = ctx.value_mutator.mutate(ctx.node.value, ctx.inferred_type)
    mutated = _clamp_pow_exponent(ctx, mutated)
    mutated = _clamp_subscript_index(ctx, mutated, original)
    ctx.node.value = _clamp_int_bounds(ctx, mutated, original)
    return ctx.node

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


def _check_subscript_context(ctx: MutationCtx) -> None:
    """If this Int is a subscript slice, validate using our semantic rules.

    Applies our rules:
    1. Negative indices are invalid in Vyper
    2. Out of bounds indices are invalid for static arrays
    3. Array/bytes sizes must be > 0 in type annotations

    Note: We use the node's .value directly, not .reduced(), because
    reduced() returns a cached folded value from original analysis
    that doesn't reflect our mutations.
    """
    parent = ctx.node.get_ancestor()
    if not isinstance(parent, ast.Subscript) or parent.slice is not ctx.node:
        return

    value = ctx.node.value

    # Check if this is a type annotation subscript (e.g., Bytes[N], String[N], uint256[N])
    # In type annotations, the subscript value must be > 0
    grandparent = parent.get_ancestor() if hasattr(parent, "get_ancestor") else None
    is_type_annotation = isinstance(parent.value, ast.Name) and parent.value.id in (
        "Bytes", "String", "DynArray"
    ) or (
        isinstance(grandparent, (ast.AnnAssign, ast.VariableDecl, ast.arg))
    )

    if is_type_annotation:
        # Array/bytes sizes must be > 0
        if value <= 0:
            ctx.context.compilation_xfails.append(
                XFailExpectation(kind="compilation", reason="array size must be > 0")
            )
        return

    # Rule 1: Negative indices are invalid
    if value < 0:
        ctx.context.compilation_xfails.append(
            XFailExpectation(kind="compilation", reason="negative index")
        )
        return

    # Rule 2: Out of bounds for static arrays
    base_type = (
        parent.value._metadata.get("type")
        if hasattr(parent.value, "_metadata")
        else None
    )
    if base_type and hasattr(base_type, "length"):
        if value >= base_type.length:
            ctx.context.compilation_xfails.append(
                XFailExpectation(kind="compilation", reason="index out of bounds")
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
    _check_subscript_context(ctx)
    return ctx.node


def _subtract_one(*, ctx: MutationCtx, **_) -> ast.Int:
    new_value = ctx.node.value - 1
    _check_bounds_and_xfail(ctx, new_value)
    ctx.node.value = new_value
    _check_subscript_context(ctx)
    return ctx.node


def _set_zero(*, ctx: MutationCtx, **_) -> ast.Int:
    ctx.node.value = 0
    _check_subscript_context(ctx)
    return ctx.node


def _type_aware_mutate(*, ctx: MutationCtx, **_) -> ast.Int:
    ctx.node.value = ctx.value_mutator.mutate(ctx.node.value, ctx.inferred_type)
    _check_subscript_context(ctx)
    return ctx.node

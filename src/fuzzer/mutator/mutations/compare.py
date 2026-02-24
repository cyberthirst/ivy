from __future__ import annotations

from vyper.ast import nodes as ast
from vyper.semantics.types.primitives import NumericT

from fuzzer.mutator.strategy import strategy
from fuzzer.mutator.mutations.base import MutationCtx


def _is_numeric_compare(*, ctx: MutationCtx, **_) -> bool:
    if isinstance(ctx.node.op, (ast.In, ast.NotIn)):
        return False
    return ctx.inferred_type is not None and isinstance(ctx.inferred_type, NumericT)


def _is_eq_or_neq(*, ctx: MutationCtx, **_) -> bool:
    return isinstance(ctx.node.op, (ast.Eq, ast.NotEq))


def _is_in_compare(*, ctx: MutationCtx, **_) -> bool:
    return isinstance(ctx.node.op, (ast.In, ast.NotIn))


@strategy(
    name="compare.swap_numeric_op",
    type_classes=(ast.Compare,),
    tags=frozenset({"mutation", "compare"}),
    is_applicable="_is_numeric_compare",
)
def _swap_numeric_op(*, ctx: MutationCtx, **_) -> ast.Compare:
    ops = [ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq]
    new_op_type = ctx.rng.choice(ops)
    ctx.node.op = new_op_type()
    return ctx.node


@strategy(
    name="compare.toggle_eq",
    type_classes=(ast.Compare,),
    tags=frozenset({"mutation", "compare"}),
    is_applicable="_is_eq_or_neq",
)
def _toggle_eq(*, ctx: MutationCtx, **_) -> ast.Compare:
    if isinstance(ctx.node.op, ast.Eq):
        ctx.node.op = ast.NotEq()
    else:
        ctx.node.op = ast.Eq()
    return ctx.node


@strategy(
    name="compare.toggle_in",
    type_classes=(ast.Compare,),
    tags=frozenset({"mutation", "compare"}),
    is_applicable="_is_in_compare",
)
def _toggle_in(*, ctx: MutationCtx, **_) -> ast.Compare:
    if isinstance(ctx.node.op, ast.In):
        ctx.node.op = ast.NotIn()
    else:
        ctx.node.op = ast.In()
    return ctx.node

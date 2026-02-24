from __future__ import annotations

from vyper.ast import nodes as ast

from fuzzer.mutator.ast_utils import expr_type
from fuzzer.mutator.augassign_utils import augassign_ops_for_type
from fuzzer.mutator.mutations.base import MutationCtx
from fuzzer.mutator.strategy import strategy


def _target_type(ctx: MutationCtx):
    return expr_type(ctx.node.target)


def _has_swappable_op(*, ctx: MutationCtx, **_) -> bool:
    typ = _target_type(ctx)
    if typ is None:
        return False
    return len(augassign_ops_for_type(typ)) >= 2


def _has_target_type(*, ctx: MutationCtx, **_) -> bool:
    return _target_type(ctx) is not None


@strategy(
    name="augassign.swap_op",
    type_classes=(ast.AugAssign,),
    tags=frozenset({"mutation", "augassign"}),
    is_applicable="_has_swappable_op",
)
def _swap_op(*, ctx: MutationCtx, **_) -> ast.AugAssign:
    typ = _target_type(ctx)
    assert typ is not None
    current_op = type(ctx.node.op)
    ops = [op for op in augassign_ops_for_type(typ) if op is not current_op]
    if not ops:
        return ctx.node

    new_op = ctx.rng.choice(ops)
    ctx.node.op = new_op()
    ctx.node.value = ctx.stmt_gen.generate_augassign_rhs(new_op, typ, ctx.context)
    return ctx.node


@strategy(
    name="augassign.generate_new_rhs",
    type_classes=(ast.AugAssign,),
    tags=frozenset({"mutation", "augassign"}),
    is_applicable="_has_target_type",
)
def _generate_new_rhs(*, ctx: MutationCtx, **_) -> ast.AugAssign:
    typ = _target_type(ctx)
    assert typ is not None
    op_class = type(ctx.node.op)
    ctx.node.value = ctx.stmt_gen.generate_augassign_rhs(op_class, typ, ctx.context)
    return ctx.node

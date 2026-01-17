from __future__ import annotations

from vyper.ast import nodes as ast

from fuzzer.mutator.strategy import strategy
from fuzzer.mutator.mutations.base import MutationCtx


def _has_rhs_type(*, ctx: MutationCtx, **_) -> bool:
    return ctx.inferred_type is not None


def _has_matching_var(*, ctx: MutationCtx, **_) -> bool:
    return ctx.inferred_type is not None and bool(
        ctx.context.find_matching_vars(ctx.inferred_type)
    )


@strategy(
    name="assign.use_var_as_rhs",
    type_classes=(ast.Assign,),
    tags=frozenset({"mutation", "assign"}),
    is_applicable="_has_matching_var",
)
def _use_var_as_rhs(*, ctx: MutationCtx, **_) -> ast.Assign:
    assert ctx.inferred_type is not None
    other_var = ctx.expr_gen.random_var_ref(ctx.inferred_type, ctx.context)
    assert other_var is not None
    ctx.node.value = other_var
    return ctx.node


@strategy(
    name="assign.generate_new_expr",
    type_classes=(ast.Assign,),
    tags=frozenset({"mutation", "assign"}),
    is_applicable="_has_rhs_type",
)
def _generate_new_expr(*, ctx: MutationCtx, **_) -> ast.Assign:
    new_expr = ctx.expr_gen.generate(ctx.inferred_type, ctx.context, depth=2)
    ctx.node.value = new_expr
    return ctx.node

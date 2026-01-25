from __future__ import annotations

from vyper.ast import nodes as ast

from fuzzer.mutator.ast_utils import ast_equivalent
from fuzzer.mutator.strategy import strategy
from fuzzer.mutator.mutations.base import MutationCtx


def _has_rhs_type(*, ctx: MutationCtx, **_) -> bool:
    return ctx.inferred_type is not None


def _has_matching_var(*, ctx: MutationCtx, **_) -> bool:
    return ctx.inferred_type is not None and bool(
        ctx.context.find_matching_vars(ctx.inferred_type)
    )


def _assign_target(node: ast.Assign) -> ast.VyperNode:
    if hasattr(node, "target"):
        return node.target
    return node.targets[0]


@strategy(
    name="assign.use_var_as_rhs",
    type_classes=(ast.Assign,),
    tags=frozenset({"mutation", "assign"}),
    is_applicable="_has_matching_var",
)
def _use_var_as_rhs(*, ctx: MutationCtx, **_) -> ast.Assign:
    assert ctx.inferred_type is not None
    cfg = ctx.stmt_gen.cfg
    target_node = _assign_target(ctx.node)
    matches = ctx.context.find_matching_vars(ctx.inferred_type)
    self_candidates = []
    other_candidates = []
    for match in matches:
        var_ref = ctx.expr_gen._generate_variable_ref(match, ctx.context)
        if ast_equivalent(var_ref, target_node):
            self_candidates.append(var_ref)
        else:
            other_candidates.append(var_ref)

    if other_candidates:
        if self_candidates and ctx.rng.random() < cfg.self_assign_prob:
            ctx.node.value = ctx.rng.choice(self_candidates)
        else:
            ctx.node.value = ctx.rng.choice(other_candidates)
        return ctx.node

    ctx.node.value = ctx.rng.choice(self_candidates)
    return ctx.node


@strategy(
    name="assign.generate_new_expr",
    type_classes=(ast.Assign,),
    tags=frozenset({"mutation", "assign"}),
    is_applicable="_has_rhs_type",
)
def _generate_new_expr(*, ctx: MutationCtx, **_) -> ast.Assign:
    assert ctx.inferred_type is not None
    target_node = _assign_target(ctx.node)
    new_expr = ctx.stmt_gen._generate_assign_value(
        ctx.context, target_node, ctx.inferred_type, rng=ctx.rng
    )
    ctx.node.value = new_expr
    return ctx.node

from __future__ import annotations

from vyper.ast import nodes as ast

from fuzzer.mutator.constant_folding import constant_folds_to_zero
from fuzzer.mutator.mutations.base import MutationCtx
from fuzzer.mutator.strategy import strategy


OP_SWAPS = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.FloorDiv,
    ast.FloorDiv: ast.Mult,
    ast.Mod: ast.FloorDiv,
    ast.BitAnd: ast.BitOr,
    ast.BitOr: ast.BitAnd,
    ast.BitXor: ast.BitAnd,
    ast.LShift: ast.RShift,
    ast.RShift: ast.LShift,
}


def _can_swap(*, ctx: MutationCtx, **_) -> bool:
    op_type = type(ctx.node.op)
    if op_type not in OP_SWAPS:
        return False
    new_op = OP_SWAPS[op_type]
    if new_op in (ast.FloorDiv, ast.Mod, ast.Div):
        return not constant_folds_to_zero(ctx.node.right, ctx.context.constants) or bool(
            getattr(ctx.node.right, "_metadata", {}).get("type")
        )
    return True


@strategy(
    name="binop.swap_operator",
    type_classes=(ast.BinOp,),
    tags=frozenset({"mutation", "binop"}),
    is_applicable="_can_swap",
)
def _swap_operator(*, ctx: MutationCtx, **_) -> ast.BinOp:
    op_type = type(ctx.node.op)
    ctx.node.op = OP_SWAPS[op_type]()
    if (
        isinstance(ctx.node.op, (ast.FloorDiv, ast.Mod, ast.Div))
        and constant_folds_to_zero(ctx.node.right, ctx.context.constants)
    ):
        rhs_type = getattr(ctx.node.right, "_metadata", {}).get("type")
        if rhs_type is not None:
            ctx.node.right = ctx.expr_gen.generate_nonzero_expr(
                rhs_type, ctx.context, depth=ctx.expr_gen.root_depth()
            )
    return ctx.node

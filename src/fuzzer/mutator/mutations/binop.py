from __future__ import annotations

from vyper.ast import nodes as ast

from fuzzer.mutator.strategy import Strategy, StrategyRegistry
from fuzzer.mutator.mutations.base import MutationCtx


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


def register(registry: StrategyRegistry) -> None:
    registry.register(
        Strategy(
            name="binop.swap_operator",
            type_classes=(ast.BinOp,),
            tags=frozenset({"mutation", "binop"}),
            is_applicable=_can_swap,
            weight=lambda **_: 1.0,
            run=_swap_operator,
        )
    )


def _can_swap(*, ctx: MutationCtx, **_) -> bool:
    return type(ctx.node.op) in OP_SWAPS


def _swap_operator(*, ctx: MutationCtx, **_) -> ast.BinOp:
    op_type = type(ctx.node.op)
    ctx.node.op = OP_SWAPS[op_type]()
    return ctx.node

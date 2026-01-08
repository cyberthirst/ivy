from __future__ import annotations

from vyper.ast import nodes as ast

from fuzzer.mutator.strategy import Strategy, StrategyRegistry
from fuzzer.mutator.mutations.base import MutationCtx


def register(registry: StrategyRegistry) -> None:
    registry.register(
        Strategy(
            name="unaryop.remove",
            type_classes=(ast.UnaryOp,),
            tags=frozenset({"mutation", "unaryop"}),
            is_applicable=lambda **_: True,
            weight=lambda **_: 1.0,
            run=_remove_operator,
        )
    )
    registry.register(
        Strategy(
            name="unaryop.double",
            type_classes=(ast.UnaryOp,),
            tags=frozenset({"mutation", "unaryop"}),
            is_applicable=lambda **_: True,
            weight=lambda **_: 1.0,
            run=_double_operator,
        )
    )


def _remove_operator(*, ctx: MutationCtx, **_) -> ast.VyperNode:
    """Remove the unary operator, returning just the operand."""
    return ctx.node.operand


def _double_operator(*, ctx: MutationCtx, **_) -> ast.UnaryOp:
    """Double the operator: -x → -(-x), ~x → ~~x, not x → not not x."""
    inner = ast.UnaryOp(op=ctx.node.op.__class__(), operand=ctx.node.operand)
    ctx.node.operand = inner
    return ctx.node

from __future__ import annotations

from vyper.ast import nodes as ast

from fuzzer.mutator.strategy import strategy
from fuzzer.mutator.mutations.base import MutationCtx


@strategy(
    name="unaryop.remove",
    type_classes=(ast.UnaryOp,),
    tags=frozenset({"mutation", "unaryop"}),
)
def _remove_operator(*, ctx: MutationCtx, **_) -> ast.VyperNode:
    """Remove the unary operator, returning just the operand."""
    return ctx.node.operand


@strategy(
    name="unaryop.double",
    type_classes=(ast.UnaryOp,),
    tags=frozenset({"mutation", "unaryop"}),
)
def _double_operator(*, ctx: MutationCtx, **_) -> ast.UnaryOp:
    """Double the operator: -x → -(-x), ~x → ~~x, not x → not not x."""
    inner = ast.UnaryOp(op=ctx.node.op.__class__(), operand=ctx.node.operand)
    ctx.node.operand = inner
    return ctx.node

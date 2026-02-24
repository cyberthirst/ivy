from __future__ import annotations

from vyper.ast import nodes as ast

from fuzzer.mutator.strategy import strategy
from fuzzer.mutator.mutations.base import MutationCtx


@strategy(
    name="continue.swap_to_break",
    type_classes=(ast.Continue,),
    tags=frozenset({"mutation", "continue_break"}),
)
def _continue_to_break(*, ctx: MutationCtx, **_) -> ast.Break:
    return ast.Break()


@strategy(
    name="break.swap_to_continue",
    type_classes=(ast.Break,),
    tags=frozenset({"mutation", "continue_break"}),
)
def _break_to_continue(*, ctx: MutationCtx, **_) -> ast.Continue:
    return ast.Continue()

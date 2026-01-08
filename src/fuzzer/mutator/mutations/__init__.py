from __future__ import annotations

from fuzzer.mutator.strategy import StrategyRegistry
from fuzzer.mutator.mutations import (
    module,
    int_literal,
    function_def,
    binop,
    if_stmt,
    assign,
    compare,
    unaryop,
    boolop,
    subscript,
    for_loop,
)


def register_all(registry: StrategyRegistry) -> None:
    module.register(registry)
    int_literal.register(registry)
    function_def.register(registry)
    binop.register(registry)
    if_stmt.register(registry)
    assign.register(registry)
    compare.register(registry)
    unaryop.register(registry)
    boolop.register(registry)
    subscript.register(registry)
    for_loop.register(registry)

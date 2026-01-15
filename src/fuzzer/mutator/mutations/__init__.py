from __future__ import annotations

from fuzzer.mutator.strategy import StrategyRegistry, register_decorated
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
    # Keep registration explicit to avoid import-time side effects and to keep
    # strategy ordering stable across modules.
    for mod in (
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
    ):
        register_decorated(registry, mod)

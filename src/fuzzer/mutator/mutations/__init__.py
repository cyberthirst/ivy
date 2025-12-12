from __future__ import annotations

from ..strategy import StrategyRegistry
from . import module, int_literal, function_def


def register_all(registry: StrategyRegistry) -> None:
    module.register(registry)
    int_literal.register(registry)
    function_def.register(registry)

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Optional

from vyper.ast import nodes as ast

from fuzzer.mutator.config import DepthConfig
from fuzzer.mutator.strategy import (
    StrategyRegistry,
    StrategySelector,
    StrategyExecutor,
    register_decorated,
)


class BaseGenerator(ABC):
    def __init__(
        self,
        rng: random.Random,
        depth_cfg: Optional[DepthConfig] = None,
    ):
        self.rng = rng
        self.depth_cfg = depth_cfg or DepthConfig()

        self._strategy_registry = StrategyRegistry()
        self._strategy_selector = StrategySelector(self.rng)
        self._strategy_executor = StrategyExecutor(self._strategy_selector)

        self._register_strategies()

    def _register_strategies(self) -> None:
        register_decorated(self._strategy_registry, self)

    def root_depth(self) -> int:
        return self.depth_cfg.root_depth

    def child_depth(self, depth: int) -> int:
        return depth + 1

    def at_max_depth(self, depth: int) -> bool:
        return depth >= self.depth_cfg.max_depth

    def should_continue(self, depth: int) -> bool:
        if self.at_max_depth(depth):
            return False
        return self.rng.random() < self.depth_cfg.decay_base ** depth

    @abstractmethod
    def generate(self, *args, **kwargs) -> ast.VyperNode:
        ...

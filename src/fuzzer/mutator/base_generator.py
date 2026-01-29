from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Callable, Optional, TypeVar

from vyper.ast import nodes as ast

from fuzzer.mutator.config import DepthConfig
from fuzzer.mutator.strategy import (
    StrategyRegistry,
    StrategySelector,
    StrategyExecutor,
    register_decorated,
)


T = TypeVar("T")


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

    def _retry(
        self,
        *,
        make_candidate: Callable[[], T],
        reject_if: Callable[[T], bool],
        retries: int,
        fallback: Optional[Callable[[], T]] = None,
    ) -> T:
        attempts = max(0, retries)
        if attempts == 0 and fallback is not None:
            return fallback()

        if attempts == 0:
            attempts = 1

        last: Optional[T] = None
        for _ in range(attempts):
            candidate = make_candidate()
            last = candidate
            if not reject_if(candidate):
                return candidate

        if fallback is not None:
            return fallback()
        if last is None:
            raise RuntimeError("retry produced no candidates")
        return last

    @abstractmethod
    def generate(self, *args, **kwargs) -> ast.VyperNode:
        ...

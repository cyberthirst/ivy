from __future__ import annotations

import random

from vyper.ast import nodes as ast

from fuzzer.mutator.strategy import StrategyRegistry, StrategySelector, StrategyExecutor
from fuzzer.mutator.mutations.base import MutationCtx


class MutationEngine:
    def __init__(self, registry: StrategyRegistry, rng: random.Random):
        self.registry = registry
        self.selector = StrategySelector(rng)
        self.executor = StrategyExecutor(self.selector)

    def mutate(self, ctx: MutationCtx) -> ast.VyperNode:
        strategies = self.registry.collect(
            type_class=type(ctx.node),
            include_tags=("mutation",),
            context={"ctx": ctx},
        )

        if not strategies:
            return ctx.node

        def _noop():
            return ctx.node

        return self.executor.execute_with_retry(
            strategies,
            policy="weighted_random",
            fallback=_noop,
            context={"ctx": ctx},
        )

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


Runner = Callable[..., Any]
WeightFn = Callable[..., float]
PredicateFn = Callable[..., bool]


@dataclass(slots=True, frozen=True)
class Strategy:
    """
    Strategy is metadata plus three callables:
    - run(**ctx): executes the strategy using keyword-only context
    - is_applicable(**ctx): availability check given context
    - weight(**ctx): dynamic weight for selection
    """

    name: str
    tags: frozenset[str] = field(default_factory=frozenset)
    type_classes: Optional[tuple[type, ...]] = None
    priority: int = 0

    # Predicates and scoring
    is_applicable: PredicateFn | None = None
    weight: WeightFn | None = None
    # Runner invoked by the executor; must accept keyword-only context
    run: Runner | None = None

    def applicable(self, /, **ctx: Any) -> bool:
        """Return True if the strategy is usable in the provided context."""

        if self.is_applicable is None:
            return True
        try:
            return bool(self.is_applicable(**ctx))
        except Exception:
            return False

    def score(self, /, **ctx: Any) -> float:
        """Return a non-negative weight for selection.

        If unset, defaults to 1.0. Negative values are treated as 0.
        """

        if self.weight is None:
            return 1.0
        try:
            w = float(self.weight(**ctx))
        except Exception:
            return 0.0
        return w if w > 0 else 0.0


class StrategyRegistry:
    """Simple registry with direct filtering.

    Filters Strategy objects directly without index indirection. This keeps
    the implementation straightforward and is performant for our expected
    strategy counts.
    """

    __slots__ = ("_strategies",)

    def __init__(self) -> None:
        self._strategies: list[Strategy] = []

    def register(self, strategy: Strategy) -> None:
        self._strategies.append(strategy)

    def collect(
        self,
        *,
        # Filtering
        type_class: Optional[type] = None,
        include_tags: Optional[Iterable[str]] = None,
        exclude_tags: Optional[Iterable[str]] = None,
        # Context used by applicability/weight functions
        context: Optional[Mapping[str, Any]] = None,
    ) -> list[Strategy]:
        """Return strategies that pass filters"""

        ctx = dict(context or {})

        include = set(include_tags or ())
        exclude = set(exclude_tags or ())

        out: list[Strategy] = []
        for s in self._strategies:
            if type_class is not None and s.type_classes is not None:
                if type_class not in s.type_classes:
                    continue

            if include and not include.issubset(s.tags):
                continue

            if exclude and (s.tags & exclude):
                continue

            if s.applicable(**ctx):
                out.append(s)

        return out

    def iter_all(self) -> Iterable[Strategy]:
        return iter(self._strategies)


class StrategySelector:
    """Select a candidate according to a policy.

    Supported policies:
    - 'weighted_random' (default): proportional to weight; falls back to uniform
    - 'uniform': uniform over strategies
    - 'priority': max priority; ties broken uniformly
    """

    __slots__ = ("rng",)

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self.rng = rng or random.Random()

    def select(
        self,
        strategies: Sequence[Strategy],
        *,
        policy: str = "weighted_random",
        context: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Strategy]:
        if not strategies:
            return None

        if policy == "uniform":
            return self.rng.choice(list(strategies))

        if policy == "priority":
            # Highest strategy.priority wins; ties uniform.
            max_p = max(s.priority for s in strategies)
            pool = [s for s in strategies if s.priority == max_p]
            return self.rng.choice(pool)

        # Default: weighted_random
        ctx = dict(context or {})
        weights = []
        for s in strategies:
            w = max(0.0, s.score(**ctx))
            # Simple, generic depth-aware scaling to avoid per-strategy depth wiring
            depth = ctx.get("depth")
            nest_decay = ctx.get("nest_decay")
            if depth is not None and nest_decay is not None and "recursive" in s.tags:
                try:
                    w *= float(nest_decay) ** int(depth)
                except Exception:
                    pass
            if depth == 0 and "terminal" in s.tags:
                w *= 2.0
            weights.append(w)
        total = sum(weights)
        if total <= 0:
            return self.rng.choice(list(strategies))

        return self.rng.choices(list(strategies), weights=weights, k=1)[0]


class StrategyExecutor:
    """Execute strategies with retry and fallback semantics.

    Each strategy's runner is invoked until one returns a non-None result.
    """

    __slots__ = ("selector",)

    def __init__(self, selector: StrategySelector) -> None:
        self.selector = selector

    def execute_with_retry(
        self,
        strategies: Sequence[Strategy],
        *,
        policy: str = "weighted_random",
        max_attempts: Optional[int] = None,
        fallback: Runner,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        pool: list[Strategy] = list(strategies)
        attempts_left = max_attempts if max_attempts is not None else len(pool)
        ctx = dict(context or {})

        while pool and attempts_left > 0:
            attempts_left -= 1
            strat = self.selector.select(pool, policy=policy, context=ctx)
            if strat is None:
                break

            result = strat.run(**ctx)
            if result is not None:
                return result

            # None result => remove and retry
            pool.remove(strat)

        return fallback()

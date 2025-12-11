import random
from typing import List, Optional, TypeVar

from fuzzer.runner.scenario import Scenario

T = TypeVar("T")


class FuzzCorpus:
    """
    Two-tier corpus: immutable seeds + bounded evolved pool.

    Seeds are never removed. Evolved items are capped at max_evolved
    with replacement when full.
    """

    def __init__(
        self,
        rng: random.Random,
        max_evolved: int = 10_000,
        seed_selection_prob: float = 0.3,
    ):
        self.rng = rng
        self.max_evolved = max_evolved
        self.seed_selection_prob = seed_selection_prob

        self._seeds: List = []
        self._evolved: List = []

    def add_seed(self, item) -> None:
        """Add an item to the immutable seed corpus."""
        self._seeds.append(item)

    def add_evolved(self, item) -> None:
        """
        Add an evolved item. If at capacity, randomly replace an existing one.
        """
        if len(self._evolved) < self.max_evolved:
            self._evolved.append(item)
        else:
            idx = self.rng.randint(0, self.max_evolved - 1)
            self._evolved[idx] = item

    def pick(self) -> Optional[Scenario]:
        """
        Pick an item from the corpus.

        With seed_selection_prob, picks from seeds. Otherwise from evolved
        (falls back to seeds if evolved is empty).
        """
        if not self._seeds and not self._evolved:
            return None

        use_seed = self.rng.random() < self.seed_selection_prob or not self._evolved

        if use_seed and self._seeds:
            return self.rng.choice(self._seeds)
        elif self._evolved:
            return self.rng.choice(self._evolved)
        elif self._seeds:
            return self.rng.choice(self._seeds)

        return None

    @property
    def seeds(self) -> List:
        return self._seeds

    @property
    def seed_count(self) -> int:
        return len(self._seeds)

    @property
    def evolved_count(self) -> int:
        return len(self._evolved)

    @property
    def total_count(self) -> int:
        return len(self._seeds) + len(self._evolved)

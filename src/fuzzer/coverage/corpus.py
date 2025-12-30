"""
In-memory corpus for coverage-guided fuzzing (single process).
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Set

from ..runner.scenario import Scenario
from ..trace_types import DeploymentTrace
from .tracker import GlobalEdgeTracker


def scenario_source_size(scenario: Scenario) -> int:
    size = 0
    for trace in scenario.traces:
        if isinstance(trace, DeploymentTrace) and trace.source_code:
            size += len(trace.source_code)
    return size


@dataclass
class CorpusEntry:
    scenario: Scenario
    edge_ids: Set[int]
    compile_time_s: float
    generation: int = 0
    parent_score: Optional[float] = None

    source_size: int = 0
    keep_forever: bool = False


class Corpus:
    def __init__(self, rng: Optional[random.Random] = None):
        self._rng = rng or random.Random()
        self._entries: List[CorpusEntry] = []
        self._reps: Dict[str, Dict[str, CorpusEntry]] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def get_representatives(self, coverage_fp: str) -> Dict[str, CorpusEntry]:
        return dict(self._reps.get(coverage_fp, {}))

    def improves_representative(
        self,
        *,
        coverage_fp: str,
        source_size: int,
        compile_time_s: float,
    ) -> bool:
        reps = self._reps.get(coverage_fp)
        if not reps:
            return True

        smallest = reps.get("smallest")
        fastest = reps.get("fastest")

        if smallest is None or source_size < smallest.source_size:
            return True
        if fastest is None or compile_time_s < fastest.compile_time_s:
            return True
        return False

    def add(self, entry: CorpusEntry, *, coverage_fp: str) -> None:
        self._entries.append(entry)
        reps = self._reps.setdefault(coverage_fp, {})

        smallest = reps.get("smallest")
        fastest = reps.get("fastest")

        if smallest is None or entry.source_size < smallest.source_size:
            reps["smallest"] = entry
            if smallest is not None:
                self._maybe_evict(smallest, reps)

        if fastest is None or entry.compile_time_s < fastest.compile_time_s:
            reps["fastest"] = entry
            if fastest is not None:
                self._maybe_evict(fastest, reps)

    def _maybe_evict(self, old: CorpusEntry, reps: Dict[str, CorpusEntry]) -> None:
        if old.keep_forever:
            return
        if old in reps.values():
            return
        try:
            self._entries.remove(old)
        except ValueError:
            return

    def select_weighted(
        self,
        tracker: GlobalEdgeTracker,
        *,
        eps: float = 1e-9,
    ) -> Optional[CorpusEntry]:
        if not self._entries:
            return None

        weights = [
            tracker.compute_rare_score(e.edge_ids) / max(e.compile_time_s, eps)
            for e in self._entries
        ]
        total = sum(weights)
        if total <= 0:
            return self._rng.choice(self._entries)
        return self._rng.choices(self._entries, weights=weights, k=1)[0]

    def select_uniform(self) -> Optional[CorpusEntry]:
        if not self._entries:
            return None
        return self._rng.choice(self._entries)

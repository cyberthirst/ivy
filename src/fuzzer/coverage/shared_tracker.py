from __future__ import annotations

import ctypes
import multiprocessing
from typing import Any
from typing import Iterable, MutableSequence


def create_shared_edge_counts(map_size: int) -> Any:
    return multiprocessing.Array(ctypes.c_uint16, map_size, lock=False)


class SharedEdgeTracker:
    def __init__(
        self,
        counts: MutableSequence[int],
        map_size: int,
        *,
        max_count: int = 0xFFFF,
    ):
        if map_size <= 0:
            raise ValueError("map_size must be positive")
        if len(counts) != map_size:
            raise ValueError("counts length must match map_size")
        self.counts = counts
        self.map_size = map_size
        self.max_count = max_count

    def compute_rare_score(self, edge_ids: Iterable[int]) -> float:
        score = 0.0
        for edge_id in edge_ids:
            score += 1.0 / (self.counts[edge_id] + 1.0)
        return score

    def count_new_edges(self, edge_ids: Iterable[int]) -> int:
        return sum(1 for edge_id in edge_ids if self.counts[edge_id] == 0)

    def merge(self, edge_ids: Iterable[int]) -> int:
        new_edges = 0
        for edge_id in edge_ids:
            current = self.counts[edge_id]
            if current == 0:
                new_edges += 1
            if current < self.max_count:
                self.counts[edge_id] = current + 1
        return new_edges

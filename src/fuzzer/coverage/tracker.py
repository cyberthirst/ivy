"""
Global edge counter map for rare-edge prioritization.
"""

from __future__ import annotations

from array import array
from typing import Iterable, MutableSequence, Optional


class GlobalEdgeTracker:
    """
    Tracks edge hit counts in a fixed-size saturating counter map.

    Counters are incremented once per scenario for each edge in the scenario's
    edge set.
    """

    def __init__(
        self,
        map_size: int,
        *,
        counts: Optional[MutableSequence[int]] = None,
        max_count: int = 0xFFFF,
    ):
        if map_size <= 0:
            raise ValueError("map_size must be positive")

        self.map_size = map_size
        self.max_count = max_count

        if counts is None:
            self.counts: MutableSequence[int] = array("H", [0]) * map_size
        else:
            if len(counts) != map_size:
                raise ValueError("counts length must match map_size")
            self.counts = counts

    def compute_rare_score(self, edge_ids: Iterable[int]) -> float:
        score = 0.0
        for edge_id in edge_ids:
            score += 1.0 / (self.counts[edge_id] + 1.0)
        return score

    def count_new_edges(self, edge_ids: Iterable[int]) -> int:
        return sum(1 for edge_id in edge_ids if self.counts[edge_id] == 0)

    def merge(self, edge_ids: Iterable[int]) -> int:
        """
        Merge an edge set into the global counter map.

        Returns the count of edges that were newly discovered (count was 0).
        """
        new_edges = 0
        for edge_id in edge_ids:
            current = self.counts[edge_id]
            if current == 0:
                new_edges += 1
            if current < self.max_count:
                self.counts[edge_id] = current + 1
        return new_edges

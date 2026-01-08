"""
Coverage-guided fuzzing utilities.
"""

from fuzzer.coverage.collector import ArcCoverageCollector
from fuzzer.coverage.corpus import Corpus, CorpusEntry
from fuzzer.coverage.edge_map import EdgeMap
from fuzzer.coverage.gatekeeper import Gatekeeper, GatekeeperDecision
from fuzzer.coverage.tracker import GlobalEdgeTracker

__all__ = [
    "ArcCoverageCollector",
    "Corpus",
    "CorpusEntry",
    "EdgeMap",
    "Gatekeeper",
    "GatekeeperDecision",
    "GlobalEdgeTracker",
]

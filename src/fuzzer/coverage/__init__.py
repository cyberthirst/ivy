"""
Coverage-guided fuzzing utilities.
"""

from fuzzer.coverage.collector import ArcCoverageCollector
from fuzzer.coverage.corpus import Corpus, CorpusEntry
from fuzzer.coverage.disk_corpus import DiskEntryMeta, DiskIndex
from fuzzer.coverage.edge_map import EdgeMap
from fuzzer.coverage.gatekeeper import Gatekeeper, GatekeeperDecision
from fuzzer.coverage.scenario_serde import deserialize_scenario, serialize_scenario
from fuzzer.coverage.shared_tracker import SharedEdgeTracker, create_shared_edge_counts
from fuzzer.coverage.tracker import GlobalEdgeTracker

__all__ = [
    "ArcCoverageCollector",
    "Corpus",
    "CorpusEntry",
    "DiskEntryMeta",
    "DiskIndex",
    "EdgeMap",
    "Gatekeeper",
    "GatekeeperDecision",
    "GlobalEdgeTracker",
    "SharedEdgeTracker",
    "create_shared_edge_counts",
    "serialize_scenario",
    "deserialize_scenario",
]

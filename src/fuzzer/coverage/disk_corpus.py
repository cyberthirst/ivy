from __future__ import annotations

from array import array
from collections import deque
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import struct
import sys
import time
import uuid
from typing import Any, MutableSequence, Optional

from fuzzer.coverage.scenario_serde import deserialize_scenario, serialize_scenario
from fuzzer.runner.scenario import Scenario

_EPS = 1e-9


def _queue_dir(corpus_dir: Path) -> Path:
    return corpus_dir / "queue"


def _snapshots_dir(corpus_dir: Path) -> Path:
    return corpus_dir / "snapshots"


def _scenario_path(corpus_dir: Path, entry_id: str) -> Path:
    return _queue_dir(corpus_dir) / f"{entry_id}.scenario.json"


def _edges_path(corpus_dir: Path, entry_id: str) -> Path:
    return _queue_dir(corpus_dir) / f"{entry_id}.edges.bin"


def _meta_path(corpus_dir: Path, entry_id: str) -> Path:
    return _queue_dir(corpus_dir) / f"{entry_id}.meta.json"


def _entry_id_from_meta_path(path: Path) -> str:
    return path.name[: -len(".meta.json")]


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with open(tmp_path, "wb") as f:
        f.write(payload)
    os.replace(tmp_path, path)


def _meta_from_dict(entry_id: str, data: dict[str, Any]) -> "DiskEntryMeta":
    return DiskEntryMeta(
        entry_id=entry_id,
        cycle_time_s=float(data["cycle_time_s"]),
        source_size=int(data["source_size"]),
        generation=int(data["generation"]),
        coverage_fp=str(data["coverage_fp"]),
        n_edges=int(data["n_edges"]),
        timestamp=float(data["timestamp"]),
        worker_id=int(data["worker_id"]),
    )


@dataclass(frozen=True)
class DiskEntryMeta:
    entry_id: str
    cycle_time_s: float
    source_size: int
    generation: int
    coverage_fp: str
    n_edges: int
    timestamp: float
    worker_id: int


class DiskIndex:
    def __init__(self, corpus_dir: Path):
        self.corpus_dir = Path(corpus_dir)
        self.queue_dir = _queue_dir(self.corpus_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)

        self._entries: list[DiskEntryMeta] = []
        self._known_ids: set[str] = set()
        self._fresh_ids: deque[str] = deque()
        self._reps: dict[str, dict[str, DiskEntryMeta]] = {}

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[DiskEntryMeta]:
        return list(self._entries)

    def scan_new(self) -> int:
        added = 0
        for meta_file in sorted(self.queue_dir.glob("*.meta.json")):
            entry_id = _entry_id_from_meta_path(meta_file)
            if entry_id in self._known_ids:
                continue

            with open(meta_file, "r") as f:
                meta_data = json.load(f)

            meta = _meta_from_dict(entry_id, meta_data)
            self._entries.append(meta)
            self._known_ids.add(entry_id)
            self._fresh_ids.append(entry_id)
            self._update_representatives(meta)
            added += 1

        return added

    def pop_fresh(self, rng: random.Random) -> Optional[DiskEntryMeta]:
        del rng
        while self._fresh_ids:
            entry_id = self._fresh_ids.popleft()
            meta = self._find_entry(entry_id)
            if meta is None:
                continue
            if not _meta_path(self.corpus_dir, entry_id).exists():
                continue
            return meta
        return None

    def purge_deleted(self) -> int:
        removed_ids = {
            meta.entry_id
            for meta in self._entries
            if not _meta_path(self.corpus_dir, meta.entry_id).exists()
        }
        if not removed_ids:
            return 0

        self._entries = [
            meta for meta in self._entries if meta.entry_id not in removed_ids
        ]
        self._known_ids.difference_update(removed_ids)
        self._fresh_ids = deque(
            entry_id for entry_id in self._fresh_ids if entry_id not in removed_ids
        )
        self._rebuild_representatives()
        return len(removed_ids)

    def improves_representative(
        self,
        *,
        coverage_fp: str,
        source_size: int,
        cycle_time_s: float,
    ) -> bool:
        reps = self._reps.get(coverage_fp)
        if not reps:
            return True

        smallest = reps.get("smallest")
        fastest = reps.get("fastest")
        if smallest is None or source_size < smallest.source_size:
            return True
        if fastest is None or cycle_time_s < fastest.cycle_time_s:
            return True
        return False

    def _find_entry(self, entry_id: str) -> Optional[DiskEntryMeta]:
        for meta in self._entries:
            if meta.entry_id == entry_id:
                return meta
        return None

    def _update_representatives(self, meta: DiskEntryMeta) -> None:
        reps = self._reps.setdefault(meta.coverage_fp, {})
        smallest = reps.get("smallest")
        fastest = reps.get("fastest")
        if smallest is None or meta.source_size < smallest.source_size:
            reps["smallest"] = meta
        if fastest is None or meta.cycle_time_s < fastest.cycle_time_s:
            reps["fastest"] = meta

    def _rebuild_representatives(self) -> None:
        self._reps = {}
        for meta in self._entries:
            self._update_representatives(meta)


def write_corpus_entry(
    corpus_dir: Path,
    *,
    scenario: Scenario,
    edge_ids: set[int],
    cycle_time_s: float,
    source_size: int,
    generation: int,
    coverage_fp: str,
    worker_id: int,
    timestamp: Optional[float] = None,
) -> str:
    corpus_dir = Path(corpus_dir)
    entry_id = str(uuid.uuid4())
    timestamp = time.time() if timestamp is None else timestamp

    scenario_payload = json.dumps(
        serialize_scenario(scenario),
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")

    ordered_edges = sorted(edge_ids)
    if ordered_edges:
        edge_payload = struct.pack(f"<{len(ordered_edges)}I", *ordered_edges)
    else:
        edge_payload = b""

    meta_payload = json.dumps(
        {
            "cycle_time_s": float(cycle_time_s),
            "source_size": int(source_size),
            "generation": int(generation),
            "coverage_fp": coverage_fp,
            "n_edges": len(ordered_edges),
            "timestamp": float(timestamp),
            "worker_id": int(worker_id),
        },
        separators=(",", ":"),
    ).encode("utf-8")

    _atomic_write_bytes(_scenario_path(corpus_dir, entry_id), scenario_payload)
    _atomic_write_bytes(_edges_path(corpus_dir, entry_id), edge_payload)
    _atomic_write_bytes(_meta_path(corpus_dir, entry_id), meta_payload)
    return entry_id


def read_edge_ids(corpus_dir: Path, entry_id: str) -> set[int]:
    payload = _edges_path(Path(corpus_dir), entry_id).read_bytes()
    if not payload:
        return set()
    if len(payload) % 4 != 0:
        raise ValueError(f"Invalid edge payload for entry {entry_id}")
    n_edges = len(payload) // 4
    return set(struct.unpack(f"<{n_edges}I", payload))


def read_scenario(corpus_dir: Path, entry_id: str) -> Scenario:
    with open(_scenario_path(Path(corpus_dir), entry_id), "r") as f:
        return deserialize_scenario(json.load(f))


def delete_corpus_entry(corpus_dir: Path, entry_id: str) -> None:
    _meta_path(Path(corpus_dir), entry_id).unlink(missing_ok=True)
    _edges_path(Path(corpus_dir), entry_id).unlink(missing_ok=True)
    _scenario_path(Path(corpus_dir), entry_id).unlink(missing_ok=True)


def read_all_meta(corpus_dir: Path) -> list[DiskEntryMeta]:
    entries: list[DiskEntryMeta] = []
    for meta_file in sorted(_queue_dir(Path(corpus_dir)).glob("*.meta.json")):
        entry_id = _entry_id_from_meta_path(meta_file)
        with open(meta_file, "r") as f:
            entries.append(_meta_from_dict(entry_id, json.load(f)))
    return entries


def tournament_select(
    index: DiskIndex,
    corpus_dir: Path,
    edge_counts: MutableSequence[int],
    rng: random.Random,
    *,
    k: int = 8,
) -> Optional[tuple[DiskEntryMeta, Scenario]]:
    entries = index.entries
    if not entries:
        return None

    sample_size = min(k, len(entries))
    candidates = rng.sample(entries, sample_size)

    scored: list[tuple[float, DiskEntryMeta]] = []
    for candidate in candidates:
        try:
            edge_ids = read_edge_ids(corpus_dir, candidate.entry_id)
        except FileNotFoundError:
            continue

        rare_score = 0.0
        for edge_id in edge_ids:
            rare_score += 1.0 / (edge_counts[edge_id] + 1.0)

        score = rare_score / max(candidate.cycle_time_s, _EPS)
        scored.append((score, candidate))

    for _, candidate in sorted(scored, key=lambda x: x[0], reverse=True):
        try:
            scenario = read_scenario(corpus_dir, candidate.entry_id)
            return candidate, scenario
        except FileNotFoundError:
            continue

    return None


def select_parent(
    index: DiskIndex,
    corpus_dir: Path,
    edge_counts: MutableSequence[int],
    rng: random.Random,
    *,
    k: int = 8,
) -> Optional[tuple[DiskEntryMeta, Scenario]]:
    while True:
        fresh = index.pop_fresh(rng)
        if fresh is None:
            break
        try:
            return fresh, read_scenario(corpus_dir, fresh.entry_id)
        except FileNotFoundError:
            continue

    return tournament_select(index, corpus_dir, edge_counts, rng, k=k)


def compute_energy(
    parent_meta: DiskEntryMeta,
    rng: random.Random,
    *,
    min_e: int = 4,
    max_e: int = 16,
) -> int:
    del parent_meta
    return rng.randint(min_e, max_e)


def save_edge_count_snapshot(
    corpus_dir: Path,
    edge_counts: MutableSequence[int],
) -> None:
    snapshot_path = _snapshots_dir(Path(corpus_dir)) / "edge_counts.bin"
    arr = array("H", edge_counts)
    if sys.byteorder != "little":
        arr.byteswap()
    _atomic_write_bytes(snapshot_path, arr.tobytes())


def load_edge_count_snapshot(corpus_dir: Path, map_size: int) -> Optional[array]:
    snapshot_path = _snapshots_dir(Path(corpus_dir)) / "edge_counts.bin"
    if not snapshot_path.exists():
        return None

    payload = snapshot_path.read_bytes()
    if len(payload) != map_size * 2:
        return None

    arr = array("H")
    arr.frombytes(payload)
    if sys.byteorder != "little":
        arr.byteswap()
    if len(arr) != map_size:
        return None
    return arr

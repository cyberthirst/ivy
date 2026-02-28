from __future__ import annotations

import argparse
import ctypes
import hashlib
import logging
import multiprocessing
from pathlib import Path
import random
import resource
import time
from typing import Any, MutableSequence, Optional

from fuzzer.base_fuzzer import BaseFuzzer
from fuzzer.deduper import Deduper
from fuzzer.coverage.collector import ArcCoverageCollector
from fuzzer.coverage.disk_corpus import (
    DiskEntryMeta,
    DiskIndex,
    compute_energy,
    delete_corpus_entry,
    load_edge_count_snapshot,
    read_all_meta,
    select_parent,
    write_corpus_entry,
    save_edge_count_snapshot,
)
from fuzzer.coverage.edge_map import EdgeMap
from fuzzer.coverage.gatekeeper import (
    Gatekeeper,
    coverage_fingerprint,
)
from fuzzer.coverage.shared_tracker import SharedEdgeTracker, create_shared_edge_counts
from fuzzer.export_utils import (
    TestFilter,
    exclude_unsupported_patterns,
    solc_json_source_size,
)
from fuzzer.issue_filter import IssueFilter, default_issue_filter
from fuzzer.runtime_engine import HarnessConfig
from fuzzer.generator import generate_scenario
from fuzzer.runner.scenario import Scenario, create_scenario_from_item
from fuzzer.trace_types import DeploymentTrace

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_EPS = 1e-9
_LOG_INTERVAL = 100
_STATE_IDLE = 0
_STATE_RUNNING = 1
_STATE_BOOTSTRAPPING = 2
_STATE_STOPPED = 3
_DEFAULT_WORKER_MEMORY_LIMIT_BYTES = 2 * 1024**3


def _apply_memory_limit(limit_bytes: int, worker_id: int) -> None:
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        logger.info(
            f"[w{worker_id}] memory limit set to {limit_bytes / 1024**3:.1f} GiB"
        )
    except (ValueError, OSError) as exc:
        logger.warning(f"[w{worker_id}] failed to set memory limit: {exc}")


def _scenario_source_size(scenario: Scenario) -> int:
    size = 0
    for trace in scenario.traces:
        if isinstance(trace, DeploymentTrace):
            size += solc_json_source_size(trace.solc_json)
    return size


def build_default_coverage_test_filter() -> TestFilter:
    test_filter = TestFilter(exclude_multi_module=True, exclude_deps=True)
    exclude_unsupported_patterns(test_filter)
    test_filter.exclude_source(r"\bsend\s*\(")
    test_filter.include_path("functional/codegen/")
    test_filter.exclude_name("zero_length_side_effects")
    return test_filter


def build_default_coverage_issue_filter() -> IssueFilter:
    return default_issue_filter()


def _derive_seed(seed: int, *parts: object) -> int:
    h = hashlib.blake2b(digest_size=16)
    h.update(str(seed).encode("utf-8"))
    for part in parts:
        h.update(b"|")
        h.update(str(part).encode("utf-8"))
    return int.from_bytes(h.digest(), byteorder="big", signed=False)


def _touch_heartbeat(
    worker_id: int,
    last_progress_ts: MutableSequence[float],
    last_iter: MutableSequence[int],
    worker_state: MutableSequence[int],
    *,
    iteration: int,
    state: Optional[int] = None,
) -> None:
    last_progress_ts[worker_id] = time.monotonic()
    last_iter[worker_id] = iteration
    if state is not None:
        worker_state[worker_id] = state


def _hash_collected_arcs(
    collector: ArcCoverageCollector, edge_map: EdgeMap
) -> set[int]:
    if edge_map.tag_with_config:
        return edge_map.hash_arcs_by_config(collector.get_arcs_by_config())
    return edge_map.hash_arcs(collector.get_scenario_arcs())


def _admit_to_corpus(
    *,
    scenario: Scenario,
    artifacts: Any,
    edge_ids: set[int],
    cycle_time_s: float,
    generation: int,
    worker_id: int,
    gatekeeper: Gatekeeper,
    disk_index: DiskIndex,
    corpus_dir: Path,
) -> bool:
    coverage_fp = coverage_fingerprint(edge_ids)
    source_size = _scenario_source_size(scenario)

    decision = gatekeeper.decide_and_update(
        edge_ids=edge_ids,
        cycle_time_s=cycle_time_s,
        analysis=artifacts.analysis,
        ivy_result=artifacts.ivy_result,
        improves_representative=disk_index.improves_representative(
            coverage_fp=coverage_fp,
            source_size=source_size,
            cycle_time_s=cycle_time_s,
        ),
        coverage_fp=coverage_fp,
    )
    if not decision.accept:
        return False

    write_corpus_entry(
        corpus_dir,
        scenario=scenario,
        edge_ids=edge_ids,
        cycle_time_s=cycle_time_s,
        source_size=source_size,
        generation=generation,
        coverage_fp=decision.coverage_fingerprint,
        worker_id=worker_id,
    )
    disk_index.scan_new()
    return True


def _bootstrap_worker(
    *,
    worker_id: int,
    bootstrap_seed: int,
    worker_seed: int,
    base_fuzzer: BaseFuzzer,
    test_filter: Optional[TestFilter],
    bootstrap_shard_index: int,
    bootstrap_shard_count: int,
    resume_from_processed: int,
    bootstrap_progress: MutableSequence[int],
    collector: ArcCoverageCollector,
    edge_map: EdgeMap,
    gatekeeper: Gatekeeper,
    disk_index: DiskIndex,
    corpus_dir: Path,
    stop_event: Any,
    last_progress_ts: MutableSequence[float],
    last_iter: MutableSequence[int],
    worker_state: MutableSequence[int],
) -> bool:
    reporter = base_fuzzer.reporter
    exports = base_fuzzer.load_filtered_exports(test_filter)

    bootstrap_items: list[tuple[str, str, Any]] = []
    for export_path in sorted(exports, key=lambda p: str(p)):
        export = exports[export_path]
        for item_name in sorted(export.items):
            item = export.items[item_name]
            if item.item_type == "fixture":
                continue
            bootstrap_items.append((str(export_path), item_name, item))

    added = 0
    processed = 0  # number of items processed in this worker's deterministic shard
    for export_path, item_name, item in bootstrap_items:
        assigned_worker = (
            _derive_seed(
                bootstrap_seed,
                "bootstrap_assign",
                export_path,
                item_name,
            )
            % bootstrap_shard_count
        )
        if assigned_worker != bootstrap_shard_index:
            continue
        processed += 1
        if processed <= resume_from_processed:
            continue
        if stop_event.is_set():
            return False

        try:
            cycle_start = time.perf_counter()
            scenario = create_scenario_from_item(item, use_python_args=True)
            scenario_seed = _derive_seed(
                worker_seed,
                "bootstrap",
                export_path,
                item_name,
            )

            reporter.set_context(
                f"w{worker_id}_bootstrap_{processed}",
                processed,
                worker_seed,
                scenario_seed,
            )

            collector.start_scenario()
            artifacts = base_fuzzer.run_scenario_with_artifacts(
                scenario,
                seed=scenario_seed,
                coverage_collector=collector,
            )
            cycle_time_s = max(time.perf_counter() - cycle_start, _EPS)

            reporter.ingest_run(artifacts.analysis, artifacts, debug_mode=False)

            edge_ids = _hash_collected_arcs(collector, edge_map)
            if _admit_to_corpus(
                scenario=scenario,
                artifacts=artifacts,
                edge_ids=edge_ids,
                cycle_time_s=cycle_time_s,
                generation=0,
                worker_id=worker_id,
                gatekeeper=gatekeeper,
                disk_index=disk_index,
                corpus_dir=corpus_dir,
            ):
                added += 1
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"[w{worker_id}] bootstrap item failed "
                f"({export_path}::{item_name}): "
                f"{type(exc).__name__}: {exc}"
            )
            stop_event.set()
            return False
        finally:
            bootstrap_progress[worker_id] = processed
            _touch_heartbeat(
                worker_id,
                last_progress_ts,
                last_iter,
                worker_state,
                iteration=processed,
                state=_STATE_BOOTSTRAPPING,
            )

    logger.info(
        f"[w{worker_id}] bootstrap complete: {added}/{processed} seeds added "
        f"(shard={bootstrap_shard_index}/{bootstrap_shard_count}, "
        f"total={len(bootstrap_items)})"
    )
    return True


def _worker_main(
    *,
    worker_id: int,
    corpus_dir: Path,
    shared_counts: MutableSequence[int],
    map_size: int,
    exports_dir: Path,
    seed: int,
    worker_epoch: int,
    test_filter: Optional[TestFilter],
    stop_event: Any,
    last_progress_ts: MutableSequence[float],
    last_iter: MutableSequence[int],
    worker_state: MutableSequence[int],
    harness_config: HarnessConfig,
    reports_dir: Path,
    tag_edges_with_config: bool = False,
    scan_interval_s: float = 5.0,
    min_energy: int = 4,
    max_energy: int = 16,
    max_iterations: Optional[int] = None,
    log_interval: int = _LOG_INTERVAL,
    stagnation_threshold: int = 200,
    seen_crashes: Optional[dict[str, bool]] = None,
    seen_compile_failures: Optional[dict[str, bool]] = None,
    seen_compilation_timeouts: Optional[dict[str, bool]] = None,
    seen_divergences: Optional[dict[str, bool]] = None,
    worker_memory_limit: Optional[int] = None,
    bootstrap_required: bool = False,
    bootstrap_done_event: Optional[Any] = None,
    bootstrap_done_flags: Optional[MutableSequence[int]] = None,
    bootstrap_progress: Optional[MutableSequence[int]] = None,
    bootstrap_shard_count: int = 1,
    drop_initial_fresh: bool = False,
) -> None:
    import boa  # pyright: ignore[reportMissingImports]

    boa.interpret.disable_cache()  # pyright: ignore[reportAttributeAccessIssue]

    worker_seed = _derive_seed(seed, "worker", worker_id, worker_epoch)
    rng = random.Random(worker_seed)

    collector = ArcCoverageCollector()
    edge_map = EdgeMap(map_size, tag_with_config=tag_edges_with_config)
    tracker = SharedEdgeTracker(shared_counts, map_size)
    gatekeeper = Gatekeeper(tracker=tracker)  # pyright: ignore[reportArgumentType]
    disk_index = DiskIndex(corpus_dir)
    deduper = Deduper(
        seen_crashes=seen_crashes,
        seen_compile_failures=seen_compile_failures,
        seen_compilation_timeouts=seen_compilation_timeouts,
        seen_divergences=seen_divergences,
    )

    base_fuzzer = BaseFuzzer(
        exports_dir=exports_dir,
        seed=worker_seed,
        debug_mode=False,
        issue_filter=build_default_coverage_issue_filter(),
        harness_config=harness_config,
        deduper=deduper,
    )
    reporter = base_fuzzer.reporter
    reporter.reports_dir = reports_dir
    reporter.start_timer()
    reporter.start_metrics_stream(harness_config.enable_interval_metrics)

    iteration = 0
    batch_id = 0
    iters_since_accept = 0
    last_scan = time.monotonic()
    last_purge = last_scan

    _touch_heartbeat(
        worker_id,
        last_progress_ts,
        last_iter,
        worker_state,
        iteration=iteration,
        state=_STATE_IDLE,
    )

    try:
        disk_index.scan_new()

        if bootstrap_required:
            if (
                bootstrap_done_event is None
                or bootstrap_done_flags is None
                or bootstrap_progress is None
            ):
                raise ValueError("bootstrap synchronization primitives are required")
            _touch_heartbeat(
                worker_id,
                last_progress_ts,
                last_iter,
                worker_state,
                iteration=iteration,
                state=_STATE_BOOTSTRAPPING,
            )
            if bootstrap_done_flags[worker_id] == 0:
                bootstrap_ok = _bootstrap_worker(
                    worker_id=worker_id,
                    bootstrap_seed=seed,
                    worker_seed=worker_seed,
                    base_fuzzer=base_fuzzer,
                    test_filter=test_filter,
                    bootstrap_shard_index=worker_id,
                    bootstrap_shard_count=bootstrap_shard_count,
                    resume_from_processed=int(bootstrap_progress[worker_id]),
                    bootstrap_progress=bootstrap_progress,
                    collector=collector,
                    edge_map=edge_map,
                    gatekeeper=gatekeeper,
                    disk_index=disk_index,
                    corpus_dir=corpus_dir,
                    stop_event=stop_event,
                    last_progress_ts=last_progress_ts,
                    last_iter=last_iter,
                    worker_state=worker_state,
                )
                if not bootstrap_ok or stop_event.is_set():
                    return
                disk_index.scan_new()
                bootstrap_done_flags[worker_id] = 1
                if all(int(flag) == 1 for flag in bootstrap_done_flags):
                    bootstrap_done_event.set()

            while not bootstrap_done_event.is_set() and not stop_event.is_set():
                time.sleep(0.5)
                disk_index.scan_new()
                _touch_heartbeat(
                    worker_id,
                    last_progress_ts,
                    last_iter,
                    worker_state,
                    iteration=iteration,
                    state=_STATE_BOOTSTRAPPING,
                )

            disk_index.scan_new()
        if drop_initial_fresh:
            cleared = disk_index.clear_fresh()
            if cleared > 0:
                logger.info(f"[w{worker_id}] cleared {cleared} fresh bootstrap entries")

        if worker_memory_limit is not None:
            _apply_memory_limit(worker_memory_limit, worker_id)

        while len(disk_index) == 0 and not stop_event.is_set():
            time.sleep(0.5)
            disk_index.scan_new()
            _touch_heartbeat(
                worker_id,
                last_progress_ts,
                last_iter,
                worker_state,
                iteration=iteration,
                state=_STATE_IDLE,
            )

        logger.info(f"[w{worker_id}] starting fuzz loop, corpus={len(disk_index)}")

        while not stop_event.is_set():
            batch_id += 1
            _touch_heartbeat(
                worker_id,
                last_progress_ts,
                last_iter,
                worker_state,
                iteration=iteration,
                state=_STATE_RUNNING,
            )

            now = time.monotonic()
            if now - last_scan >= scan_interval_s:
                disk_index.scan_new()
                last_scan = now
                if now - last_purge >= (scan_interval_s * 6.0):
                    disk_index.purge_deleted()
                    last_purge = now
                _touch_heartbeat(
                    worker_id,
                    last_progress_ts,
                    last_iter,
                    worker_state,
                    iteration=iteration,
                    state=_STATE_RUNNING,
                )

            # Pick a starting scenario: generate from scratch or select from corpus
            if iters_since_accept >= stagnation_threshold:
                gen_seed = _derive_seed(worker_seed, "gen", batch_id)
                parent_scenario = generate_scenario(seed=gen_seed)
                if parent_scenario is None:
                    continue
                energy = rng.randint(min_energy, max_energy)
                generation = 1
            else:
                parent = select_parent(disk_index, corpus_dir, shared_counts, rng, k=8)
                if parent is None:
                    _touch_heartbeat(
                        worker_id,
                        last_progress_ts,
                        last_iter,
                        worker_state,
                        iteration=iteration,
                        state=_STATE_IDLE,
                    )
                    time.sleep(0.05)
                    continue

                parent_meta, parent_scenario = parent
                energy = compute_energy(
                    parent_meta, rng, min_e=min_energy, max_e=max_energy
                )
                generation = parent_meta.generation + 1

            for e_idx in range(energy):
                if stop_event.is_set():
                    break

                iteration += 1
                scenario_seed = _derive_seed(
                    worker_seed, "cov", batch_id, e_idx, iteration
                )

                cycle_start = time.perf_counter()
                mutated = base_fuzzer.mutate_scenario(
                    parent_scenario, scenario_seed=scenario_seed
                )

                reporter.set_context(
                    f"w{worker_id}_cov_{iteration}",
                    iteration,
                    worker_seed,
                    scenario_seed,
                )

                collector.start_scenario()
                artifacts = base_fuzzer.run_scenario_with_artifacts(
                    mutated,
                    seed=scenario_seed,
                    coverage_collector=collector,
                )
                cycle_time_s = max(time.perf_counter() - cycle_start, _EPS)

                reporter.ingest_run(artifacts.analysis, artifacts, debug_mode=False)

                edge_ids = _hash_collected_arcs(collector, edge_map)
                if _admit_to_corpus(
                    scenario=mutated,
                    artifacts=artifacts,
                    edge_ids=edge_ids,
                    cycle_time_s=cycle_time_s,
                    generation=generation,
                    worker_id=worker_id,
                    gatekeeper=gatekeeper,
                    disk_index=disk_index,
                    corpus_dir=corpus_dir,
                ):
                    iters_since_accept = 0
                else:
                    iters_since_accept += 1

                _touch_heartbeat(
                    worker_id,
                    last_progress_ts,
                    last_iter,
                    worker_state,
                    iteration=iteration,
                    state=_STATE_RUNNING,
                )

                if iteration % log_interval == 0:
                    snapshot = reporter.record_interval_metrics(
                        iteration=iteration,
                        corpus_seed_count=len(disk_index),
                        corpus_evolved_count=0,
                        corpus_max_evolved=0,
                        debug_mode=False,
                    )
                    _log_worker_progress(
                        worker_id, iteration, len(disk_index), reporter, snapshot
                    )

                if max_iterations is not None and iteration >= max_iterations:
                    stop_event.set()
                    break

    finally:
        reporter.stop_timer()
        reporter.save_statistics()
        logger.info(
            f"[w{worker_id}] stopped after {iteration} iterations, "
            f"divergences={reporter.divergences}"
        )
        _touch_heartbeat(
            worker_id,
            last_progress_ts,
            last_iter,
            worker_state,
            iteration=iteration,
            state=_STATE_STOPPED,
        )


def _log_worker_progress(
    worker_id: int,
    iteration: int,
    corpus_size: int,
    reporter: Any,
    snapshot: Optional[dict[str, Any]],  # noqa: ARG001
) -> None:
    del snapshot
    elapsed = reporter.get_elapsed_time()
    rate = iteration / elapsed if elapsed > 0 else 0
    deploy_ok = reporter.get_runtime_deployment_success_rate()
    call_ok = reporter.get_runtime_call_success_rate()
    logger.info(
        f"[w{worker_id}] iter={iteration} | "
        f"corpus={corpus_size} | "
        f"divergences={reporter.divergences} | "
        f"crashes={reporter.compiler_crashes} | "
        f"deploy_ok={deploy_ok:.1f}% | "
        f"call_ok={call_ok:.1f}% | "
        f"rate={rate:.1f}/s"
    )


class ParallelFuzzer:
    def __init__(
        self,
        exports_dir: Path,
        corpus_dir: Path,
        *,
        num_workers: int = 4,
        seed: Optional[int] = None,
        map_size: int = 1 << 20,
        max_corpus_size: int = 10_000,
        harness_config: HarnessConfig = HarnessConfig(),
        tag_edges_with_config: bool = False,
        compaction_interval_s: float = 60.0,
        snapshot_interval_s: float = 120.0,
        monitor_interval_s: float = 2.0,
        worker_stall_timeout_s: float = 45.0,
        min_energy: int = 4,
        max_energy: int = 16,
        max_runtime_s: Optional[float] = None,
        max_iterations_per_worker: Optional[int] = None,
        reports_dir: Path = Path("reports"),
        worker_memory_limit: Optional[int] = _DEFAULT_WORKER_MEMORY_LIMIT_BYTES,
    ):
        self.exports_dir = Path(exports_dir)
        self.corpus_dir = Path(corpus_dir)
        self.num_workers = num_workers
        self.seed = seed if seed is not None else random.getrandbits(64)
        self.map_size = map_size
        self.max_corpus_size = max_corpus_size
        self.harness_config = harness_config
        self.tag_edges_with_config = tag_edges_with_config
        self.compaction_interval_s = compaction_interval_s
        self.snapshot_interval_s = snapshot_interval_s
        self.monitor_interval_s = monitor_interval_s
        self.worker_stall_timeout_s = worker_stall_timeout_s
        self.min_energy = min_energy
        self.max_energy = max_energy
        self.max_runtime_s = max_runtime_s
        self.max_iterations_per_worker = max_iterations_per_worker
        self.reports_dir = Path(reports_dir)
        self.worker_memory_limit = worker_memory_limit

        self.ctx = multiprocessing.get_context("spawn")
        self.stop_event = self.ctx.Event()

        self.shared_counts = create_shared_edge_counts(self.map_size)
        self.last_progress_ts = self.ctx.Array(
            ctypes.c_double,
            self.num_workers,
            lock=False,
        )
        self.last_iter = self.ctx.Array(
            ctypes.c_uint64,
            self.num_workers,
            lock=False,
        )
        self.worker_state = self.ctx.Array(
            ctypes.c_uint8,
            self.num_workers,
            lock=False,
        )
        self._bootstrap_done_event = self.ctx.Event()
        self._bootstrap_done_flags = self.ctx.Array(
            ctypes.c_uint8,
            self.num_workers,
            lock=False,
        )
        self._bootstrap_progress = self.ctx.Array(
            ctypes.c_uint64,
            self.num_workers,
            lock=False,
        )

        self._manager = self.ctx.Manager()
        self._shared_seen_crashes = self._manager.dict()
        self._shared_seen_compile_failures = self._manager.dict()
        self._shared_seen_compilation_timeouts = self._manager.dict()
        self._shared_seen_divergences = self._manager.dict()

        self._worker_epochs = [0 for _ in range(self.num_workers)]
        self._test_filter: Optional[TestFilter] = None
        self._bootstrap_required = False
        self._drop_initial_fresh = False

        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        (self.corpus_dir / "queue").mkdir(parents=True, exist_ok=True)
        (self.corpus_dir / "snapshots").mkdir(parents=True, exist_ok=True)

    def _spawn_worker(self, worker_id: int) -> Any:
        process = self.ctx.Process(
            target=_worker_main,
            kwargs={
                "worker_id": worker_id,
                "corpus_dir": self.corpus_dir,
                "shared_counts": self.shared_counts,
                "map_size": self.map_size,
                "exports_dir": self.exports_dir,
                "seed": self.seed,
                "worker_epoch": self._worker_epochs[worker_id],
                "test_filter": self._test_filter,
                "stop_event": self.stop_event,
                "last_progress_ts": self.last_progress_ts,
                "last_iter": self.last_iter,
                "worker_state": self.worker_state,
                "harness_config": self.harness_config,
                "reports_dir": self.reports_dir,
                "tag_edges_with_config": self.tag_edges_with_config,
                "min_energy": self.min_energy,
                "max_energy": self.max_energy,
                "max_iterations": self.max_iterations_per_worker,
                "seen_crashes": self._shared_seen_crashes,
                "seen_compile_failures": self._shared_seen_compile_failures,
                "seen_compilation_timeouts": self._shared_seen_compilation_timeouts,
                "seen_divergences": self._shared_seen_divergences,
                "worker_memory_limit": self.worker_memory_limit,
                "bootstrap_required": self._bootstrap_required,
                "bootstrap_done_event": self._bootstrap_done_event,
                "bootstrap_done_flags": self._bootstrap_done_flags,
                "bootstrap_progress": self._bootstrap_progress,
                "bootstrap_shard_count": self.num_workers,
                "drop_initial_fresh": self._drop_initial_fresh,
            },
            name=f"parallel-fuzzer-worker-{worker_id}",
        )
        process.start()
        self.last_progress_ts[worker_id] = time.monotonic()
        self.last_iter[worker_id] = 0
        self.worker_state[worker_id] = _STATE_IDLE
        return process

    def _queue_size(self) -> int:
        return sum(1 for _ in (self.corpus_dir / "queue").glob("*.meta.json"))

    def _cleanup_orphans(self) -> int:
        queue_dir = self.corpus_dir / "queue"
        deleted = 0
        cutoff = time.time() - 60.0

        for suffix in (".scenario.json", ".edges.bin"):
            for path in queue_dir.glob(f"*{suffix}"):
                if path.stat().st_mtime >= cutoff:
                    continue
                entry_id = path.name[: -len(suffix)]
                if (queue_dir / f"{entry_id}.meta.json").exists():
                    continue
                path.unlink(missing_ok=True)
                deleted += 1

        return deleted

    def _compact_corpus(self) -> int:
        all_entries = read_all_meta(self.corpus_dir)
        if not all_entries:
            return self._cleanup_orphans()

        by_fp: dict[str, list[DiskEntryMeta]] = {}
        for entry in all_entries:
            by_fp.setdefault(entry.coverage_fp, []).append(entry)

        to_delete: list[str] = []
        for entries in by_fp.values():
            keep_ids: set[str] = set()
            smallest = min(entries, key=lambda e: e.source_size)
            fastest = min(entries, key=lambda e: e.cycle_time_s)
            keep_ids.add(smallest.entry_id)
            keep_ids.add(fastest.entry_id)

            for entry in entries:
                if entry.entry_id not in keep_ids:
                    to_delete.append(entry.entry_id)

        to_delete_set = set(to_delete)
        remaining = [
            entry for entry in all_entries if entry.entry_id not in to_delete_set
        ]
        if len(remaining) > self.max_corpus_size:
            evictable = sorted(
                remaining,
                key=lambda e: e.timestamp,
            )
            excess = len(remaining) - self.max_corpus_size
            for entry in evictable[:excess]:
                to_delete_set.add(entry.entry_id)

        for entry_id in to_delete_set:
            delete_corpus_entry(self.corpus_dir, entry_id)

        return len(to_delete_set) + self._cleanup_orphans()

    def _snapshot_edge_counts(self, shared_counts: MutableSequence[int]) -> None:
        save_edge_count_snapshot(self.corpus_dir, shared_counts)

    def _monitor_and_respawn_workers(
        self,
        workers: list[Any],
    ) -> bool:
        restarted = False
        now = time.monotonic()

        for worker_id, process in enumerate(workers):
            if not process.is_alive():
                if self.stop_event.is_set():
                    continue
                self._worker_epochs[worker_id] += 1
                workers[worker_id] = self._spawn_worker(worker_id)
                restarted = True
                continue

            if self.worker_state[worker_id] == _STATE_BOOTSTRAPPING:
                continue

            last_ts = self.last_progress_ts[worker_id]
            if now - last_ts <= self.worker_stall_timeout_s:
                continue

            process.terminate()
            process.join(timeout=3.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)

            if self.stop_event.is_set():
                continue

            self._worker_epochs[worker_id] += 1
            workers[worker_id] = self._spawn_worker(worker_id)
            restarted = True

        return restarted

    def run(self, test_filter: Optional[TestFilter] = None) -> None:
        self._test_filter = test_filter

        queue_has_entries = any((self.corpus_dir / "queue").glob("*.meta.json"))
        self._bootstrap_required = not queue_has_entries
        self._drop_initial_fresh = self._bootstrap_required
        if self._bootstrap_required:
            self._bootstrap_done_event.clear()
            for i in range(self.num_workers):
                self._bootstrap_done_flags[i] = 0
                self._bootstrap_progress[i] = 0
        else:
            self._bootstrap_done_event.set()
            for i in range(self.num_workers):
                self._bootstrap_done_flags[i] = 1
                self._bootstrap_progress[i] = 0

        if queue_has_entries:
            snapshot = load_edge_count_snapshot(self.corpus_dir, self.map_size)
            if snapshot is not None:
                for i, count in enumerate(snapshot):
                    self.shared_counts[i] = count

        mem_str = (
            f"{self.worker_memory_limit / 1024**3:.1f} GiB"
            if self.worker_memory_limit is not None
            else "unlimited"
        )
        logger.info(
            f"starting {self.num_workers} workers, seed={self.seed}, "
            f"corpus={self.corpus_dir}, reports={self.reports_dir}, "
            f"worker_memory_limit={mem_str}, "
            f"bootstrap_required={self._bootstrap_required}"
        )

        workers = [
            self._spawn_worker(worker_id) for worker_id in range(self.num_workers)
        ]
        start_time = time.monotonic()
        last_compaction = start_time
        last_snapshot = start_time

        try:
            while not self.stop_event.is_set():
                now = time.monotonic()

                if (now - last_compaction) >= self.compaction_interval_s:
                    self._compact_corpus()
                    last_compaction = now
                elif self._queue_size() > self.max_corpus_size:
                    self._compact_corpus()
                    last_compaction = now

                if (now - last_snapshot) >= self.snapshot_interval_s:
                    self._snapshot_edge_counts(self.shared_counts)
                    last_snapshot = now

                self._monitor_and_respawn_workers(workers)

                if (
                    self.max_runtime_s is not None
                    and (now - start_time) >= self.max_runtime_s
                ):
                    self.stop_event.set()
                    break

                time.sleep(self.monitor_interval_s)

        except KeyboardInterrupt:
            self.stop_event.set()
        finally:
            self.stop_event.set()
            for process in workers:
                process.join(timeout=5.0)
            for process in workers:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2.0)
            for process in workers:
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1.0)

            self._snapshot_edge_counts(self.shared_counts)
            elapsed = time.monotonic() - start_time
            total_iters = sum(int(self.last_iter[i]) for i in range(self.num_workers))
            logger.info(
                f"stopped after {elapsed:.1f}s, "
                f"total_iterations={total_iters}, "
                f"corpus={self._queue_size()}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exports-dir", type=Path, default=Path("tests/vyper-exports"))
    parser.add_argument("--corpus-dir", type=Path, default=Path(".parallel-corpus"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--map-size", type=int, default=1 << 20)
    parser.add_argument("--max-corpus-size", type=int, default=10_000)
    parser.add_argument("--max-runtime-s", type=float, default=None)
    parser.add_argument("--max-iterations-per-worker", type=int, default=None)
    args = parser.parse_args()

    if args.num_workers < 1:
        parser.error("--num-workers must be >= 1")

    fuzzer = ParallelFuzzer(
        exports_dir=args.exports_dir,
        corpus_dir=args.corpus_dir,
        num_workers=args.num_workers,
        seed=args.seed,
        map_size=args.map_size,
        max_corpus_size=args.max_corpus_size,
        max_runtime_s=args.max_runtime_s,
        max_iterations_per_worker=args.max_iterations_per_worker,
        reports_dir=args.reports_dir,
        worker_memory_limit=_DEFAULT_WORKER_MEMORY_LIMIT_BYTES,
    )
    fuzzer.run(test_filter=build_default_coverage_test_filter())


if __name__ == "__main__":
    main()

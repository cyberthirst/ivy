from __future__ import annotations

import argparse
import ctypes
import hashlib
import logging
import multiprocessing
from pathlib import Path
import random
import time
from typing import Any, MutableSequence, Optional

from fuzzer.base_fuzzer import BaseFuzzer
from fuzzer.coverage.collector import ArcCoverageCollector
from fuzzer.coverage.corpus import scenario_source_size
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
    had_any_successful_compile,
)
from fuzzer.coverage.shared_tracker import SharedEdgeTracker, create_shared_edge_counts
from fuzzer.coverage_fuzzer import (
    build_default_coverage_issue_filter,
    build_default_coverage_test_filter,
)
from fuzzer.export_utils import TestFilter
from fuzzer.runtime_engine import HarnessConfig
from fuzzer.runner.scenario import create_scenario_from_item

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_EPS = 1e-9
_LOG_INTERVAL = 100
_STATE_IDLE = 0
_STATE_RUNNING = 1
_STATE_BOOTSTRAPPING = 2
_STATE_STOPPED = 3


def _disable_alarm_timeouts() -> None:
    from fuzzer import compilation as fuzzer_compilation

    fuzzer_compilation.compilation_timeout.__defaults__ = (0,)  # pyright: ignore[reportAttributeAccessIssue]


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


def _serialize_test_filter(
    test_filter: Optional[TestFilter],
) -> Optional[dict[str, Any]]:
    if test_filter is None:
        return None
    return {
        "exclude_multi_module": test_filter.exclude_multi_module,
        "exclude_deps": test_filter.exclude_deps,
        "path_includes": [p.pattern for p in test_filter.path_includes],
        "path_excludes": [p.pattern for p in test_filter.path_excludes],
        "source_excludes": [p.pattern for p in test_filter.source_excludes],
        "source_includes": [p.pattern for p in test_filter.source_includes],
        "name_includes": [p.pattern for p in test_filter.name_includes],
        "name_excludes": [p.pattern for p in test_filter.name_excludes],
    }


def _deserialize_test_filter(
    data: Optional[dict[str, Any]],
) -> Optional[TestFilter]:
    if data is None:
        return None

    test_filter = TestFilter(
        exclude_multi_module=bool(data.get("exclude_multi_module", False)),
        exclude_deps=bool(data.get("exclude_deps", False)),
    )

    for pattern in data.get("path_includes", []):
        test_filter.include_path(str(pattern))
    for pattern in data.get("path_excludes", []):
        test_filter.exclude_path(str(pattern))
    for pattern in data.get("source_excludes", []):
        test_filter.exclude_source(str(pattern))
    for pattern in data.get("source_includes", []):
        test_filter.include_source(str(pattern))
    for pattern in data.get("name_includes", []):
        test_filter.include_name(str(pattern))
    for pattern in data.get("name_excludes", []):
        test_filter.exclude_name(str(pattern))

    return test_filter


def _hash_collected_arcs(
    collector: ArcCoverageCollector, edge_map: EdgeMap
) -> set[int]:
    if edge_map.tag_with_config:
        return edge_map.hash_arcs_by_config(collector.get_arcs_by_config())
    return edge_map.hash_arcs(collector.get_scenario_arcs())


def _bootstrap_worker(
    *,
    worker_id: int,
    worker_seed: int,
    base_fuzzer: BaseFuzzer,
    test_filter: Optional[TestFilter],
    collector: ArcCoverageCollector,
    edge_map: EdgeMap,
    tracker: SharedEdgeTracker,
    corpus_dir: Path,
    stop_event: Any,
    last_progress_ts: MutableSequence[float],
    last_iter: MutableSequence[int],
    worker_state: MutableSequence[int],
) -> int:
    reporter = base_fuzzer.reporter
    exports = base_fuzzer.load_filtered_exports(test_filter)

    added = 0
    processed = 0
    for export in exports.values():
        for item in export.items.values():
            if stop_event.is_set():
                return added
            if item.item_type == "fixture":
                continue

            processed += 1
            scenario = create_scenario_from_item(item, use_python_args=True)
            scenario_seed = _derive_seed(worker_seed, "bootstrap", processed)

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

            reporter.ingest_run(artifacts.analysis, artifacts, debug_mode=False)

            if not had_any_successful_compile(artifacts.ivy_result):
                _touch_heartbeat(
                    worker_id,
                    last_progress_ts,
                    last_iter,
                    worker_state,
                    iteration=processed,
                    state=_STATE_BOOTSTRAPPING,
                )
                continue

            edge_ids = _hash_collected_arcs(collector, edge_map)
            if not edge_ids and not (
                artifacts.analysis.crashes or artifacts.analysis.divergences
            ):
                _touch_heartbeat(
                    worker_id,
                    last_progress_ts,
                    last_iter,
                    worker_state,
                    iteration=processed,
                    state=_STATE_BOOTSTRAPPING,
                )
                continue

            tracker.merge(edge_ids)
            write_corpus_entry(
                corpus_dir,
                scenario=scenario,
                edge_ids=edge_ids,
                compile_time_s=max(collector.compile_time_s, _EPS),
                source_size=scenario_source_size(scenario),
                generation=0,
                keep_forever=True,
                coverage_fp=coverage_fingerprint(edge_ids),
                worker_id=worker_id,
            )
            added += 1
            _touch_heartbeat(
                worker_id,
                last_progress_ts,
                last_iter,
                worker_state,
                iteration=processed,
                state=_STATE_BOOTSTRAPPING,
            )

    logger.info(f"[w{worker_id}] bootstrap complete: {added}/{processed} seeds added")
    return added


def _worker_main(
    *,
    worker_id: int,
    corpus_dir: Path,
    shared_counts: MutableSequence[int],
    map_size: int,
    exports_dir: Path,
    seed: int,
    worker_epoch: int,
    test_filter_config: Optional[dict[str, Any]],
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
) -> None:
    import boa  # pyright: ignore[reportMissingImports]

    boa.interpret.disable_cache()  # pyright: ignore[reportAttributeAccessIssue]
    _disable_alarm_timeouts()

    worker_seed = _derive_seed(seed, "worker", worker_id, worker_epoch)
    rng = random.Random(worker_seed)

    collector = ArcCoverageCollector()
    edge_map = EdgeMap(map_size, tag_with_config=tag_edges_with_config)
    tracker = SharedEdgeTracker(shared_counts, map_size)
    gatekeeper = Gatekeeper(tracker=tracker)  # pyright: ignore[reportArgumentType]
    disk_index = DiskIndex(corpus_dir)
    test_filter = _deserialize_test_filter(test_filter_config)

    base_fuzzer = BaseFuzzer(
        exports_dir=exports_dir,
        seed=worker_seed,
        debug_mode=False,
        issue_filter=build_default_coverage_issue_filter(),
        harness_config=harness_config,
    )
    reporter = base_fuzzer.reporter
    reporter.reports_dir = reports_dir
    reporter.start_timer()
    reporter.start_metrics_stream(harness_config.enable_interval_metrics)

    iteration = 0
    batch_id = 0
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

        if worker_id == 0 and len(disk_index) == 0:
            _touch_heartbeat(
                worker_id,
                last_progress_ts,
                last_iter,
                worker_state,
                iteration=iteration,
                state=_STATE_BOOTSTRAPPING,
            )
            _bootstrap_worker(
                worker_id=worker_id,
                worker_seed=worker_seed,
                base_fuzzer=base_fuzzer,
                test_filter=test_filter,
                collector=collector,
                edge_map=edge_map,
                tracker=tracker,
                corpus_dir=corpus_dir,
                stop_event=stop_event,
                last_progress_ts=last_progress_ts,
                last_iter=last_iter,
                worker_state=worker_state,
            )
            disk_index.scan_new()

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

            parent = select_parent(
                disk_index,
                corpus_dir,
                shared_counts,
                rng,
                k=8,
            )
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

            for e_idx in range(energy):
                if stop_event.is_set():
                    break

                iteration += 1
                scenario_seed = _derive_seed(
                    worker_seed,
                    "cov",
                    batch_id,
                    e_idx,
                    iteration,
                )

                reporter.set_context(
                    f"w{worker_id}_cov_{iteration}",
                    iteration,
                    worker_seed,
                    scenario_seed,
                )

                mutated = base_fuzzer.mutate_scenario(
                    parent_scenario,
                    scenario_seed=scenario_seed,
                )

                collector.start_scenario()
                artifacts = base_fuzzer.run_scenario_with_artifacts(
                    mutated,
                    seed=scenario_seed,
                    coverage_collector=collector,
                )

                reporter.ingest_run(artifacts.analysis, artifacts, debug_mode=False)

                edge_ids = _hash_collected_arcs(collector, edge_map)
                compile_time_s = max(collector.compile_time_s, _EPS)
                coverage_fp = coverage_fingerprint(edge_ids)
                source_size = scenario_source_size(mutated)

                decision = gatekeeper.decide_and_update(
                    edge_ids=edge_ids,
                    compile_time_s=compile_time_s,
                    analysis=artifacts.analysis,
                    ivy_result=artifacts.ivy_result,
                    improves_representative=disk_index.improves_representative(
                        coverage_fp=coverage_fp,
                        source_size=source_size,
                        compile_time_s=compile_time_s,
                    ),
                    coverage_fp=coverage_fp,
                )

                if decision.accept:
                    write_corpus_entry(
                        corpus_dir,
                        scenario=mutated,
                        edge_ids=edge_ids,
                        compile_time_s=compile_time_s,
                        source_size=source_size,
                        generation=parent_meta.generation + 1,
                        keep_forever=decision.reason in ("issue", "new_edge"),
                        coverage_fp=decision.coverage_fingerprint,
                        worker_id=worker_id,
                    )
                    disk_index.scan_new()

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

        self._worker_epochs = [0 for _ in range(self.num_workers)]
        self._test_filter_config: Optional[dict[str, Any]] = None

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
                "test_filter_config": self._test_filter_config,
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
            keep_ids = {entry.entry_id for entry in entries if entry.keep_forever}

            smallest = min(entries, key=lambda e: e.source_size)
            fastest = min(entries, key=lambda e: e.compile_time_s)
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
                [entry for entry in remaining if not entry.keep_forever],
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
        self._test_filter_config = _serialize_test_filter(test_filter)

        queue_has_entries = any((self.corpus_dir / "queue").glob("*.meta.json"))
        if queue_has_entries:
            snapshot = load_edge_count_snapshot(self.corpus_dir, self.map_size)
            if snapshot is not None:
                for i, count in enumerate(snapshot):
                    self.shared_counts[i] = count

        logger.info(
            f"starting {self.num_workers} workers, seed={self.seed}, "
            f"corpus={self.corpus_dir}, reports={self.reports_dir}"
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
    )
    fuzzer.run(test_filter=build_default_coverage_test_filter())


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Set, TYPE_CHECKING
import time
import json
import logging
import traceback
from pathlib import Path
from datetime import datetime

from fuzzer.coverage_types import RuntimeBranchOutcome, RuntimeStmtSite

if TYPE_CHECKING:
    from .result_analyzer import AnalysisResult
    from fuzzer.runner.base_scenario_runner import ScenarioResult

UNPARSABLE_INTEGRITY_SUM = "0" * 64


@dataclass
class RuntimeMetricsTotals:
    enumeration_calls: int = 0
    replay_calls: int = 0
    fuzz_calls: int = 0
    timeouts: int = 0
    new_coverage_calls: int = 0
    state_modified_calls: int = 0
    interesting_calls: int = 0
    skipped_replay: int = 0
    deployment_attempts: int = 0
    deployment_successes: int = 0
    deployment_failures: int = 0
    call_attempts: int = 0
    call_successes: int = 0
    call_failures: int = 0
    calls_to_no_code: int = 0


def _make_json_serializable(obj) -> Any:
    """Convert nested structures into JSON-serializable equivalents."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if isinstance(k, bytes):
                k = "0x" + k.hex()
            elif not isinstance(k, (str, int, float, bool, type(None))):
                k = str(k)
            result[k] = _make_json_serializable(v)
        return result
    if isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    if isinstance(obj, bytes):
        return "0x" + obj.hex()
    return obj


def _format_traceback(error: Optional[BaseException]) -> Optional[str]:
    if error is None:
        return None
    return "".join(traceback.format_exception(type(error), error, error.__traceback__))


def build_divergence_record(
    divergence: Any,
    *,
    item_name: str,
    scenario_num: int,
    seed: Optional[int],
    scenario_seed: Optional[int],
) -> Dict[str, Any]:
    """Create a JSON-serializable divergence payload for saving or printing."""
    divergence_data = {
        **divergence.as_dict,
        "timestamp": datetime.now().isoformat(),
        "item_name": item_name,
        "scenario_num": scenario_num,
        "seed": seed,
        "scenario_seed": scenario_seed,
    }
    return _make_json_serializable(divergence_data)


@dataclass
class FuzzerReporter:
    # Deployment statistics
    successful_deployments: int = 0
    deployment_failures: int = 0
    compilation_failures: int = 0
    compiler_crashes: int = 0

    # Call statistics
    successful_calls: int = 0
    call_failures: int = 0

    # Overall statistics
    total_scenarios: int = 0
    divergences: int = 0

    # Xfail validation statistics
    xfail_violations: int = 0  # Count of xfail expectation violations

    # Timing statistics
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    # Context for reporting
    current_item_name: Optional[str] = None
    current_scenario_num: Optional[int] = None
    seed: Optional[int] = None
    reports_dir: Path = field(default_factory=lambda: Path("reports"))
    _run_started_at: Optional[datetime] = None
    _run_reports_dir: Optional[Path] = None
    _file_counter: int = 0
    _metrics_enabled: bool = False
    _metrics_path: Optional[Path] = None
    _last_snapshot_elapsed_s: float = 0.0
    _last_snapshot_total_scenarios: int = 0
    _last_snapshot_total_calls: int = 0
    _last_snapshot_calls_to_no_code: int = 0
    _last_snapshot_total_deployments: int = 0
    _last_snapshot_compilation_failures: int = 0
    _last_snapshot_compiler_crashes: int = 0
    _seen_runtime_edges: Set[int] = field(default_factory=set)
    _seen_contract_fingerprints: Set[str] = field(default_factory=set)
    _seen_stmt_sites: Set[RuntimeStmtSite] = field(default_factory=set)
    _seen_branch_outcomes: Set[RuntimeBranchOutcome] = field(default_factory=set)
    _stmt_sites_total: Set[RuntimeStmtSite] = field(default_factory=set)
    _branch_outcomes_total: Set[RuntimeBranchOutcome] = field(default_factory=set)
    _runtime_totals: RuntimeMetricsTotals = field(default_factory=RuntimeMetricsTotals)
    _pending_runtime_edge_ids: Set[int] = field(default_factory=set)
    _pending_contract_fingerprints: Set[str] = field(default_factory=set)
    _pending_stmt_sites_seen: Set[RuntimeStmtSite] = field(default_factory=set)
    _pending_branch_outcomes_seen: Set[RuntimeBranchOutcome] = field(
        default_factory=set
    )
    _pending_stmt_sites_total: Set[RuntimeStmtSite] = field(default_factory=set)
    _pending_branch_outcomes_total: Set[RuntimeBranchOutcome] = field(
        default_factory=set
    )

    def record_compilation_failure(self):
        self.compilation_failures += 1

    def record_compiler_crash(self):
        self.compiler_crashes += 1

    def record_deployment(self, success: bool):
        if success:
            self.successful_deployments += 1
        else:
            self.deployment_failures += 1

    def record_call(self, success: bool):
        if success:
            self.successful_calls += 1
        else:
            self.call_failures += 1

    def record_scenario(self):
        self.total_scenarios += 1

    def record_divergence(self):
        self.divergences += 1

    def set_context(
        self,
        item_name: str,
        scenario_num: int,
        seed: Optional[int] = None,
        scenario_seed: Optional[int] = None,
    ):
        """Set the current test context for reporting."""
        self.current_item_name = item_name
        self.current_scenario_num = scenario_num
        if seed is not None:
            self.seed = seed
        # Attach scenario seed to the instance for saving alongside reports
        self._current_scenario_seed = scenario_seed

    def update_from_scenario_result(self, result: ScenarioResult):
        for trace_idx, deployment_result in result.get_deployment_results():
            if deployment_result.is_compiler_crash:
                # Compiler crash (CompilerPanic, CodegenPanic, etc.)
                self.record_compiler_crash()
                self.save_compiler_crash(
                    deployment_result.solc_json,
                    deployment_result.error,
                    type(deployment_result.error).__name__,
                    compiler_settings=deployment_result.compiler_settings,
                )
            elif deployment_result.is_compilation_failure:
                # Normal compilation failure (syntax, type errors, etc.)
                self.record_compilation_failure()
                self.save_compilation_failure(
                    deployment_result.solc_json,
                    deployment_result.error,
                    type(deployment_result.error).__name__,
                    compiler_settings=deployment_result.compiler_settings,
                )
            elif deployment_result.is_runtime_failure:
                # Actual runtime deployment failure (constructor failed)
                self.record_deployment(False)
            else:
                self.record_deployment(deployment_result.success)

        for trace_idx, call_result in result.get_call_results():
            self.record_call(call_result.success)

    def report(self, analysis: AnalysisResult, debug_mode: bool = False):
        """
        Report results from an AnalysisResult.

        Handles all reporting: stats, item stats, file saving, and logging.
        Saves unique items to filtered/, all items to unfiltered/ if debug_mode.
        """
        from .divergence_detector import DivergenceType

        item_name = self.current_item_name or "unknown"
        scenario_num = self.current_scenario_num or 0

        # Update scenario stats
        self.total_scenarios += 1

        # Update stats from analysis
        self.successful_deployments += analysis.successful_deployments
        self.deployment_failures += analysis.failed_deployments
        self.successful_calls += analysis.successful_calls
        self.call_failures += analysis.failed_calls

        # Report crashes
        for deployment_result, decision in analysis.crashes:
            self.compiler_crashes += 1
            solc_json = deployment_result.solc_json
            error = deployment_result.error
            error_type = type(error).__name__

            status = "new" if decision.keep else "dup"
            logging.error(
                f"crash| {status} | {item_name} | mut#{scenario_num} | {error_type}"
            )

            if decision.keep:
                self.save_compiler_crash(
                    solc_json,
                    error,
                    error_type,
                    compiler_settings=deployment_result.compiler_settings,
                    subfolder="filtered",
                )
            if debug_mode:
                self.save_compiler_crash(
                    solc_json,
                    error,
                    error_type,
                    compiler_settings=deployment_result.compiler_settings,
                    subfolder="unfiltered",
                )

        # Report compilation failures
        for deployment_result, decision in analysis.compile_failures:
            self.compilation_failures += 1
            solc_json = deployment_result.solc_json
            error = deployment_result.error
            error_type = type(error).__name__

            if decision.keep:
                logging.debug(
                    f"compile_fail| new | {item_name} | mut#{scenario_num} | {error_type}"
                )
                self.save_compilation_failure(
                    solc_json,
                    error,
                    error_type,
                    compiler_settings=deployment_result.compiler_settings,
                    subfolder="filtered",
                )
            if debug_mode:
                self.save_compilation_failure(
                    solc_json,
                    error,
                    error_type,
                    compiler_settings=deployment_result.compiler_settings,
                    subfolder="unfiltered",
                )

        # Report divergences
        for divergence, decision in analysis.divergences:
            self.divergences += 1

            status = "new" if decision.keep else "dup"
            if divergence.type == DivergenceType.DEPLOYMENT:
                logging.error(
                    f"diff| {status} | {item_name} | mut#{scenario_num} | step {divergence.step} | {divergence.divergent_runner} | deployment"
                )
            else:
                logging.error(
                    f"diff| {status} | {item_name} | mut#{scenario_num} | step {divergence.step} | {divergence.divergent_runner} | {divergence.function}"
                )

            if decision.keep:
                self.save_divergence(divergence, subfolder="filtered")
            if debug_mode:
                self.save_divergence(divergence, subfolder="unfiltered")

        # Log success if no divergences
        if not analysis.divergences:
            logging.info(f"ok  | {item_name} | mut#{scenario_num}")

    def get_deployment_success_rate(self) -> float:
        total = self.successful_deployments + self.deployment_failures
        if total == 0:
            return 0.0
        return (self.successful_deployments / total) * 100

    def get_call_success_rate(self) -> float:
        total = self.successful_calls + self.call_failures
        if total == 0:
            return 0.0
        return (self.successful_calls / total) * 100

    def _build_run_reports_dir(self, started_at: datetime) -> Path:
        seed_part = str(self.seed) if self.seed is not None else "noseed"
        base_name = f"{started_at.strftime('%Y-%m-%d_%H-%M-%S')}_seed-{seed_part}"
        run_dir = self.reports_dir / base_name
        suffix = 1
        while run_dir.exists():
            run_dir = self.reports_dir / f"{base_name}_{suffix}"
            suffix += 1
        return run_dir

    def _get_run_reports_dir(self) -> Path:
        if self._run_reports_dir is None:
            started_at = self._run_started_at or datetime.now()
            self._run_started_at = started_at
            self._run_reports_dir = self._build_run_reports_dir(started_at)

        self._run_reports_dir.mkdir(parents=True, exist_ok=True)
        return self._run_reports_dir

    def start_timer(self):
        self._run_started_at = datetime.now()
        self._run_reports_dir = self._build_run_reports_dir(self._run_started_at)
        self._run_reports_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        self.end_time = None

    def stop_timer(self):
        self.end_time = time.time()

    def get_elapsed_time(self) -> float:
        if self.start_time is None:
            return 0.0
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    def get_scenarios_per_second(self) -> float:
        elapsed = self.get_elapsed_time()
        if elapsed == 0:
            return 0.0
        return self.total_scenarios / elapsed

    @staticmethod
    def _safe_pct(numerator: int, denominator: int) -> float:
        if denominator <= 0:
            return 0.0
        return (numerator / denominator) * 100.0

    @staticmethod
    def _safe_rate(count: int, denominator: int | float) -> float:
        if denominator <= 0:
            return 0.0
        return count / denominator

    @staticmethod
    def _resolve_error_type(
        error: Optional[Exception], error_type: Optional[str]
    ) -> str:
        if error_type:
            return error_type
        if error is None:
            return "UnknownError"
        return type(error).__name__

    @staticmethod
    def _merge_harness_stats(dst: RuntimeMetricsTotals, src: Any) -> None:
        fields = (
            "enumeration_calls",
            "replay_calls",
            "fuzz_calls",
            "timeouts",
            "new_coverage_calls",
            "state_modified_calls",
            "interesting_calls",
            "skipped_replay",
            "deployment_attempts",
            "deployment_successes",
            "deployment_failures",
            "call_attempts",
            "call_successes",
            "call_failures",
            "calls_to_no_code",
        )
        for field_name in fields:
            setattr(
                dst,
                field_name,
                getattr(dst, field_name) + int(getattr(src, field_name, 0)),
            )

    @staticmethod
    def _clear_pending_metrics_state(
        pending_runtime_edge_ids: Set[int],
        pending_contract_fingerprints: Set[str],
        pending_stmt_sites_seen: Set[RuntimeStmtSite],
        pending_branch_outcomes_seen: Set[RuntimeBranchOutcome],
        pending_stmt_sites_total: Set[RuntimeStmtSite],
        pending_branch_outcomes_total: Set[RuntimeBranchOutcome],
    ) -> None:
        pending_runtime_edge_ids.clear()
        pending_contract_fingerprints.clear()
        pending_stmt_sites_seen.clear()
        pending_branch_outcomes_seen.clear()
        pending_stmt_sites_total.clear()
        pending_branch_outcomes_total.clear()

    def _reset_metrics_state(self) -> None:
        self._last_snapshot_elapsed_s = 0.0
        self._last_snapshot_total_scenarios = 0
        self._last_snapshot_total_calls = 0
        self._last_snapshot_calls_to_no_code = 0
        self._last_snapshot_total_deployments = 0
        self._last_snapshot_compilation_failures = 0
        self._last_snapshot_compiler_crashes = 0

        self._seen_runtime_edges.clear()
        self._seen_contract_fingerprints.clear()
        self._seen_stmt_sites.clear()
        self._seen_branch_outcomes.clear()
        self._stmt_sites_total.clear()
        self._branch_outcomes_total.clear()

        self._runtime_totals = RuntimeMetricsTotals()
        self._clear_pending_metrics_state(
            self._pending_runtime_edge_ids,
            self._pending_contract_fingerprints,
            self._pending_stmt_sites_seen,
            self._pending_branch_outcomes_seen,
            self._pending_stmt_sites_total,
            self._pending_branch_outcomes_total,
        )

    def ingest_run(self, analysis: AnalysisResult, artifacts: Any, *, debug_mode: bool):
        self.report(analysis, debug_mode=debug_mode)

        harness_stats = getattr(artifacts, "harness_stats", None)
        if harness_stats is not None:
            self._merge_harness_stats(self._runtime_totals, harness_stats)

        # Avoid accumulating pending state when the metrics stream is disabled or
        # not initialized.
        if not self._metrics_enabled or self._metrics_path is None:
            self._clear_pending_metrics_state(
                self._pending_runtime_edge_ids,
                self._pending_contract_fingerprints,
                self._pending_stmt_sites_seen,
                self._pending_branch_outcomes_seen,
                self._pending_stmt_sites_total,
                self._pending_branch_outcomes_total,
            )
            return

        self._pending_runtime_edge_ids.update(
            getattr(artifacts, "runtime_edge_ids", set())
        )
        self._pending_stmt_sites_seen.update(
            getattr(artifacts, "runtime_stmt_sites_seen", set())
        )
        self._pending_branch_outcomes_seen.update(
            getattr(artifacts, "runtime_branch_outcomes_seen", set())
        )
        self._pending_stmt_sites_total.update(
            getattr(artifacts, "runtime_stmt_sites_total", set())
        )
        self._pending_branch_outcomes_total.update(
            getattr(artifacts, "runtime_branch_outcomes_total", set())
        )
        self._pending_contract_fingerprints.update(
            getattr(artifacts, "contract_fingerprints", set())
        )

    def get_runtime_deployment_success_rate(self) -> float:
        return self._safe_pct(
            self._runtime_totals.deployment_successes,
            self._runtime_totals.deployment_attempts,
        )

    def get_runtime_call_success_rate(self) -> float:
        return self._safe_pct(
            self._runtime_totals.call_successes,
            self._runtime_totals.call_attempts,
        )

    def start_metrics_stream(self, enabled: bool) -> None:
        self._metrics_enabled = enabled
        self._reset_metrics_state()
        # If the reporter instance is reused across campaigns, treat the current
        # aggregate counters as the baseline so new intervals don't include prior runs.
        self._last_snapshot_total_scenarios = self.total_scenarios
        self._last_snapshot_total_deployments = (
            self.successful_deployments
            + self.deployment_failures
            + self.compilation_failures
            + self.compiler_crashes
        )
        self._last_snapshot_compilation_failures = self.compilation_failures
        self._last_snapshot_compiler_crashes = self.compiler_crashes
        if not enabled:
            self._metrics_path = None
            return

        stats_dir = self._get_run_reports_dir()
        stats_dir.mkdir(parents=True, exist_ok=True)
        seed_part = str(self.seed) if self.seed is not None else "noseed"
        timestamp = datetime.now().strftime("%H%M%S")
        self._metrics_path = stats_dir / f"metrics_{seed_part}_{timestamp}.jsonl"

        logging.info(f"Interval metrics enabled at {self._metrics_path}")

    def record_interval_metrics(
        self,
        *,
        iteration: int,
        corpus_seed_count: int,
        corpus_evolved_count: int,
        corpus_max_evolved: int,
        include_coverage_percentages: bool,
    ) -> Optional[Dict[str, Any]]:
        if not self._metrics_enabled or self._metrics_path is None:
            self._clear_pending_metrics_state(
                self._pending_runtime_edge_ids,
                self._pending_contract_fingerprints,
                self._pending_stmt_sites_seen,
                self._pending_branch_outcomes_seen,
                self._pending_stmt_sites_total,
                self._pending_branch_outcomes_total,
            )
            return None

        elapsed = self.get_elapsed_time()
        delta_elapsed = max(elapsed - self._last_snapshot_elapsed_s, 1e-9)

        scenarios_interval = self.total_scenarios - self._last_snapshot_total_scenarios
        call_attempts_total = self._runtime_totals.call_attempts
        phase_calls_total = (
            self._runtime_totals.enumeration_calls
            + self._runtime_totals.replay_calls
            + self._runtime_totals.fuzz_calls
        )
        if call_attempts_total != phase_calls_total:
            logging.debug(
                "Runtime call counter mismatch: call_attempts=%s phase_calls=%s",
                call_attempts_total,
                phase_calls_total,
            )
        calls_interval = call_attempts_total - self._last_snapshot_total_calls
        calls_to_no_code_total = self._runtime_totals.calls_to_no_code
        calls_to_no_code_interval = (
            calls_to_no_code_total - self._last_snapshot_calls_to_no_code
        )

        all_deployments_total = (
            self.successful_deployments
            + self.deployment_failures
            + self.compilation_failures
            + self.compiler_crashes
        )
        deployments_interval = (
            all_deployments_total - self._last_snapshot_total_deployments
        )
        compilation_failures_interval = (
            self.compilation_failures - self._last_snapshot_compilation_failures
        )
        compiler_crashes_interval = (
            self.compiler_crashes - self._last_snapshot_compiler_crashes
        )

        prev_runtime_edges_total = len(self._seen_runtime_edges)
        self._seen_runtime_edges.update(self._pending_runtime_edge_ids)
        runtime_edges_total = len(self._seen_runtime_edges)
        new_runtime_edges_interval = runtime_edges_total - prev_runtime_edges_total

        prev_contracts_total = len(self._seen_contract_fingerprints)
        self._seen_contract_fingerprints.update(self._pending_contract_fingerprints)
        contracts_seen_total = len(self._seen_contract_fingerprints)
        new_contracts_interval = contracts_seen_total - prev_contracts_total

        self._seen_stmt_sites.update(self._pending_stmt_sites_seen)
        self._seen_branch_outcomes.update(self._pending_branch_outcomes_seen)
        self._stmt_sites_total.update(self._pending_stmt_sites_total)
        self._branch_outcomes_total.update(self._pending_branch_outcomes_total)

        stmt_coverage_pct: Optional[float] = None
        branch_coverage_pct: Optional[float] = None
        if include_coverage_percentages:
            if self._stmt_sites_total:
                stmt_coverage_pct = self._safe_pct(
                    len(self._seen_stmt_sites), len(self._stmt_sites_total)
                )
            if self._branch_outcomes_total:
                branch_coverage_pct = self._safe_pct(
                    len(self._seen_branch_outcomes), len(self._branch_outcomes_total)
                )

        deployment_attempts_total = self._runtime_totals.deployment_attempts
        deployment_successes_total = self._runtime_totals.deployment_successes
        deployment_failures_total = self._runtime_totals.deployment_failures
        call_successes_total = self._runtime_totals.call_successes
        call_failures_total = self._runtime_totals.call_failures

        snapshot: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "seed": self.seed,
            "iteration": iteration,
            "elapsed_s": elapsed,
            "throughput": {
                "scenarios_per_sec": self._safe_rate(scenarios_interval, delta_elapsed),
                "calls_per_sec": self._safe_rate(calls_interval, delta_elapsed),
                "scenarios_total": self.total_scenarios,
                "calls_total": call_attempts_total,
            },
            "deployments": {
                "attempts_total": deployment_attempts_total,
                "success_total": deployment_successes_total,
                "failure_total": deployment_failures_total,
                "success_rate_pct": self._safe_pct(
                    deployment_successes_total, deployment_attempts_total
                ),
            },
            "calls": {
                "attempts_total": call_attempts_total,
                "success_total": call_successes_total,
                "failure_total": call_failures_total,
                "timeouts_total": self._runtime_totals.timeouts,
                "calls_to_no_code_total": calls_to_no_code_total,
                "calls_to_no_code_interval": calls_to_no_code_interval,
                "calls_to_no_code_pct": self._safe_pct(
                    calls_to_no_code_total, call_attempts_total
                ),
                "success_rate_pct": self._safe_pct(
                    call_successes_total, call_attempts_total
                ),
            },
            "compile": {
                "compilation_failures_total": self.compilation_failures,
                "compiler_crashes_total": self.compiler_crashes,
                "compilation_failures_interval": compilation_failures_interval,
                "compiler_crashes_interval": compiler_crashes_interval,
                "compilation_failures_per_scenario": self._safe_rate(
                    compilation_failures_interval, scenarios_interval
                ),
                "compiler_crashes_per_scenario": self._safe_rate(
                    compiler_crashes_interval, scenarios_interval
                ),
                "compilation_failures_per_deployment": self._safe_rate(
                    compilation_failures_interval, deployments_interval
                ),
                "compiler_crashes_per_deployment": self._safe_rate(
                    compiler_crashes_interval, deployments_interval
                ),
            },
            "coverage": {
                "runtime_edges_total": runtime_edges_total,
                "new_runtime_edges_interval": new_runtime_edges_interval,
                "new_runtime_edges_per_sec": (
                    new_runtime_edges_interval / delta_elapsed
                ),
                "stmt_coverage_pct": stmt_coverage_pct,
                "branch_coverage_pct": branch_coverage_pct,
            },
            "novelty": {
                "contracts_seen_total": contracts_seen_total,
                "new_contracts_interval": new_contracts_interval,
            },
            "runtime_phase": {
                "enumeration_calls": self._runtime_totals.enumeration_calls,
                "replay_calls": self._runtime_totals.replay_calls,
                "fuzz_calls": self._runtime_totals.fuzz_calls,
                "interesting_calls": self._runtime_totals.interesting_calls,
                "new_coverage_calls": self._runtime_totals.new_coverage_calls,
                "state_modified_calls": self._runtime_totals.state_modified_calls,
                "skipped_replay": self._runtime_totals.skipped_replay,
            },
            "corpus": {
                "seed_count": corpus_seed_count,
                "evolved_count": corpus_evolved_count,
                "max_evolved": corpus_max_evolved,
            },
            "issues": {
                "divergences_total": self.divergences,
                "compilation_failures_total": self.compilation_failures,
                "compiler_crashes_total": self.compiler_crashes,
            },
        }

        with open(self._metrics_path, "a") as f:
            json.dump(snapshot, f, default=str)
            f.write("\n")

        self._last_snapshot_elapsed_s = elapsed
        self._last_snapshot_total_scenarios = self.total_scenarios
        self._last_snapshot_total_calls = call_attempts_total
        self._last_snapshot_calls_to_no_code = calls_to_no_code_total
        self._last_snapshot_total_deployments = all_deployments_total
        self._last_snapshot_compilation_failures = self.compilation_failures
        self._last_snapshot_compiler_crashes = self.compiler_crashes
        self._clear_pending_metrics_state(
            self._pending_runtime_edge_ids,
            self._pending_contract_fingerprints,
            self._pending_stmt_sites_seen,
            self._pending_branch_outcomes_seen,
            self._pending_stmt_sites_total,
            self._pending_branch_outcomes_total,
        )

        return snapshot

    def log_generative_progress(
        self,
        *,
        iteration: int,
        corpus_seed_count: int,
        corpus_evolved_count: int,
        snapshot: Optional[Dict[str, Any]],
    ) -> None:
        """Log interval progress for the generative fuzzer."""
        elapsed = self.get_elapsed_time()
        rate = iteration / elapsed if elapsed > 0 else 0
        deployment_success_rate = self.get_runtime_deployment_success_rate()
        call_success_rate = self.get_runtime_call_success_rate()
        new_edges_interval = 0
        new_contracts_interval = 0
        if snapshot is not None:
            coverage = snapshot.get("coverage", {})
            novelty = snapshot.get("novelty", {})
            new_edges_interval = int(coverage.get("new_runtime_edges_interval", 0))
            new_contracts_interval = int(novelty.get("new_contracts_interval", 0))

        logging.info(
            f"iter={iteration} | "
            f"seeds={corpus_seed_count} | "
            f"evolved={corpus_evolved_count} | "
            f"divergences={self.divergences} | "
            f"deploy_ok={deployment_success_rate:.1f}% | "
            f"call_ok={call_success_rate:.1f}% | "
            f"new_edges={new_edges_interval} | "
            f"new_contracts={new_contracts_interval} | "
            f"rate={rate:.1f}/s"
        )

    def print_summary(self):
        print("\n" + "=" * 60)
        print("FUZZING CAMPAIGN STATISTICS")
        print("=" * 60)

        # Time and throughput statistics
        elapsed = self.get_elapsed_time()
        print(f"Total running time: {self._format_duration(elapsed)}")
        print(f"Throughput: {self.get_scenarios_per_second():.2f} scenarios/second")
        print(f"\nTotal scenarios executed: {self.total_scenarios}")
        print(f"Total divergences found: {self.divergences}")

        print("\nDeployment Statistics:")
        print(f"  Successful deployments: {self.successful_deployments}")
        print(f"  Failed deployments: {self.deployment_failures}")
        print(f"  Compilation failures: {self.compilation_failures}")
        print(f"  Compiler crashes: {self.compiler_crashes}")

        print("\nCall Statistics:")
        print(f"  Successful calls: {self.successful_calls}")
        print(f"  Failed calls: {self.call_failures}")

        # Print success rates
        print("\nSuccess Rates:")
        print(f"  Deployment success rate: {self.get_deployment_success_rate():.2f}%")
        print(f"  Call success rate: {self.get_call_success_rate():.2f}%")

        print("=" * 60 + "\n")

    def _format_duration(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} hours"

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to a dictionary for serialization."""
        return {
            "successful_deployments": self.successful_deployments,
            "deployment_failures": self.deployment_failures,
            "compilation_failures": self.compilation_failures,
            "compiler_crashes": self.compiler_crashes,
            "successful_calls": self.successful_calls,
            "call_failures": self.call_failures,
            "total_scenarios": self.total_scenarios,
            "divergences": self.divergences,
            "deployment_success_rate": self.get_deployment_success_rate(),
            "call_success_rate": self.get_call_success_rate(),
            "elapsed_time_seconds": self.get_elapsed_time(),
            "scenarios_per_second": self.get_scenarios_per_second(),
        }

    def save_divergence(self, divergence: Any, subfolder: Optional[str] = None):
        """Save a divergence between Ivy and Boa execution."""
        reports_dir = self._get_run_reports_dir()
        if subfolder:
            reports_dir = reports_dir / subfolder
        reports_dir = reports_dir / "divergences"
        reports_dir.mkdir(parents=True, exist_ok=True)

        item_name = self.current_item_name or "unknown"
        scenario_num = self.current_scenario_num or 0

        self._file_counter += 1
        filename = f"divergence_{self._file_counter}.json"
        filepath = reports_dir / filename

        divergence_data = build_divergence_record(
            divergence,
            item_name=item_name,
            scenario_num=scenario_num,
            seed=self.seed,
            scenario_seed=getattr(self, "_current_scenario_seed", None),
        )

        with open(filepath, "w") as f:
            json.dump(divergence_data, f, indent=2, default=str)

        logging.error(f"Divergence saved to {filepath}")

    def save_compiler_crash(
        self,
        solc_json: Optional[Dict[str, Any]],
        error: Optional[Exception],
        error_type: Optional[str] = None,
        compiler_settings: Optional[Dict[str, Any]] = None,
        subfolder: Optional[str] = None,
    ):
        """Save a compiler crash with the source code that caused it."""
        error_type = self._resolve_error_type(error, error_type)

        crash_dir = self._get_run_reports_dir()
        if subfolder:
            crash_dir = crash_dir / subfolder
        crash_dir = crash_dir / "compiler_crashes"
        crash_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
        filename = f"crash_{error_type}_{timestamp}.json"
        filepath = crash_dir / filename

        crash_data = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": str(error),
            "error_traceback": _format_traceback(error),
            "solc_json": solc_json,
            "compiler_settings": compiler_settings,
            "reproduction_info": {
                "seed": self.seed,
                "item_name": self.current_item_name or "unknown",
                "scenario_num": self.current_scenario_num or -1,
                "scenario_seed": getattr(self, "_current_scenario_seed", None),
            },
        }

        with open(filepath, "w") as f:
            json.dump(crash_data, f, indent=2, default=str)

        logging.error(f"Compiler crash saved to {filepath}")

    def save_compilation_failure(
        self,
        solc_json: Optional[Dict[str, Any]],
        error: Optional[Exception],
        error_type: Optional[str] = None,
        compiler_settings: Optional[Dict[str, Any]] = None,
        subfolder: Optional[str] = None,
    ):
        """Save a compilation failure with the source code that caused it."""
        error_type = self._resolve_error_type(error, error_type)

        failure_dir = self._get_run_reports_dir()
        if subfolder:
            failure_dir = failure_dir / subfolder
        failure_dir = failure_dir / "compilation_failures"
        failure_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
        filename = f"failure_{error_type}_{timestamp}.json"
        filepath = failure_dir / filename

        failure_data = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": str(error),
            "error_traceback": _format_traceback(error),
            "solc_json": solc_json,
            "compiler_settings": compiler_settings,
            "reproduction_info": {
                "seed": self.seed,
                "item_name": self.current_item_name or "unknown",
                "scenario_num": self.current_scenario_num or -1,
                "scenario_seed": getattr(self, "_current_scenario_seed", None),
            },
        }

        with open(filepath, "w") as f:
            json.dump(failure_data, f, indent=2, default=str)

        logging.debug(f"Compilation failure saved to {filepath}")

    def save_statistics(self):
        """Save the campaign statistics to a JSON file."""
        stats_dir = self._get_run_reports_dir()
        stats_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%H%M%S")
        filename = f"stats_{timestamp}.json"
        filepath = stats_dir / filename

        stats_data = self.to_dict()
        stats_data["timestamp"] = datetime.now().isoformat()
        stats_data["seed"] = self.seed

        with open(filepath, "w") as f:
            json.dump(stats_data, f, indent=2, default=str)

        logging.info(f"Statistics saved to {filepath}")

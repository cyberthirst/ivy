from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, TYPE_CHECKING
import time
import json
import logging
from pathlib import Path
from datetime import datetime

from src.fuzzer.runner.base_scenario_runner import ScenarioResult

if TYPE_CHECKING:
    from .result_analyzer import AnalysisResult


def _make_json_serializable(obj):
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

    # Per-item statistics
    item_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Timing statistics
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    # Context for reporting
    current_item_name: Optional[str] = None
    current_scenario_num: Optional[int] = None
    seed: Optional[int] = None
    reports_dir: Path = field(default_factory=lambda: Path("reports"))
    _file_counter: int = 0

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
                    deployment_result.source_code,
                    deployment_result.error,
                    type(deployment_result.error).__name__,
                )
            elif deployment_result.is_compilation_failure:
                # Normal compilation failure (syntax, type errors, etc.)
                self.record_compilation_failure()
                self.save_compilation_failure(
                    deployment_result.source_code,
                    deployment_result.error,
                    type(deployment_result.error).__name__,
                )
            elif deployment_result.is_runtime_failure:
                # Actual runtime deployment failure (constructor failed)
                self.record_deployment(False)
            else:
                self.record_deployment(deployment_result.success)

        for trace_idx, call_result in result.get_call_results():
            self.record_call(call_result.success)

    def record_item_stats(self, item_name: str, stat_type: str):
        """Record statistics for a specific test item."""
        if item_name not in self.item_stats:
            self.item_stats[item_name] = {
                "scenarios": 0,
                "divergences": 0,
                "deployments": 0,
                "calls": 0,
                "failures": 0,
            }

        if stat_type == "scenario":
            self.item_stats[item_name]["scenarios"] += 1
        elif stat_type == "divergence":
            self.item_stats[item_name]["divergences"] += 1
        elif stat_type == "deployment":
            self.item_stats[item_name]["deployments"] += 1
        elif stat_type == "call":
            self.item_stats[item_name]["calls"] += 1
        elif stat_type == "failure":
            self.item_stats[item_name]["failures"] += 1

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
        self.record_item_stats(item_name, "scenario")

        # Update stats from analysis
        self.successful_deployments += analysis.successful_deployments
        self.deployment_failures += analysis.failed_deployments
        self.successful_calls += analysis.successful_calls
        self.call_failures += analysis.failed_calls

        # Report crashes
        for deployment_result, decision in analysis.crashes:
            self.compiler_crashes += 1
            source_code = deployment_result.source_code
            error = deployment_result.error
            error_type = type(error).__name__

            status = "new" if decision.keep else "dup"
            logging.error(
                f"crash| {status} | {item_name} | mut#{scenario_num} | {error_type}"
            )

            if decision.keep:
                self.save_compiler_crash(
                    source_code, error, error_type, subfolder="filtered"
                )
            if debug_mode:
                self.save_compiler_crash(
                    source_code, error, error_type, subfolder="unfiltered"
                )

        # Report compilation failures
        for deployment_result, decision in analysis.compile_failures:
            self.compilation_failures += 1
            source_code = deployment_result.source_code
            error = deployment_result.error
            error_type = type(error).__name__

            if decision.keep:
                logging.debug(
                    f"compile_fail| new | {item_name} | mut#{scenario_num} | {error_type}"
                )
                self.save_compilation_failure(
                    source_code, error, error_type, subfolder="filtered"
                )
            if debug_mode:
                self.save_compilation_failure(
                    source_code, error, error_type, subfolder="unfiltered"
                )

        # Report divergences
        for divergence, decision in analysis.divergences:
            self.divergences += 1
            self.record_item_stats(item_name, "divergence")

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

    def start_timer(self):
        self.start_time = time.time()

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

        # Print per-item statistics if we have any
        if self.item_stats:
            print("\nPer-Item Statistics:")
            print(
                f"{'Item Name':<50} {'Scenarios':>10} {'Divergences':>12} {'Div Rate':>10}"
            )
            print("-" * 85)

            for item_name, stats in sorted(self.item_stats.items()):
                div_rate = 0.0
                if stats["scenarios"] > 0:
                    div_rate = (stats["divergences"] / stats["scenarios"]) * 100

                # Truncate long names
                display_name = (
                    item_name if len(item_name) <= 50 else item_name[:47] + "..."
                )
                print(
                    f"{display_name:<50} {stats['scenarios']:>10} {stats['divergences']:>12} {div_rate:>9.1f}%"
                )

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
            "item_stats": self.item_stats,
            "elapsed_time_seconds": self.get_elapsed_time(),
            "scenarios_per_second": self.get_scenarios_per_second(),
        }

    def save_divergence(self, divergence: Any, subfolder: Optional[str] = None):
        """Save a divergence between Ivy and Boa execution."""
        reports_dir = self.reports_dir / datetime.now().strftime("%Y-%m-%d")
        if subfolder:
            reports_dir = reports_dir / subfolder
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
        source_code: str,
        error: Exception,
        error_type: Optional[str] = None,
        subfolder: Optional[str] = None,
    ):
        """Save a compiler crash with the source code that caused it."""
        error_type = error_type or type(error).__name__

        crash_dir = self.reports_dir / datetime.now().strftime("%Y-%m-%d")
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
            "source_code": source_code,
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
        source_code: str,
        error: Exception,
        error_type: Optional[str] = None,
        subfolder: Optional[str] = None,
    ):
        """Save a compilation failure with the source code that caused it."""
        error_type = error_type or type(error).__name__

        failure_dir = self.reports_dir / datetime.now().strftime("%Y-%m-%d")
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
            "source_code": source_code,
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
        stats_dir = self.reports_dir / datetime.now().strftime("%Y-%m-%d")
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

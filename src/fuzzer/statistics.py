from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import time


@dataclass
class FuzzerStatistics:
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

    # Per-item statistics
    item_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Timing statistics
    start_time: Optional[float] = None
    end_time: Optional[float] = None

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

    def update_from_scenario_result(self, result: Any):
        for trace_idx, deployment_result in result.get_deployment_results():
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

"""
Result analyzer for differential fuzzing.

Analyzes scenario results, detects crashes/failures/divergences,
applies deduplication, and returns structured results for the fuzzer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from .runner.scenario import Scenario
from .runner.base_scenario_runner import ScenarioResult, DeploymentResult
from .divergence_detector import DivergenceDetector, Divergence
from .deduper import Deduper, KeepDecision
from .issue_filter import IssueFilter, IssueType as IT

if TYPE_CHECKING:
    from .runner.multi_runner import CompilerConfig


@dataclass
class AnalysisResult:
    """Result of analyzing a scenario run."""

    # Detected issues with dedup decisions
    crashes: List[Tuple[DeploymentResult, KeepDecision]] = field(default_factory=list)
    compile_failures: List[Tuple[DeploymentResult, KeepDecision]] = field(
        default_factory=list
    )
    divergences: List[Tuple[Divergence, KeepDecision]] = field(default_factory=list)

    # Stats for reporter
    successful_deployments: int = 0
    failed_deployments: int = 0  # runtime failures, not compile
    successful_calls: int = 0
    failed_calls: int = 0

    def has_any_unique(self) -> bool:
        """Check if any unique (non-duplicate) issues were found."""
        return (
            any(decision.keep for _, decision in self.crashes)
            or any(decision.keep for _, decision in self.compile_failures)
            or any(decision.keep for _, decision in self.divergences)
        )

    def unique_crash_count(self) -> int:
        return sum(1 for _, decision in self.crashes if decision.keep)

    def unique_compile_failure_count(self) -> int:
        return sum(1 for _, decision in self.compile_failures if decision.keep)

    def unique_divergence_count(self) -> int:
        return sum(1 for _, decision in self.divergences if decision.keep)


class ResultAnalyzer:
    """
    Analyzes scenario results and applies deduplication and filtering.

    Detects compiler crashes, compilation failures, and divergences.
    Uses IssueFilter to filter known issues, then Deduper to filter
    duplicates. Returns AnalysisResult for the fuzzer to use for
    corpus management and reporting.
    """

    def __init__(self, deduper: Deduper, issue_filter: Optional[IssueFilter] = None):
        self.deduper = deduper
        self.issue_filter = issue_filter
        self.divergence_detector = DivergenceDetector()

    def _check_filter(self, issue_dict: dict, issue_type: IT) -> Optional[KeepDecision]:
        """Check if issue should be filtered. Returns DedupDecision if filtered, None otherwise."""
        if not self.issue_filter:
            return None
        filter_reason = self.issue_filter.should_filter(issue_dict, issue_type)
        if filter_reason:
            return KeepDecision(
                keep=False, reason=f"filtered:{filter_reason}", fingerprint=""
            )
        return None

    def analyze_run(
        self,
        scenario: Scenario,
        ivy_result: ScenarioResult,
        boa_results: Dict[str, Tuple[CompilerConfig, ScenarioResult]],
    ) -> AnalysisResult:
        """
        Analyze a completed scenario run.

        Detects crashes, failures, divergences and applies deduplication.
        Returns AnalysisResult containing all findings with dedup decisions.
        """
        result = AnalysisResult()

        # Process Ivy results
        self._process_scenario_result(ivy_result, result)

        # Process all Boa results
        for _, (_, boa_result) in boa_results.items():
            self._process_scenario_result(boa_result, result)

        # Detect divergences
        divergences = self.divergence_detector.compare_all_results(
            ivy_result, boa_results, scenario
        )

        # Apply filter and dedup to divergences
        for divergence in divergences:
            decision = self._check_filter(divergence.as_dict, IT.DIVERGENCE)
            if decision is None:
                decision = self.deduper.check_divergence(divergence)
            result.divergences.append((divergence, decision))

        return result

    def _process_scenario_result(
        self, scenario_result: ScenarioResult, result: AnalysisResult
    ):
        """Process a single scenario result (Ivy or Boa)."""
        for _, deployment_result in scenario_result.get_deployment_results():
            if deployment_result.is_compiler_crash:
                decision = self._check_filter(deployment_result.to_dict(), IT.CRASH)
                if decision is None:
                    decision = self.deduper.check_compiler_crash(
                        deployment_result.error
                    )
                result.crashes.append((deployment_result, decision))
            elif deployment_result.is_compilation_failure:
                decision = self._check_filter(
                    deployment_result.to_dict(), IT.COMPILE_FAILURE
                )
                if decision is None:
                    decision = self.deduper.check_compilation_failure(
                        deployment_result.error
                    )
                result.compile_failures.append((deployment_result, decision))
            elif deployment_result.is_runtime_failure:
                result.failed_deployments += 1
            elif deployment_result.success:
                result.successful_deployments += 1
            else:
                result.failed_deployments += 1

        for _, call_result in scenario_result.get_call_results():
            if call_result.success:
                result.successful_calls += 1
            else:
                result.failed_calls += 1

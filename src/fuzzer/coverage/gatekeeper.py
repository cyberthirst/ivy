"""
Acceptance logic (the "gatekeeper") for coverage-guided fuzzing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

from fuzzer.runner.base_scenario_runner import ScenarioResult, DeploymentResult
from fuzzer.runner.scenario import Scenario
from fuzzer.result_analyzer import AnalysisResult
from fuzzer.coverage.tracker import GlobalEdgeTracker


def coverage_fingerprint(edge_ids: Set[int]) -> str:
    ordered = sorted(edge_ids)
    h = hashlib.blake2b(digest_size=16)
    for edge_id in ordered:
        h.update(edge_id.to_bytes(4, "little", signed=False))
    return h.hexdigest()


def had_any_successful_compile(result: ScenarioResult) -> bool:
    for _, deployment_result in result.get_deployment_results():
        assert isinstance(deployment_result, DeploymentResult)
        if deployment_result.success or deployment_result.is_runtime_failure:
            return True
    return False


def all_boa_configs_failed_to_compile(
    boa_results: Dict[str, Tuple[object, ScenarioResult]],
) -> bool:
    if not boa_results:
        return True
    return all(
        not had_any_successful_compile(result)
        for _, result in boa_results.values()
    )


@dataclass(frozen=True)
class GatekeeperDecision:
    accept: bool
    reason: str
    coverage_fingerprint: str
    new_edges: int
    rare_edge_score: float
    selection_weight: float


class Gatekeeper:
    def __init__(
        self,
        tracker: GlobalEdgeTracker,
        *,
        eps: float = 1e-9,
    ):
        self._tracker = tracker
        self._eps = eps

    def decide_and_update(
        self,
        *,
        edge_ids: Set[int],
        cycle_time_s: float,
        analysis: AnalysisResult,
        ivy_result: ScenarioResult,
        boa_results: Optional[Dict[str, Tuple[object, ScenarioResult]]] = None,
        improves_representative: bool,
        post_processed_scenario: Optional[Scenario] = None,
        coverage_fp: Optional[str] = None,
    ) -> GatekeeperDecision:
        # Note: queue admission semantics may depend on post-processing
        # (e.g. traces removed before corpus insertion), so callers can pass the
        # post-processed scenario even if it's not consumed yet.
        del post_processed_scenario, analysis

        # Reject if nothing compiled under Ivy. The AST mutator depends on the
        # compiler frontend to parse and annotate the source, so a scenario that
        # can't compile is a dead end — we won't be able to mutate it further.
        if not had_any_successful_compile(ivy_result):
            return GatekeeperDecision(
                accept=False,
                reason="no_ivy_compile",
                coverage_fingerprint=coverage_fp or "",
                new_edges=0,
                rare_edge_score=0.0,
                selection_weight=0.0,
            )

        fp = coverage_fp or coverage_fingerprint(edge_ids)

        rare_edge_score = self._tracker.compute_rare_score(edge_ids)
        selection_weight = rare_edge_score / max(cycle_time_s, self._eps)
        new_edges = self._tracker.merge(edge_ids)

        if boa_results is not None and all_boa_configs_failed_to_compile(boa_results):
            return GatekeeperDecision(
                accept=False,
                reason="all_configs_failed_compile",
                coverage_fingerprint=fp,
                new_edges=new_edges,
                rare_edge_score=rare_edge_score,
                selection_weight=selection_weight,
            )

        if new_edges > 0:
            accept = True
            reason = "new_edge"
        elif improves_representative:
            accept = True
            reason = "better_representative"
        else:
            accept = False
            reason = "no_new_coverage"

        return GatekeeperDecision(
            accept=accept,
            reason=reason,
            coverage_fingerprint=fp,
            new_edges=new_edges,
            rare_edge_score=rare_edge_score,
            selection_weight=selection_weight,
        )

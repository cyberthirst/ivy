"""
Acceptance logic (the "gatekeeper") for coverage-guided fuzzing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

from ..runner.base_scenario_runner import ScenarioResult, DeploymentResult
from ..result_analyzer import AnalysisResult
from .tracker import GlobalEdgeTracker


def coverage_fingerprint(edge_ids: Set[int]) -> str:
    ordered = sorted(edge_ids)
    h = hashlib.blake2b(digest_size=16)
    for edge_id in ordered:
        h.update(edge_id.to_bytes(4, "little", signed=False))
    return h.hexdigest()


def _boa_config_had_any_successful_compile(boa_result: ScenarioResult) -> bool:
    for _, deployment_result in boa_result.get_deployment_results():
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
        not _boa_config_had_any_successful_compile(result)
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
        edge_ids: Optional[Set[int]],
        compile_time_s: Optional[float],
        analysis: AnalysisResult,
        boa_results: Dict[str, Tuple[object, ScenarioResult]],
        improves_representative: bool,
        coverage_fp: Optional[str] = None,
    ) -> GatekeeperDecision:
        is_issue = bool(analysis.crashes or analysis.divergences)

        if edge_ids is None or compile_time_s is None:
            self._tracker.merge(edge_ids or set())
            fp = coverage_fp or coverage_fingerprint(edge_ids or set())
            return GatekeeperDecision(
                accept=is_issue,
                reason="issue" if is_issue else "missing_coverage",
                coverage_fingerprint=fp,
                new_edges=0,
                rare_edge_score=0.0,
                selection_weight=0.0,
            )

        fp = coverage_fp or coverage_fingerprint(edge_ids)

        rare_edge_score = self._tracker.compute_rare_score(edge_ids)
        selection_weight = rare_edge_score / max(compile_time_s, self._eps)
        new_edges = self._tracker.merge(edge_ids)

        if is_issue:
            return GatekeeperDecision(
                accept=True,
                reason="issue",
                coverage_fingerprint=fp,
                new_edges=new_edges,
                rare_edge_score=rare_edge_score,
                selection_weight=selection_weight,
            )

        if all_boa_configs_failed_to_compile(boa_results):
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

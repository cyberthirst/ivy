from __future__ import annotations

from types import SimpleNamespace

from fuzzer.coverage.disk_corpus import DiskIndex, read_scenario
from fuzzer.coverage.gatekeeper import Gatekeeper, GatekeeperDecision
from fuzzer.coverage.tracker import GlobalEdgeTracker
from fuzzer.deduper import KeepDecision
from fuzzer.parallel_fuzzer import (
    _admit_to_corpus,
    _has_unique_issue,
    _update_stagnation_counters,
)
from fuzzer.result_analyzer import AnalysisResult
from fuzzer.runner.base_scenario_runner import (
    DeploymentResult,
    ScenarioResult,
    TraceResult,
)
from fuzzer.runner.scenario import Scenario
from fuzzer.trace_types import DeploymentTrace, Env, Tx


def _make_deployment_trace(
    *, deployed_address: str = "0x0000000000000000000000000000000000000001"
) -> DeploymentTrace:
    return DeploymentTrace(
        deployment_type="source",
        calldata=None,
        value=0,
        solc_json={"sources": {"test.vy": {"content": "@external\ndef foo(): pass"}}},
        blueprint_initcode_prefix=None,
        deployed_address=deployed_address,
        deployment_succeeded=True,
        env=Env(tx=Tx(origin="0xabc")),
    )


def _successful_ivy_result() -> ScenarioResult:
    return ScenarioResult(
        results=[
            TraceResult(
                trace_type="deployment",
                trace_index=0,
                result=DeploymentResult(success=True),
            )
        ]
    )


def _make_artifacts(
    analysis: AnalysisResult, ivy_result: ScenarioResult | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        analysis=analysis,
        ivy_result=ivy_result or _successful_ivy_result(),
    )


def test_admit_to_corpus_returns_decision_for_accepted_entry(tmp_path):
    corpus_dir = tmp_path / "corpus"
    gatekeeper = Gatekeeper(GlobalEdgeTracker(64))
    disk_index = DiskIndex(corpus_dir)

    decision = _admit_to_corpus(
        scenario=Scenario(traces=[_make_deployment_trace()]),
        artifacts=_make_artifacts(AnalysisResult()),
        edge_ids={1, 2, 3},
        cycle_time_s=0.1,
        generation=1,
        worker_id=0,
        gatekeeper=gatekeeper,
        disk_index=disk_index,
        corpus_dir=corpus_dir,
    )

    assert decision is not None
    assert decision.reason == "new_edge"
    assert len(disk_index) == 1


def test_admit_to_corpus_returns_none_for_rejected_entry(tmp_path):
    corpus_dir = tmp_path / "corpus"
    gatekeeper = Gatekeeper(GlobalEdgeTracker(64))
    disk_index = DiskIndex(corpus_dir)
    artifacts = _make_artifacts(AnalysisResult())

    first = _admit_to_corpus(
        scenario=Scenario(traces=[_make_deployment_trace()]),
        artifacts=artifacts,
        edge_ids={7},
        cycle_time_s=0.5,
        generation=1,
        worker_id=0,
        gatekeeper=gatekeeper,
        disk_index=disk_index,
        corpus_dir=corpus_dir,
    )
    assert first is not None

    second = _admit_to_corpus(
        scenario=Scenario(traces=[_make_deployment_trace()]),
        artifacts=artifacts,
        edge_ids={7},
        cycle_time_s=0.5,
        generation=2,
        worker_id=0,
        gatekeeper=gatekeeper,
        disk_index=disk_index,
        corpus_dir=corpus_dir,
    )

    assert second is None
    assert len(disk_index) == 1


def test_admit_to_corpus_sanitizes_compile_failing_deployments(tmp_path):
    corpus_dir = tmp_path / "corpus"
    gatekeeper = Gatekeeper(GlobalEdgeTracker(64))
    disk_index = DiskIndex(corpus_dir)

    bad = _make_deployment_trace(
        deployed_address="0x000000000000000000000000000000000000000A"
    )
    good = _make_deployment_trace(
        deployed_address="0x000000000000000000000000000000000000000B"
    )
    scenario = Scenario(traces=[bad, good])
    ivy_result = ScenarioResult(
        results=[
            TraceResult(
                trace_type="deployment",
                trace_index=0,
                result=DeploymentResult(
                    success=False,
                    error=RuntimeError("compile failure"),
                    error_phase="compile",
                ),
            ),
            TraceResult(
                trace_type="deployment",
                trace_index=1,
                result=DeploymentResult(success=True),
            ),
        ]
    )

    decision = _admit_to_corpus(
        scenario=scenario,
        artifacts=_make_artifacts(AnalysisResult(), ivy_result=ivy_result),
        edge_ids={8, 9},
        cycle_time_s=0.1,
        generation=1,
        worker_id=0,
        gatekeeper=gatekeeper,
        disk_index=disk_index,
        corpus_dir=corpus_dir,
    )

    assert decision is not None
    assert len(disk_index) == 1
    written = read_scenario(corpus_dir, disk_index.entries[0].entry_id)
    written_deployments = [
        trace for trace in written.traces if isinstance(trace, DeploymentTrace)
    ]
    assert len(written_deployments) == 1
    assert written_deployments[0].deployed_address == good.deployed_address


def test_admit_to_corpus_rejects_when_sanitization_removes_all_deployments(tmp_path):
    corpus_dir = tmp_path / "corpus"
    gatekeeper = Gatekeeper(GlobalEdgeTracker(64))
    disk_index = DiskIndex(corpus_dir)
    scenario = Scenario(traces=[_make_deployment_trace()])
    ivy_result = ScenarioResult(
        results=[
            TraceResult(
                trace_type="deployment",
                trace_index=0,
                result=DeploymentResult(
                    success=False,
                    error=RuntimeError("compile failure"),
                    error_phase="compile",
                ),
            )
        ]
    )

    decision = _admit_to_corpus(
        scenario=scenario,
        artifacts=_make_artifacts(AnalysisResult(), ivy_result=ivy_result),
        edge_ids={10},
        cycle_time_s=0.1,
        generation=1,
        worker_id=0,
        gatekeeper=gatekeeper,
        disk_index=disk_index,
        corpus_dir=corpus_dir,
    )

    assert decision is None
    assert len(disk_index) == 0


def test_has_unique_issue_ignores_compile_failure_only():
    analysis = AnalysisResult()
    analysis.compile_failures.append(
        (
            DeploymentResult(success=False),
            KeepDecision(
                keep=True,
                reason="new_compile_failure",
                fingerprint="compile",
            ),
        )
    )

    assert _has_unique_issue(analysis) is False


def test_update_stagnation_counters_does_not_reset_for_better_representative():
    decision = GatekeeperDecision(
        accept=True,
        reason="better_representative",
        coverage_fingerprint="fp",
        new_edges=0,
        rare_edge_score=0.0,
        selection_weight=0.0,
    )

    cov_iters, issue_iters = _update_stagnation_counters(
        decision=decision,
        analysis=AnalysisResult(),
        iters_since_new_compiler_cov=9,
        iters_since_unique_issue=4,
    )

    assert cov_iters == 10
    assert issue_iters == 5


def test_update_stagnation_counters_resets_on_new_edge_and_unique_issue():
    analysis = AnalysisResult()
    analysis.crashes.append(
        (
            DeploymentResult(
                success=False, error=RuntimeError("boom"), error_phase="compile"
            ),
            KeepDecision(
                keep=True,
                reason="new_crash",
                fingerprint="crash",
            ),
        )
    )
    decision = GatekeeperDecision(
        accept=True,
        reason="new_edge",
        coverage_fingerprint="fp",
        new_edges=1,
        rare_edge_score=0.0,
        selection_weight=0.0,
    )

    cov_iters, issue_iters = _update_stagnation_counters(
        decision=decision,
        analysis=analysis,
        iters_since_new_compiler_cov=9,
        iters_since_unique_issue=4,
    )

    assert cov_iters == 0
    assert issue_iters == 0

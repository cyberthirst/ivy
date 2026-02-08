from __future__ import annotations

import json
from types import SimpleNamespace

from fuzzer.reporter import FuzzerReporter, UNPARSABLE_INTEGRITY_SUM
from fuzzer.runner.scenario import Scenario
from fuzzer.trace_types import DeploymentTrace, Env, Tx
from fuzzer.runtime_engine.runtime_fuzz_engine import HarnessStats


def _make_deployment_trace(solc_json):
    return DeploymentTrace(
        deployment_type="source",
        calldata=None,
        value=0,
        solc_json=solc_json,
        blueprint_initcode_prefix=None,
        deployed_address="0x123",
        deployment_succeeded=False,
        env=Env(tx=Tx(origin="0xabc")),
    )


def _make_artifacts(
    *,
    harness_stats: HarnessStats,
    runtime_edge_ids: set[int],
    runtime_stmt_sites_seen: set[tuple[str, int]],
    runtime_branch_outcomes_seen: set[tuple[str, int, bool]],
    runtime_stmt_sites_total: set[tuple[str, int]],
    runtime_branch_outcomes_total: set[tuple[str, int, bool]],
    finalized_scenario: Scenario,
):
    return SimpleNamespace(
        harness_stats=harness_stats,
        runtime_edge_ids=runtime_edge_ids,
        runtime_stmt_sites_seen=runtime_stmt_sites_seen,
        runtime_branch_outcomes_seen=runtime_branch_outcomes_seen,
        runtime_stmt_sites_total=runtime_stmt_sites_total,
        runtime_branch_outcomes_total=runtime_branch_outcomes_total,
        finalized_scenario=finalized_scenario,
    )


def _empty_analysis():
    return SimpleNamespace(
        successful_deployments=0,
        failed_deployments=0,
        successful_calls=0,
        failed_calls=0,
        crashes=[],
        compile_failures=[],
        divergences=[],
    )


def test_reporter_writes_interval_metrics_jsonl(tmp_path, monkeypatch):
    reporter = FuzzerReporter(seed=7, reports_dir=tmp_path)
    reporter.start_timer()
    assert reporter.start_time is not None
    reporter.start_time -= 5
    reporter.start_metrics_stream(True)

    monkeypatch.setattr(
        "fuzzer.reporter.loads_from_solc_json",
        lambda *_args, **_kwargs: SimpleNamespace(integrity_sum="abc"),
    )

    artifacts = _make_artifacts(
        harness_stats=HarnessStats(
            enumeration_calls=2,
            replay_calls=3,
            fuzz_calls=4,
            timeouts=1,
            new_coverage_calls=2,
            state_modified_calls=1,
            interesting_calls=2,
            skipped_replay=1,
            deployment_attempts=3,
            deployment_successes=2,
            deployment_failures=1,
            call_attempts=9,
            call_successes=6,
            call_failures=3,
        ),
        runtime_edge_ids={11, 22, 33},
        runtime_stmt_sites_seen={("0x1", 1), ("0x1", 2)},
        runtime_branch_outcomes_seen={("0x1", 10, True)},
        runtime_stmt_sites_total={("0x1", 1), ("0x1", 2), ("0x1", 3), ("0x1", 4)},
        runtime_branch_outcomes_total={("0x1", 10, True), ("0x1", 10, False)},
        finalized_scenario=Scenario(
            traces=[_make_deployment_trace(solc_json={"sources": {}})],
            dependencies=[],
            use_python_args=True,
        ),
    )
    reporter.ingest_run(_empty_analysis(), artifacts, debug_mode=False)

    reporter.total_scenarios = 3
    reporter.successful_deployments = 5
    reporter.deployment_failures = 1
    reporter.compilation_failures = 2
    reporter.compiler_crashes = 1
    reporter.divergences = 4

    snapshot = reporter.record_interval_metrics(
        iteration=10,
        corpus_seed_count=20,
        corpus_evolved_count=15,
        corpus_max_evolved=40,
        include_coverage_percentages=True,
    )

    assert snapshot is not None
    assert snapshot["deployments"]["success_rate_pct"] == (2 / 3) * 100
    assert snapshot["calls"]["success_rate_pct"] == (6 / 9) * 100
    assert snapshot["coverage"]["runtime_edges_total"] == 3
    assert snapshot["coverage"]["new_runtime_edges_interval"] == 3
    assert snapshot["coverage"]["stmt_coverage_pct"] == 50.0
    assert snapshot["coverage"]["branch_coverage_pct"] == 50.0
    assert snapshot["novelty"]["new_contracts_interval"] == 1

    assert reporter._metrics_path is not None
    lines = reporter._metrics_path.read_text().strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["iteration"] == 10
    assert payload["corpus"]["seed_count"] == 20


def test_reporter_interval_delta_and_empty_coverage_denominator(tmp_path, monkeypatch):
    reporter = FuzzerReporter(seed=11, reports_dir=tmp_path)
    reporter.start_timer()
    assert reporter.start_time is not None
    reporter.start_time -= 4
    reporter.start_metrics_stream(True)

    monkeypatch.setattr(
        "fuzzer.reporter.loads_from_solc_json",
        lambda *_args, **_kwargs: SimpleNamespace(integrity_sum="contract-a"),
    )

    reporter.ingest_run(
        _empty_analysis(),
        _make_artifacts(
            harness_stats=HarnessStats(
                deployment_attempts=1,
                deployment_successes=1,
                call_attempts=1,
                call_successes=1,
            ),
            runtime_edge_ids={1, 2},
            runtime_stmt_sites_seen=set(),
            runtime_branch_outcomes_seen=set(),
            runtime_stmt_sites_total=set(),
            runtime_branch_outcomes_total=set(),
            finalized_scenario=Scenario(
                traces=[_make_deployment_trace(solc_json={"sources": {}})],
                dependencies=[],
                use_python_args=True,
            ),
        ),
        debug_mode=False,
    )
    first = reporter.record_interval_metrics(
        iteration=1,
        corpus_seed_count=1,
        corpus_evolved_count=0,
        corpus_max_evolved=2,
        include_coverage_percentages=True,
    )
    assert first is not None
    assert first["coverage"]["stmt_coverage_pct"] is None
    assert first["coverage"]["branch_coverage_pct"] is None
    assert first["coverage"]["new_runtime_edges_interval"] == 2

    reporter.ingest_run(
        _empty_analysis(),
        _make_artifacts(
            harness_stats=HarnessStats(
                deployment_attempts=1,
                deployment_failures=1,
                call_attempts=1,
                call_failures=1,
            ),
            runtime_edge_ids={2, 3},
            runtime_stmt_sites_seen=set(),
            runtime_branch_outcomes_seen=set(),
            runtime_stmt_sites_total=set(),
            runtime_branch_outcomes_total=set(),
            finalized_scenario=Scenario(
                traces=[_make_deployment_trace(solc_json={"sources": {}})],
                dependencies=[],
                use_python_args=True,
            ),
        ),
        debug_mode=False,
    )

    second = reporter.record_interval_metrics(
        iteration=2,
        corpus_seed_count=1,
        corpus_evolved_count=1,
        corpus_max_evolved=2,
        include_coverage_percentages=True,
    )
    assert second is not None
    assert second["coverage"]["runtime_edges_total"] == 3
    assert second["coverage"]["new_runtime_edges_interval"] == 1
    assert second["novelty"]["new_contracts_interval"] == 0


def test_reporter_uses_unparsable_integrity_sum_sentinel(tmp_path, monkeypatch):
    reporter = FuzzerReporter(seed=19, reports_dir=tmp_path)
    reporter.start_timer()
    assert reporter.start_time is not None
    reporter.start_time -= 1
    reporter.start_metrics_stream(True)

    def _raise(*_args, **_kwargs):
        raise ValueError("bad source")

    monkeypatch.setattr("fuzzer.reporter.loads_from_solc_json", _raise)

    reporter.ingest_run(
        _empty_analysis(),
        _make_artifacts(
            harness_stats=HarnessStats(),
            runtime_edge_ids=set(),
            runtime_stmt_sites_seen=set(),
            runtime_branch_outcomes_seen=set(),
            runtime_stmt_sites_total=set(),
            runtime_branch_outcomes_total=set(),
            finalized_scenario=Scenario(
                traces=[_make_deployment_trace(solc_json={"sources": {"x.vy": {}}})],
                dependencies=[],
                use_python_args=True,
            ),
        ),
        debug_mode=False,
    )
    snapshot = reporter.record_interval_metrics(
        iteration=1,
        corpus_seed_count=1,
        corpus_evolved_count=0,
        corpus_max_evolved=1,
        include_coverage_percentages=False,
    )

    assert snapshot is not None
    assert snapshot["novelty"]["contracts_seen_total"] == 1
    assert snapshot["novelty"]["new_contracts_interval"] == 1
    assert reporter._seen_contract_fingerprints == {UNPARSABLE_INTEGRITY_SUM}

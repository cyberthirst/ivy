from __future__ import annotations

import json
from types import SimpleNamespace

from fuzzer.coverage_types import RuntimeBranchOutcome, RuntimeStmtSite
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
    runtime_stmt_sites_seen: set[RuntimeStmtSite],
    runtime_branch_outcomes_seen: set[RuntimeBranchOutcome],
    runtime_stmt_sites_total: set[RuntimeStmtSite],
    runtime_branch_outcomes_total: set[RuntimeBranchOutcome],
    finalized_scenario: Scenario,
    contract_fingerprints: set[str] | None = None,
):
    return SimpleNamespace(
        harness_stats=harness_stats,
        runtime_edge_ids=runtime_edge_ids,
        runtime_stmt_sites_seen=runtime_stmt_sites_seen,
        runtime_branch_outcomes_seen=runtime_branch_outcomes_seen,
        runtime_stmt_sites_total=runtime_stmt_sites_total,
        runtime_branch_outcomes_total=runtime_branch_outcomes_total,
        contract_fingerprints=contract_fingerprints or set(),
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


def test_reporter_writes_interval_metrics_jsonl(tmp_path):
    reporter = FuzzerReporter(seed=7, reports_dir=tmp_path)
    reporter.start_timer()
    assert reporter.start_time is not None
    reporter.start_time -= 5
    reporter.start_metrics_stream(True)

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
            calls_to_no_code=2,
        ),
        runtime_edge_ids={11, 22, 33},
        runtime_stmt_sites_seen={("0x1", 0, 1), ("0x1", 0, 2)},
        runtime_branch_outcomes_seen={("0x1", 0, 10, True)},
        runtime_stmt_sites_total={
            ("0x1", 0, 1),
            ("0x1", 0, 2),
            ("0x1", 0, 3),
            ("0x1", 0, 4),
        },
        runtime_branch_outcomes_total={("0x1", 0, 10, True), ("0x1", 0, 10, False)},
        finalized_scenario=Scenario(
            traces=[_make_deployment_trace(solc_json={"sources": {}})],
            dependencies=[],
            use_python_args=True,
        ),
        contract_fingerprints={"abc"},
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
    )

    assert snapshot is not None
    assert snapshot["deployments"]["success_rate_pct"] == (2 / 3) * 100
    assert snapshot["calls"]["success_rate_pct"] == (6 / 9) * 100
    assert snapshot["calls"]["calls_to_no_code_total"] == 2
    assert snapshot["calls"]["calls_to_no_code_interval"] == 2
    assert snapshot["calls"]["calls_to_no_code_pct"] == (2 / 9) * 100
    assert snapshot["coverage"]["runtime_edges_avg_per_scenario"] == 3
    assert snapshot["coverage"]["runtime_edges_avg_per_scenario_interval"] == 3
    assert snapshot["coverage"]["stmt_coverage_pct"] == 50.0
    assert snapshot["coverage"]["branch_coverage_pct"] == 50.0
    assert snapshot["novelty"]["contracts_per_scenario_interval_avg"] == 1

    assert reporter._metrics_path is not None
    lines = reporter._metrics_path.read_text().strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["iteration"] == 10
    assert payload["corpus"]["seed_count"] == 20
    assert reporter._pending_runtime_edges_sum == 0
    assert reporter._pending_runtime_edges_samples == 0
    assert reporter._pending_contracts_observed_sum == 0
    assert reporter._pending_contracts_observed_samples == 0
    assert reporter._pending_stmt_coverage_pct_sum == 0.0
    assert reporter._pending_stmt_coverage_pct_samples == 0
    assert reporter._pending_branch_coverage_pct_sum == 0.0
    assert reporter._pending_branch_coverage_pct_samples == 0


def test_reporter_interval_delta_and_empty_coverage_denominator(tmp_path):
    reporter = FuzzerReporter(seed=11, reports_dir=tmp_path)
    reporter.start_timer()
    assert reporter.start_time is not None
    reporter.start_time -= 4
    reporter.start_metrics_stream(True)

    reporter.ingest_run(
        _empty_analysis(),
        _make_artifacts(
            harness_stats=HarnessStats(
                deployment_attempts=1,
                deployment_successes=1,
                call_attempts=1,
                call_successes=1,
                calls_to_no_code=1,
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
            contract_fingerprints={"contract-a"},
        ),
        debug_mode=False,
    )
    first = reporter.record_interval_metrics(
        iteration=1,
        corpus_seed_count=1,
        corpus_evolved_count=0,
        corpus_max_evolved=2,
    )
    assert first is not None
    assert first["coverage"]["stmt_coverage_pct"] is None
    assert first["coverage"]["branch_coverage_pct"] is None
    assert first["coverage"]["runtime_edges_avg_per_scenario_interval"] == 0
    assert first["calls"]["calls_to_no_code_total"] == 1
    assert first["calls"]["calls_to_no_code_interval"] == 1

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
            contract_fingerprints={"contract-a"},
        ),
        debug_mode=False,
    )

    second = reporter.record_interval_metrics(
        iteration=2,
        corpus_seed_count=1,
        corpus_evolved_count=1,
        corpus_max_evolved=2,
    )
    assert second is not None
    assert second["coverage"]["runtime_edges_avg_per_scenario"] == 0
    assert second["coverage"]["runtime_edges_avg_per_scenario_interval"] == 0
    assert second["novelty"]["contracts_per_scenario_interval_avg"] == 1
    assert second["calls"]["calls_to_no_code_total"] == 1
    assert second["calls"]["calls_to_no_code_interval"] == 0


def test_reporter_uses_unparsable_integrity_sum_sentinel(tmp_path):
    reporter = FuzzerReporter(seed=19, reports_dir=tmp_path)
    reporter.start_timer()
    assert reporter.start_time is not None
    reporter.start_time -= 1
    reporter.start_metrics_stream(True)

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
            contract_fingerprints={UNPARSABLE_INTEGRITY_SUM},
        ),
        debug_mode=False,
    )
    snapshot = reporter.record_interval_metrics(
        iteration=1,
        corpus_seed_count=1,
        corpus_evolved_count=0,
        corpus_max_evolved=1,
    )

    assert snapshot is not None
    assert snapshot["novelty"]["contracts_per_scenario_avg"] == 1
    assert snapshot["novelty"]["contracts_per_scenario_interval_avg"] == 1
    assert reporter._contracts_observed_sum == 1
    assert reporter._contracts_observed_samples == 1


def test_reporter_start_metrics_stream_reuse_resets_interval_baselines(tmp_path):
    reporter = FuzzerReporter(seed=31, reports_dir=tmp_path)
    reporter.start_timer()
    reporter.start_metrics_stream(True)

    # Simulate previously accumulated campaign totals on a reused reporter.
    reporter.total_scenarios = 10
    reporter.successful_deployments = 7
    reporter.deployment_failures = 2
    reporter.compilation_failures = 3
    reporter.compiler_crashes = 1

    reporter.start_metrics_stream(True)

    # New campaign increments should be measured against the new baseline only.
    reporter.total_scenarios += 1
    reporter.successful_deployments += 1
    reporter.compilation_failures += 1
    reporter.compiler_crashes += 1

    snapshot = reporter.record_interval_metrics(
        iteration=1,
        corpus_seed_count=0,
        corpus_evolved_count=0,
        corpus_max_evolved=0,
    )

    assert snapshot is not None
    assert snapshot["throughput"]["scenarios_total"] == 11
    assert snapshot["compile"]["compilation_failures_interval"] == 1
    assert snapshot["compile"]["compiler_crashes_interval"] == 1
    assert snapshot["compile"]["compilation_failures_per_deployment"] == (1 / 3)
    assert snapshot["compile"]["compiler_crashes_per_deployment"] == (1 / 3)


def test_reporter_does_not_accumulate_pending_state_when_metrics_disabled(
    tmp_path,
):
    reporter = FuzzerReporter(seed=23, reports_dir=tmp_path)
    reporter.start_timer()
    reporter.start_metrics_stream(False)

    reporter.ingest_run(
        _empty_analysis(),
        _make_artifacts(
            harness_stats=HarnessStats(
                deployment_attempts=1,
                deployment_successes=1,
                call_attempts=2,
                call_successes=1,
                call_failures=1,
                calls_to_no_code=2,
            ),
            runtime_edge_ids={1, 2, 3},
            runtime_stmt_sites_seen={("0x1", 0, 10)},
            runtime_branch_outcomes_seen={("0x1", 0, 20, True)},
            runtime_stmt_sites_total={("0x1", 0, 10), ("0x1", 0, 11)},
            runtime_branch_outcomes_total={
                ("0x1", 0, 20, True),
                ("0x1", 0, 20, False),
            },
            finalized_scenario=Scenario(
                traces=[_make_deployment_trace(solc_json={"sources": {"x.vy": {}}})],
                dependencies=[],
                use_python_args=True,
            ),
        ),
        debug_mode=False,
    )

    assert reporter._metrics_path is None
    assert reporter.get_runtime_deployment_success_rate() == 100.0
    assert reporter.get_runtime_call_success_rate() == 50.0
    assert reporter._pending_runtime_edges_sum == 0
    assert reporter._pending_runtime_edges_samples == 0
    assert reporter._pending_contracts_observed_sum == 0
    assert reporter._pending_contracts_observed_samples == 0
    assert reporter._pending_stmt_coverage_pct_sum == 0.0
    assert reporter._pending_stmt_coverage_pct_samples == 0
    assert reporter._pending_branch_coverage_pct_sum == 0.0
    assert reporter._pending_branch_coverage_pct_samples == 0
    assert reporter._runtime_edges_sum == 0
    assert reporter._contracts_observed_sum == 0
    assert reporter._runtime_totals.calls_to_no_code == 2

    snapshot = reporter.record_interval_metrics(
        iteration=1,
        corpus_seed_count=1,
        corpus_evolved_count=1,
        corpus_max_evolved=1,
    )
    assert snapshot is None
    assert reporter._pending_runtime_edges_sum == 0
    assert reporter._pending_contracts_observed_sum == 0


def test_reporter_spools_contract_fingerprints_periodically(tmp_path):
    reporter = FuzzerReporter(seed=37, reports_dir=tmp_path)
    reporter.start_timer()
    reporter.start_metrics_stream(True)
    reporter._contract_fingerprint_flush_max_buffer_bytes = 1

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
                traces=[_make_deployment_trace(solc_json={"sources": {}})],
                dependencies=[],
                use_python_args=True,
            ),
            contract_fingerprints={"a", "b"},
        ),
        debug_mode=False,
    )
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
                traces=[_make_deployment_trace(solc_json={"sources": {}})],
                dependencies=[],
                use_python_args=True,
            ),
            contract_fingerprints={"a"},
        ),
        debug_mode=False,
    )

    assert reporter._contract_fingerprints_path is not None
    lines = reporter._contract_fingerprints_path.read_text().splitlines()
    assert len(lines) == 3
    assert lines.count("a") == 2
    assert lines.count("b") == 1


def test_reporter_contract_fingerprint_buffer_survives_metric_snapshot(tmp_path):
    reporter = FuzzerReporter(seed=41, reports_dir=tmp_path)
    reporter.start_timer()
    reporter.start_metrics_stream(True)
    reporter._contract_fingerprint_flush_max_buffer_bytes = 20

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
                traces=[_make_deployment_trace(solc_json={"sources": {}})],
                dependencies=[],
                use_python_args=True,
            ),
            contract_fingerprints={"persist-a"},
        ),
        debug_mode=False,
    )
    _ = reporter.record_interval_metrics(
        iteration=1,
        corpus_seed_count=1,
        corpus_evolved_count=0,
        corpus_max_evolved=1,
    )

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
                traces=[_make_deployment_trace(solc_json={"sources": {}})],
                dependencies=[],
                use_python_args=True,
            ),
            contract_fingerprints={"persist-b"},
        ),
        debug_mode=False,
    )

    assert reporter._contract_fingerprints_path is not None
    lines = reporter._contract_fingerprints_path.read_text().splitlines()
    assert len(lines) == 2
    assert set(lines) == {"persist-a", "persist-b"}


def test_reporter_flushes_contract_fingerprints_on_save_statistics(tmp_path):
    reporter = FuzzerReporter(seed=43, reports_dir=tmp_path)
    reporter.start_timer()
    reporter.start_metrics_stream(True)
    reporter._contract_fingerprint_flush_max_buffer_bytes = 10_000

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
                traces=[_make_deployment_trace(solc_json={"sources": {}})],
                dependencies=[],
                use_python_args=True,
            ),
            contract_fingerprints={"final-only"},
        ),
        debug_mode=False,
    )

    assert reporter._contract_fingerprints_path is not None
    assert not reporter._contract_fingerprints_path.exists()

    reporter.save_statistics()

    assert reporter._contract_fingerprints_path.exists()
    assert reporter._contract_fingerprints_path.read_text().splitlines() == [
        "final-only"
    ]
    assert reporter._pending_unique_contract_fingerprints == set()

def test_reporter_uses_unknown_error_type_when_error_missing(tmp_path):
    reporter = FuzzerReporter(seed=29, reports_dir=tmp_path)
    reporter.save_compiler_crash(solc_json={"sources": {}}, error=None)
    reporter.save_compilation_failure(solc_json={"sources": {}}, error=None)

    crash_files = list(tmp_path.rglob("crash_UnknownError_*.json"))
    failure_files = list(tmp_path.rglob("failure_UnknownError_*.json"))
    assert len(crash_files) == 1
    assert len(failure_files) == 1

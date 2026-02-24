from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from fuzzer.reporter import FuzzerReporter
from fuzzer.runner.scenario import Scenario
from fuzzer.runtime_engine.runtime_fuzz_engine import (
    HarnessConfig,
    HarnessStats,
    RuntimeFuzzEngine,
)


def _empty_analysis() -> Any:
    return SimpleNamespace(
        successful_deployments=0,
        failed_deployments=0,
        successful_calls=0,
        failed_calls=0,
        crashes=[],
        compile_failures=[],
        compilation_timeouts=[],
        divergences=[],
    )


def _runtime_stmt_sites_seen(env) -> set[tuple[str, int, int]]:
    return {
        (str(addr), int(source_id), int(node_id))
        for addr, source_node_ids in env.execution_metadata.coverage.items()
        for source_id, node_id in source_node_ids
    }


def _runtime_branch_outcomes_seen(env) -> set[tuple[str, int, int, bool]]:
    return {
        (str(addr), int(source_id), int(node_id), taken)
        for addr, source_id, node_id, taken in env.execution_metadata.branches
    }


@pytest.mark.parametrize(
    ("call_sequence", "expected_stmt_coverage_pct"),
    [
        (("hit_left", "hit_right"), 100.0),
        (("hit_left",), 50.0),
    ],
)
def test_reporter_runtime_stmt_coverage_for_multimodule_contract(
    env,
    get_contract,
    make_input_bundle,
    tmp_path,
    call_sequence,
    expected_stmt_coverage_pct,
):
    main = """
import lib1

initializes: lib1

@external
def hit_left():
    lib1.left()

@external
def hit_right():
    lib1.right()
    """
    lib1 = """
a: uint256
b: uint256

def left():
    self.a = 1

def right():
    self.b = 2
    """

    contract = get_contract(
        main,
        input_bundle=make_input_bundle({"lib1.vy": lib1}),
    )

    coverage_engine = RuntimeFuzzEngine(
        HarnessConfig(
            max_total_calls=0,
            max_enumeration_calls=0,
            max_replay_calls=0,
            max_fuzz_calls=0,
        ),
        seed=1,
    )
    runtime_stmt_sites_total, runtime_branch_outcomes_total = (
        coverage_engine._collect_static_coverage_sites_for_contract(contract)
    )

    env.reset_execution_metadata()
    for function_name in call_sequence:
        getattr(contract, function_name)()

    reporter = FuzzerReporter(seed=7, reports_dir=tmp_path)
    reporter.start_timer()
    reporter.start_metrics_stream(True)

    artifacts = SimpleNamespace(
        harness_stats=HarnessStats(
            call_attempts=len(call_sequence),
            call_successes=len(call_sequence),
            call_failures=0,
        ),
        runtime_edge_ids=set(),
        runtime_stmt_sites_seen=_runtime_stmt_sites_seen(env),
        runtime_branch_outcomes_seen=_runtime_branch_outcomes_seen(env),
        runtime_stmt_sites_total=runtime_stmt_sites_total,
        runtime_branch_outcomes_total=runtime_branch_outcomes_total,
        finalized_scenario=Scenario(
            traces=[],
            dependencies=[],
            use_python_args=True,
        ),
    )

    reporter.ingest_run(_empty_analysis(), artifacts, debug_mode=False)

    snapshot = reporter.record_interval_metrics(
        iteration=1,
        corpus_seed_count=1,
        corpus_evolved_count=0,
        corpus_max_evolved=1,
    )

    assert snapshot is not None
    assert snapshot["coverage"]["stmt_coverage_pct"] == expected_stmt_coverage_pct
    assert snapshot["coverage"]["branch_coverage_pct"] is None

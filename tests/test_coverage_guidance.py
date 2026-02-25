from __future__ import annotations

import time

import pytest

from fuzzer.coverage.collector import ArcCoverageCollector
from fuzzer.coverage.edge_map import EdgeMap
from fuzzer.coverage.gatekeeper import Gatekeeper, all_boa_configs_failed_to_compile
from fuzzer.coverage.tracker import GlobalEdgeTracker
from fuzzer.result_analyzer import AnalysisResult
from fuzzer.runner.base_scenario_runner import (
    DeploymentResult,
    ScenarioResult,
    TraceResult,
)
from vyper.exceptions import ParserException


def test_arc_coverage_collector_records_arcs_and_time():
    collector = ArcCoverageCollector(
        guidance_targets=("tests/test_coverage_guidance.py",)
    )
    collector.start_scenario()

    def f(x: bool) -> int:
        if x:
            y = 1
        else:
            y = 2
        return y

    with collector.collect_compile(config_name="c1"):
        _ = f(True)

    assert collector.compile_time_s >= 0
    assert collector.compile_time_by_config_s["c1"] >= 0
    assert collector.get_scenario_arcs()
    assert collector.get_arcs_by_config()["c1"]
    line_analysis = collector.get_scenario_line_analysis()
    assert line_analysis
    for executable, missing in line_analysis.values():
        assert missing <= executable


def test_arc_coverage_collector_drops_arcs_on_exception():
    collector = ArcCoverageCollector(
        guidance_targets=("tests/test_coverage_guidance.py",)
    )
    collector.start_scenario()

    t0 = time.perf_counter()
    with pytest.raises(RuntimeError):
        with collector.collect_compile(config_name="c1"):
            raise RuntimeError("boom")
    assert time.perf_counter() - t0 >= 0

    assert collector.compile_time_s >= 0
    assert collector.get_scenario_arcs() == set()


def test_edge_map_hashing_is_stable_and_bounded():
    edge_map = EdgeMap(1 << 8)
    arc = ("/tmp/vyper/ir/foo.py", 10, 20)
    a = edge_map.hash_arc(arc)
    b = edge_map.hash_arc(arc)
    assert a == b
    assert 0 <= a < edge_map.map_size


def test_global_edge_tracker_saturates():
    tracker = GlobalEdgeTracker(8, max_count=2)
    assert tracker.merge({1}) == 1
    assert tracker.merge({1}) == 0
    assert tracker.merge({1}) == 0
    assert tracker.counts[1] == 2


def test_gatekeeper_accepts_new_edges_and_updates_global_counts():
    tracker = GlobalEdgeTracker(32)
    gatekeeper = Gatekeeper(tracker)
    analysis = AnalysisResult()

    boa_results: dict[str, tuple[object, ScenarioResult]] = {
        "cfg": (
            object(),
            ScenarioResult(
                results=[
                    TraceResult(
                        trace_type="deployment",
                        trace_index=0,
                        result=DeploymentResult(success=True),
                    )
                ]
            ),
        )
    }

    ivy_result = ScenarioResult(
        results=[
            TraceResult(
                trace_type="deployment",
                trace_index=0,
                result=DeploymentResult(success=True),
            )
        ]
    )

    decision = gatekeeper.decide_and_update(
        edge_ids={1, 2, 3},
        compile_time_s=0.01,
        analysis=analysis,
        ivy_result=ivy_result,
        boa_results=boa_results,
        improves_representative=False,
    )

    assert decision.accept is True
    assert decision.reason == "new_edge"
    assert tracker.counts[1] == 1


def test_gatekeeper_rejects_when_all_boa_configs_fail_compile():
    tracker = GlobalEdgeTracker(32)
    gatekeeper = Gatekeeper(tracker)

    boa_results: dict[str, tuple[object, ScenarioResult]] = {
        "cfg": (
            object(),
            ScenarioResult(
                results=[
                    TraceResult(
                        trace_type="deployment",
                        trace_index=0,
                        result=DeploymentResult(
                            success=False,
                            error=ParserException("nope"),
                            error_phase="compile",
                        ),
                    )
                ]
            ),
        )
    }
    assert all_boa_configs_failed_to_compile(boa_results) is True

    ivy_result = ScenarioResult(
        results=[
            TraceResult(
                trace_type="deployment",
                trace_index=0,
                result=DeploymentResult(success=True),
            )
        ]
    )

    decision = gatekeeper.decide_and_update(
        edge_ids=set(),
        compile_time_s=0.01,
        analysis=AnalysisResult(),
        ivy_result=ivy_result,
        boa_results=boa_results,
        improves_representative=True,
    )
    assert decision.accept is False
    assert decision.reason == "all_configs_failed_compile"

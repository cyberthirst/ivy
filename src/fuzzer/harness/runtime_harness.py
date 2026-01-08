from __future__ import annotations

import hashlib
import random
from array import array
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import ivy
from ivy.execution_metadata import ExecutionMetadata

from ..runner.base_scenario_runner import CallResult, ScenarioResult, TraceResult
from ..runner.scenario import Scenario
from ..trace_types import (
    CallTrace,
    ClearTransientStorageTrace,
    DeploymentTrace,
    SetBalanceTrace,
)

from .call_generator import CallGenerator
from .timeout import CallTimeout, call_with_timeout
from ..runner.ivy_scenario_runner import IvyScenarioRunner


@dataclass
class HarnessConfig:
    # Harness behavior
    max_fuzz_calls: int = 100
    plateau_calls: int = 20
    call_timeout_s: float = 5.0
    map_size: int = 1 << 16
    # Runner behavior
    collect_storage_dumps: bool = True
    no_solc_json: bool = True


@dataclass
class HarnessStats:
    enumeration_calls: int = 0
    replay_calls: int = 0
    fuzz_calls: int = 0
    timeouts: int = 0
    new_coverage_calls: int = 0


@dataclass
class HarnessResult:
    finalized_scenario: Scenario
    ivy_result: ScenarioResult
    stats: HarnessStats
    runtime_edge_ids: Set[int]

    @property
    def finalized_traces(self) -> List[Any]:
        return self.finalized_scenario.traces


class RuntimeEdgeMap:
    def __init__(self, map_size: int):
        self.map_size = map_size
        self._mask = map_size - 1

    def hash_tuple(self, tup: tuple) -> int:
        key = repr(tup).encode("utf-8")
        digest = hashlib.blake2b(key, digest_size=8).digest()
        return int.from_bytes(digest, "little") & self._mask

    def hash_metadata(self, metadata: ExecutionMetadata) -> Set[int]:
        edge_ids: Set[int] = set()
        for edge in metadata.edges:
            edge_ids.add(self.hash_tuple(edge))
        for branch in metadata.branches:
            edge_ids.add(self.hash_tuple(branch))
        for boolop in metadata.boolops:
            edge_ids.add(self.hash_tuple(boolop))
        for loop in metadata.loops:
            edge_ids.add(self.hash_tuple(loop))
        return edge_ids


class RuntimeCoverageTracker:
    def __init__(self, map_size: int):
        self.map_size = map_size
        self.counts: array[int] = array("H", [0]) * map_size
        self._seen: Set[int] = set()

    def count_new_edges(self, edge_ids: Set[int]) -> int:
        return sum(1 for eid in edge_ids if self.counts[eid] == 0)

    def merge(self, edge_ids: Set[int]) -> int:
        new_count = 0
        for eid in edge_ids:
            if self.counts[eid] == 0:
                new_count += 1
            if self.counts[eid] < 0xFFFF:
                self.counts[eid] += 1
        self._seen.update(edge_ids)
        return new_count


class RuntimeHarness:
    def __init__(
        self,
        config: HarnessConfig | None = None,
        seed: int | None = None,
    ):
        self.config = config or HarnessConfig()
        self.rng = random.Random(seed)
        self.runner = IvyScenarioRunner(
            collect_storage_dumps=self.config.collect_storage_dumps,
            no_solc_json=self.config.no_solc_json,
        )
        self.edge_map = RuntimeEdgeMap(self.config.map_size)
        self.call_generator = CallGenerator(self.rng)

    def run(self, scenario: Scenario) -> HarnessResult:
        with self.runner.env.anchor():
            self.runner.deployed_contracts = {}
            self.runner.executed_dependencies = set()

            stats = HarnessStats()
            tracker = RuntimeCoverageTracker(self.config.map_size)
            finalized_traces: List[Any] = []
            trace_results: List[TraceResult] = []

            deployed_contracts: Dict[str, Any] = {}
            trace_index = 0

            for dep_scenario in scenario.dependencies:
                self.runner._execute_dependency_scenario(dep_scenario)

            for trace in scenario.traces:
                if isinstance(trace, DeploymentTrace):
                    finalized_traces.append(trace)
                    trace_result = self.runner.execute_trace(
                        trace, trace_index, scenario.use_python_args
                    )
                    trace_results.append(trace_result)
                    if trace_result.result and trace_result.result.success:
                        if (
                            hasattr(trace_result.result, "contract")
                            and trace_result.result.contract
                        ):
                            deployed_contracts[trace.deployed_address] = (
                                trace_result.result.contract
                            )
                    trace_index += 1

                elif isinstance(trace, (SetBalanceTrace, ClearTransientStorageTrace)):
                    finalized_traces.append(trace)
                    trace_result = self.runner.execute_trace(
                        trace, trace_index, scenario.use_python_args
                    )
                    trace_results.append(trace_result)
                    trace_index += 1

            ivy.env.reset_execution_metadata()

            self._seed_enumerate_externals(
                deployed_contracts,
                self.runner,
                tracker,
                finalized_traces,
                trace_results,
                stats,
            )

            self._seed_replay_parent_traces(
                scenario,
                self.runner,
                tracker,
                finalized_traces,
                trace_results,
                stats,
            )

            self._fuzz_calls(
                deployed_contracts,
                self.runner,
                tracker,
                finalized_traces,
                trace_results,
                stats,
            )

            ivy_result = ScenarioResult(results=trace_results)
            finalized_scenario = Scenario(
                traces=finalized_traces,
                dependencies=scenario.dependencies,
                use_python_args=True,
            )

            return HarnessResult(
                finalized_scenario=finalized_scenario,
                ivy_result=ivy_result,
                stats=stats,
                runtime_edge_ids=tracker._seen,
            )

    def _execute_trace_with_timeout(
        self,
        runner: IvyScenarioRunner,
        trace: CallTrace,
        trace_index: int,
        use_python_args: bool,
    ) -> Optional[TraceResult]:
        with call_with_timeout(self.config.call_timeout_s):
            trace_result = runner.execute_trace(trace, trace_index, use_python_args)

        if isinstance(trace_result.result, CallResult):
            if isinstance(trace_result.result.error, CallTimeout):
                return None

        return trace_result

    def _get_coverage_delta(self, tracker: RuntimeCoverageTracker) -> int:
        metadata = ivy.env.execution_metadata
        edge_ids = self.edge_map.hash_metadata(metadata)
        new_count = tracker.merge(edge_ids)
        return new_count

    def _seed_enumerate_externals(
        self,
        deployed_contracts: Dict[str, Any],
        runner: IvyScenarioRunner,
        tracker: RuntimeCoverageTracker,
        finalized_traces: List[Any],
        trace_results: List[TraceResult],
        stats: HarnessStats,
    ) -> None:
        for addr, contract in deployed_contracts.items():
            functions = self.call_generator.get_external_functions(contract)
            for fn_name, func_t in functions:
                generated = self.call_generator.generate_call_for_function(
                    addr, fn_name, func_t
                )
                trace = self.call_generator.call_trace_from_generated(generated, addr)

                trace_result = self._execute_trace_with_timeout(
                    runner, trace, len(finalized_traces), True
                )
                stats.enumeration_calls += 1

                if trace_result is not None:
                    finalized_traces.append(trace)
                    trace_results.append(trace_result)
                    new_cov = self._get_coverage_delta(tracker)
                    if new_cov > 0:
                        stats.new_coverage_calls += 1
                else:
                    stats.timeouts += 1

    def _seed_replay_parent_traces(
        self,
        scenario: Scenario,
        runner: IvyScenarioRunner,
        tracker: RuntimeCoverageTracker,
        finalized_traces: List[Any],
        trace_results: List[TraceResult],
        stats: HarnessStats,
    ) -> None:
        for trace in scenario.traces:
            if not isinstance(trace, CallTrace):
                continue

            trace_copy = deepcopy(trace)
            trace_result = self._execute_trace_with_timeout(
                runner, trace_copy, len(finalized_traces), scenario.use_python_args
            )
            stats.replay_calls += 1

            if trace_result is not None:
                finalized_traces.append(trace_copy)
                trace_results.append(trace_result)
                new_cov = self._get_coverage_delta(tracker)
                if new_cov > 0:
                    stats.new_coverage_calls += 1
            else:
                stats.timeouts += 1

    def _fuzz_calls(
        self,
        deployed_contracts: Dict[str, Any],
        runner: IvyScenarioRunner,
        tracker: RuntimeCoverageTracker,
        finalized_traces: List[Any],
        trace_results: List[TraceResult],
        stats: HarnessStats,
    ) -> None:
        if not deployed_contracts:
            return

        contract_list = list(deployed_contracts.items())
        all_functions: List[tuple] = []
        for addr, contract in contract_list:
            for fn_name, func_t in self.call_generator.get_external_functions(contract):
                all_functions.append((addr, fn_name, func_t))

        if not all_functions:
            return

        consecutive_no_coverage = 0
        for _ in range(self.config.max_fuzz_calls):
            if consecutive_no_coverage >= self.config.plateau_calls:
                break

            addr, fn_name, func_t = self.rng.choice(all_functions)
            generated = self.call_generator.generate_call_for_function(
                addr, fn_name, func_t
            )
            generated = self.call_generator.mutate_call(generated)
            trace = self.call_generator.call_trace_from_generated(generated, addr)

            trace_result = self._execute_trace_with_timeout(
                runner, trace, len(finalized_traces), True
            )
            stats.fuzz_calls += 1

            if trace_result is not None:
                finalized_traces.append(trace)
                trace_results.append(trace_result)
                new_cov = self._get_coverage_delta(tracker)
                if new_cov > 0:
                    stats.new_coverage_calls += 1
                    consecutive_no_coverage = 0
                else:
                    consecutive_no_coverage += 1
            else:
                stats.timeouts += 1
                consecutive_no_coverage += 1

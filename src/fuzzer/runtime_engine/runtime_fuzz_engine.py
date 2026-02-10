from __future__ import annotations

import hashlib
import random
from array import array
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import ivy
from ivy.execution_metadata import ExecutionMetadata
from ivy.exceptions import CallTimeout
from ivy.types import Address

from fuzzer.coverage_types import RuntimeBranchOutcome, RuntimeStmtSite
from fuzzer.runner.base_scenario_runner import (
    CallResult,
    DeploymentResult,
    ScenarioResult,
    TraceResult,
)
from fuzzer.runner.scenario import Scenario
from fuzzer.trace_types import (
    CallTrace,
    ClearTransientStorageTrace,
    DeploymentTrace,
    SetBalanceTrace,
)

from fuzzer.mutator.value_mutator import BOUNDARY_ADDRESSES
from fuzzer.runtime_engine.call_generator import (
    CallGenerator,
    CallKey,
    Corpus,
    GeneratedCall,
)
from fuzzer.runtime_engine.static_coverage import (
    collect_static_coverage_sites_for_contract,
)
from fuzzer.runner.ivy_scenario_runner import IvyScenarioRunner


@dataclass
class HarnessConfig:
    # Call budget limits
    max_total_calls: int = 1000
    max_enumeration_calls: int = 200
    max_replay_calls: int = 200
    max_fuzz_calls: int = 600

    # Plateau and timeout behavior
    plateau_calls: int = 8
    call_timeout_s: float = 0.5
    max_timeouts_per_func: int = 3

    # Coverage map
    map_size: int = 1 << 16

    # Runner behavior
    collect_storage_dumps: bool = False  # Disabled during exploration
    max_deploy_retries: int = 30

    # Corpus settings
    max_seeds_per_func: int = 16
    max_seeds_total: int = 512

    # Reporting toggles
    enable_interval_metrics: bool = True
    enable_coverage_percentages: bool = True


@dataclass
class HarnessStats:
    enumeration_calls: int = 0
    replay_calls: int = 0
    fuzz_calls: int = 0
    timeouts: int = 0
    new_coverage_calls: int = 0
    state_modified_calls: int = 0
    interesting_calls: int = 0
    skipped_replay: int = 0
    deployment_attempts: int = 0
    deployment_successes: int = 0
    deployment_failures: int = 0
    call_attempts: int = 0
    call_successes: int = 0
    call_failures: int = 0
    calls_to_no_code: int = 0

    def record_deployment_result(self, result: DeploymentResult) -> None:
        self.deployment_attempts += 1
        if result.success:
            self.deployment_successes += 1
        else:
            self.deployment_failures += 1

    def record_call_outcome(self, outcome: CallOutcome) -> None:
        self.call_attempts += 1
        if outcome.target_no_code:
            self.calls_to_no_code += 1
        if outcome.timed_out or outcome.trace_result is None:
            self.call_failures += 1
            return
        if (
            isinstance(outcome.trace_result.result, CallResult)
            and outcome.trace_result.result.success
        ):
            self.call_successes += 1
        else:
            self.call_failures += 1


@dataclass
class CallOutcome:
    new_cov: int
    state_modified: bool
    timed_out: bool
    trace_result: Optional[TraceResult]
    stmt_sites: Set[RuntimeStmtSite] = field(default_factory=set)
    branch_outcomes: Set[RuntimeBranchOutcome] = field(default_factory=set)
    target_no_code: bool = False

    @property
    def is_interesting(self) -> bool:
        return self.new_cov > 0 or self.state_modified

    @property
    def is_progress(self) -> bool:
        return self.new_cov > 0


@dataclass
class HarnessResult:
    finalized_scenario: Scenario
    ivy_result: ScenarioResult
    stats: HarnessStats
    runtime_edge_ids: Set[int]
    runtime_stmt_sites_seen: Set[RuntimeStmtSite]
    runtime_branch_outcomes_seen: Set[RuntimeBranchOutcome]
    runtime_stmt_sites_total: Set[RuntimeStmtSite]
    runtime_branch_outcomes_total: Set[RuntimeBranchOutcome]
    contract_fingerprints: Set[str]

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


@dataclass
class FunctionInfo:
    addr: str
    fn_name: str
    func_t: Any
    timeout_count: int = 0


class RuntimeFuzzEngine:
    def __init__(
        self,
        config: HarnessConfig | None = None,
        seed: int | None = None,
    ):
        self.config = config or HarnessConfig()
        self.rng = random.Random(seed)
        self.runner = IvyScenarioRunner(
            collect_storage_dumps=self.config.collect_storage_dumps,
        )
        self.edge_map = RuntimeEdgeMap(self.config.map_size)
        self.call_generator = CallGenerator(self.rng)
        self.sender_pool = BOUNDARY_ADDRESSES

    def run(self, scenario: Scenario) -> HarnessResult:
        with self.runner.env.anchor():
            self.runner.deployed_contracts = {}
            self.runner.executed_dependencies = set()

            stats = HarnessStats()
            tracker = RuntimeCoverageTracker(self.config.map_size)
            finalized_traces: List[Any] = []
            trace_results: List[TraceResult] = []
            runtime_stmt_sites_seen: Set[RuntimeStmtSite] = set()
            runtime_branch_outcomes_seen: Set[RuntimeBranchOutcome] = set()
            runtime_stmt_sites_total: Set[RuntimeStmtSite] = set()
            runtime_branch_outcomes_total: Set[RuntimeBranchOutcome] = set()
            contract_fingerprints: Set[str] = set()

            deployed_contracts: Dict[str, Any] = {}
            trace_index = 0

            # Execute dependencies
            for dep_scenario in scenario.dependencies:
                self.runner._execute_dependency_scenario(dep_scenario)

            # Execute deployment and setup traces
            for trace in scenario.traces:
                if isinstance(trace, DeploymentTrace):
                    finalized_traces.append(trace)
                    trace_result = self._execute_deployment_with_retries(
                        trace=trace,
                        trace_index=trace_index,
                        use_python_args=scenario.use_python_args,
                        stats=stats,
                        contract_fingerprints=contract_fingerprints,
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

            # Build function info with timeout tracking
            all_functions: List[FunctionInfo] = []
            for addr, contract in deployed_contracts.items():
                for fn_name, func_t in self.call_generator.get_external_functions(
                    contract
                ):
                    all_functions.append(FunctionInfo(addr, fn_name, func_t))

            if self.config.enable_coverage_percentages:
                for contract in deployed_contracts.values():
                    stmt_sites, branch_outcomes = (
                        self._collect_static_coverage_sites_for_contract(contract)
                    )
                    runtime_stmt_sites_total.update(stmt_sites)
                    runtime_branch_outcomes_total.update(branch_outcomes)

            # Create corpus for mutation-based fuzzing
            corpus = Corpus(
                max_seeds_per_func=self.config.max_seeds_per_func,
                max_seeds_total=self.config.max_seeds_total,
            )

            # Track total calls across all phases
            total_calls = [0]  # Use list for mutable reference in closures

            def calls_remaining() -> int:
                return self.config.max_total_calls - total_calls[0]

            # Phase 1: Enumeration
            self._seed_enumerate_externals(
                all_functions,
                tracker,
                finalized_traces,
                trace_results,
                stats,
                corpus,
                total_calls,
                runtime_stmt_sites_seen,
                runtime_branch_outcomes_seen,
            )

            # Phase 2: Replay parent traces (bounded, no deepcopy)
            self._seed_replay_parent_traces(
                scenario,
                tracker,
                finalized_traces,
                trace_results,
                stats,
                corpus,
                total_calls,
                runtime_stmt_sites_seen,
                runtime_branch_outcomes_seen,
            )

            # Phase 3: Corpus-guided fuzzing with plateau escape
            self._fuzz_calls(
                all_functions,
                tracker,
                finalized_traces,
                trace_results,
                stats,
                corpus,
                total_calls,
                runtime_stmt_sites_seen,
                runtime_branch_outcomes_seen,
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
                runtime_stmt_sites_seen=runtime_stmt_sites_seen,
                runtime_branch_outcomes_seen=runtime_branch_outcomes_seen,
                runtime_stmt_sites_total=runtime_stmt_sites_total,
                runtime_branch_outcomes_total=runtime_branch_outcomes_total,
                contract_fingerprints=contract_fingerprints,
            )

    def _get_constructor_function(self, compiled: Any) -> Any:
        return getattr(getattr(compiled, "global_ctx", None), "init_function", None)

    def _execute_deployment_with_retries(
        self,
        trace: DeploymentTrace,
        trace_index: int,
        use_python_args: bool,
        stats: HarnessStats,
        contract_fingerprints: Set[str],
    ) -> TraceResult:
        prepared = self.runner.prepare_deployment_context(trace, trace_index)
        if isinstance(prepared, TraceResult):
            assert isinstance(prepared.result, DeploymentResult)
            stats.record_deployment_result(prepared.result)
            return prepared

        deployment_ctx = prepared
        contract_fingerprints.add(deployment_ctx.contract_fingerprint)
        sender = self.runner._get_sender(trace.env.tx.origin if trace.env else None)
        attempt_budget = max(1, self.config.max_deploy_retries)
        last_trace_result: Optional[TraceResult] = None

        base_args = (
            list(trace.python_args.get("args", []))
            if use_python_args and trace.python_args
            else []
        )
        base_kwargs = (
            dict(trace.python_args.get("kwargs", {}))
            if use_python_args and trace.python_args
            else {}
        )
        base_value = trace.value
        init_function = (
            self._get_constructor_function(deployment_ctx.compiled)
            if use_python_args and trace.python_args
            else None
        )

        for attempt in range(attempt_budget):
            if attempt == 0:
                attempt_args = list(base_args)
                attempt_value = base_value
            else:
                attempt_args, attempt_value = (
                    self.call_generator.argument_mutator.mutate_deployment_args(
                        init_function=init_function,
                        deploy_args=base_args,
                        deploy_value=base_value,
                    )
                )
            if use_python_args and trace.python_args is not None:
                trace.python_args["args"] = list(attempt_args)
                trace.python_args["kwargs"] = dict(base_kwargs)
            trace.value = attempt_value

            nonce_before = self.runner._get_nonce(sender)
            balance_before = self.runner._get_balance(sender)

            trace_result = self.runner.execute_trace(
                trace=trace,
                trace_index=trace_index,
                use_python_args=use_python_args,
                deployment_ctx=deployment_ctx,
            )
            last_trace_result = trace_result

            deployment_result = trace_result.result
            assert isinstance(deployment_result, DeploymentResult)
            stats.record_deployment_result(deployment_result)

            if deployment_result.success:
                if use_python_args and trace.python_args is not None:
                    trace.python_args["kwargs"] = dict(base_kwargs)
                return trace_result

            if deployment_result.error_phase == "compile":
                return trace_result

            if attempt + 1 < attempt_budget:
                self.runner._set_nonce(sender, nonce_before)
                self.runner._set_balance(sender, balance_before)

        assert last_trace_result is not None
        return last_trace_result

    def _execute_call_and_measure(
        self,
        trace: CallTrace,
        trace_index: int,
        tracker: RuntimeCoverageTracker,
        use_python_args: bool = True,
    ) -> CallOutcome:
        """Execute a single call with per-call metadata reset and measurement."""
        # Reset metadata BEFORE the call
        ivy.env.reset_execution_metadata()

        target_no_code = self._call_targets_no_code(trace)

        # Execute with timeout
        timed_out = False
        trace_result: Optional[TraceResult] = None
        interpreter = ivy.env.interpreter

        interpreter.set_call_timeout_seconds(self.config.call_timeout_s)
        try:
            trace_result = self.runner.execute_trace(
                trace, trace_index, use_python_args
            )
        except CallTimeout:
            timed_out = True
        finally:
            interpreter.clear_call_timeout()

        if trace_result is not None:
            if isinstance(trace_result.result, CallResult):
                if isinstance(trace_result.result.error, CallTimeout):
                    timed_out = True
                    trace_result = None

        # Read per-call metadata (now represents only THIS call)
        metadata = ivy.env.execution_metadata
        edge_ids = self.edge_map.hash_metadata(metadata)
        new_cov = tracker.merge(edge_ids)
        state_modified = metadata.state_modified
        stmt_sites: Set[RuntimeStmtSite] = set()
        branch_outcomes: Set[RuntimeBranchOutcome] = set()
        if self.config.enable_coverage_percentages:
            for addr, source_node_ids in metadata.coverage.items():
                addr_str = str(addr)
                for source_id, node_id in source_node_ids:
                    stmt_sites.add((addr_str, int(source_id), int(node_id)))
            for addr, source_id, node_id, taken in metadata.branches:
                branch_outcomes.add((str(addr), int(source_id), int(node_id), taken))

        return CallOutcome(
            new_cov=new_cov,
            state_modified=state_modified,
            timed_out=timed_out,
            trace_result=trace_result,
            stmt_sites=stmt_sites,
            branch_outcomes=branch_outcomes,
            target_no_code=target_no_code,
        )

    def _call_targets_no_code(self, trace: CallTrace) -> bool:
        to_address = trace.call_args.get("to", "")
        if not to_address:
            return False

        return self.runner.env.state.get_code(Address(to_address)) is None

    def _collect_static_coverage_sites_for_contract(
        self, contract: Any
    ) -> tuple[Set[RuntimeStmtSite], Set[RuntimeBranchOutcome]]:
        return collect_static_coverage_sites_for_contract(contract)

    def _seed_enumerate_externals(
        self,
        all_functions: List[FunctionInfo],
        tracker: RuntimeCoverageTracker,
        finalized_traces: List[Any],
        trace_results: List[TraceResult],
        stats: HarnessStats,
        corpus: Corpus,
        total_calls: List[int],
        runtime_stmt_sites_seen: Set[RuntimeStmtSite],
        runtime_branch_outcomes_seen: Set[RuntimeBranchOutcome],
    ) -> None:
        """Phase 1: Touch each external function once to seed corpus."""
        # Randomize order to avoid bias
        shuffled = list(all_functions)
        self.rng.shuffle(shuffled)

        for func_info in shuffled:
            if stats.enumeration_calls >= self.config.max_enumeration_calls:
                break
            if total_calls[0] >= self.config.max_total_calls:
                break

            sender = self.rng.choice(self.sender_pool)
            generated = self.call_generator.generate_call_for_function(
                func_info.addr, func_info.fn_name, func_info.func_t, sender=sender
            )
            trace = self.call_generator.call_trace_from_generated(
                generated, func_info.addr
            )

            outcome = self._execute_call_and_measure(
                trace, len(finalized_traces), tracker
            )
            stats.enumeration_calls += 1
            total_calls[0] += 1
            stats.record_call_outcome(outcome)
            runtime_stmt_sites_seen.update(outcome.stmt_sites)
            runtime_branch_outcomes_seen.update(outcome.branch_outcomes)

            if outcome.timed_out:
                stats.timeouts += 1
                func_info.timeout_count += 1
                # Still seed corpus with low weight for timeout functions
                corpus.add_seed(generated, score=0, step=total_calls[0])
            else:
                # Calculate score: new_cov + bonus for state_modified
                score = outcome.new_cov + (1 if outcome.state_modified else 0)

                # Always seed corpus (even if not interesting) to ensure coverage
                corpus.add_seed(generated, score=max(score, 1), step=total_calls[0])

                if outcome.new_cov > 0:
                    stats.new_coverage_calls += 1
                if outcome.state_modified:
                    stats.state_modified_calls += 1

                # Only retain in finalized traces if interesting
                if outcome.is_interesting and outcome.trace_result is not None:
                    finalized_traces.append(trace)
                    trace_results.append(outcome.trace_result)
                    stats.interesting_calls += 1

    def _seed_replay_parent_traces(
        self,
        scenario: Scenario,
        tracker: RuntimeCoverageTracker,
        finalized_traces: List[Any],
        trace_results: List[TraceResult],
        stats: HarnessStats,
        corpus: Corpus,
        total_calls: List[int],
        runtime_stmt_sites_seen: Set[RuntimeStmtSite],
        runtime_branch_outcomes_seen: Set[RuntimeBranchOutcome],
    ) -> None:
        """Phase 2: Replay parent CallTraces (bounded, no deepcopy)."""
        call_traces = [t for t in scenario.traces if isinstance(t, CallTrace)]

        # Cap replay to budget
        max_replay = min(
            len(call_traces),
            self.config.max_replay_calls,
            self.config.max_total_calls - total_calls[0],
        )

        # Replay from the beginning (prefix replay for state prerequisites)
        for i, trace in enumerate(call_traces[:max_replay]):
            if total_calls[0] >= self.config.max_total_calls:
                break

            # NO deepcopy - traces are not mutated during execution
            outcome = self._execute_call_and_measure(
                trace, len(finalized_traces), tracker, scenario.use_python_args
            )
            stats.replay_calls += 1
            total_calls[0] += 1
            stats.record_call_outcome(outcome)
            runtime_stmt_sites_seen.update(outcome.stmt_sites)
            runtime_branch_outcomes_seen.update(outcome.branch_outcomes)

            if outcome.timed_out:
                stats.timeouts += 1
                continue

            if outcome.new_cov > 0:
                stats.new_coverage_calls += 1
            if outcome.state_modified:
                stats.state_modified_calls += 1

            # Only retain interesting replayed traces
            if outcome.is_interesting and outcome.trace_result is not None:
                finalized_traces.append(trace)
                trace_results.append(outcome.trace_result)
                stats.interesting_calls += 1

                # Add to corpus if we have python_args to reconstruct GeneratedCall
                if trace.python_args and trace.function_name:
                    addr = trace.call_args.get("to", "")
                    generated = GeneratedCall(
                        contract_address=addr,
                        function_name=trace.function_name,
                        args=trace.python_args.get("args", []),
                        kwargs={"value": trace.call_args.get("value", 0)},
                        func_t=None,  # May not have type info from parent
                        sender=trace.env.tx.origin if trace.env else None,
                    )
                    score = outcome.new_cov + (1 if outcome.state_modified else 0)
                    corpus.add_seed(generated, score=score, step=total_calls[0])
            else:
                stats.skipped_replay += 1

    def _fuzz_calls(
        self,
        all_functions: List[FunctionInfo],
        tracker: RuntimeCoverageTracker,
        finalized_traces: List[Any],
        trace_results: List[TraceResult],
        stats: HarnessStats,
        corpus: Corpus,
        total_calls: List[int],
        runtime_stmt_sites_seen: Set[RuntimeStmtSite],
        runtime_branch_outcomes_seen: Set[RuntimeBranchOutcome],
    ) -> None:
        """Phase 3: Corpus-guided fuzzing with tiered plateau escape."""
        if not all_functions:
            return

        # Build lookup for function info
        func_lookup: Dict[CallKey, FunctionInfo] = {}
        for fi in all_functions:
            func_lookup[(fi.addr, fi.fn_name)] = fi

        # Filter to functions that haven't exceeded timeout threshold
        def get_available_functions() -> List[FunctionInfo]:
            return [
                fi
                for fi in all_functions
                if fi.timeout_count < self.config.max_timeouts_per_func
            ]

        consecutive_no_progress = 0
        plateau_level = 0  # 0=normal, 1=switch seed, 2=switch func, 3=havoc, 4=fresh

        fuzz_step = 0
        while (
            stats.fuzz_calls < self.config.max_fuzz_calls
            and total_calls[0] < self.config.max_total_calls
        ):
            fuzz_step += 1
            available_funcs = get_available_functions()
            if not available_funcs:
                break

            # Plateau escape logic
            if consecutive_no_progress >= self.config.plateau_calls:
                plateau_level = min(plateau_level + 1, 4)
                consecutive_no_progress = 0

            generated: Optional[GeneratedCall] = None
            func_info: Optional[FunctionInfo] = None

            if plateau_level == 0:
                # Normal: pick seed from corpus and mutate single arg
                seed = corpus.get_any_seed(self.rng)
                if seed and seed.call.func_t is not None:
                    seed.times_mutated += 1
                    seed.last_used_step = fuzz_step
                    generated = self.call_generator.mutate_single_arg(seed.call)
                    key = (seed.call.contract_address, seed.call.function_name)
                    func_info = func_lookup.get(key)
            elif plateau_level == 1:
                # Switch seed: try different seed from same function
                seed = corpus.get_any_seed(self.rng)
                if seed:
                    key = (seed.call.contract_address, seed.call.function_name)
                    other_seed = corpus.get_seed_for_func(key, self.rng)
                    if other_seed and other_seed is not seed:
                        other_seed.times_mutated += 1
                        generated = self.call_generator.mutate_single_arg(
                            other_seed.call
                        )
                    else:
                        generated = self.call_generator.mutate_call(seed.call)
                    func_info = func_lookup.get(key)
            elif plateau_level == 2:
                # Switch function: different function in corpus
                keys = list(corpus.seeds_by_func.keys())
                if keys:
                    key = self.rng.choice(keys)
                    seed = corpus.get_seed_for_func(key, self.rng)
                    if seed:
                        seed.times_mutated += 1
                        generated = self.call_generator.mutate_single_arg(seed.call)
                        func_info = func_lookup.get(key)
            elif plateau_level == 3:
                # Havoc: mutate multiple args aggressively
                seed = corpus.get_any_seed(self.rng)
                if seed and seed.call.func_t is not None:
                    seed.times_mutated += 1
                    generated = self.call_generator.mutate_havoc(seed.call)
                    key = (seed.call.contract_address, seed.call.function_name)
                    func_info = func_lookup.get(key)
            else:
                # Fresh random: generate entirely new call (reset plateau)
                func_info = self.rng.choice(available_funcs)
                generated = self.call_generator.generate_call_for_function(
                    func_info.addr, func_info.fn_name, func_info.func_t
                )
                plateau_level = 0

            # Fallback: generate fresh call if no corpus seed available
            if generated is None or func_info is None:
                func_info = self.rng.choice(available_funcs)
                generated = self.call_generator.generate_call_for_function(
                    func_info.addr, func_info.fn_name, func_info.func_t
                )

            # 10% chance to swap sender
            if self.rng.random() < 0.1:
                generated = self.call_generator.mutate_sender(
                    generated, self.sender_pool
                )

            trace = self.call_generator.call_trace_from_generated(
                generated, func_info.addr
            )

            outcome = self._execute_call_and_measure(
                trace, len(finalized_traces), tracker
            )
            stats.fuzz_calls += 1
            total_calls[0] += 1
            stats.record_call_outcome(outcome)
            runtime_stmt_sites_seen.update(outcome.stmt_sites)
            runtime_branch_outcomes_seen.update(outcome.branch_outcomes)

            if outcome.timed_out:
                stats.timeouts += 1
                func_info.timeout_count += 1
                consecutive_no_progress += 1
                continue

            if outcome.new_cov > 0:
                stats.new_coverage_calls += 1
            if outcome.state_modified:
                stats.state_modified_calls += 1

            if outcome.is_progress:
                consecutive_no_progress = 0
                plateau_level = 0  # Reset plateau on progress

                # Add successful call to corpus
                score = outcome.new_cov + (1 if outcome.state_modified else 0)
                corpus.add_seed(generated, score=score, step=fuzz_step)
            else:
                consecutive_no_progress += 1

            # Only retain interesting traces
            if outcome.is_interesting and outcome.trace_result is not None:
                finalized_traces.append(trace)
                trace_results.append(outcome.trace_result)
                stats.interesting_calls += 1

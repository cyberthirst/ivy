"""
Base fuzzer infrastructure shared between DifferentialFuzzer and GenerativeFuzzer.
"""

import logging
import random
import hashlib
import secrets
import typing
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

from fuzzer.runtime_engine.runtime_fuzz_engine import HarnessConfig

from fuzzer.mutator.ast_mutator import AstMutator
from fuzzer.mutator.value_mutator import ValueMutator
from fuzzer.mutator.trace_mutator import TraceMutator
from fuzzer.mutator.argument_mutator import ArgumentMutator
from fuzzer.export_utils import (
    load_all_exports,
    filter_exports,
    TestFilter,
    settings_to_kwargs,
)
from fuzzer.trace_types import (
    DeploymentTrace,
    CallTrace,
    SetBalanceTrace,
    ClearTransientStorageTrace,
    TestExport,
)
from ivy.frontend.loader import loads_from_solc_json

from fuzzer.runner.scenario import Scenario
from fuzzer.runner.multi_runner import MultiRunner
from fuzzer.deduper import Deduper
from fuzzer.result_analyzer import ResultAnalyzer
from fuzzer.reporter import FuzzerReporter
from fuzzer.issue_filter import IssueFilter

from vyper.compiler.phases import CompilerData


DEFAULT_AST_MUTATIONS = 8

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class BaseFuzzer:
    """Base class with shared fuzzer infrastructure."""

    def __init__(
        self,
        exports_dir: Path = Path("tests/vyper-exports"),
        seed: Optional[int] = None,
        debug_mode: bool = True,
        issue_filter: Optional[IssueFilter] = None,
        harness_config: Optional[HarnessConfig] = None,
    ):
        self.exports_dir = exports_dir
        self.seed = seed if seed is not None else secrets.randbits(64)
        self.debug_mode = debug_mode
        self.rng = random.Random(self.seed)
        self.harness_config = harness_config

        self.deduper = Deduper()
        self.reporter = FuzzerReporter(seed=self.seed)
        self.issue_filter = issue_filter
        self.result_analyzer = ResultAnalyzer(self.deduper, issue_filter)

        # Mutation probabilities
        self.call_drop_prob = 0.1
        self.call_mutate_args_prob = 0.3
        self.call_duplicate_prob = 0.1

        # Multi-runner (created lazily or by subclass)
        self._multi_runner: Optional[MultiRunner] = None

    @property
    def multi_runner(self) -> MultiRunner:
        if self._multi_runner is None:
            self._multi_runner = MultiRunner(
                collect_storage_dumps=True,
            )
        return self._multi_runner

    def load_filtered_exports(
        self, test_filter: Optional[TestFilter] = None
    ) -> Dict[Path, TestExport]:
        exports = load_all_exports(self.exports_dir, include_compiler_settings=False)
        if test_filter:
            exports = filter_exports(exports, test_filter=test_filter)
        return exports

    def get_compiler_data(self, trace: DeploymentTrace) -> Optional[CompilerData]:
        """Get CompilerData for a deployment trace."""
        if not trace.solc_json:
            return None

        try:
            return typing.cast(
                CompilerData,
                loads_from_solc_json(trace.solc_json, get_compiler_data=True),
            )
        except Exception as e:
            logging.debug(
                f"Failed to load compiler data ({type(e).__name__}): {e}"
            )
            return None

    def derive_scenario_seed(self, base: str, num: int) -> int:
        """Derive a deterministic per-scenario seed."""
        h = hashlib.blake2b(
            f"{self.seed}|{base}|{num}".encode("utf-8"), digest_size=16
        ).digest()
        return int.from_bytes(h, byteorder="big", signed=False)

    def mutate_scenario(
        self,
        scenario: Scenario,
        scenario_seed: Optional[int] = None,
        *,
        n_mutations: int = DEFAULT_AST_MUTATIONS,
    ) -> Scenario:
        """
        Apply mutations to a scenario's traces.

        Takes a scenario and returns a new scenario with mutated traces.
        """
        rng = random.Random(scenario_seed) if scenario_seed else self.rng

        ast_mutator = AstMutator(rng, max_mutations=n_mutations)
        value_mutator = ValueMutator(rng)
        argument_mutator = ArgumentMutator(rng, value_mutator)
        trace_mutator = TraceMutator(
            rng, value_mutator, argument_mutator, ast_mutator=ast_mutator
        )

        new_scenario = deepcopy(scenario)
        new_traces = []
        deployment_compiler_data = {}

        for trace in new_scenario.traces:
            if isinstance(trace, DeploymentTrace) and trace.deployment_type == "source":
                compiler_data = self.get_compiler_data(trace)

                trace_mutator.mutate_deployment_trace(trace, compiler_data)

                if compiler_data and hasattr(compiler_data, "settings"):
                    settings = settings_to_kwargs(compiler_data.settings)
                    # TODO the ast mutator should return the settings
                    # this is a temporary fix to avoid compilation failures
                    settings["enable_decimals"] = True
                    trace.compiler_settings = settings

                new_traces.append(trace)

                if compiler_data:
                    deployment_compiler_data[trace.deployed_address] = compiler_data

            # TODO disable mutations when harness is on
            # maybe we should remove the mutation all together
            elif isinstance(trace, CallTrace):
                if rng.random() < self.call_drop_prob:
                    continue

                mutate_args = rng.random() < self.call_mutate_args_prob

                trace_mutator.mutate_call_args(
                    trace, mutate_args, deployment_compiler_data
                )
                new_traces.append(trace)

                if rng.random() < self.call_duplicate_prob:
                    new_traces.append(deepcopy(trace))

            else:
                assert isinstance(trace, (SetBalanceTrace, ClearTransientStorageTrace))
                new_traces.append(trace)

        new_scenario.traces = new_traces
        return new_scenario

    def run_scenario(
        self,
        scenario: Scenario,
        *,
        seed: Optional[int] = None,
    ):
        """Run a scenario and analyze results."""
        if self.harness_config is not None:
            from fuzzer.runtime_engine.runtime_fuzz_engine import RuntimeFuzzEngine

            harness = RuntimeFuzzEngine(self.harness_config, seed)
            harness_result = harness.run(scenario)

            results = self.multi_runner.run_boa_only(
                harness_result.finalized_scenario,
                harness_result.ivy_result,
            )
            scenario_for_analysis = harness_result.finalized_scenario
            ivy_result = harness_result.ivy_result
        else:
            results = self.multi_runner.run(scenario)
            scenario_for_analysis = scenario
            ivy_result = results.ivy_result

        analysis = self.result_analyzer.analyze_run(
            scenario_for_analysis, ivy_result, results.boa_results
        )

        return analysis

    def finalize(self):
        """Stop timer and output final reports."""
        self.reporter.stop_timer()
        self.reporter.print_summary()
        self.reporter.save_statistics()  #

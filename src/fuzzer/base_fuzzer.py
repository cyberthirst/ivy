"""
Base fuzzer infrastructure shared between DifferentialFuzzer and GenerativeFuzzer.
"""

import logging
import random
import hashlib
import secrets
import typing
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional
from copy import deepcopy

from .mutator.ast_mutator import AstMutator
from .mutator.value_mutator import ValueMutator
from .mutator.trace_mutator import TraceMutator
from .mutator.argument_mutator import ArgumentMutator
from .export_utils import load_all_exports, filter_exports, TestFilter
from .trace_types import (
    DeploymentTrace,
    CallTrace,
    SetBalanceTrace,
    ClearTransientStorageTrace,
    TestExport,
)
from src.ivy.frontend.loader import loads_from_solc_json

from .runner.scenario import Scenario
from .runner.multi_runner import MultiRunner
from .deduper import Deduper
from .result_analyzer import ResultAnalyzer
from .reporter import FuzzerReporter
from .issue_filter import IssueFilter

from vyper.compiler.phases import CompilerData
from vyper.exceptions import CompilerPanic, VyperException


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
    ):
        self.exports_dir = exports_dir
        self.seed = seed if seed is not None else secrets.randbits(64)
        self.debug_mode = debug_mode
        self.rng = random.Random(self.seed)

        # Core components
        self.deduper = Deduper()
        self.reporter = FuzzerReporter(seed=self.seed)
        self.issue_filter = issue_filter
        self.result_analyzer = ResultAnalyzer(self.deduper, issue_filter)

        # Cache for CompilerData objects
        self._compiler_data_cache: Dict[int, CompilerData] = {}

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
                no_solc_json=True,
            )
        return self._multi_runner

    def load_filtered_exports(
        self, test_filter: Optional[TestFilter] = None
    ) -> Dict[Path, TestExport]:
        """Load and filter test exports."""
        exports = load_all_exports(self.exports_dir)
        if test_filter:
            exports = filter_exports(exports, test_filter=test_filter)
        return exports

    def get_compiler_data(self, trace: DeploymentTrace) -> Optional[CompilerData]:
        """Get CompilerData for a deployment trace, using cache if available."""
        if not trace.solc_json:
            return None

        cache_key = id(trace.solc_json)

        if cache_key in self._compiler_data_cache:
            return self._compiler_data_cache[cache_key]

        try:
            compiler_data = typing.cast(
                CompilerData,
                loads_from_solc_json(trace.solc_json, get_compiler_data=True),
            )
            self._compiler_data_cache[cache_key] = compiler_data
            return compiler_data
        except CompilerPanic as e:
            logging.error(f"Compiler panic: {e}")
            self.reporter.record_compiler_crash()
            return None
        except VyperException as e:
            logging.debug(f"Compilation failure (VyperException): {e}")
            self.reporter.record_compilation_failure()
            return None
        except Exception as e:
            logging.error(f"Compiler crash ({type(e).__name__}): {e}")
            self.reporter.record_compiler_crash()
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

                mutated_trace = trace_mutator.mutate_deployment_trace(
                    trace, compiler_data
                )

                # Capture compiler settings for runners to use.
                # TODO: This unconditionally enables decimals because the mutator
                # might generate decimal-using code. Ideally, we'd track whether
                # decimals were actually used and only enable if needed.
                if compiler_data and hasattr(compiler_data, "settings"):
                    # Use dataclasses.asdict to preserve enum types (Settings.as_dict
                    # converts enums to strings, which Settings() doesn't accept)
                    settings = {
                        k: v
                        for k, v in asdict(compiler_data.settings).items()
                        if v is not None
                    }
                    settings["enable_decimals"] = True
                    mutated_trace.compiler_settings = settings

                new_traces.append(mutated_trace)

                if compiler_data:
                    deployment_compiler_data[mutated_trace.deployed_address] = (
                        compiler_data
                    )

            elif isinstance(trace, CallTrace):
                if rng.random() < self.call_drop_prob:
                    continue

                mutate_args = rng.random() < self.call_mutate_args_prob

                mutated_trace = trace_mutator.mutate_and_normalize_call_args(
                    trace, mutate_args, deployment_compiler_data
                )
                new_traces.append(mutated_trace)

                if rng.random() < self.call_duplicate_prob:
                    new_traces.append(deepcopy(mutated_trace))

            else:
                assert isinstance(trace, (SetBalanceTrace, ClearTransientStorageTrace))
                new_traces.append(trace)

        new_scenario.traces = new_traces
        return new_scenario

    def run_scenario(self, scenario: Scenario):
        """Run a scenario and analyze results."""
        results = self.multi_runner.run(scenario)

        analysis = self.result_analyzer.analyze_run(
            scenario, results.ivy_result, results.boa_results
        )

        return analysis

    def finalize(self):
        """Stop timer and output final reports."""
        self.reporter.stop_timer()
        self.reporter.print_summary()
        self.reporter.save_statistics()

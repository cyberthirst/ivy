"""
Differential fuzzer for Vyper using test exports.

This module loads test exports, mutates them, and compares
execution between Ivy and the Vyper compiler (via Boa).
"""

import logging
import random
import hashlib
import secrets
from pathlib import Path
from typing import Dict, Optional
from copy import deepcopy

from .mutator.ast_mutator import AstMutator
from .mutator.value_mutator import ValueMutator
from .mutator.trace_mutator import TraceMutator
from .mutator.argument_mutator import ArgumentMutator
from .export_utils import (
    load_all_exports,
    filter_exports,
    TestFilter,
    TestItem,
    DeploymentTrace,
    CallTrace,
    SetBalanceTrace,
    ClearTransientStorageTrace,
)
from src.ivy.frontend.loader import loads_from_solc_json

from .runner.scenario import Scenario, create_scenario_from_item
from .runner.multi_runner import MultiRunner
from .deduper import Deduper
from .result_analyzer import ResultAnalyzer
from .reporter import FuzzerReporter

from vyper.compiler.phases import CompilerData
from vyper.exceptions import CompilerPanic, VyperException


# Configuration constants from spec
MAX_SCENARIOS_PER_ITEM = 30
MAX_AST_MUTATIONS = 8
TIMEOUT_PER_SCENARIO = 120  # seconds
LOG_LEVEL = logging.INFO

logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s: %(message)s")


class DifferentialFuzzer:
    """Fuzzer that uses Vyper test exports for differential testing."""

    def __init__(
        self,
        exports_dir: Path = Path("tests/vyper-exports"),
        seed: Optional[int] = None,
        debug_mode: bool = True,
    ):
        self.exports_dir = exports_dir
        # global campaign seed for total reproducibility
        self.seed = seed if seed is not None else secrets.randbits(64)
        self.debug_mode = debug_mode

        # Core components
        self.deduper = Deduper()
        self.reporter = FuzzerReporter(seed=self.seed)
        self.result_analyzer = ResultAnalyzer(self.deduper)

        # Cache for CompilerData objects, keyed by id of solc_json
        self._compiler_data_cache: Dict[int, CompilerData] = {}

        # Call trace mutation probabilities
        self.call_drop_prob = 0.1
        self.call_mutate_args_prob = 0.3
        self.call_duplicate_prob = 0.1

    def load_filtered_exports(self, test_filter: Optional[TestFilter] = None) -> Dict:
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
            compiler_data = loads_from_solc_json(
                trace.solc_json, get_compiler_data=True
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

    def create_mutated_scenario(
        self,
        item: TestItem,
        *,
        scenario_seed: Optional[int] = None,
    ) -> Scenario:
        """Create a scenario from a test item with optional mutations."""
        # Create base scenario using shared utility
        # TODO use_python_args should probably set on per-trace basis
        scenario = create_scenario_from_item(item, use_python_args=True)

        # per-scenario RNG for easy reproducibility
        rng = random.Random(
            scenario_seed
            if scenario_seed is not None
            else self._derive_scenario_seed(item.name, 0)
        )
        ast_mutator = AstMutator(rng, mutate_prob=0.5, max_mutations=MAX_AST_MUTATIONS)
        value_mutator = ValueMutator(rng)
        argument_mutator = ArgumentMutator(rng, value_mutator)
        trace_mutator = TraceMutator(rng, value_mutator, argument_mutator, ast_mutator)

        mutated_traces = []

        deployment_compiler_data = {}

        for trace in scenario.traces:
            if isinstance(trace, DeploymentTrace) and trace.deployment_type == "source":
                compiler_data = self.get_compiler_data(trace)

                # Mutate the deployment trace, xfail flags are set on the trace
                mutated_trace = trace_mutator.mutate_deployment_trace(
                    trace, compiler_data
                )
                mutated_traces.append(mutated_trace)

                # Add to deployment compiler data map
                if compiler_data:
                    deployment_compiler_data[mutated_trace.deployed_address] = (
                        compiler_data
                    )

            elif isinstance(trace, CallTrace):
                if rng.random() < self.call_drop_prob:
                    continue  # Drop this trace

                mutate_args = False
                if rng.random() < self.call_mutate_args_prob:
                    mutate_args = True

                mutated_trace = trace_mutator.mutate_and_normalize_call_args(
                    trace, mutate_args, deployment_compiler_data
                )

                mutated_traces.append(mutated_trace)

                # Check if we should duplicate
                if rng.random() < self.call_duplicate_prob:
                    # Append duplicate as-is (no further mutations)
                    mutated_traces.append(deepcopy(mutated_trace))

            else:
                assert isinstance(trace, (SetBalanceTrace, ClearTransientStorageTrace))
                mutated_traces.append(trace)

        scenario.mutated_traces = mutated_traces

        return scenario

    def _derive_scenario_seed(self, item_name: str, scenario_num: int) -> int:
        """Derive a deterministic per-scenario seed"""
        h = hashlib.blake2b(
            f"{self.seed}|{item_name}|{scenario_num}".encode("utf-8"), digest_size=16
        ).digest()
        return int.from_bytes(h, byteorder="big", signed=False)

    def fuzz_exports(
        self,
        test_filter: Optional[TestFilter] = None,
        max_scenarios: int = MAX_SCENARIOS_PER_ITEM,
    ):
        """Main fuzzing loop following the spec structure."""
        # Load and filter exports
        exports = self.load_filtered_exports(test_filter)
        logging.info(
            f"Loaded {sum(len(e.items) for e in exports.values())} test items from {len(exports)} files"
        )

        self.reporter.start_timer()

        items_processed = 0

        # Create multi-runner (with storage dumps enabled for comparison)
        # Use no_solc_json=True for Ivy to compile mutated source code
        multi_runner = MultiRunner(
            collect_storage_dumps=True,
            no_solc_json=True,
        )

        # Process each export file
        for export_path, export in exports.items():
            for item_name, item in export.items.items():
                # Skip fixtures - they'll be executed as dependencies
                if item.item_type == "fixture":
                    continue

                items_processed += 1
                logging.info(f"Testing {item_name} ({items_processed})")

                # Set context for reporting
                self.reporter.set_context(item_name, 0, self.seed, scenario_seed=None)

                # Run mutation scenarios
                for scenario_num in range(max_scenarios):
                    # Update context for this scenario
                    scenario_seed = self._derive_scenario_seed(item_name, scenario_num)
                    self.reporter.set_context(
                        item_name, scenario_num, self.seed, scenario_seed
                    )
                    # Create scenario with mutations using the per-scenario seed
                    scenario = self.create_mutated_scenario(
                        item, scenario_seed=scenario_seed
                    )

                    # Run in all environments
                    results = multi_runner.run(scenario)

                    # Analyze results (detects crashes, failures, divergences + dedup)
                    analysis = self.result_analyzer.analyze_run(
                        scenario, results.ivy_result, results.boa_results
                    )

                    # Report (handles stats, logging, file saving)
                    self.reporter.report(analysis, debug_mode=self.debug_mode)

        self.reporter.stop_timer()
        self.reporter.print_summary()
        self.reporter.save_statistics()


def main():
    """Run differential fuzzing with test exports."""
    # Create test filter - exclude multi-module contracts for now
    test_filter = TestFilter(exclude_multi_module=True)
    # Include tests with certain patterns
    test_filter.include_path("functional/builtins/codegen/test_slice")
    test_filter.exclude_source(r"\.code")
    test_filter.exclude_name("zero_length_side_effects")

    # Create and run fuzzer
    fuzzer = DifferentialFuzzer()
    fuzzer.fuzz_exports(test_filter=test_filter, max_scenarios=20)


if __name__ == "__main__":
    main()

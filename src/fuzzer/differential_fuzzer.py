"""
Differential fuzzer for Vyper using test exports.

This module loads test exports, mutates them, and compares
execution between Ivy and the Vyper compiler (via Boa).
"""

import logging
import random
from pathlib import Path
from typing import Dict, Optional
import json
from datetime import datetime
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
from .runner.ivy_scenario_runner import IvyScenarioRunner
from .runner.boa_scenario_runner import BoaScenarioRunner
from .divergence_detector import Divergence, DivergenceDetector

from vyper.compiler.phases import CompilerData


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
    ):
        self.exports_dir = exports_dir
        self.rng = random.Random(seed)
        self.ast_mutator = AstMutator(
            self.rng, mutate_prob=0.5, max_mutations=MAX_AST_MUTATIONS
        )
        self.value_mutator = ValueMutator(self.rng)
        self.argument_mutator = ArgumentMutator(self.rng, self.value_mutator)
        self.trace_mutator = TraceMutator(
            self.rng, self.value_mutator, self.argument_mutator, self.ast_mutator
        )
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
        # TODO this might be a compiler crash, we should use some filtering
        except Exception as e:
            logging.debug(f"Failed to load CompilerData: {e}")
            return None

    def create_mutated_scenario(
        self,
        item: TestItem,
    ) -> Scenario:
        """Create a scenario from a test item with optional mutations."""
        # Create base scenario using shared utility
        # TODO use_python_args should probably set on per-trace basis
        scenario = create_scenario_from_item(item, use_python_args=True)

        mutated_traces = []

        deployment_compiler_data = {}

        for trace in scenario.traces:
            if isinstance(trace, DeploymentTrace) and trace.deployment_type == "source":
                compiler_data = self.get_compiler_data(trace)

                # Mutate the deployment trace
                mutated_trace = self.trace_mutator.mutate_deployment_trace(
                    trace, compiler_data
                )
                mutated_traces.append(mutated_trace)

                # Add to deployment compiler data map
                if compiler_data:
                    deployment_compiler_data[mutated_trace.deployed_address] = (
                        compiler_data
                    )

            elif isinstance(trace, CallTrace):
                if self.rng.random() < self.call_drop_prob:
                    continue  # Drop this trace

                mutate_args = False
                if self.rng.random() < self.call_mutate_args_prob:
                    mutate_args = True

                mutated_trace = self.trace_mutator.mutate_and_normalize_call_args(
                    trace, mutate_args, deployment_compiler_data
                )

                mutated_traces.append(mutated_trace)

                # Check if we should duplicate
                if self.rng.random() < self.call_duplicate_prob:
                    # Append duplicate as-is (no further mutations)
                    mutated_traces.append(deepcopy(mutated_trace))

            else:
                assert isinstance(trace, (SetBalanceTrace, ClearTransientStorageTrace))
                mutated_traces.append(trace)

        scenario.mutated_traces = mutated_traces

        return scenario

    def save_divergence(
        self, divergence: Divergence, item_name: str, scenario_num: int
    ):
        """Save divergence to file."""
        # Create reports directory with date
        reports_dir = Path("reports") / datetime.now().strftime("%Y-%m-%d")
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Create filename
        filename = f"{item_name.replace('::', '_')}_{scenario_num}.divergence"
        filepath = reports_dir / filename

        # Convert to dict and add metadata
        divergence_data = divergence.to_dict()
        divergence_data["timestamp"] = datetime.now().isoformat()
        divergence_data["item_name"] = item_name
        divergence_data["scenario_num"] = scenario_num

        # Write to file
        with open(filepath, "w") as f:
            json.dump(divergence_data, f, indent=2, default=str)

        logging.error(f"Divergence saved to {filepath}")

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

        divergence_count = 0
        items_processed = 0

        # Create runners and detector (with storage dumps enabled for comparison)
        ivy_runner = IvyScenarioRunner(collect_storage_dumps=True)
        boa_runner = BoaScenarioRunner(collect_storage_dumps=True)
        detector = DivergenceDetector()

        # Process each export file
        for export_path, export in exports.items():
            for item_name, item in export.items.items():
                # Skip fixtures - they'll be executed as dependencies
                if item.item_type == "fixture":
                    continue

                items_processed += 1
                logging.info(f"Testing {item_name} ({items_processed})")

                # Run mutation scenarios
                for scenario_num in range(max_scenarios):
                    # Create scenario with mutations
                    scenario = self.create_mutated_scenario(item)

                    # Run in both environments
                    ivy_result = ivy_runner.run(scenario)
                    boa_result = boa_runner.run(scenario)

                    # Compare results
                    divergence = detector.compare_results(
                        ivy_result, boa_result, scenario
                    )

                    if divergence:
                        divergence_count += 1
                        logging.error(
                            f"diff| item {item_name} | mut#{scenario_num} | step {divergence.step}"
                        )
                        if divergence.type == "deployment":
                            logging.error("  Deployment divergence")
                        else:
                            logging.error(
                                f"  Execution divergence at function {divergence.function}"
                            )

                        # Save divergence
                        self.save_divergence(divergence, item_name, scenario_num)
                    else:
                        logging.info(f"ok  | item {item_name} | mut#{scenario_num}")

        logging.info(f"Fuzzing complete. Found {divergence_count} divergences.")


def main():
    """Run differential fuzzing with test exports."""
    # Create test filter - exclude multi-module contracts for now
    test_filter = TestFilter(exclude_multi_module=True)
    # Include tests with certain patterns
    test_filter.include_path("functional/builtins/codegen/test_concat")
    test_filter.exclude_source(r"\.code")
    test_filter.exclude_name("test_concat_zero_length_side_effects")

    # Create and run fuzzer
    fuzzer = DifferentialFuzzer()
    fuzzer.fuzz_exports(test_filter=test_filter)


if __name__ == "__main__":
    main()

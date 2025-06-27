"""
Differential fuzzer for Vyper using test exports.

This module loads test exports, mutates them, and compares
execution between Ivy and the Vyper compiler (via Boa).
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
from datetime import datetime
from copy import deepcopy

from .mutator import AstMutator
from .value_mutator import ValueMutator
from .trace_mutator import TraceMutator
from .type_retrieval import abi_type_to_vyper_type
from .export_utils import (
    load_all_exports,
    filter_exports,
    TestFilter,
    TestItem,
    DeploymentTrace,
)
from src.unparser.unparser import unparse

from .scenario import Scenario, create_scenario_from_item
from .base_scenario_runner import ScenarioResult, DeploymentResult, CallResult
from .ivy_scenario_runner import IvyScenarioRunner
from .boa_scenario_runner import BoaScenarioRunner
from .divergence_detector import Divergence, DivergenceDetector


# Configuration constants from spec
MAX_SCENARIOS_PER_ITEM = 30
MAX_AST_MUTATIONS = 8
MAX_CALLS = 12
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
        self.trace_mutator = TraceMutator(self.rng, self.value_mutator)

    def load_filtered_exports(self, test_filter: Optional[TestFilter] = None) -> Dict:
        """Load and filter test exports."""
        exports = load_all_exports(self.exports_dir)

        if test_filter:
            exports = filter_exports(exports, test_filter=test_filter)

        return exports

    def get_boundary_values(self, abi_type: str) -> List[Any]:
        """Get boundary values for a given ABI type."""
        vyper_type = abi_type_to_vyper_type(abi_type)
        if vyper_type:
            return self.value_mutator.get_boundary_values(vyper_type)
        else:
            # Fallback for unknown types
            return [0, 1, -1]

    def mutate_arguments(
        self, inputs: List[Dict[str, Any]], args: List[Any], mutation_prob: float = 0.3
    ) -> List[Any]:
        """Generalized function to mutate arguments based on ABI inputs."""
        mutated_args = args.copy()

        for i, input_spec in enumerate(inputs):
            if i < len(mutated_args) and self.rng.random() < mutation_prob:
                abi_type = input_spec.get("type", "")
                boundary_values = self.get_boundary_values(abi_type)
                if boundary_values:
                    mutated_args[i] = self.rng.choice(boundary_values)

        return mutated_args

    def mutate_value(self, is_payable: bool, mutation_prob: float = 0.3) -> int:
        """Mutate ETH value for payable functions."""
        if is_payable and self.rng.random() < mutation_prob:
            value_choices = [0, 1, 10**18, 2**128 - 1]  # 0, 1 wei, 1 ether, 2^128-1
            return self.rng.choice(value_choices)
        return 0

    def mutate_deployment(
        self, abi: List[Dict[str, Any]], deploy_args: List[Any], deploy_value: int
    ) -> Tuple[List[Any], int]:
        """Mutate deployment arguments and value according to spec."""
        # Find constructor in ABI
        constructor = None
        for item in abi:
            if item.get("type") == "constructor":
                constructor = item
                break

        # Mutate constructor arguments
        mutated_args = deploy_args.copy()
        if constructor and constructor.get("inputs"):
            mutated_args = self.mutate_arguments(constructor["inputs"], deploy_args)

        # Mutate deployment value
        is_payable = constructor and constructor.get("stateMutability") == "payable"
        mutated_value = self.mutate_value(is_payable) if is_payable else deploy_value
        if not is_payable and self.rng.random() < 0.1:
            # For non-payable constructors, sometimes try sending value anyway
            mutated_value = self.mutate_value(True)

        return mutated_args, mutated_value

    def mutate_source(self, source: str) -> Optional[str]:
        """Mutate source code and return the mutated version."""
        try:
            # Parse the source into AST
            import vyper

            ast = vyper.ast.parse_to_ast(source)
            logging.debug(f"Parsed AST successfully")

            # Mutate the AST
            mutated_ast = self.ast_mutator.mutate(ast)
            logging.debug(f"Mutation completed")

            # Unparse back to source
            result = unparse(mutated_ast)

            # Preserve pragma version if present in original source
            if source.lstrip().startswith("#pragma"):
                pragma_line = source.split("\n")[0]
                result = pragma_line + "\n\n" + result

            logging.debug(f"Unparsed successfully, result differs: {result != source}")
            return result
        except Exception as e:
            logging.info(f"Failed to mutate source: {e}")
            import traceback

            logging.debug(traceback.format_exc())
            return None


    def create_mutated_scenario(
        self,
        item: TestItem,
        enable_mutations: bool,
    ) -> Scenario:
        """Create a scenario from a test item with optional mutations."""
        # Create base scenario using shared utility
        scenario = create_scenario_from_item(item, use_python_args=True)

        # Apply mutations if enabled
        if enable_mutations:
            # Copy traces for potential mutation
            mutated_traces = []
            any_mutation = False
            
            for trace in scenario.traces:
                if isinstance(trace, DeploymentTrace) and trace.deployment_type == "source":
                    # Check if we should mutate this deployment
                    mutated_deployment = None
                    
                    # Try to mutate source code
                    if trace.source_code:
                        mutated_source = self.mutate_source(trace.source_code)
                        if mutated_source and mutated_source != trace.source_code:
                            # Need to create a mutated deployment
                            mutated_deployment = deepcopy(trace)
                            mutated_deployment.source_code = mutated_source
                            any_mutation = True
                    
                    # Try to mutate deployment args
                    if trace.python_args:
                        deploy_args = trace.python_args.get("args", [])
                        mutated_args, mutated_value = self.mutate_deployment(
                            trace.contract_abi,
                            deploy_args,
                            trace.value,
                        )
                        
                        if mutated_args != deploy_args or mutated_value != trace.value:
                            # Create or update mutated deployment
                            if not mutated_deployment:
                                mutated_deployment = deepcopy(trace)
                            
                            # Update python_args with mutations
                            mutated_deployment.python_args = deepcopy(trace.python_args)
                            mutated_deployment.python_args["args"] = mutated_args
                            mutated_deployment.value = mutated_value
                            any_mutation = True
                    
                    # Add the appropriate trace
                    mutated_traces.append(mutated_deployment if mutated_deployment else trace)
                else:
                    # For non-deployment traces, just append
                    mutated_traces.append(trace)
            
            # Apply trace mutations (calls, etc.)
            final_traces = self.trace_mutator.mutate_trace_sequence(
                mutated_traces,
                max_traces=MAX_CALLS,
            )
            
            # Only store if actually mutated
            if final_traces != scenario.traces:
                scenario.mutated_traces = final_traces
            elif any_mutation:
                # We mutated deployments but trace_mutator didn't change anything else
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
        enable_mutations: bool = True,
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
                    scenario = self.create_mutated_scenario(
                        item, enable_mutations
                    )

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
                            logging.error(f"  Deployment divergence")
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
    test_filter.include_path("functional/codegen/calling_convention/test_internal")
    test_filter.exclude_source(r"\.code")

    # Create and run fuzzer
    fuzzer = DifferentialFuzzer()
    fuzzer.fuzz_exports(
        test_filter=test_filter, max_scenarios=200, enable_mutations=True
    )


if __name__ == "__main__":
    main()

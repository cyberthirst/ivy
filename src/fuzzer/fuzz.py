"""
Differential fuzzer for Vyper using test exports.

This module loads test exports, optionally mutates them, and compares
execution between Ivy and the Vyper compiler (via Boa).
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
from datetime import datetime

from .mutator import AstMutator
from .value_mutator import ValueMutator
from .type_retrieval import (
    abi_type_to_vyper_type,
    get_function_types_from_abi,
    get_constructor_types_from_abi,
)
from .export_utils import (
    load_all_exports,
    filter_exports,
    TestFilter,
    DeploymentTrace,
    CallTrace,
    TestExport,
    TestItem,
)
from src.unparser.unparser import unparse

from .runner import Scenario, Call, DivergenceDetector, Divergence
from .ivy_runner import IvyRunner
from .boa_runner import BoaRunner


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
        self.mutator = AstMutator(
            self.rng, mutate_prob=0.5, max_mutations=MAX_AST_MUTATIONS
        )
        self.value_mutator = ValueMutator(self.rng)

    def load_filtered_exports(self, test_filter: Optional[TestFilter] = None) -> Dict:
        """Load and filter test exports."""
        exports = load_all_exports(self.exports_dir)

        if test_filter:
            exports = filter_exports(exports, test_filter=test_filter)

        return exports

    def get_boundary_values(self, abi_type: str) -> List[Any]:
        """Get boundary values for a given ABI type."""
        # Convert ABI type to Vyper type
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
            mutated_ast = self.mutator.mutate(ast)
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

    def generate_schedule(
        self, contract_abi: List[Dict[str, Any]], num_calls: Optional[int] = None
    ) -> List[Call]:
        """Generate a call schedule for stateful ABI fuzzing according to spec."""
        if num_calls is None:
            num_calls = self.rng.randint(1, MAX_CALLS)

        call_schedule = []

        # Separate functions by type
        state_changing_fns = []
        view_pure_fns = []

        for item in contract_abi:
            if item.get("type") == "function":
                state_mutability = item.get("stateMutability", "nonpayable")
                if state_mutability in ["view", "pure"]:
                    view_pure_fns.append(item)
                else:
                    state_changing_fns.append(item)

        # Generate calls
        for _ in range(num_calls):
            # 70% chance for state-changing, 30% for view/pure
            if self.rng.random() < 0.7 and state_changing_fns:
                fn = self.rng.choice(state_changing_fns)
            elif view_pure_fns:
                fn = self.rng.choice(view_pure_fns)
            elif state_changing_fns:
                fn = self.rng.choice(state_changing_fns)
            else:
                continue

            # Generate arguments using the generalized mutation function
            # First create base arguments with boundary values
            base_args = []
            for input_spec in fn.get("inputs", []):
                abi_type = input_spec.get("type", "")
                boundary_values = self.get_boundary_values(abi_type)
                if boundary_values:
                    base_args.append(self.rng.choice(boundary_values))
                else:
                    base_args.append(0)  # Default value

            # Apply mutations to the arguments
            args = self.mutate_arguments(
                fn.get("inputs", []), base_args, mutation_prob=0.5
            )

            # Generate kwargs (value for payable functions)
            kwargs = {}
            is_payable = fn.get("stateMutability") == "payable"
            value = self.mutate_value(
                is_payable, mutation_prob=1.0 if is_payable else 0.1
            )
            if value > 0:
                kwargs["value"] = value

            # Random msg.sender from a pool of 3 addresses
            senders = [
                "0x0000000000000000000000000000000000000001",
                "0x0000000000000000000000000000000000000002",
                "0x0000000000000000000000000000000000000003",
            ]

            call = Call(
                fn_name=fn["name"],
                args=args,
                kwargs=kwargs,
                msg_sender=self.rng.choice(senders),
            )
            call_schedule.append(call)

        return call_schedule

    def get_source_deployments(self, exports: Dict[Path, Any]) -> List[tuple]:
        """Extract test items that have source deployments."""
        deployments = []

        for path, export in exports.items():
            for item_name, item in export.items.items():
                # Find source deployment traces
                for trace in item.traces:
                    if (
                        isinstance(trace, DeploymentTrace)
                        and trace.deployment_type == "source"
                        and trace.source_code
                    ):
                        deployments.append((export, item_name, trace))
                        break  # Only take first deployment per item

        return deployments

    def run_scenario(self, scenario: Scenario) -> Optional[Divergence]:
        """Run a complete scenario and check for divergences."""
        # Create runners
        ivy_runner = IvyRunner()
        boa_runner = BoaRunner()

        # Run scenarios in both environments
        ivy_result = ivy_runner.run(scenario)
        boa_result = boa_runner.run(scenario)

        # Compare results
        detector = DivergenceDetector()
        return detector.compare_results(ivy_result, boa_result, scenario)

    def save_divergence(
        self, divergence: Dict[str, Any], item_name: str, scenario_num: int
    ):
        """Save divergence to file."""
        # Create reports directory with date
        reports_dir = Path("reports") / datetime.now().strftime("%Y-%m-%d")
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Create filename
        filename = f"{item_name.replace('::', '_')}_{scenario_num}.divergence"
        filepath = reports_dir / filename

        # Add metadata
        divergence["timestamp"] = datetime.now().isoformat()
        divergence["seed"] = self.rng.getstate()[1][
            0
        ]  # Get first element of random state

        # Write to file
        with open(filepath, "w") as f:
            json.dump(divergence, f, indent=2, default=str)

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

        # Get source deployments
        deployments = self.get_source_deployments(exports)
        logging.info(f"Found {len(deployments)} test items with source deployments")

        divergence_count = 0

        # Run differential testing
        for i, (export, item_name, deployment_trace) in enumerate(deployments):
            logging.info(f"Testing {item_name} ({i + 1}/{len(deployments)})")

            # Extract deployment args from trace
            deploy_args = []
            deploy_kwargs = {"value": deployment_trace.value}
            if deployment_trace.python_args:
                deploy_args = deployment_trace.python_args.get("args", [])
                deploy_kwargs.update(deployment_trace.python_args.get("kwargs", {}))
                deploy_kwargs["value"] = deployment_trace.value  # Ensure value is set

            # Debug log
            logging.debug(f"Deploy args: {deploy_args}")
            logging.debug(f"Deploy kwargs: {deploy_kwargs}")
            logging.debug(f"Python args: {deployment_trace.python_args}")

            # Step 0: Baseline run (no mutation)
            baseline_scenario = Scenario(
                mutated_source=deployment_trace.source_code,
                deploy_args=deploy_args,
                deploy_kwargs=deploy_kwargs,
                call_schedule=[],  # No calls for baseline
            )

            baseline_divergence = self.run_scenario(baseline_scenario)
            if baseline_divergence:
                logging.error(f"Baseline failure for {item_name} - skipping")
                logging.error(f"  Divergence type: {baseline_divergence.type}")
                if (
                    baseline_divergence.ivy_result
                    and baseline_divergence.ivy_result.error
                ):
                    logging.error(
                        f"  Ivy error: {baseline_divergence.ivy_result.error}"
                    )
                if (
                    baseline_divergence.boa_result
                    and baseline_divergence.boa_result.error
                ):
                    logging.error(
                        f"  Boa error: {baseline_divergence.boa_result.error}"
                    )
                continue

            # Run mutation scenarios
            scenarios_run = 0
            for scenario_num in range(max_scenarios):
                # Step 1: Mutate source (AST layer)
                mutated_source = deployment_trace.source_code
                if enable_mutations:
                    mutated_source = self.mutate_source(deployment_trace.source_code)
                    if not mutated_source:
                        mutated_source = deployment_trace.source_code

                # Step 2: Mutate deployment (ctor args + value)
                mutated_args, mutated_value = self.mutate_deployment(
                    deployment_trace.contract_abi,
                    deploy_args.copy(),
                    deployment_trace.value,
                )
                mutated_kwargs = deploy_kwargs.copy()
                mutated_kwargs["value"] = mutated_value

                # Step 3: Build call schedule (stateful ABI fuzz)
                call_schedule = self.generate_schedule(deployment_trace.contract_abi)

                # Create scenario
                scenario = Scenario(
                    mutated_source=mutated_source,
                    deploy_args=mutated_args,
                    deploy_kwargs=mutated_kwargs,
                    call_schedule=call_schedule,
                )

                # Step 4 & 5: Run in Ivy/Boa and compare
                divergence = self.run_scenario(scenario)

                if divergence:
                    divergence_count += 1
                    logging.error(
                        f"diff| item {item_name} | mut#{scenario_num} | calls {len(call_schedule)} | step {divergence.step}"
                    )
                    if divergence.type == "deployment":
                        logging.error(f"  Deployment divergence")
                    else:
                        logging.error(
                            f"  Execution divergence at function {divergence.function}"
                        )

                    # Save divergence
                    self.save_divergence(divergence.to_dict(), item_name, scenario_num)
                else:
                    logging.info(
                        f"ok  | item {item_name} | mut#{scenario_num} | calls {len(call_schedule)}"
                    )

                scenarios_run += 1

                # Continue to next scenario (don't stop on first divergence)

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

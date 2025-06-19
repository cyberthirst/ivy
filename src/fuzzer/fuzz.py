"""
Differential fuzzer for Vyper using test exports.

This module loads test exports, optionally mutates them, and compares
execution between Ivy and the Vyper compiler (via Boa).
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ivy.frontend.loader import loads as ivy_loads, loads_from_solc_json
from ivy.frontend.env import Env
from ivy.types import Address
from boa import loads as boa_loads
import boa

from .mutator import AstMutator
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
from tests.test_replay import TestReplay


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@dataclass
class MutatedScenario:
    """Test scenario with mutated source code"""

    export: TestExport
    item_name: str
    original_source: str
    mutated_source: str
    deployment_trace: DeploymentTrace


class DifferentialFuzzer:
    """Fuzzer that uses Vyper test exports for differential testing."""

    def __init__(
        self,
        exports_dir: Path = Path("tests/vyper-exports"),
        seed: Optional[int] = None,
    ):
        self.exports_dir = exports_dir
        self.rng = random.Random(seed)
        self.mutator = AstMutator(self.rng, mutate_prob=0.5, max_mutations=5)

    def load_filtered_exports(self, test_filter: Optional[TestFilter] = None) -> Dict:
        """Load and filter test exports."""
        exports = load_all_exports(self.exports_dir)

        if test_filter:
            exports = filter_exports(exports, test_filter=test_filter)

        return exports

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

    def run_ivy_with_mutated_source(self, scenario: MutatedScenario) -> dict:
        """
        Execute test with mutated source code using Ivy.
        Returns deployment status and execution results.
        """
        env = Env()

        # Create a modified export with mutated source
        modified_export = TestExport(path=scenario.export.path, items={})

        # Deep copy the test item
        import copy

        item = copy.deepcopy(scenario.export.items[scenario.item_name])

        # Replace the source code in deployment trace
        for i, trace in enumerate(item.traces):
            if trace == scenario.deployment_trace:
                # Create new deployment trace with mutated source
                new_trace = DeploymentTrace(
                    deployment_type=trace.deployment_type,
                    deployer=trace.deployer,
                    deployed_address=trace.deployed_address,
                    value=trace.value,
                    calldata=trace.calldata,
                    source_code=scenario.mutated_source,
                    solc_json=trace.solc_json,
                    deployment_succeeded=trace.deployment_succeeded,
                    contract_abi=trace.contract_abi,
                    initcode=trace.initcode,
                    annotated_ast=trace.annotated_ast,
                    raw_ir=trace.raw_ir,
                    blueprint_initcode_prefix=trace.blueprint_initcode_prefix,
                    runtime_bytecode=trace.runtime_bytecode,
                    python_args=trace.python_args,
                )
                item.traces[i] = new_trace
                break

        modified_export.items[scenario.item_name] = item

        # Use TestReplay to execute with python_args
        replay = TestReplay(env, use_python_args=True)

        try:
            with env.anchor():
                replay.execute_item(modified_export, scenario.item_name)
            return {"success": True, "env": env}
        except Exception as e:
            return {"error": e, "env": env}

    def run_boa_with_source(
        self,
        source_code: str,
        python_args: Optional[Dict[str, Any]],
        deployment_value: int,
    ) -> dict:
        """
        Compile and deploy source code with Boa.
        Returns deployment status.
        """
        try:
            # Deploy with Boa using python args
            if python_args:
                args = python_args.get("args", [])
                kwargs = python_args.get("kwargs", {})
                kwargs["value"] = deployment_value
                contract = boa_loads(source_code, *args, **kwargs)
            else:
                contract = boa_loads(source_code, value=deployment_value)

            return {"success": True, "contract": contract}
        except Exception as e:
            return {"error": e}

    def compare_deployment(self, scenario: MutatedScenario):
        """Compare deployment between Ivy and Boa with mutated source."""
        # Run Ivy
        ivy_res = self.run_ivy_with_mutated_source(scenario)

        # Run Boa with just the mutated source
        deployment_trace = scenario.deployment_trace
        boa_res = self.run_boa_with_source(
            scenario.mutated_source,
            deployment_trace.python_args,
            deployment_trace.value,
        )

        # Compare results
        ivy_err = ivy_res.get("error")
        boa_err = boa_res.get("error")

        if (ivy_err is None) != (boa_err is None):
            # Skip known risky overlap errors
            if (
                boa_err
                and hasattr(boa_err, "message")
                and "risky overlap" in str(boa_err)
            ):
                return

            logging.error("Deployment mismatch:")
            logging.error("Original source:\n%s", scenario.original_source)
            logging.error("Mutated source:\n%s", scenario.mutated_source)
            logging.error("  Ivy error: %r", ivy_err)
            logging.error("  Boa error: %r", boa_err)

    def fuzz_exports(
        self,
        test_filter: Optional[TestFilter] = None,
        max_mutations_per_test: int = 5,
        enable_mutations: bool = True,
    ):
        """Main fuzzing loop using test exports."""
        # Load and filter exports
        exports = self.load_filtered_exports(test_filter)
        logging.info(
            f"Loaded {sum(len(e.items) for e in exports.values())} test items from {len(exports)} files"
        )

        # Get source deployments
        deployments = self.get_source_deployments(exports)
        logging.info(f"Found {len(deployments)} test items with source deployments")

        # Run differential testing
        for i, (export, item_name, deployment_trace) in enumerate(deployments):
            logging.info(f"Testing {item_name} ({i + 1}/{len(deployments)})")

            # First, test without mutations to ensure baseline correctness
            baseline_scenario = MutatedScenario(
                export=export,
                item_name=item_name,
                original_source=deployment_trace.source_code,
                mutated_source=deployment_trace.source_code,
                deployment_trace=deployment_trace,
            )
            self.compare_deployment(baseline_scenario)

            # Then test with mutations if enabled
            if enable_mutations:
                for mutation_round in range(max_mutations_per_test):
                    mutated_source = self.mutate_source(deployment_trace.source_code)
                    if (
                        mutated_source
                        and mutated_source != deployment_trace.source_code
                    ):
                        logging.info(f"Testing mutation {mutation_round + 1}")
                        # Log first line of mutation to see what changed
                        first_change = None
                        orig_lines = deployment_trace.source_code.split("\n")
                        mut_lines = mutated_source.split("\n")
                        for i, (orig, mut) in enumerate(zip(orig_lines, mut_lines)):
                            if orig != mut:
                                first_change = f"Line {i + 1}: '{orig}' -> '{mut}'"
                                break
                        if first_change:
                            logging.info(f"  Mutation: {first_change}")
                        mutated_scenario = MutatedScenario(
                            export=export,
                            item_name=item_name,
                            original_source=deployment_trace.source_code,
                            mutated_source=mutated_source,
                            deployment_trace=deployment_trace,
                        )
                        self.compare_deployment(mutated_scenario)


def main():
    """Run differential fuzzing with test exports."""
    # Create test filter - exclude multi-module contracts for now
    test_filter = TestFilter(exclude_multi_module=True)
    # Exclude tests with certain patterns
    test_filter.include_path("functional/examples/tokens/test_erc20")

    # Create and run fuzzer
    fuzzer = DifferentialFuzzer()
    fuzzer.fuzz_exports(
        test_filter=test_filter, max_mutations_per_test=3, enable_mutations=True
    )


if __name__ == "__main__":
    main()

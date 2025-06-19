"""
Differential fuzzer for Vyper using test exports.

This module loads test exports, optionally mutates them, and compares
execution between Ivy and the Vyper compiler (via Boa).
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json
from datetime import datetime

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


# Configuration constants from spec
MAX_SCENARIOS_PER_ITEM = 30
MAX_AST_MUTATIONS = 8
MAX_CALLS = 12
TIMEOUT_PER_SCENARIO = 120  # seconds
LOG_LEVEL = logging.INFO

logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s: %(message)s")


@dataclass
class Call:
    """Represents a function call in the call schedule"""

    fn_name: str
    args: List[Any]
    kwargs: Dict[str, Any]
    msg_sender: Optional[str] = None


@dataclass
class Scenario:
    """Fuzzing scenario as specified in the spec"""

    mutated_source: str
    deploy_args: List[Any]
    deploy_kwargs: Dict[str, Any]  # contains "value"
    call_schedule: List[Call]


@dataclass
class MutatedScenario:
    """Test scenario with mutated source code"""

    export: TestExport
    item_name: str
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
        self.mutator = AstMutator(
            self.rng, mutate_prob=0.5, max_mutations=MAX_AST_MUTATIONS
        )

    def load_filtered_exports(self, test_filter: Optional[TestFilter] = None) -> Dict:
        """Load and filter test exports."""
        exports = load_all_exports(self.exports_dir)

        if test_filter:
            exports = filter_exports(exports, test_filter=test_filter)

        return exports

    def get_boundary_values(self, abi_type: str) -> List[Any]:
        """Get boundary values for a given ABI type."""
        if "uint" in abi_type:
            # Extract bit size (default to 256)
            bits = 256
            if abi_type != "uint":
                try:
                    bits = int(abi_type[4:])
                except:
                    pass
            max_val = (2**bits) - 1
            return [
                0,
                1,
                max_val,
                max_val - 1,
                2 ** (bits - 1),
                self.rng.randint(0, max_val),
            ]
        elif "int" in abi_type:
            # Signed integers
            bits = 256
            if abi_type != "int":
                try:
                    bits = int(abi_type[3:])
                except:
                    pass
            max_val = (2 ** (bits - 1)) - 1
            min_val = -(2 ** (bits - 1))
            return [0, 1, -1, max_val, min_val, self.rng.randint(min_val, max_val)]
        elif "address" in abi_type:
            return [
                "0x0000000000000000000000000000000000000000",
                "0x0000000000000000000000000000000000000001",
                "0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
                f"0x{self.rng.randbytes(20).hex()}",
            ]
        elif "bool" in abi_type:
            return [True, False]
        elif "bytes" in abi_type:
            if abi_type == "bytes":
                # Dynamic bytes
                return [
                    b"",
                    b"\\x00",
                    b"\\xff" * 32,
                    self.rng.randbytes(self.rng.randint(1, 100)),
                ]
            else:
                # Fixed bytes
                try:
                    size = int(abi_type[5:])
                    return [b"\\x00" * size, b"\\xff" * size, self.rng.randbytes(size)]
                except:
                    return [b""]
        elif "string" in abi_type:
            return [
                "",
                "a",
                "A" * 1000,
                "\\x00",
                "test_" + str(self.rng.randint(0, 1000)),
            ]
        else:
            # Default/unknown types
            return [0, 1, -1]

    def mutate_deployment(
        self, abi: List[Dict[str, Any]], deploy_args: List[Any], deploy_value: int
    ) -> Tuple[List[Any], int]:
        """Mutate deployment arguments and value according to spec."""
        mutated_args = deploy_args.copy()

        # Find constructor in ABI
        constructor = None
        for item in abi:
            if item.get("type") == "constructor":
                constructor = item
                break

        if constructor and constructor.get("inputs"):
            # Mutate constructor arguments
            for i, input_spec in enumerate(constructor["inputs"]):
                if (
                    i < len(mutated_args) and self.rng.random() < 0.3
                ):  # 30% chance to mutate each arg
                    abi_type = input_spec.get("type", "")
                    boundary_values = self.get_boundary_values(abi_type)
                    if boundary_values:
                        mutated_args[i] = self.rng.choice(boundary_values)

        # Mutate deployment value
        if self.rng.random() < 0.3:  # 30% chance to mutate value
            value_choices = [0, 1, 10**18, 2**128 - 1]  # 0, 1 wei, 1 ether, 2^128-1
            deploy_value = self.rng.choice(value_choices)

        return mutated_args, deploy_value

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

            # Generate arguments
            args = []
            for input_spec in fn.get("inputs", []):
                abi_type = input_spec.get("type", "")
                boundary_values = self.get_boundary_values(abi_type)
                if boundary_values:
                    args.append(self.rng.choice(boundary_values))
                else:
                    args.append(0)  # Default value

            # Generate kwargs (value for payable functions)
            kwargs = {}
            if fn.get("stateMutability") == "payable":
                value_choices = [0, 1, 10**18, 2**128 - 1]  # 0, 1 wei, 1 ether, 2^128-1
                kwargs["value"] = self.rng.choice(value_choices)

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

    def compare_step(
        self, ivy_res: Any, boa_res: Any, ivy_env: Env, boa_env: Any
    ) -> bool:
        """Compare execution step results between Ivy and Boa."""
        # Check if both reverted or both succeeded
        ivy_reverted = isinstance(ivy_res, Exception) or (
            hasattr(ivy_res, "reverted") and ivy_res.reverted
        )
        boa_reverted = isinstance(boa_res, Exception) or (
            hasattr(boa_res, "reverted") and boa_res.reverted
        )

        if ivy_reverted != boa_reverted:
            return False

        # If both reverted, we consider them equal (for now)
        if ivy_reverted:
            return True

        # Compare outputs (if not reverted)
        ivy_output = ivy_res if not hasattr(ivy_res, "output") else ivy_res.output
        boa_output = boa_res if not hasattr(boa_res, "output") else boa_res.output

        if ivy_output != boa_output:
            return False

        # TODO: Compare state - this would require accessing storage from both environments
        # For now, we only compare deployment and outputs

        return True

    def run_scenario(self, scenario: Scenario) -> Optional[Dict[str, Any]]:
        """Run a complete scenario and check for divergences."""
        # Deploy in both Ivy and Boa
        ivy_env = Env()

        # Deploy with Ivy - ivy_loads already deploys the contract
        try:
            # Set up environment
            deployer_addr = Address("0x0000000000000000000000000000000000000001")
            ivy_env.set_balance(deployer_addr, 10**21)  # Give deployer some ETH
            # Set deployer as tx origin if available
            if hasattr(ivy_env, "evm") and hasattr(ivy_env.evm, "set_tx_origin"):
                ivy_env.evm.set_tx_origin(deployer_addr)

            # ivy_loads compiles and deploys the contract with constructor args
            if scenario.deploy_args:
                ivy_address = ivy_loads(
                    scenario.mutated_source,
                    env=ivy_env,
                    *scenario.deploy_args,
                    value=scenario.deploy_kwargs.get("value", 0),
                )
            else:
                ivy_address = ivy_loads(
                    scenario.mutated_source,
                    env=ivy_env,
                    value=scenario.deploy_kwargs.get("value", 0),
                )

            ivy_deployed = True
            ivy_deploy_error = None
        except Exception as e:
            ivy_deployed = False
            ivy_deploy_error = e
            ivy_address = None

        # Deploy with Boa
        try:
            boa_contract = boa_loads(
                scenario.mutated_source, *scenario.deploy_args, **scenario.deploy_kwargs
            )
            boa_deployed = True
            boa_deploy_error = None
        except Exception as e:
            boa_deployed = False
            boa_deploy_error = e
            boa_contract = None

        # Check deployment divergence
        if ivy_deployed != boa_deployed:
            # Skip known risky overlap errors
            if boa_deploy_error and "risky overlap" in str(boa_deploy_error):
                return None

            return {
                "type": "deployment",
                "step": 0,
                "mutated_source": scenario.mutated_source,
                "deploy_args": scenario.deploy_args,
                "deploy_kwargs": scenario.deploy_kwargs,
                "ivy_error": ivy_deploy_error,
                "boa_error": boa_deploy_error,
            }

        # If both failed to deploy, no divergence
        if not ivy_deployed:
            return None

        # Execute call schedule
        for step, call in enumerate(scenario.call_schedule):
            # Execute in Ivy
            try:
                if call.msg_sender:
                    ivy_env.set_balance(
                        Address(call.msg_sender), 10**20
                    )  # Give sender some ETH

                ivy_result = getattr(ivy_address, call.fn_name)(
                    *call.args,
                    value=call.kwargs.get("value", 0),
                    sender=Address(call.msg_sender) if call.msg_sender else None,
                )
                ivy_error = None
            except Exception as e:
                ivy_result = None
                ivy_error = e

            # Execute in Boa
            try:
                if call.msg_sender:
                    boa.env.set_balance(call.msg_sender, 10**20)  # Give sender some ETH
                    with boa.env.prank(call.msg_sender):
                        boa_fn = getattr(boa_contract, call.fn_name)
                        boa_result = boa_fn(*call.args, **call.kwargs)
                else:
                    boa_fn = getattr(boa_contract, call.fn_name)
                    boa_result = boa_fn(*call.args, **call.kwargs)
                boa_error = None
            except Exception as e:
                boa_result = None
                boa_error = e

            # Compare results
            if not self.compare_step(
                ivy_result if ivy_error is None else ivy_error,
                boa_result if boa_error is None else boa_error,
                ivy_env,
                boa,
            ):
                return {
                    "type": "execution",
                    "step": step + 1,  # +1 because step 0 is deployment
                    "function": call.fn_name,
                    "mutated_source": scenario.mutated_source,
                    "deploy_args": scenario.deploy_args,
                    "deploy_kwargs": scenario.deploy_kwargs,
                    "call_schedule": [
                        {
                            "fn_name": c.fn_name,
                            "args": c.args,
                            "kwargs": c.kwargs,
                            "msg_sender": c.msg_sender,
                        }
                        for c in scenario.call_schedule[: step + 1]
                    ],  # Include calls up to divergence
                    "ivy_result": str(ivy_result) if ivy_error is None else None,
                    "ivy_error": str(ivy_error) if ivy_error else None,
                    "boa_result": str(boa_result) if boa_error is None else None,
                    "boa_error": str(boa_error) if boa_error else None,
                }

        return None  # No divergence found

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
                logging.error(f"  Divergence type: {baseline_divergence.get('type')}")
                if baseline_divergence.get("ivy_error"):
                    logging.error(
                        f"  Ivy error: {baseline_divergence.get('ivy_error')}"
                    )
                if baseline_divergence.get("boa_error"):
                    logging.error(
                        f"  Boa error: {baseline_divergence.get('boa_error')}"
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
                        f"diff| item {item_name} | mut#{scenario_num} | calls {len(call_schedule)} | step {divergence['step']}"
                    )
                    if divergence["type"] == "deployment":
                        logging.error(f"  Deployment divergence")
                    else:
                        logging.error(
                            f"  Execution divergence at function {divergence['function']}"
                        )

                    # Save divergence
                    self.save_divergence(divergence, item_name, scenario_num)
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
    test_filter.include_path("functional/codegen/calling_convention/test_erc20")

    # Create and run fuzzer
    fuzzer = DifferentialFuzzer()
    fuzzer.fuzz_exports(
        test_filter=test_filter, max_scenarios=10, enable_mutations=True
    )


if __name__ == "__main__":
    main()

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Add src to path for this module
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ivy.frontend.env import Env
from ivy.frontend.loader import loads
from ivy.types import Address
from fuzzer.export_utils import (
    DeploymentTrace,
    CallTrace,
    TestExport,
    TestItem,
    TestFilter,
    load_export,
    load_all_exports,
    filter_exports,
)


class TestReplay:
    """Executes test traces from Vyper test exports."""

    def __init__(self, env: Optional[Env] = None):
        self.env = env or Env.get_singleton()
        self.deployed_contracts: Dict[str, Any] = {}  # address -> contract instance
        self.executed_fixtures: Dict[str, TestItem] = {}  # fixture path -> TestItem

    def load_export(self, export_path: Union[str, Path]) -> TestExport:
        """Load test export from JSON file."""
        return load_export(export_path)

    def execute_item(self, export: TestExport, item_name: str) -> None:
        """Execute a test item and all its dependencies."""
        if item_name not in export.items:
            raise ValueError(f"Test item '{item_name}' not found in export")

        item = export.items[item_name]

        # Execute dependencies first
        for dep in item.deps:
            dep_path, dep_name = dep.rsplit("/", 1)
            dep_export_path = Path(dep_path)

            # Check if we already executed this fixture
            if dep not in self.executed_fixtures:
                dep_export = self.load_export(dep_export_path)
                self.execute_item(dep_export, dep_name)
                self.executed_fixtures[dep] = dep_export.items[dep_name]

        # Execute the item's traces
        for trace in item.traces:
            if isinstance(trace, DeploymentTrace):
                self._execute_deployment(trace)
            elif isinstance(trace, CallTrace):
                self._execute_call(trace)

    def _execute_deployment(self, trace: DeploymentTrace) -> None:
        """Execute a deployment trace."""
        # Only support source deployments
        if trace.deployment_type != "source":
            raise NotImplementedError(
                f"Only source deployments are supported, got: {trace.deployment_type}"
            )

        # Deploy from source code
        if not trace.source_code:
            raise ValueError("Source code required for source deployment")

        # Set the sender for deployment
        # Note: env.set_sender doesn't exist, we'll pass sender as a parameter
        sender = Address(trace.deployer)

        # Extract constructor args if provided
        constructor_args = None
        if trace.calldata:
            constructor_args = bytes.fromhex(trace.calldata)

        contract = loads(
            trace.source_code,
            value=trace.value,
            encoded_constructor_args=constructor_args,
            # Pass any additional source files from solc_json if needed
            input_bundle=self._create_input_bundle(trace) if trace.solc_json else None,
        )

        # Verify deployment address matches expected
        if contract.address != Address(trace.deployed_address):
            raise AssertionError(
                f"Deployment address mismatch: "
                f"expected {trace.deployed_address}, got {contract.address}"
            )

        # Store deployed contract
        self.deployed_contracts[trace.deployed_address] = contract

    def _create_input_bundle(self, trace: DeploymentTrace):
        """Create an input bundle from solc_json for module imports."""
        # TODO: Implement input bundle creation from solc_json
        # This would allow tests with imports to work
        return None

    def _execute_call(self, trace: CallTrace) -> None:
        """Execute a call trace."""
        call_args = trace.call_args

        # Get the contract
        to_address = call_args["to"]
        contract = self.deployed_contracts.get(to_address)

        if contract is None:
            raise ValueError(f"Contract at {to_address} not found")

        # Execute the call
        calldata = bytes.fromhex(call_args["calldata"])

        # Use raw_call which accepts sender parameter
        output = self.env.raw_call(
            to_address=Address(to_address),
            sender=Address(call_args["sender"]),
            calldata=calldata,
            value=call_args["value"],
            is_modifying=call_args.get("is_modifying", True),
        )

        # Verify output matches expected
        if trace.output is not None:
            expected_output = bytes.fromhex(trace.output)
            if output != expected_output:
                raise AssertionError(
                    f"Call output mismatch: expected {trace.output}, got {output.hex()}"
                )


def replay_test(
    export_path: Union[str, Path], test_name: str, env: Optional[Env] = None
) -> TestReplay:
    """Convenience function to replay a single test."""
    replay = TestReplay(env)
    export = replay.load_export(export_path)
    replay.execute_item(export, test_name)
    return replay


def validate_exports(
    exports_dir: Union[str, Path] = "tests/vyper-exports",
    test_filter: Optional[TestFilter] = None,
) -> Dict[str, bool]:
    exports = load_all_exports(exports_dir)

    if test_filter:
        exports = filter_exports(exports, test_filter=test_filter)

    results = {}

    for path, export in exports.items():
        for item_name, item in export.items.items():
            test_key = f"{path}::{item_name}"
            try:
                replay = TestReplay()
                replay.execute_item(export, item_name)
                results[test_key] = True
            except Exception as e:
                results[test_key] = False
                print(f"Failed to replay {test_key}: {e}")

    return results

from pathlib import Path
from typing import Any, Dict, Optional, Union

from ivy.frontend.env import Env
from ivy.frontend.loader import loads_from_solc_json
from ivy.types import Address
from fuzzer.export_utils import (
    DeploymentTrace,
    CallTrace,
    SetBalanceTrace,
    ClearTransientStorageTrace,
    TestExport,
    TestItem,
    TestFilter,
    load_export,
    load_all_exports,
    filter_exports,
)


class TestReplay:
    """Executes test traces from Vyper test exports."""

    def __init__(self, env: Optional[Env] = None, use_python_args: bool = False):
        self.env = env or Env.get_singleton()
        self.deployed_contracts: Dict[str, Any] = {}  # address -> contract instance
        self.executed_fixtures: Dict[str, TestItem] = {}  # fixture path -> TestItem
        self.use_python_args = use_python_args  # Flag to use python_args instead of calldata

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
            # Fix the path prefix from "tests/export/" to "tests/vyper-exports/"
            if dep_path.startswith("tests/export/"):
                dep_path = dep_path.replace("tests/export/", "tests/vyper-exports/", 1)
            dep_export_path = Path(dep_path)

            # Check if we already executed this fixture
            dep_key = f"{dep_export_path}::{dep_name}"
            if dep_key not in self.executed_fixtures:
                try:
                    # If this is the same export file, use the same instance
                    if dep_export_path == export.path:
                        dep_export = export
                    else:
                        dep_export = self.load_export(dep_export_path)
                    self.execute_item(dep_export, dep_name)
                    self.executed_fixtures[dep_key] = dep_export.items[dep_name]
                except Exception as e:
                    raise Exception(f"Failed to execute dependency {dep}: {e}") from e

        # Execute the item's traces
        for i, trace in enumerate(item.traces):
            try:
                if isinstance(trace, DeploymentTrace):
                    self._execute_deployment(trace)
                elif isinstance(trace, CallTrace):
                    self._execute_call(trace)
                elif isinstance(trace, SetBalanceTrace):
                    self._execute_set_balance(trace)
                elif isinstance(trace, ClearTransientStorageTrace):
                    self._execute_clear_transient_storage(trace)
            except Exception as e:
                raise Exception(
                    f"Failed to execute trace {i} of {item_name}: {e}"
                ) from e

    def _execute_deployment(self, trace: DeploymentTrace) -> None:
        """Execute a deployment trace."""
        # Only support source deployments
        if trace.deployment_type != "source":
            raise NotImplementedError(
                f"Only source deployments are supported, got: {trace.deployment_type}"
            )

        # Check if we have solc_json (always available according to user)
        if not trace.solc_json:
            raise ValueError("solc_json is required for deployment")

        # Set the sender for deployment
        sender = Address(trace.deployer)

        # Ensure the deployer account exists with proper balance
        # This will create the account if it doesn't exist (with nonce 0)
        deployer_balance = self.env.get_balance(sender)
        if deployer_balance < trace.value:
            # Give the deployer enough balance to deploy
            self.env.set_balance(sender, trace.value + 10**18)  # Add 1 ETH extra

        # Save current eoa and set the deployer as eoa temporarily
        original_eoa = self.env.eoa
        self.env.eoa = sender

        # Extract constructor args based on use_python_args flag
        deployment_kwargs = {
            "value": trace.value,
            "env": self.env,
        }
        
        if self.use_python_args and trace.python_args is not None:
            # Use python_args for constructor
            deployment_kwargs["constructor_args"] = trace.python_args
        elif trace.calldata:
            # Use encoded calldata
            deployment_kwargs["encoded_constructor_args"] = bytes.fromhex(trace.calldata)

        deployment_succeeded = True
        contract = None

        try:
            # Always use loads_from_solc_json since solc_json is always available
            contract = loads_from_solc_json(
                trace.solc_json,
                **deployment_kwargs
            )
        except Exception as e:
            print(f"deployment failed: {e}")
            deployment_succeeded = False
            if trace.deployment_succeeded is True:
                # Deployment was expected to succeed but failed
                raise Exception(f"Deployment failed unexpectedly: {e}") from e
            # else: deployment was expected to fail, which it did
        finally:
            # Restore original eoa
            self.env.eoa = original_eoa

        # Check if deployment success matches expected
        if (
            trace.deployment_succeeded is not None
            and trace.deployment_succeeded != deployment_succeeded
        ):
            raise AssertionError(
                f"Deployment success mismatch: expected {trace.deployment_succeeded}, got {deployment_succeeded}"
            )

        # Only store contract if deployment succeeded
        if deployment_succeeded and contract:
            # Store deployed contract by both expected and actual addresses
            self.deployed_contracts[trace.deployed_address] = contract

            # Warn about address mismatch but continue
            if contract.address != Address(trace.deployed_address):
                print(
                    f"Warning: deployment address mismatch - "
                    f"expected {trace.deployed_address}, got {contract.address}"
                )
                # Also store by actual address for cross-references
                self.deployed_contracts[str(contract.address)] = contract

    def _execute_call(self, trace: CallTrace) -> None:
        """Execute a call trace."""
        call_args = trace.call_args

        # Extract call parameters
        to_address = call_args["to"]

        # Set sender and ensure they have enough balance
        original_eoa = self.env.eoa
        self.env.eoa = Address(call_args["sender"])

        sender_balance = self.env.get_balance(Address(call_args["sender"]))
        if sender_balance < call_args["value"]:
            self.env.set_balance(
                Address(call_args["sender"]), call_args["value"] + 10**18
            )

        call_succeeded = True
        output = b""

        try:
            if self.use_python_args and trace.python_args is not None and trace.function_name is not None:
                # Use python_args - call contract method directly
                if to_address in self.deployed_contracts:
                    contract = self.deployed_contracts[to_address]
                    method = getattr(contract, trace.function_name)
                    # Call the method with python args
                    args = trace.python_args.get("args", [])
                    kwargs = trace.python_args.get("kwargs", {})
                    kwargs["value"] = call_args["value"]
                    method(*args, **kwargs)
                    # For python_args calls, we don't compare output yet
                else:
                    raise Exception(f"Contract at {to_address} not found for python_args call")
            else:
                # Use calldata
                calldata = bytes.fromhex(call_args["calldata"])
                output = self.env.message_call(
                    to_address=to_address,
                    data=calldata,
                    value=call_args["value"],
                )
        except Exception as e:
            call_succeeded = False
            if trace.call_succeeded is True:
                # Call was expected to succeed but failed
                raise Exception(f"Call failed unexpectedly: {e}") from e
            # else: call was expected to fail, which it did
        
        self.env.eoa = original_eoa

        # Check if call success matches expected
        if trace.call_succeeded is not None and trace.call_succeeded != call_succeeded:
            raise AssertionError(
                f"Call success mismatch: expected {trace.call_succeeded}, got {call_succeeded}"
            )

        # Only verify output if not using python_args
        if not self.use_python_args and call_succeeded and trace.output is not None:
            expected_output = bytes.fromhex(trace.output)
            if output != expected_output:
                raise AssertionError(
                    f"Call output mismatch: expected {trace.output}, got {output.hex()}"
                )

    def _execute_set_balance(self, trace: SetBalanceTrace) -> None:
        """Execute a set_balance trace."""
        address = Address(trace.address)
        self.env.set_balance(address, trace.value)

    def _execute_clear_transient_storage(self, _: ClearTransientStorageTrace) -> None:
        """Execute a clear_transient_storage trace."""
        self.env.clear_transient_storage()


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
    test_modes: Optional[list] = None,
) -> Dict[str, bool]:
    """Validate test exports.
    Args:
        exports_dir: Directory containing test exports
        test_filter: Optional filter to select tests
        test_modes: Optional list of (mode_name, use_python_args) tuples.
                   Defaults to [("calldata", False), ("python_args", True)]
    """
    exports = load_all_exports(exports_dir)

    if test_filter:
        exports = filter_exports(exports, test_filter=test_filter)

    results = {}
    
    # Default to running both modes
    if test_modes is None:
        test_modes = [("calldata", False), ("python_args", True)]

    for path, export in exports.items():
        # Process each test independently with its own environment
        for item_name, item in export.items.items():
            # Skip fixtures when processing top-level items (they'll be executed as dependencies)
            if item.item_type == "fixture":
                continue

            for mode_name, use_python_args_flag in test_modes:
                test_key = f"{path}::{item_name}::{mode_name}"

                try:
                    # Create fresh environment for each test
                    env = Env()
                    replay = TestReplay(env, use_python_args=use_python_args_flag)

                    # Execute the test and all its dependencies in an anchored context
                    # This ensures complete isolation between tests
                    with env.anchor():
                        replay.execute_item(export, item_name)

                    results[test_key] = True
                except Exception as e:
                    results[test_key] = False
                    print(f"Failed to replay {test_key}: {e}")
                    # Add more debugging for specific error
                    if "encoded_constructor_args" in str(e):
                        print(f"  Item type: {item.item_type}")
                        print(f"  Deps: {item.deps}")
                        if item.traces and hasattr(item.traces[0], "deployment_type"):
                            print(f"  Deployment type: {item.traces[0].deployment_type}")

    return results


def test_replay_exports():
    test_filter = TestFilter(exclude_multi_module=False)
    test_filter.include_path(r"functional/codegen/")

    # ---- unsupported features
    test_filter.exclude_source(r"pragma nonreentrancy")
    test_filter.exclude_source(r"import math")
    test_filter.exclude_source(r"raw_log")
    test_filter.exclude_source(r"selfdestruct")
    test_filter.exclude_source(r"gas=")
    test_filter.exclude_name("test_tx_gasprice")
    test_filter.exclude_name("test_blockhash")
    test_filter.exclude_name("test_blobbasefee")
    test_filter.exclude_name("test_block_number")
    test_filter.exclude_name("test_gas_call")
    test_filter.exclude_name("test_mana")
    # ---- unsupported features

    results = validate_exports("tests/vyper-exports", test_filter=test_filter)

    # Report summary
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    print(f"\nSummary: {passed} passed, {failed} failed out of {len(results)} tests")

    assert all(results.values()), f"{failed} tests failed"

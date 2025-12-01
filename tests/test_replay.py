from pathlib import Path
from typing import Any, Dict, Optional, Union

import pytest

from fuzzer.runner.base_scenario_runner import ScenarioResult
from fuzzer.export_utils import (
    TestExport,
    TestFilter,
    load_export,
    load_all_exports,
    filter_exports,
)
from fuzzer.runner.scenario import Scenario, create_scenario_from_export
from fuzzer.runner.ivy_scenario_runner import IvyScenarioRunner


class TestReplay:
    """Executes test traces from Vyper test exports using the scenario runner."""

    def __init__(self, use_python_args: bool = False):
        self.runner = IvyScenarioRunner()
        self.use_python_args = use_python_args

    def load_export(self, export_path: Union[str, Path]) -> TestExport:
        """Load test export from JSON file."""
        return load_export(export_path)

    def validate_result(self, scenario: Scenario, result: ScenarioResult) -> None:
        """Validate execution result matches expected outcomes."""
        # Get all traces to execute (respects mutations if any)
        traces = scenario.get_traces_to_execute()

        # Validate each trace result
        for trace_result in result.results:
            trace = traces[trace_result.trace_index]

            if trace_result.trace_type == "deployment":
                deployment_trace = trace
                deployment_result = trace_result.result

                # Validate deployment success matches expected
                if deployment_trace.deployment_succeeded != deployment_result.success:
                    raise AssertionError(
                        f"Deployment success mismatch: expected {deployment_trace.deployment_succeeded}, "
                        f"got {deployment_result.success}"
                    )

                # Validate deployed address
                if deployment_result.success:
                    # The deployment_result.contract is the VyperContract object
                    # Use the address property to get the actual address
                    contract = deployment_result.contract
                    actual_address_str = str(contract.address)
                else:
                    # Failed deployments should have zero address
                    actual_address_str = "0x0000000000000000000000000000000000000000"

                if actual_address_str != deployment_trace.deployed_address:
                    raise AssertionError(
                        f"Deployed address mismatch: expected {deployment_trace.deployed_address}, "
                        f"got {actual_address_str}"
                    )

            elif trace_result.trace_type == "call":
                call_trace = trace
                call_result = trace_result.result

                # Check if call success matches expected (only if call_succeeded is specified)
                if call_trace.call_succeeded is not None:
                    if call_trace.call_succeeded != call_result.success:
                        raise AssertionError(
                            f"Call success mismatch: expected {call_trace.call_succeeded}, "
                            f"got {call_result.success}"
                        )

                # Only verify output if not using python_args and output is specified
                if not self.use_python_args and call_trace.output is not None:
                    if call_result.success:
                        expected_output = bytes.fromhex(call_trace.output)
                        if call_result.output != expected_output:
                            raise AssertionError(
                                f"Call output mismatch: expected {call_trace.output}, "
                                f"got {call_result.output.hex()}"
                            )
                    # Note: We don't check output for failed calls as it may be undefined

    def execute_item(self, export: TestExport, item_name: str) -> None:
        """Execute a test item and validate results."""
        # Create scenario from test item using the shared utility
        scenario = create_scenario_from_export(export, item_name, self.use_python_args)

        # Run scenario
        result = self.runner.run(scenario)

        # Validate results match expectations
        self.validate_result(scenario, result)


def replay_test(export_path: Union[str, Path], test_name: str) -> TestReplay:
    """Convenience function to replay a single test."""
    replay = TestReplay(use_python_args=True)
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
    print("Loading exports...")
    exports = load_all_exports(exports_dir)
    print(f"Loaded {len(exports)} export files")

    if test_filter:
        print("Filtering exports...")
        exports = filter_exports(exports, test_filter=test_filter)
        print(f"After filtering: {len(exports)} export files")

    results = {}

    # Default to running both modes
    if test_modes is None:
        test_modes = [("calldata", False), ("python_args", True)]

    test_count = 0
    for path, export in exports.items():
        # Process each test independently
        for item_name, item in export.items.items():
            # Skip fixtures when processing top-level items (they'll be executed as dependencies)
            if item.item_type == "fixture":
                continue

            for mode_name, use_python_args_flag in test_modes:
                test_key = f"{path}::{item_name}::{mode_name}"
                test_count += 1
                if test_count % 100 == 0:
                    print(f"Processing test {test_count}...")

                try:
                    # Create replay with the appropriate mode
                    replay = TestReplay(use_python_args=use_python_args_flag)

                    # Execute the test
                    replay.execute_item(export, item_name)

                    results[test_key] = True
                except Exception as e:
                    results[test_key] = False
                    print(f"Failed to replay {test_key}: {e}")

    return results


def test_replay_exports():
    print("Starting test_replay_exports...")
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

    # Only run python_args mode for now
    test_modes = [("python_args", True)]
    results = validate_exports(
        "tests/vyper-exports", test_filter=test_filter, test_modes=test_modes
    )

    # Report summary
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    print(f"\nSummary: {passed} passed, {failed} failed out of {len(results)} tests")

    assert all(results.values()), f"{failed} tests failed"

def get_replay_test_cases():
    test_filter = get_replay_test_filter()
    exports = load_all_exports("tests/vyper-exports")
    exports = filter_exports(exports, test_filter=test_filter)

    cases = []
    for path, export in exports.items():
        for item_name, item in export.items.items():
            if item.item_type == "fixture":
                continue
            test_id = f"{Path(path).stem}::{item_name}"
            cases.append(pytest.param(export, item_name, id=test_id))
    return cases


@pytest.mark.parametrize("export,item_name", get_replay_test_cases())
def test_replay(export, item_name):
    replay = TestReplay(use_python_args=True)
    replay.execute_item(export, item_name)

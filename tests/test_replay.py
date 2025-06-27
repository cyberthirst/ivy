from pathlib import Path
from typing import Any, Dict, Optional, Union

from fuzzer.export_utils import (
    CallTrace,
    TestExport,
    TestFilter,
    load_export,
    load_all_exports,
    filter_exports,
)
from fuzzer.scenario import Scenario, create_scenario_from_export
from fuzzer.ivy_scenario_runner import IvyScenarioRunner


class TestReplay:
    """Executes test traces from Vyper test exports using the scenario runner."""

    def __init__(self, use_python_args: bool = False):
        self.runner = IvyScenarioRunner()
        self.use_python_args = use_python_args

    def load_export(self, export_path: Union[str, Path]) -> TestExport:
        """Load test export from JSON file."""
        return load_export(export_path)


    def validate_result(self, scenario: Scenario, result) -> None:
        """Validate execution result matches expected outcomes."""
        # Check deployment result
        if scenario.deployment_trace and scenario.deployment_trace.deployment_succeeded is not None:
            if scenario.deployment_trace.deployment_succeeded and not result.deployment.success:
                raise Exception(f"Deployment failed unexpectedly: {result.deployment.error}")
            elif not scenario.deployment_trace.deployment_succeeded and result.deployment.success:
                raise AssertionError(
                    f"Deployment success mismatch: expected {scenario.deployment_trace.deployment_succeeded}, "
                    f"got {result.deployment.success}"
                )

        # Check call results
        call_idx = 0
        for trace in scenario.traces:
            if isinstance(trace, CallTrace):
                if call_idx >= len(result.calls):
                    break
                    
                call_result = result.calls[call_idx]
                call_idx += 1

                # Check if call success matches expected
                if trace.call_succeeded is not None:
                    if trace.call_succeeded and not call_result.success:
                        raise Exception(f"Call failed unexpectedly: {call_result.error}")
                    elif not trace.call_succeeded and call_result.success:
                        raise AssertionError(
                            f"Call success mismatch: expected {trace.call_succeeded}, "
                            f"got {call_result.success}"
                        )

                # Only verify output if not using python_args
                if not self.use_python_args and call_result.success and trace.output is not None:
                    expected_output = bytes.fromhex(trace.output)
                    if call_result.output != expected_output:
                        raise AssertionError(
                            f"Call output mismatch: expected {trace.output}, "
                            f"got {call_result.output.hex()}"
                        )

    def execute_item(self, export: TestExport, item_name: str) -> None:
        """Execute a test item and validate results."""
        # Create scenario from test item using the shared utility
        scenario = create_scenario_from_export(export, item_name, self.use_python_args)

        # Run scenario
        result = self.runner.run(scenario)

        # Validate results match expectations
        self.validate_result(scenario, result)


def replay_test(
    export_path: Union[str, Path], test_name: str
) -> TestReplay:
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
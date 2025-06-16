#!/usr/bin/env python3
"""
Example script demonstrating differential testing between Ivy and Vyper compiler.

This shows how to:
1. Load Vyper test exports
2. Apply filters to select specific tests
3. Execute test traces within Ivy interpreter
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fuzzer.fuzz import DifferentialFuzzer
from fuzzer.export_utils import (
    TestFilter,
    CallTrace,
    filter_exports,
    extract_test_cases,
)


def example_basic_fuzzing():
    """Basic example: Run fuzzer with default settings."""
    print("=== Basic Fuzzing Example ===")

    fuzzer = DifferentialFuzzer()

    # Create a simple filter to skip problematic tests
    test_filter = TestFilter()
    test_filter.exclude_path(r"test_raw_call")  # Skip raw call tests
    test_filter.exclude_source(r"@nonreentrant")  # Skip nonreentrant tests

    # Run fuzzing with mutations disabled first
    print("\n1. Testing without mutations (baseline check)...")
    fuzzer.fuzz_exports(test_filter=test_filter, enable_mutations=False)

    print("\n2. Testing with mutations...")
    fuzzer.fuzz_exports(
        test_filter=test_filter, max_mutations_per_test=3, enable_mutations=True
    )


def example_targeted_fuzzing():
    """Example: Target specific types of contracts."""
    print("\n=== Targeted Fuzzing Example ===")

    fuzzer = DifferentialFuzzer()

    # Create filter that only includes contracts with specific features
    test_filter = TestFilter()
    test_filter.include_source(
        r"def transfer\("
    )  # Only test contracts with transfer function
    test_filter.exclude_source(r"raw_log\(")  # But exclude those with raw logs

    print("Fuzzing only contracts with transfer functions...")
    fuzzer.fuzz_exports(
        test_filter=test_filter, max_mutations_per_test=5, enable_mutations=True
    )


def example_validate_exports():
    """Example: Validate that exports can be replayed correctly."""
    print("\n=== Export Validation Example ===")

    # Import here to avoid circular imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from tests.test_replay import validate_exports

    # Create a filter for specific tests
    test_filter = TestFilter()
    test_filter.include_path(r"tokens/test_erc20")  # Only ERC20 tests

    print("Validating ERC20 test exports...")
    results = validate_exports("tests/vyper-exports", test_filter=test_filter)

    total = len(results)
    successful = sum(1 for success in results.values() if success)

    print(f"\nValidation Results:")
    print(f"  Total tests: {total}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {total - successful}")

    # Show some failed tests if any
    failed_tests = [test for test, success in results.items() if not success]
    if failed_tests:
        print(f"\nFirst 5 failed tests:")
        for test in failed_tests[:5]:
            print(f"  - {test}")


def example_custom_filter():
    """Example: Create custom filtering logic."""
    print("\n=== Custom Filter Example ===")

    fuzzer = DifferentialFuzzer()

    # Custom filter function that selects tests based on complexity
    def complex_contracts_only(item, path):
        # Skip tests with less than 3 traces
        if len(item.traces) < 3:
            return True

        # Skip tests without any call traces
        has_calls = any(isinstance(trace, CallTrace) for trace in item.traces)
        return not has_calls

    # Run with custom filter
    exports = fuzzer.load_filtered_exports()
    filtered = filter_exports(exports, filter_fn=complex_contracts_only)

    print(f"Filtered to {sum(len(e.items) for e in filtered.values())} complex tests")

    # Extract and fuzz the filtered tests
    test_cases = extract_test_cases(filtered)
    print(f"Running differential testing on {len(test_cases)} test cases...")

    for source, calldatas in test_cases[:5]:  # Just run first 5
        fuzzer.compare_runs(source, calldatas)


def main():
    """Run all examples."""
    print("Differential Testing Examples")
    print("=" * 50)

    # Check if test exports exist
    exports_dir = Path("tests/vyper-exports")
    if not exports_dir.exists():
        print(f"Error: Test exports directory not found at {exports_dir}")
        print("Please ensure Vyper test exports are available.")
        return

    # Run examples
    example_validate_exports()
    example_basic_fuzzing()
    example_targeted_fuzzing()
    example_custom_filter()

    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    main()

"""
Utilities for working with Vyper test exports.

This module provides data structures and functions for loading,
filtering, and extracting test cases from Vyper test exports.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field


@dataclass
class DeploymentTrace:
    """Represents a contract deployment trace."""

    deployer: str
    deployment_type: str  # "source", "ir", "blueprint", "raw_bytecode"
    contract_abi: List[Dict[str, Any]]
    initcode: str
    calldata: Optional[str]
    value: int
    source_code: Optional[str]
    annotated_ast: Optional[Dict[str, Any]]
    solc_json: Optional[Dict[str, Any]]
    raw_ir: Optional[str]
    blueprint_initcode_prefix: Optional[str]
    deployed_address: str
    runtime_bytecode: str
    deployment_succeeded: Optional[bool] = None


@dataclass
class CallTrace:
    """Represents a function call trace."""

    output: Optional[str]
    call_args: Dict[str, Any]
    call_succeeded: Optional[bool] = None


@dataclass
class SetBalanceTrace:
    """Represents a set_balance trace."""

    address: str
    value: int


@dataclass
class ClearTransientStorageTrace:
    """Represents a clear_transient_storage trace."""

    # No fields needed for this trace type


@dataclass
class TestItem:
    """Represents a test or fixture with its traces."""

    name: str
    item_type: str  # "test" or "fixture"
    deps: List[str]
    traces: List[
        Union[DeploymentTrace, CallTrace, SetBalanceTrace, ClearTransientStorageTrace]
    ]


@dataclass
class TestExport:
    """Container for all test exports from a file."""

    path: Path
    items: Dict[str, TestItem] = field(default_factory=dict)


class TestFilter:
    """Filters for selecting which tests to use."""

    def __init__(self, exclude_multi_module: bool = False):
        self.path_includes: List[Union[str, re.Pattern]] = []
        self.path_excludes: List[Union[str, re.Pattern]] = []
        self.source_excludes: List[Union[str, re.Pattern]] = []
        self.source_includes: List[Union[str, re.Pattern]] = []
        self.name_includes: List[Union[str, re.Pattern]] = []
        self.name_excludes: List[Union[str, re.Pattern]] = []
        self.exclude_multi_module = exclude_multi_module

    def include_path(self, pattern: Union[str, re.Pattern]) -> "TestFilter":
        """Only include tests from paths matching the pattern."""
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
        self.path_includes.append(pattern)
        return self

    def exclude_path(self, pattern: Union[str, re.Pattern]) -> "TestFilter":
        """Exclude tests from paths matching the pattern."""
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
        self.path_excludes.append(pattern)
        return self

    def exclude_source(self, pattern: Union[str, re.Pattern]) -> "TestFilter":
        """Exclude tests with source code matching the pattern."""
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
        self.source_excludes.append(pattern)
        return self

    def include_source(self, pattern: Union[str, re.Pattern]) -> "TestFilter":
        """Only include tests with source code matching the pattern."""
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
        self.source_includes.append(pattern)
        return self

    def include_name(self, pattern: Union[str, re.Pattern]) -> "TestFilter":
        """Only include tests with names matching the pattern."""
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
        self.name_includes.append(pattern)
        return self

    def exclude_name(self, pattern: Union[str, re.Pattern]) -> "TestFilter":
        """Exclude tests with names matching the pattern."""
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
        self.name_excludes.append(pattern)
        return self

    def should_skip_path(self, path: Path) -> bool:
        """Check if a path should be skipped."""
        path_str = str(path)

        # If includes are specified, path must match at least one
        if self.path_includes:
            matched = False
            for pattern in self.path_includes:
                if pattern.search(path_str):
                    matched = True
                    break
            if not matched:
                return True

        # Check excludes (these override includes)
        for pattern in self.path_excludes:
            if pattern.search(path_str):
                return True

        return False

    def should_skip_item(
        self,
        item: TestItem,
        export_path: Path,
        export: Optional["TestExport"] = None,
        checked_deps: Optional[set] = None,
    ) -> bool:
        """Check if a test item should be skipped, including checking all dependencies."""
        if checked_deps is None:
            checked_deps = set()

        # Avoid infinite recursion
        dep_key = f"{export_path}::{item.name}"
        if dep_key in checked_deps:
            return False
        checked_deps.add(dep_key)

        # Check path filters
        if self.should_skip_path(export_path):
            return True

        # Check name filters
        # If includes are specified, name must match at least one
        if self.name_includes:
            matched = False
            for pattern in self.name_includes:
                if pattern.search(item.name):
                    matched = True
                    break
            if not matched:
                return True

        # Check excludes (these override includes)
        for pattern in self.name_excludes:
            if pattern.search(item.name):
                return True

        # Check source code filters
        for trace in item.traces:
            if isinstance(trace, DeploymentTrace):
                # Check if multi-module contracts should be excluded
                if (
                    self.exclude_multi_module
                    and trace.solc_json
                    and "sources" in trace.solc_json
                ):
                    sources = trace.solc_json["sources"]
                    if len(sources) > 1:
                        # This contract has multiple modules/imports
                        return True

                # Check source code filters
                if trace.source_code:
                    # Check excludes
                    for pattern in self.source_excludes:
                        if pattern.search(trace.source_code):
                            return True

                    # Check includes (if any specified, must match at least one)
                    if self.source_includes:
                        matched = False
                        for pattern in self.source_includes:
                            if pattern.search(trace.source_code):
                                matched = True
                                break
                        if not matched:
                            return True

        # Check dependencies recursively
        for dep in item.deps:
            dep_path_str, dep_name = dep.rsplit("/", 1)
            # Fix the path prefix from "tests/export/" to "tests/vyper-exports/"
            if dep_path_str.startswith("tests/export/"):
                dep_path_str = dep_path_str.replace(
                    "tests/export/", "tests/vyper-exports/", 1
                )
            dep_path = Path(dep_path_str)

            # Load dependency export if needed
            if export and dep_path == export_path:
                # Dependency is in the same export file
                if dep_name in export.items:
                    dep_item = export.items[dep_name]
                    if self.should_skip_item(dep_item, dep_path, export, checked_deps):
                        return True
            else:
                # Need to load the dependency export file
                try:
                    dep_export = load_export(dep_path)
                    if dep_name in dep_export.items:
                        dep_item = dep_export.items[dep_name]
                        if self.should_skip_item(
                            dep_item, dep_path, dep_export, checked_deps
                        ):
                            return True
                except Exception:
                    # If we can't load the dependency, skip this test
                    return True

        return False


def load_export(export_path: Union[str, Path]) -> TestExport:
    """Load test export from JSON file."""
    path = Path(export_path)
    with open(path, "r") as f:
        data = json.load(f)

    export = TestExport(path=path)

    for name, item_data in data.items():
        traces = []
        for trace_data in item_data["traces"]:
            if trace_data["trace_type"] == "deployment":
                trace = DeploymentTrace(
                    deployer=trace_data["deployer"],
                    deployment_type=trace_data["deployment_type"],
                    contract_abi=trace_data["contract_abi"],
                    initcode=trace_data["initcode"],
                    calldata=trace_data.get("calldata"),
                    value=trace_data["value"],
                    source_code=trace_data.get("source_code"),
                    annotated_ast=trace_data.get("annotated_ast"),
                    solc_json=trace_data.get("solc_json"),
                    raw_ir=trace_data.get("raw_ir"),
                    blueprint_initcode_prefix=trace_data.get(
                        "blueprint_initcode_prefix"
                    ),
                    deployed_address=trace_data["deployed_address"],
                    runtime_bytecode=trace_data["runtime_bytecode"],
                    deployment_succeeded=trace_data.get("deployment_succeeded"),
                )
            elif trace_data["trace_type"] == "call":
                trace = CallTrace(
                    output=trace_data.get("output"),
                    call_args=trace_data["call_args"],
                    call_succeeded=trace_data.get("call_succeeded"),
                )
            elif trace_data["trace_type"] == "set_balance":
                trace = SetBalanceTrace(
                    address=trace_data["address"],
                    value=trace_data["value"],
                )
            elif trace_data["trace_type"] == "clear_transient_storage":
                trace = ClearTransientStorageTrace()
            else:
                raise ValueError(f"Unknown trace type: {trace_data['trace_type']}")

            traces.append(trace)

        item = TestItem(
            name=name,
            item_type=item_data["item_type"],
            deps=item_data["deps"],
            traces=traces,
        )
        export.items[name] = item

    return export


def load_all_exports(
    exports_dir: Union[str, Path] = "tests/vyper-exports",
) -> Dict[Path, TestExport]:
    """Load all test exports from a directory."""
    exports_dir = Path(exports_dir)

    if not exports_dir.is_absolute():
        exports_dir = exports_dir.absolute()

    exports = {}

    for json_file in exports_dir.rglob("*.json"):
        try:
            export = load_export(json_file)
            exports[json_file] = export
        except Exception as e:
            print(f"Failed to load {json_file}: {e}")

    return exports


def filter_exports(
    exports: Dict[Path, TestExport],
    filter_fn: Optional[Callable[[TestItem, Path], bool]] = None,
    test_filter: Optional[TestFilter] = None,
) -> Dict[Path, TestExport]:
    """Filter test exports based on criteria."""
    filtered = {}

    for path, export in exports.items():
        filtered_export = TestExport(path=export.path)

        for name, item in export.items.items():
            # Apply custom filter function
            if filter_fn and filter_fn(item, path):
                continue

            # Apply test filter
            if test_filter and test_filter.should_skip_item(item, path):
                continue

            # Check if test is source-only
            has_non_source = any(
                isinstance(t, DeploymentTrace) and t.deployment_type != "source"
                for t in item.traces
            )

            if not has_non_source:
                filtered_export.items[name] = item

        if filtered_export.items:
            filtered[path] = filtered_export

    return filtered


def extract_test_cases(exports: Dict[Path, TestExport]) -> List[tuple[str, List[str]]]:
    """Extract (source_code, calldatas) pairs from test exports."""
    test_cases = []

    for path, export in exports.items():
        for item_name, item in export.items.items():
            # Group traces by deployment
            current_source = None
            current_calldatas = []

            for trace in item.traces:
                if isinstance(trace, DeploymentTrace):
                    # If we have accumulated calldatas, save the previous test case
                    if current_source and current_calldatas:
                        test_cases.append((current_source, current_calldatas))

                    # Start new test case
                    if trace.deployment_type == "source" and trace.source_code:
                        current_source = trace.source_code
                        current_calldatas = []
                    else:
                        current_source = None

                elif isinstance(trace, CallTrace) and current_source:
                    # Add calldata from this call
                    calldata = trace.call_args.get("calldata", "")
                    if calldata:
                        current_calldatas.append(calldata)

            # Don't forget the last accumulated test case
            if current_source and current_calldatas:
                test_cases.append((current_source, current_calldatas))

    return test_cases

"""
Enhanced scenario structure for unified execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

from ..trace_types import Trace, TestItem, TestExport
from ..export_utils import load_export


@dataclass
class Scenario:
    """
    Scenario that can represent both fuzzer mutations and replay traces.

    This structure stores all traces (deployments, calls, etc.) in execution order
    and handles mutations uniformly for any trace type.
    """

    traces: List[Trace] = field(default_factory=list)
    mutated_traces: Optional[List[Trace]] = None

    # Dependencies as fully-resolved Scenario objects (executed depth-first)
    dependencies: List[Scenario] = field(default_factory=list)

    # Unique identifier for deduplication during execution (e.g., "path::item_name")
    scenario_id: Optional[str] = None

    # Configuration
    use_python_args: bool = True  # Default to python args for fuzzing

    def active_traces(self) -> List[Trace]:
        """
        Get the active traces (mutated if available, otherwise original).

        Used for both execution and as the base for further mutations.
        """
        if self.mutated_traces is not None:
            return self.mutated_traces
        return self.traces


def _fixup_dep_path(dep_path_str: str) -> Path:
    """Fix path prefix from 'tests/export/' to 'tests/vyper-exports/'."""
    if dep_path_str.startswith("tests/export/"):
        dep_path_str = dep_path_str.replace("tests/export/", "tests/vyper-exports/", 1)
    return Path(dep_path_str)


def _load_dependency_scenario(
    dep_path: Path,
    dep_item_name: str,
    use_python_args: bool,
) -> Scenario:
    """Load a dependency from disk and create a Scenario."""
    dep_export = load_export(dep_path)

    if dep_item_name not in dep_export.items:
        raise ValueError(f"Dependency {dep_item_name} not found in {dep_path}")

    scenario_id = f"{dep_path}::{dep_item_name}"
    return create_scenario_from_item(
        dep_export.items[dep_item_name],
        use_python_args,
        scenario_id=scenario_id,
    )


def create_scenario_from_item(
    item: TestItem,
    use_python_args: bool = True,
    scenario_id: Optional[str] = None,
) -> Scenario:
    """Create a Scenario from a test item, recursively loading all dependencies."""
    dependencies = []
    for dep in item.deps:
        dep_path_str, dep_name = dep.rsplit("/", 1)
        dep_path = _fixup_dep_path(dep_path_str)
        dep_scenario = _load_dependency_scenario(dep_path, dep_name, use_python_args)
        dependencies.append(dep_scenario)

    return Scenario(
        traces=item.traces,
        dependencies=dependencies,
        scenario_id=scenario_id,
        use_python_args=use_python_args,
    )


def create_scenario_from_export(
    export: TestExport,
    item_name: str,
    use_python_args: bool = True,
) -> Scenario:
    """
    Create a Scenario from a test export and item name.

    Convenience function that looks up the item and creates a scenario.
    """
    if item_name not in export.items:
        raise ValueError(f"Test item '{item_name}' not found in export")

    item = export.items[item_name]
    scenario_id = f"{export.path}::{item_name}"
    return create_scenario_from_item(item, use_python_args, scenario_id=scenario_id)

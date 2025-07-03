"""
Enhanced scenario structure for unified execution.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Tuple
from pathlib import Path

from ..export_utils import (
    DeploymentTrace,
    CallTrace,
    SetBalanceTrace,
    ClearTransientStorageTrace,
    TestItem,
    TestExport,
)


@dataclass
class Scenario:
    """
    Scenario that can represent both fuzzer mutations and replay traces.

    This structure stores all traces (deployments, calls, etc.) in execution order
    and handles mutations uniformly for any trace type.
    """

    # All traces in execution order (deployments, calls, set_balance, clear_transient)
    traces: List[
        Union[DeploymentTrace, CallTrace, SetBalanceTrace, ClearTransientStorageTrace]
    ] = field(default_factory=list)

    # Mutated traces (if traces were mutated, this replaces the original traces entirely)
    # This preserves the exact order of execution
    mutated_traces: Optional[
        List[
            Union[
                DeploymentTrace, CallTrace, SetBalanceTrace, ClearTransientStorageTrace
            ]
        ]
    ] = None

    # Dependencies to execute first
    dependencies: List[tuple[Path, str]] = field(
        default_factory=list
    )  # (export_path, item_name)

    # Configuration
    use_python_args: bool = True  # Default to python args for fuzzing

    def get_traces_to_execute(
        self,
    ) -> List[
        Union[CallTrace, SetBalanceTrace, ClearTransientStorageTrace, DeploymentTrace]
    ]:
        """
        Get the traces to execute, using mutated traces if available.

        Returns either the mutated traces (if mutations were applied) or the original traces.
        The order is always preserved exactly as specified.
        """
        if self.mutated_traces is not None:
            return self.mutated_traces
        return self.traces


def build_dependencies_from_item(item: TestItem) -> List[Tuple[Path, str]]:
    """
    Build list of dependencies from a test item.

    Handles path fixup from "tests/export/" to "tests/vyper-exports/".
    """
    dependencies = []

    for dep in item.deps:
        dep_path_str, dep_name = dep.rsplit("/", 1)
        # Fix the path prefix from "tests/export/" to "tests/vyper-exports/"
        if dep_path_str.startswith("tests/export/"):
            dep_path_str = dep_path_str.replace(
                "tests/export/", "tests/vyper-exports/", 1
            )
        dep_path = Path(dep_path_str)
        dependencies.append((dep_path, dep_name))

    return dependencies


def create_scenario_from_item(
    item: TestItem,
    use_python_args: bool = True,
) -> Scenario:
    """
    Create a Scenario from a test item.

    This is the base scenario creation without any mutations.
    Used by both test_replay (for exact replay) and fuzzer (as base for mutations).

    Args:
        item: The test item to convert
        use_python_args: Whether to use python args (True) or calldata (False)

    Returns:
        A Scenario ready for execution
    """
    # Build dependencies
    dependencies = build_dependencies_from_item(item)

    # Create scenario with all traces in execution order
    return Scenario(
        traces=item.traces,
        dependencies=dependencies,
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

    Args:
        export: The test export containing the item
        item_name: Name of the test item to convert
        use_python_args: Whether to use python args (True) or calldata (False)

    Returns:
        A Scenario ready for execution

    Raises:
        ValueError: If the item is not found in the export
    """
    if item_name not in export.items:
        raise ValueError(f"Test item '{item_name}' not found in export")

    item = export.items[item_name]
    return create_scenario_from_item(item, use_python_args)

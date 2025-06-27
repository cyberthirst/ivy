"""
Enhanced scenario structure for unified execution.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path

from .export_utils import (
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

    This structure stores the original traces and any mutations separately
    to minimize memory usage during long fuzzing campaigns.
    """

    # Deployment info (optional - might only exist in dependencies)
    deployment_trace: Optional[DeploymentTrace] = None
    mutated_source: Optional[str] = None  # Only if source was mutated
    mutated_deploy_args: Optional[List[Any]] = None  # Only if args were mutated
    mutated_deploy_kwargs: Optional[Dict[str, Any]] = (
        None  # Only if kwargs were mutated
    )

    # Sequential traces (includes calls, set_balance, clear_transient, and additional deployments)
    traces: List[Union[CallTrace, SetBalanceTrace, ClearTransientStorageTrace, DeploymentTrace]] = field(
        default_factory=list
    )

    # Mutated traces (if traces were mutated, this replaces the original traces entirely)
    # This preserves the exact order of execution
    mutated_traces: Optional[
        List[Union[CallTrace, SetBalanceTrace, ClearTransientStorageTrace, DeploymentTrace]]
    ] = None

    # Dependencies to execute first
    dependencies: List[tuple[Path, str]] = field(
        default_factory=list
    )  # (export_path, item_name)

    # Configuration
    use_python_args: bool = True  # Default to python args for fuzzing

    def get_traces_to_execute(
        self,
    ) -> List[Union[CallTrace, SetBalanceTrace, ClearTransientStorageTrace, DeploymentTrace]]:
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


def find_deployment_trace(item: TestItem) -> Optional[DeploymentTrace]:
    """
    Find the first deployment trace in a test item.
    
    Only returns source deployments (not create2 deployments).
    """
    for trace in item.traces:
        if isinstance(trace, DeploymentTrace) and trace.deployment_type == "source":
            return trace
    return None


def extract_non_deployment_traces(
    item: TestItem,
) -> List[Union[CallTrace, SetBalanceTrace, ClearTransientStorageTrace, DeploymentTrace]]:
    """
    Extract all traces after the first deployment.
    
    The first deployment (if any) is treated as the "primary" deployment for the scenario.
    All subsequent traces, including additional deployments, are returned in order.
    
    This separation allows the fuzzer to mutate the primary deployment's source/args
    while preserving the exact execution order of all traces.
    """
    non_deployment_traces = []
    deployment_found = False
    
    for trace in item.traces:
        if isinstance(trace, DeploymentTrace) and not deployment_found:
            # Skip the first deployment trace
            deployment_found = True
        else:
            non_deployment_traces.append(trace)
    
    return non_deployment_traces


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
    
    # Find deployment trace (if any)
    deployment_trace = find_deployment_trace(item)
    
    # Extract other traces
    other_traces = extract_non_deployment_traces(item)
    
    # Create scenario
    return Scenario(
        deployment_trace=deployment_trace,
        traces=other_traces,
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

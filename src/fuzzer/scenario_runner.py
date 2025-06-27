"""
Scenario runner that handles dependencies and all trace types.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
from pathlib import Path

from .scenario import Scenario
from .runner import ScenarioResult, DeploymentResult, CallResult
from .trace_executor import (
    ExecutionEnvironment,
    execute_deployment,
    execute_call,
    execute_set_balance,
    execute_clear_transient_storage,
)
from .export_utils import (
    DeploymentTrace,
    CallTrace,
    SetBalanceTrace,
    ClearTransientStorageTrace,
    load_export,
)


class BaseScenarioRunner(ABC):
    """Base class for scenario runners that handle all trace types and dependencies."""

    def __init__(self, collect_storage_dumps: bool = False):
        self.env: Optional[ExecutionEnvironment] = None
        self.deployed_contracts: Dict[str, Any] = {}
        self.executed_dependencies: Set[str] = set()
        self.collect_storage_dumps = collect_storage_dumps

    @abstractmethod
    def create_environment(self) -> ExecutionEnvironment:
        """Create the execution environment (Ivy or Boa)."""
        pass

    def run(self, scenario: Scenario) -> ScenarioResult:
        """Run a complete scenario including dependencies."""
        # Create fresh environment
        self.env = self.create_environment()
        self.deployed_contracts = {}
        self.executed_dependencies = set()  # Reset for each scenario

        # Execute dependencies first
        for dep_path, dep_item_name in scenario.dependencies:
            self._execute_dependency(dep_path, dep_item_name)

        # Check if we have a deployment trace in the scenario
        if scenario.deployment_trace:
            # Deploy the main contract
            deployment_result = execute_deployment(
                trace=scenario.deployment_trace,
                env=self.env,
                deployed_contracts=self.deployed_contracts,
                mutated_source=scenario.mutated_source,
                mutated_args=scenario.mutated_deploy_args,
                mutated_kwargs=scenario.mutated_deploy_kwargs,
                use_python_args=scenario.use_python_args,
                collect_storage_dump=self.collect_storage_dumps,
            )

            # If deployment failed, return early
            if not deployment_result.success:
                return ScenarioResult(deployment=deployment_result)
        else:
            # No deployment in main scenario (deployment might be in dependencies)
            deployment_result = DeploymentResult(success=True)

        # Execute all traces in order
        call_results = []
        traces_to_execute = scenario.get_traces_to_execute()

        for trace in traces_to_execute:
            if isinstance(trace, DeploymentTrace):
                # Execute additional deployment
                deploy_result = execute_deployment(
                    trace=trace,
                    env=self.env,
                    deployed_contracts=self.deployed_contracts,
                    use_python_args=scenario.use_python_args,
                    collect_storage_dump=self.collect_storage_dumps,
                )
                # Continue even if deployment fails to match test behavior
                
            elif isinstance(trace, CallTrace):
                # Execute call and record result
                call_result = execute_call(
                    trace=trace,
                    env=self.env,
                    deployed_contracts=self.deployed_contracts,
                    use_python_args=scenario.use_python_args,
                    collect_storage_dump=self.collect_storage_dumps,
                )
                call_results.append(call_result)

                # Stop if call failed (for consistency with original runner)
                if not call_result.success:
                    break

            elif isinstance(trace, SetBalanceTrace):
                # Execute set_balance (no result to record)
                execute_set_balance(trace, self.env)

            elif isinstance(trace, ClearTransientStorageTrace):
                # Execute clear_transient_storage (no result to record)
                execute_clear_transient_storage(trace, self.env)

        return ScenarioResult(deployment=deployment_result, calls=call_results)

    def _execute_dependency(self, dep_path: Path, dep_item_name: str) -> None:
        """Execute a dependency if not already executed."""
        dep_key = f"{dep_path}::{dep_item_name}"

        # Skip if already executed
        if dep_key in self.executed_dependencies:
            return

        # Mark as executed
        self.executed_dependencies.add(dep_key)

        # Load the dependency export
        try:
            dep_export = load_export(dep_path)
        except Exception as e:
            raise ValueError(f"Failed to load dependency export {dep_path}: {e}")

        if dep_item_name not in dep_export.items:
            raise ValueError(f"Dependency {dep_item_name} not found in {dep_path}")

        dep_item = dep_export.items[dep_item_name]

        # Execute the dependency's dependencies first (recursive)
        for nested_dep in dep_item.deps:
            nested_dep_path_str, nested_dep_name = nested_dep.rsplit("/", 1)
            # Fix the path prefix from "tests/export/" to "tests/vyper-exports/"
            if nested_dep_path_str.startswith("tests/export/"):
                nested_dep_path_str = nested_dep_path_str.replace(
                    "tests/export/", "tests/vyper-exports/", 1
                )
            nested_dep_path = Path(nested_dep_path_str)
            self._execute_dependency(nested_dep_path, nested_dep_name)

        # Execute the dependency's traces
        for trace in dep_item.traces:
            if isinstance(trace, DeploymentTrace):
                execute_deployment(
                    trace=trace,
                    env=self.env,
                    deployed_contracts=self.deployed_contracts,
                    use_python_args=True,  # Dependencies typically use python args
                    collect_storage_dump=False,  # No need for dumps in dependencies
                )
            elif isinstance(trace, CallTrace):
                execute_call(
                    trace=trace,
                    env=self.env,
                    deployed_contracts=self.deployed_contracts,
                    use_python_args=True,
                    collect_storage_dump=False,  # No need for dumps in dependencies
                )
            elif isinstance(trace, SetBalanceTrace):
                execute_set_balance(trace, self.env)
            elif isinstance(trace, ClearTransientStorageTrace):
                execute_clear_transient_storage(trace, self.env)

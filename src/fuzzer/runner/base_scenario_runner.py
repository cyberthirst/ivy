"""
Base scenario runner that handles dependencies and all trace types.

This module provides the base class for running scenarios across different
execution environments (Ivy, Boa) with all shared execution logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path

from .scenario import Scenario
from ..export_utils import (
    DeploymentTrace,
    CallTrace,
    SetBalanceTrace,
    ClearTransientStorageTrace,
    load_export,
)


@dataclass
class DeploymentResult:
    """Result of deploying a contract."""

    success: bool
    contract: Optional[Any] = None  # Contract instance (VyperContract)
    error: Optional[Exception] = None
    storage_dump: Optional[Dict[str, Any]] = None
    deployed_address: Optional[str] = (
        None  # Expected address where contract was deployed
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "error": str(self.error) if self.error else None,
            "address": str(getattr(self.contract, "address", self.contract))
            if self.success
            else None,
            "storage_dump": self.storage_dump,
        }


@dataclass
class CallResult:
    """Result of a single function call."""

    success: bool
    output: Any = None
    error: Optional[Exception] = None
    storage_dump: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output if self.output else None,
            "error": str(self.error) if self.error else None,
            "storage_dump": self.storage_dump,
        }


@dataclass
class TraceResult:
    """Result of executing any trace type."""

    trace_type: str  # "deployment", "call", "set_balance", "clear_transient_storage"
    trace_index: int  # Index in the original traces list
    result: Optional[Union[DeploymentResult, CallResult]] = (
        None  # None for set_balance/clear_transient
    )


@dataclass
class ScenarioResult:
    """Complete result of running a scenario."""

    results: List[TraceResult] = field(default_factory=list)

    def get_deployment_results(self) -> List[Tuple[int, DeploymentResult]]:
        """Get all deployment results with their trace indices."""
        return [
            (r.trace_index, r.result)
            for r in self.results
            if r.trace_type == "deployment" and r.result is not None
        ]

    def get_call_results(self) -> List[Tuple[int, CallResult]]:
        """Get all call results with their trace indices."""
        return [
            (r.trace_index, r.result)
            for r in self.results
            if r.trace_type == "call" and r.result is not None
        ]


class BaseScenarioRunner(ABC):
    """Base class for scenario runners that handle all trace types and dependencies."""

    def __init__(self, collect_storage_dumps: bool = False):
        self.deployed_contracts: Dict[str, Any] = {}
        self.executed_dependencies: Set[str] = set()
        self.collect_storage_dumps = collect_storage_dumps

    @abstractmethod
    def _deploy_from_source(
        self,
        source: str,
        solc_json: Dict[str, Any],
        args: List[Any],
        kwargs: Dict[str, Any],
        sender: Optional[str] = None,
    ) -> Any:
        """Deploy a contract from source. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _call_method(
        self,
        contract: Any,
        method_name: str,
        args: List[Any],
        kwargs: Dict[str, Any],
        sender: Optional[str] = None,
    ) -> Any:
        """Call a contract method. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _message_call(
        self,
        to_address: str,
        data: bytes,
        value: int = 0,
        sender: Optional[str] = None,
    ) -> bytes:
        """Low-level message call. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _set_balance(self, address: str, value: int) -> None:
        """Set balance of an address. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _get_balance(self, address: str) -> int:
        """Get balance of an address. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _clear_transient_storage(self) -> None:
        """Clear transient storage. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _get_storage_dump(self, contract: Any) -> Optional[Dict[str, Any]]:
        """Get storage dump from a contract. Must be implemented by subclasses."""
        pass

    def run(self, scenario: Scenario) -> ScenarioResult:
        """Run a complete scenario including dependencies."""
        # Reset state for each scenario
        self.deployed_contracts = {}
        self.executed_dependencies = set()

        # Execute dependencies first
        for dep_path, dep_item_name in scenario.dependencies:
            self._execute_dependency(dep_path, dep_item_name)

        # Execute all traces in order
        result = ScenarioResult()
        traces_to_execute = scenario.get_traces_to_execute()

        for trace_index, trace in enumerate(traces_to_execute):
            if isinstance(trace, DeploymentTrace):
                # Execute deployment
                deployment_result = self._execute_deployment(
                    trace=trace,
                    use_python_args=scenario.use_python_args,
                )
                result.results.append(
                    TraceResult(
                        trace_type="deployment",
                        trace_index=trace_index,
                        result=deployment_result,
                    )
                )

                # Continue even if deployment fails to match test behavior

            elif isinstance(trace, CallTrace):
                # Execute call
                call_result = self._execute_call(
                    trace=trace,
                    use_python_args=scenario.use_python_args,
                )
                result.results.append(
                    TraceResult(
                        trace_type="call",
                        trace_index=trace_index,
                        result=call_result,
                    )
                )

            elif isinstance(trace, SetBalanceTrace):
                # Execute set_balance (no result to record)
                self._execute_set_balance(trace)
                result.results.append(
                    TraceResult(
                        trace_type="set_balance",
                        trace_index=trace_index,
                        result=None,
                    )
                )

            elif isinstance(trace, ClearTransientStorageTrace):
                # Execute clear_transient_storage (no result to record)
                self._execute_clear_transient_storage(trace)
                result.results.append(
                    TraceResult(
                        trace_type="clear_transient_storage",
                        trace_index=trace_index,
                        result=None,
                    )
                )

        return result

    def _execute_deployment(
        self,
        trace: DeploymentTrace,
        mutated_source: Optional[str] = None,
        mutated_args: Optional[List[Any]] = None,
        mutated_kwargs: Optional[Dict[str, Any]] = None,
        use_python_args: bool = False,
    ) -> DeploymentResult:
        """Execute a deployment trace."""
        try:
            # Use mutated source if provided
            source_to_deploy = (
                mutated_source if mutated_source is not None else trace.source_code
            )
            if not source_to_deploy:
                raise ValueError("No source code available for deployment")

            # Prepare constructor arguments
            if use_python_args and trace.python_args:
                # Use python args from trace or mutations
                args = (
                    mutated_args
                    if mutated_args is not None
                    else trace.python_args.get("args", [])
                )
                kwargs = (
                    mutated_kwargs
                    if mutated_kwargs is not None
                    else trace.python_args.get("kwargs", {})
                )
                # Add value from trace if not in kwargs
                if "value" not in kwargs:
                    kwargs["value"] = trace.value
            else:
                # Use empty args/kwargs for raw deployment
                args = mutated_args if mutated_args is not None else []
                kwargs = (
                    mutated_kwargs
                    if mutated_kwargs is not None
                    else {"value": trace.value}
                )

            # Deploy the contract
            contract = self._deploy_from_source(
                source=source_to_deploy,
                solc_json=trace.solc_json,
                args=args,
                kwargs=kwargs,
                sender=getattr(
                    trace, "deployer", None
                ),  # deployment traces use 'deployer'
            )

            # Store the deployed contract by its address
            deployed_addr = getattr(trace, "deployed_address", None)
            if deployed_addr:
                self.deployed_contracts[deployed_addr] = contract
            # Also store by the contract's address if available
            contract_addr = getattr(contract, "address", None)
            if contract_addr:
                self.deployed_contracts[str(contract_addr)] = contract

            # Get storage dump if requested
            storage_dump = None
            if self.collect_storage_dumps:
                storage_dump = self._get_storage_dump(contract)

            return DeploymentResult(
                success=True,
                contract=contract,
                storage_dump=storage_dump,
                deployed_address=getattr(trace, "deployed_address", None),
            )

        except Exception as e:
            return DeploymentResult(success=False, error=e)

    def _execute_call(
        self,
        trace: CallTrace,
        use_python_args: bool = False,
    ) -> CallResult:
        """Execute a call trace."""
        try:
            # Get the contract from call_args (trace structure uses call_args.to for target)
            to_address = trace.call_args.get("to")

            # Look up contract by address
            contract = self.deployed_contracts.get(to_address)

            if not contract:
                raise ValueError(
                    f"Contract at {to_address} not found in deployed contracts. Available: {list(self.deployed_contracts.keys())}"
                )

            # Get method name from function_name or extract from python_args
            method_name = trace.function_name
            if not method_name and trace.python_args and "method" in trace.python_args:
                method_name = trace.python_args["method"]

            # Check if we need to use low-level message_call
            calldata = trace.call_args.get("calldata", "")
            if not method_name and (calldata == "" or calldata == "0x"):
                # Empty calldata - this is a call to __default__
                method_name = "__default__"

            # If we still don't have a method name but have calldata, use message_call
            if not method_name and calldata:
                # Use low-level message call
                calldata_bytes = bytes.fromhex(calldata.replace("0x", ""))
                result = self._message_call(
                    to_address=to_address,
                    data=calldata_bytes,
                    value=trace.call_args.get("value", 0),
                    sender=trace.call_args.get("sender"),
                )
            else:
                # Use high-level method call
                if not method_name:
                    raise ValueError(
                        "No method name available and no calldata for message_call"
                    )

                # Prepare call arguments
                if use_python_args and trace.python_args:
                    args = trace.python_args.get("args", [])
                    kwargs = trace.python_args.get("kwargs", {})
                    # Add value from trace if not in kwargs
                    if "value" not in kwargs:
                        kwargs["value"] = trace.call_args.get("value", 0)
                else:
                    # Use empty args/kwargs for raw call
                    args = []
                    kwargs = {"value": trace.call_args.get("value", 0)}

                # Execute the call
                result = self._call_method(
                    contract=contract,
                    method_name=method_name,
                    args=args,
                    kwargs=kwargs,
                    sender=trace.call_args.get("sender"),
                )

            output = result
            if result is None:
                output = b""

            # Get storage dump if requested
            storage_dump = None
            if self.collect_storage_dumps:
                storage_dump = self._get_storage_dump(contract)

            return CallResult(
                success=True,
                output=output,
                storage_dump=storage_dump,
            )

        except Exception as e:
            return CallResult(success=False, error=e)

    def _execute_set_balance(self, trace: SetBalanceTrace) -> None:
        """Execute a set_balance trace."""
        self._set_balance(trace.address, trace.value)

    def _execute_clear_transient_storage(
        self, trace: ClearTransientStorageTrace
    ) -> None:
        """Execute a clear_transient_storage trace."""
        self._clear_transient_storage()

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
                self._execute_deployment(
                    trace=trace,
                    use_python_args=True,  # Dependencies typically use python args
                )
            elif isinstance(trace, CallTrace):
                self._execute_call(
                    trace=trace,
                    use_python_args=True,
                )
            elif isinstance(trace, SetBalanceTrace):
                self._execute_set_balance(trace)
            elif isinstance(trace, ClearTransientStorageTrace):
                self._execute_clear_transient_storage(trace)

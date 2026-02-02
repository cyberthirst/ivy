"""
Base scenario runner that handles dependencies and all trace types.

This module provides the base class for running scenarios across different
execution environments (Ivy, Boa) with all shared execution logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

from fuzzer.compilation import CompilationOutcome, classify_compilation_error
from fuzzer.runner.scenario import Scenario
from fuzzer.trace_types import (
    Trace,
    DeploymentTrace,
    CallTrace,
    SetBalanceTrace,
    ClearTransientStorageTrace,
    Env,
)
from fuzzer.xfail import XFailExpectation


@dataclass
class BaseResult(ABC):
    """Base result class with common fields."""

    success: bool
    error: Optional[Exception] = None
    storage_dump: Optional[Dict[str, Any]] = None
    transient_storage_dump: Optional[Dict[str, Any]] = None

    @property
    @abstractmethod
    def is_runtime_failure(self) -> bool:
        """Check if this is a runtime failure."""
        raise NotImplementedError

    def _format_error(self) -> Optional[str]:
        """Format error with type, message, and last 3 traceback frames."""
        if not self.error:
            return None
        import traceback

        error_type = type(self.error).__name__
        error_msg = str(self.error)
        # Get last 3 frames of traceback
        tb_lines = traceback.format_exception(
            type(self.error), self.error, self.error.__traceback__
        )
        # Keep header + last 3 frame pairs (each frame is 2 lines: location + code)
        tb_str = "".join(tb_lines[-7:]) if len(tb_lines) > 7 else "".join(tb_lines)
        return f"{error_type}: {error_msg}\n\n{tb_str}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "error": self._format_error(),
            "storage_dump": self.storage_dump,
            "transient_storage_dump": self.transient_storage_dump,
        }


@dataclass
class DeploymentResult(BaseResult):
    """Result of deploying a contract."""

    contract: Optional[Any] = None  # Contract instance (VyperContract)
    deployed_address: Optional[str] = (
        None  # Expected address where contract was deployed
    )
    source_code: Optional[str] = None  # Source code that was attempted to deploy
    error_phase: Optional[str] = None  # "compile" or "init"
    compiler_settings: Optional[Dict[str, Any]] = None

    @property
    def is_runtime_failure(self) -> bool:
        """Check if this is a runtime failure (init error)."""
        if self.success or self.error is None:
            return False
        return self.error_phase != "compile"

    @property
    def is_compilation_failure(self) -> bool:
        """Check if this is a compilation failure (not runtime)."""
        if self.success or self.error is None or self.error_phase != "compile":
            return False
        return (
            classify_compilation_error(self.error)
            is CompilationOutcome.COMPILATION_FAILURE
        )

    @property
    def is_compiler_crash(self) -> bool:
        """Check if this is a compiler crash (internal error)."""
        if self.success or self.error is None or self.error_phase != "compile":
            return False
        return (
            classify_compilation_error(self.error) is CompilationOutcome.COMPILER_CRASH
        )

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["address"] = (
            str(getattr(self.contract, "address", self.contract))
            if self.success
            else None
        )
        result["error_phase"] = self.error_phase
        result["compiler_settings"] = self.compiler_settings
        return result


@dataclass
class CallResult(BaseResult):
    """Result of a single function call."""

    output: Any = None
    contract: Optional[Any] = None

    @property
    def is_runtime_failure(self) -> bool:
        """Calls never compile; any error is runtime."""
        return not self.success and self.error is not None

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["output"] = self.output if self.output else None
        return result


@dataclass
class TraceResult:
    """Result of executing any trace type."""

    trace_type: str  # "deployment", "call", "set_balance", "clear_transient_storage"
    trace_index: int  # Index in the original traces list
    result: Optional[Union[DeploymentResult, CallResult]] = (
        None  # None for set_balance/clear_transient
    )
    compilation_xfails: List[XFailExpectation] = field(default_factory=list)
    runtime_xfails: List[XFailExpectation] = field(default_factory=list)


@dataclass
class ScenarioResult:
    """Complete result of running a scenario."""

    results: List[TraceResult] = field(default_factory=list)

    def get_deployment_results(self) -> List[Tuple[int, DeploymentResult]]:
        """Get all deployment results with their trace indices."""
        return [
            (r.trace_index, cast(DeploymentResult, r.result))
            for r in self.results
            if r.trace_type == "deployment" and r.result is not None
        ]

    def get_call_results(self) -> List[Tuple[int, CallResult]]:
        """Get all call results with their trace indices."""
        return [
            (r.trace_index, cast(CallResult, r.result))
            for r in self.results
            if r.trace_type == "call" and r.result is not None
        ]


class BaseScenarioRunner(ABC):
    """Base class for scenario runners that handle all trace types and dependencies."""

    def __init__(
        self,
        env: Any,
        collect_storage_dumps: bool = False,
        compiler_settings: Optional[Dict[str, Any]] = None,
    ):
        self.env = env
        self.deployed_contracts: Dict[str, Any] = {}
        self.executed_dependencies: Set[str] = set()
        self.collect_storage_dumps = collect_storage_dumps
        # Runner-level compiler settings (takes precedence over trace settings)
        self.compiler_settings = compiler_settings or {"enable_decimals": True}

    @abstractmethod
    def _compile_from_source(
        self,
        source: str,
        solc_json: Optional[Dict[str, Any]],
        compiler_settings: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Compile source and return a compiled artifact."""
        pass

    @abstractmethod
    def _deploy_compiled(
        self,
        compiled: Any,
        args: List[Any],
        kwargs: Dict[str, Any],
        sender: Optional[str] = None,
        compiler_settings: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Deploy a contract from a compiled artifact. Must be implemented by subclasses."""
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
    def _raw_call(
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

    @abstractmethod
    def _get_transient_storage_dump(self, contract: Any) -> Optional[Dict[str, Any]]:
        """Get transient storage dump from a contract. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _set_block_env(self, trace_env: Optional[Env]) -> None:
        """Set block environment values from trace. Must be implemented by subclasses."""
        pass

    def _get_sender(self, sender: Optional[str]) -> str:
        """Get the sender address, defaulting to env.eoa if not provided."""
        if sender:
            return sender
        return self.env.eoa

    def run(self, scenario: Scenario) -> ScenarioResult:
        """Run a complete scenario (including dependencies)"""
        # Reset state
        self.deployed_contracts = {}
        self.executed_dependencies = set()

        for dep_scenario in scenario.dependencies:
            self._execute_dependency_scenario(dep_scenario)

        # Execute main scenario traces
        results = self._execute_traces(scenario.traces, scenario.use_python_args)
        return ScenarioResult(results=results)

    def _execute_traces(
        self,
        traces: List[Trace],
        use_python_args: bool,
    ) -> List[TraceResult]:
        """Execute a list of traces and return results."""
        results: List[TraceResult] = []

        for trace_index, trace in enumerate(traces):
            result = self.execute_trace(trace, trace_index, use_python_args)
            results.append(result)

        return results

    def execute_trace(
        self,
        trace: Trace,
        trace_index: int,
        use_python_args: bool,
    ) -> TraceResult:
        if isinstance(trace, DeploymentTrace):
            deployment_result = self._execute_deployment(
                trace=trace,
                use_python_args=use_python_args,
            )
            return TraceResult(
                trace_type="deployment",
                trace_index=trace_index,
                result=deployment_result,
                compilation_xfails=list(trace.compilation_xfails),
                runtime_xfails=list(trace.runtime_xfails),
            )

        elif isinstance(trace, CallTrace):
            call_result = self._execute_call(
                trace=trace,
                use_python_args=use_python_args,
            )
            return TraceResult(
                trace_type="call",
                trace_index=trace_index,
                result=call_result,
                runtime_xfails=list(trace.runtime_xfails),
            )

        elif isinstance(trace, SetBalanceTrace):
            self._execute_set_balance(trace)
            return TraceResult(
                trace_type="set_balance",
                trace_index=trace_index,
                result=None,
            )

        elif isinstance(trace, ClearTransientStorageTrace):
            self._execute_clear_transient_storage(trace)
            return TraceResult(
                trace_type="clear_transient_storage",
                trace_index=trace_index,
                result=None,
            )

        else:
            raise ValueError(f"Unknown trace type: {type(trace)}")

    def _execute_deployment(
        self,
        trace: DeploymentTrace,
        mutated_source: Optional[str] = None,
        mutated_args: Optional[List[Any]] = None,
        mutated_kwargs: Optional[Dict[str, Any]] = None,
        use_python_args: bool = False,
    ) -> DeploymentResult:
        """Execute a deployment trace."""
        self._set_block_env(trace.env)
        merged_settings = {
            **(trace.compiler_settings or {}),
            **self.compiler_settings,
        }
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

            try:
                compiled = self._compile_from_source(
                    source=source_to_deploy,
                    solc_json=trace.solc_json,
                    compiler_settings=merged_settings,
                )
            except Exception as e:
                return DeploymentResult(
                    success=False,
                    error=e,
                    source_code=source_to_deploy,
                    error_phase="compile",
                    compiler_settings=merged_settings,
                )

            try:
                contract = self._deploy_compiled(
                    compiled=compiled,
                    args=args,
                    kwargs=kwargs,
                    sender=trace.env.tx.origin if trace.env else None,
                    compiler_settings=merged_settings,
                )
            except Exception as e:
                return DeploymentResult(
                    success=False,
                    error=e,
                    source_code=source_to_deploy,
                    error_phase="init",
                    compiler_settings=merged_settings,
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
            transient_storage_dump = None
            if self.collect_storage_dumps:
                storage_dump = self._get_storage_dump(contract)
                transient_storage_dump = self._get_transient_storage_dump(contract)

            return DeploymentResult(
                success=True,
                contract=contract,
                storage_dump=storage_dump,
                transient_storage_dump=transient_storage_dump,
                deployed_address=getattr(trace, "deployed_address", None),
                compiler_settings=merged_settings,
            )

        except Exception as e:
            return DeploymentResult(
                success=False,
                error=e,
                source_code=source_to_deploy,
                error_phase="init",
                compiler_settings=merged_settings,
            )

    def _execute_call(
        self,
        trace: CallTrace,
        use_python_args: bool = False,
    ) -> CallResult:
        """Execute a call trace."""
        self._set_block_env(trace.env)
        try:
            # Get the target address from call_args
            to_address: str = trace.call_args.get("to", "")

            # Look up contract by address
            contract = self.deployed_contracts.get(to_address) if to_address else None

            method_name = trace.function_name
            calldata = trace.call_args.get("calldata", "")

            # Determine if we should use low-level message call:
            # 1. No deployed contract at address
            # 2. No method name
            # 3. Contract exists but doesn't have the method (e.g. proxy pattern)
            use_message_call = (
                not contract
                or not method_name
                or (contract and method_name and not hasattr(contract, method_name))
            )

            if use_message_call:
                # Use low-level message call
                if calldata and calldata != "" and calldata != "0x":
                    calldata_bytes = bytes.fromhex(calldata.replace("0x", ""))
                else:
                    calldata_bytes = b""

                result = self._raw_call(
                    to_address=to_address,
                    data=calldata_bytes,
                    value=trace.call_args.get("value", 0),
                    sender=trace.env.tx.origin if trace.env else None,
                )
            else:
                # High-level method call to a contract with a specific method
                assert method_name is not None

                # Prepare call arguments
                if use_python_args and trace.python_args:
                    args = trace.python_args.get("args", [])
                    kwargs = trace.python_args.get("kwargs", {})
                    # Add value from trace if not in kwargs
                    if "value" not in kwargs:
                        kwargs["value"] = trace.call_args.get("value", 0)
                else:
                    args = []
                    kwargs = {"value": trace.call_args.get("value", 0)}

                # Execute the call
                result = self._call_method(
                    contract=contract,
                    method_name=method_name,
                    args=args,
                    kwargs=kwargs,
                    sender=trace.env.tx.origin if trace.env else None,
                )

            output = result
            if result is None:
                output = b""

            # Get storage dump if requested (only if we have a contract)
            storage_dump = None
            transient_storage_dump = None
            if self.collect_storage_dumps and contract is not None:
                storage_dump = self._get_storage_dump(contract)
                transient_storage_dump = self._get_transient_storage_dump(contract)

            return CallResult(
                success=True,
                output=output,
                storage_dump=storage_dump,
                transient_storage_dump=transient_storage_dump,
                contract=contract,
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

    def _execute_dependency_scenario(self, scenario: Scenario) -> None:
        """Execute a dependency scenario if not already executed."""
        # Skip if already executed (use scenario_id for deduplication)
        if scenario.scenario_id and scenario.scenario_id in self.executed_dependencies:
            return

        # Mark as executed
        if scenario.scenario_id:
            self.executed_dependencies.add(scenario.scenario_id)

        # Execute nested dependencies first (depth-first, same order as before)
        for nested_dep in scenario.dependencies:
            self._execute_dependency_scenario(nested_dep)

        _ = self._execute_traces(scenario.traces, scenario.use_python_args)

"""
Divergence detection for differential fuzzing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from functools import cached_property
from enum import StrEnum
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from vyper.semantics.data_locations import DataLocation

from fuzzer.runner.scenario import Scenario
from fuzzer.runner.base_scenario_runner import (
    ScenarioResult,
    DeploymentResult,
    CallResult,
)
from fuzzer.storage_normalizer import normalize_storage_dump
from fuzzer.xfail import XFailExpectation

if TYPE_CHECKING:
    from fuzzer.runner.multi_runner import CompilerConfig


class DivergenceType(StrEnum):
    DEPLOYMENT = "deployment"
    EXECUTION = "execution"
    XFAIL = "xfail"


@dataclass
class Divergence:
    """Represents a divergence between Ivy and a specific Boa runner."""

    type: DivergenceType
    step: int  # 0 for deployment, 1+ for calls
    scenario: Scenario
    # Runner identification
    divergent_runner: str = "boa"  # e.g. "boa:default", "boa:venom"
    divergent_config: Optional[CompilerConfig] = None
    # Results
    ivy_result: Optional[Union[DeploymentResult, CallResult]] = None
    boa_result: Optional[Union[DeploymentResult, CallResult]] = None
    function: Optional[str] = None  # For execution divergences
    details: Optional[str] = None  # Additional diagnostic info
    xfail_expected: Optional[str] = None
    xfail_actual: Optional[str] = None  # What actually happened
    xfail_reasons: list[str] = field(default_factory=list)

    @cached_property
    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization. Result is cached."""
        result = {
            "type": self.type,
            "step": self.step,
            "divergent_runner": self.divergent_runner,
        }
        if self.divergent_config:
            result["compiler_config"] = {
                "name": self.divergent_config.name,
                "compiler_args": self.divergent_config.compiler_args,
            }

        if self.type == DivergenceType.XFAIL:
            result["xfail_expected"] = self.xfail_expected
            result["xfail_actual"] = self.xfail_actual
            result["xfail_reasons"] = self.xfail_reasons
        elif self.type == DivergenceType.DEPLOYMENT:
            if self.ivy_result:
                result["ivy_deployment"] = self.ivy_result.to_dict()
            if self.boa_result:
                result["boa_deployment"] = self.boa_result.to_dict()
        else:  # execution
            result["function"] = self.function
            if self.ivy_result:
                result["ivy_call"] = self.ivy_result.to_dict()
            if self.boa_result:
                result["boa_call"] = self.boa_result.to_dict()

        if self.details:
            result["details"] = self.details

        # Add relevant traces up to the divergence point
        if self.scenario.traces:
            result["traces"] = []
            for trace in self.scenario.traces[: self.step + 1]:
                trace_dict = asdict(trace)
                result["traces"].append(trace_dict)

        return result


def _get_xfail_reasons(expectations: list[XFailExpectation]) -> list[str]:
    return [xf.reason or "(unspecified)" for xf in expectations]


def _deployment_outcome_label(result: DeploymentResult) -> str:
    if result.is_compilation_timeout:
        return "compilation_timeout"
    if result.is_compilation_failure:
        return "compilation_failure"
    if result.is_compiler_crash:
        return "compiler_crash"
    if result.success:
        return "success"
    return "runtime_failure"


def _execution_outcome_label(
    result: Optional[Union[DeploymentResult, CallResult]],
) -> str:
    if result is None:
        return "no_result"
    if result.is_runtime_failure:
        return "runtime_failure"
    if result.success:
        return "success"
    if isinstance(result, DeploymentResult):
        if result.is_compilation_timeout:
            return "compilation_timeout"
        if result.is_compilation_failure:
            return "compilation_failure"
        if result.is_compiler_crash:
            return "compiler_crash"
    return "runtime_failure"


class DivergenceDetector:
    """Detects divergences between execution results."""

    def compare_all_results(
        self,
        ivy_result: ScenarioResult,
        boa_results: Dict[str, tuple[CompilerConfig, ScenarioResult]],
        scenario: Scenario,
    ) -> List[Divergence]:
        """Compare Ivy results against all Boa runner results."""
        divergences = []
        for runner_name, (config, boa_result) in boa_results.items():
            divergence = self.compare_results(
                ivy_result, boa_result, scenario, runner_name, config
            )
            if divergence:
                divergences.append(divergence)
        return divergences

    def compare_results(
        self,
        ivy_result: ScenarioResult,
        boa_result: ScenarioResult,
        scenario: Scenario,
        runner_name: str = "boa",
        config: Optional[CompilerConfig] = None,
    ) -> Optional[Divergence]:
        """Compare results from two runners and identify divergences."""
        divergent_runner = f"boa:{runner_name}" if runner_name != "boa" else "boa"

        # If result counts differ, report it as a divergence
        if len(ivy_result.results) != len(boa_result.results):
            return Divergence(
                type=DivergenceType.EXECUTION,
                step=0,
                scenario=scenario,
                divergent_runner=divergent_runner,
                divergent_config=config,
                details=f"Result count mismatch: Ivy={len(ivy_result.results)}, Boa={len(boa_result.results)}",
            )

        # Compare each trace result
        for ivy_trace_result, boa_trace_result in zip(
            ivy_result.results, boa_result.results
        ):
            # Verify they're for the same trace
            if ivy_trace_result.trace_type != boa_trace_result.trace_type:
                raise ValueError(
                    f"Trace type mismatch at index {ivy_trace_result.trace_index}: "
                    f"Ivy has {ivy_trace_result.trace_type}, Boa has {boa_trace_result.trace_type}"
                )

            if ivy_trace_result.trace_index != boa_trace_result.trace_index:
                raise ValueError(
                    f"Trace index mismatch: Ivy has {ivy_trace_result.trace_index}, "
                    f"Boa has {boa_trace_result.trace_index}"
                )

            # Check xfail flags for both Ivy and Boa results
            # For xfail violations, divergent_runner is the runner that violated the expectation
            for trace_result, xfail_runner_name, xfail_config in [
                (ivy_trace_result, "ivy", None),
                (boa_trace_result, f"boa:{runner_name}", config),
            ]:
                # Check compilation expectations for deployment traces
                deployment_result = trace_result.result
                if isinstance(deployment_result, DeploymentResult):
                    reasons = _get_xfail_reasons(trace_result.compilation_xfails)
                    if reasons and not (
                        deployment_result.is_compilation_failure
                        or deployment_result.is_compilation_timeout
                        or deployment_result.is_compiler_crash
                    ):
                        return Divergence(
                            type=DivergenceType.XFAIL,
                            step=trace_result.trace_index,
                            scenario=scenario,
                            divergent_runner=xfail_runner_name,
                            divergent_config=xfail_config,
                            xfail_expected="compilation",
                            xfail_actual=(
                                f"{xfail_runner_name}: "
                                f"{_deployment_outcome_label(deployment_result)}"
                            ),
                            xfail_reasons=reasons,
                        )

                runtime_reasons = _get_xfail_reasons(trace_result.runtime_xfails)

                if runtime_reasons:
                    exec_result = trace_result.result
                    actual_runtime_fail = (
                        exec_result.is_runtime_failure if exec_result else False
                    )
                    if not actual_runtime_fail:
                        return Divergence(
                            type=DivergenceType.XFAIL,
                            step=trace_result.trace_index,
                            scenario=scenario,
                            divergent_runner=xfail_runner_name,
                            divergent_config=xfail_config,
                            xfail_expected="runtime",
                            xfail_actual=(
                                f"{xfail_runner_name}: "
                                f"{_execution_outcome_label(exec_result)}"
                            ),
                            xfail_reasons=runtime_reasons,
                        )

            # Compare deployment results
            if ivy_trace_result.trace_type == "deployment":
                ivy_res, boa_res = ivy_trace_result.result, boa_trace_result.result
                assert isinstance(ivy_res, DeploymentResult)
                assert isinstance(boa_res, DeploymentResult)
                # Compiler crashes and timeouts invalidate comparison.
                if ivy_res.is_compiler_crash or boa_res.is_compiler_crash:
                    break
                if ivy_res.is_compilation_timeout or boa_res.is_compilation_timeout:
                    break
                if not self._compare_deployment_results(ivy_res, boa_res):
                    return Divergence(
                        type=DivergenceType.DEPLOYMENT,
                        step=ivy_trace_result.trace_index,
                        scenario=scenario,
                        divergent_runner=divergent_runner,
                        divergent_config=config,
                        ivy_result=ivy_trace_result.result,
                        boa_result=boa_trace_result.result,
                    )
                # Compilation failures end comparability after this trace.
                if ivy_res.is_compilation_failure or boa_res.is_compilation_failure:
                    break

            # Compare call results
            elif ivy_trace_result.trace_type == "call":
                ivy_res, boa_res = ivy_trace_result.result, boa_trace_result.result
                assert isinstance(ivy_res, CallResult)
                assert isinstance(boa_res, CallResult)
                if not self._compare_call_results(ivy_res, boa_res):
                    # Find the function name from the scenario
                    function_name = None
                    traces = scenario.traces
                    trace = traces[ivy_trace_result.trace_index]
                    trace_info = trace.to_trace_info(ivy_trace_result.trace_index)
                    function_name = trace_info.get("function")

                    return Divergence(
                        type=DivergenceType.EXECUTION,
                        step=ivy_trace_result.trace_index,
                        scenario=scenario,
                        divergent_runner=divergent_runner,
                        divergent_config=config,
                        ivy_result=ivy_trace_result.result,
                        boa_result=boa_trace_result.result,
                        function=function_name,
                    )

        return None  # No divergence found

    def _compare_deployment_results(
        self, ivy_res: DeploymentResult, boa_res: DeploymentResult
    ) -> bool:
        """Compare two deployment results for equality."""
        # Check if both reverted or both succeeded
        if ivy_res.success != boa_res.success:
            return False

        # If both failed, consider them equal
        # TODO: catch cases where semantically different errors occur
        if not ivy_res.success:
            return True

        # Compare deployed addresses
        ivy_addr = str(getattr(ivy_res.contract, "address", ivy_res.contract))
        boa_addr = str(getattr(boa_res.contract, "address", boa_res.contract))

        if ivy_addr != boa_addr:
            return False

        # Compare storage dumps if available
        if ivy_res.storage_dump is not None and boa_res.storage_dump is not None:
            if ivy_res.storage_dump != boa_res.storage_dump:
                if not self._normalized_dumps_match(
                    ivy_res, boa_res, location=DataLocation.STORAGE
                ):
                    return False

        if (
            ivy_res.transient_storage_dump is not None
            and boa_res.transient_storage_dump is not None
        ):
            if ivy_res.transient_storage_dump != boa_res.transient_storage_dump:
                if not self._normalized_dumps_match(
                    ivy_res,
                    boa_res,
                    location=DataLocation.TRANSIENT,
                ):
                    return False

        return True

    def _compare_call_results(self, ivy_res: CallResult, boa_res: CallResult) -> bool:
        """Compare two call results for equality."""
        # Check if both reverted or both succeeded
        if ivy_res.success != boa_res.success:
            return False

        # If both reverted, consider them equal
        if not ivy_res.success:
            return True

        # Compare outputs
        if ivy_res.output != boa_res.output:
            return False

        # Compare storage dumps if available
        if ivy_res.storage_dump is not None and boa_res.storage_dump is not None:
            if ivy_res.storage_dump != boa_res.storage_dump:
                if not self._normalized_dumps_match(
                    ivy_res, boa_res, location=DataLocation.STORAGE
                ):
                    return False

        if (
            ivy_res.transient_storage_dump is not None
            and boa_res.transient_storage_dump is not None
        ):
            if ivy_res.transient_storage_dump != boa_res.transient_storage_dump:
                if not self._normalized_dumps_match(
                    ivy_res,
                    boa_res,
                    location=DataLocation.TRANSIENT,
                ):
                    return False

        return True

    @staticmethod
    def _normalized_dumps_match(
        ivy_res: Union[DeploymentResult, CallResult],
        boa_res: Union[DeploymentResult, CallResult],
        *,
        location: DataLocation,
    ) -> bool:
        """Logically equivalent storage can be produced by the runners but the dumps can differ.
        That happens e.g. when one materializes default values and the other does not. We compare
        normalized dumps lazily to avoid performance penalty only when the original doesn't compare equal.
        """
        if location == DataLocation.TRANSIENT:
            ivy_dump = ivy_res.transient_storage_dump
            boa_dump = boa_res.transient_storage_dump
        else:
            ivy_dump = ivy_res.storage_dump
            boa_dump = boa_res.storage_dump
        assert ivy_dump is not None and boa_dump is not None
        normalized_ivy = normalize_storage_dump(
            ivy_dump, ivy_res.contract, location=location
        )
        normalized_boa = normalize_storage_dump(
            boa_dump, boa_res.contract, location=location
        )
        return normalized_ivy == normalized_boa

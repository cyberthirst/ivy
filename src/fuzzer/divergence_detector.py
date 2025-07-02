"""
Divergence detection for differential fuzzing.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from .runner.scenario import Scenario
from .runner.base_scenario_runner import ScenarioResult, DeploymentResult, CallResult


@dataclass
class Divergence:
    """Represents a divergence between two runners."""

    type: str  # "deployment" or "execution"
    step: int  # 0 for deployment, 1+ for calls
    scenario: Scenario
    ivy_result: Optional[Union[DeploymentResult, CallResult]] = None
    boa_result: Optional[Union[DeploymentResult, CallResult]] = None
    function: Optional[str] = None  # For execution divergences

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "type": self.type,
            "step": self.step,
        }

        # Add trace information for the divergence step
        traces = self.scenario.get_traces_to_execute()
        if self.step < len(traces):
            trace = traces[self.step]

            # If it's a deployment trace, add source information
            if hasattr(trace, "source_code"):
                result["source_code"] = trace.source_code

            # Add mutation information if traces were mutated
            if self.scenario.mutated_traces:
                result["has_mutations"] = True

        if self.type == "deployment":
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

        # Add relevant traces up to the divergence point
        if self.scenario.get_traces_to_execute():
            result["traces"] = []
            for i, trace in enumerate(
                self.scenario.get_traces_to_execute()[: self.step + 1]
            ):
                trace_info = {
                    "type": trace.__class__.__name__,
                    "index": i,
                }
                if hasattr(trace, "function_name"):
                    trace_info["function"] = trace.function_name
                elif (
                    hasattr(trace, "python_args")
                    and trace.python_args
                    and "method" in trace.python_args
                ):
                    trace_info["function"] = trace.python_args["method"]
                if hasattr(trace, "python_args"):
                    trace_info["args"] = trace.python_args
                result["traces"].append(trace_info)

        return result


class DivergenceDetector:
    """Detects divergences between execution results."""

    def compare_results(
        self, ivy_result: ScenarioResult, boa_result: ScenarioResult, scenario: Scenario
    ) -> Optional[Divergence]:
        """Compare results from two runners and identify divergences."""
        # If result counts differ, report it as a divergence
        if len(ivy_result.results) != len(boa_result.results):
            return Divergence(
                type="execution",
                step=0,
                scenario=scenario,
                function=f"Result count mismatch: Ivy has {len(ivy_result.results)} results, Boa has {len(boa_result.results)} results",
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

            # Compare deployment results
            if ivy_trace_result.trace_type == "deployment":
                if not self._compare_deployment_results(
                    ivy_trace_result.result, boa_trace_result.result
                ):
                    return Divergence(
                        type="deployment",
                        step=ivy_trace_result.trace_index,
                        scenario=scenario,
                        ivy_result=ivy_trace_result.result,
                        boa_result=boa_trace_result.result,
                    )

            # Compare call results
            elif ivy_trace_result.trace_type == "call":
                if not self._compare_call_results(
                    ivy_trace_result.result, boa_trace_result.result
                ):
                    # Find the function name from the scenario
                    function_name = None
                    traces = scenario.get_traces_to_execute()
                    trace = traces[ivy_trace_result.trace_index]
                    if hasattr(trace, "function_name"):
                        function_name = trace.function_name
                    elif hasattr(trace, "python_args") and trace.python_args:
                        function_name = trace.python_args.get("method")

                    return Divergence(
                        type="execution",
                        step=ivy_trace_result.trace_index,
                        scenario=scenario,
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
                return False

        return True

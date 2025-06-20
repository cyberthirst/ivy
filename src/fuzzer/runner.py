"""
Scenario runner classes for differential fuzzing.

This module provides a unified API for running scenarios across different
execution environments (Ivy, Boa) and comparing their results.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


@dataclass
class DeploymentResult:
    """Result of deploying a contract."""

    success: bool
    address: Optional[Any] = None  # Contract address/instance
    error: Optional[Exception] = None
    storage_dump: Optional[Dict[str, Any]] = None

    @property
    def reverted(self) -> bool:
        """Check if deployment reverted."""
        return not self.success


@dataclass
class CallResult:
    """Result of a single function call."""

    success: bool
    output: Any = None
    error: Optional[Exception] = None
    storage_dump: Optional[Dict[str, Any]] = None

    @property
    def reverted(self) -> bool:
        """Check if call reverted."""
        return not self.success


@dataclass
class ScenarioResult:
    """Complete result of running a scenario."""

    deployment: DeploymentResult
    calls: List[CallResult] = field(default_factory=list)

    def get_step_result(self, step: int) -> Union[DeploymentResult, CallResult]:
        """Get result for a specific step (0 = deployment, 1+ = calls)."""
        if step == 0:
            return self.deployment
        return self.calls[step - 1]


@dataclass
class Call:
    """Represents a function call in the call schedule."""

    fn_name: str
    args: List[Any]
    kwargs: Dict[str, Any]
    msg_sender: Optional[str] = None


@dataclass
class Scenario:
    """Fuzzing scenario configuration."""

    mutated_source: str
    deploy_args: List[Any]
    deploy_kwargs: Dict[str, Any]
    call_schedule: List[Call]


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
            "mutated_source": self.scenario.mutated_source,
            "deploy_args": self.scenario.deploy_args,
            "deploy_kwargs": self.scenario.deploy_kwargs,
        }

        if self.type == "deployment":
            if self.ivy_result:
                result["ivy_error"] = (
                    str(self.ivy_result.error) if self.ivy_result.error else None
                )
            if self.boa_result:
                result["boa_error"] = (
                    str(self.boa_result.error) if self.boa_result.error else None
                )
        else:
            # Execution divergence
            result["function"] = self.function
            result["call_schedule"] = [
                {
                    "fn_name": c.fn_name,
                    "args": c.args,
                    "kwargs": c.kwargs,
                    "msg_sender": c.msg_sender,
                }
                for c in self.scenario.call_schedule[: self.step]
            ]

            if self.ivy_result:
                result["ivy_result"] = (
                    str(self.ivy_result.output) if self.ivy_result.success else None
                )
                result["ivy_error"] = (
                    str(self.ivy_result.error) if self.ivy_result.error else None
                )
            if self.boa_result:
                result["boa_result"] = (
                    str(self.boa_result.output) if self.boa_result.success else None
                )
                result["boa_error"] = (
                    str(self.boa_result.error) if self.boa_result.error else None
                )

        return result


class ScenarioRunner(ABC):
    """Abstract base class for scenario runners."""

    @abstractmethod
    def run(self, scenario: Scenario) -> ScenarioResult:
        """Run a scenario and return the result."""
        pass

    @abstractmethod
    def deploy(
        self, source: str, args: List[Any], kwargs: Dict[str, Any]
    ) -> DeploymentResult:
        """Deploy a contract from source code."""
        pass

    @abstractmethod
    def call(self, contract: Any, call: Call) -> CallResult:
        """Execute a function call on a deployed contract."""
        pass


class DivergenceDetector:
    """Detects divergences between scenario results."""

    def compare_results(
        self, ivy_result: ScenarioResult, boa_result: ScenarioResult, scenario: Scenario
    ) -> Optional[Divergence]:
        """Compare two scenario results and return divergence if found."""
        # Check deployment divergence
        if ivy_result.deployment.success != boa_result.deployment.success:
            # Skip known risky overlap errors
            if boa_result.deployment.error and "risky overlap" in str(
                boa_result.deployment.error
            ):
                return None

            return Divergence(
                type="deployment",
                step=0,
                scenario=scenario,
                ivy_result=ivy_result.deployment,
                boa_result=boa_result.deployment,
            )

        # If both failed to deploy, no divergence
        if not ivy_result.deployment.success:
            return None

        # Compare call results
        for step, (ivy_call, boa_call) in enumerate(
            zip(ivy_result.calls, boa_result.calls)
        ):
            if not self._compare_call_results(ivy_call, boa_call):
                return Divergence(
                    type="execution",
                    step=step + 1,  # +1 because step 0 is deployment
                    scenario=scenario,
                    ivy_result=ivy_call,
                    boa_result=boa_call,
                    function=scenario.call_schedule[step].fn_name,
                )

        return None  # No divergence found

    def _compare_call_results(self, ivy_res: CallResult, boa_res: CallResult) -> bool:
        """Compare two call results for equality."""
        # Check if both reverted or both succeeded
        if ivy_res.reverted != boa_res.reverted:
            return False

        # If both reverted, consider them equal
        if ivy_res.reverted:
            return True

        # Compare outputs
        if ivy_res.output != boa_res.output:
            return False

        # Compare storage dumps
        if ivy_res.storage_dump != boa_res.storage_dump:
            return False

        return True

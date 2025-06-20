"""
Boa scenario runner implementation.
"""

from typing import Any, Dict, List, Optional

import boa
from boa import loads as boa_loads

from .runner import (
    ScenarioRunner,
    Scenario,
    ScenarioResult,
    DeploymentResult,
    CallResult,
    Call,
)


class BoaRunner(ScenarioRunner):
    """Runner for executing scenarios in Boa."""

    def __init__(self):
        self.contract: Optional[Any] = None

    def run(self, scenario: Scenario) -> ScenarioResult:
        """Run a complete scenario in Boa."""
        # Deploy contract
        deployment_result = self.deploy(
            scenario.mutated_source, scenario.deploy_args, scenario.deploy_kwargs
        )

        # If deployment failed, return early
        if not deployment_result.success:
            return ScenarioResult(deployment=deployment_result)

        # Execute call schedule
        call_results = []
        for call in scenario.call_schedule:
            call_result = self.call(self.contract, call)
            call_results.append(call_result)

        return ScenarioResult(deployment=deployment_result, calls=call_results)

    def deploy(
        self, source: str, args: List[Any], kwargs: Dict[str, Any]
    ) -> DeploymentResult:
        """Deploy a contract from source code in Boa."""
        try:
            self.contract = boa_loads(source, *args, **kwargs)

            # Get storage dump after deployment
            storage_dump = self.contract._storage.dump()

            return DeploymentResult(
                success=True, address=self.contract, storage_dump=storage_dump
            )

        except Exception as e:
            return DeploymentResult(success=False, error=e)

    def call(self, contract: Any, call: Call) -> CallResult:
        """Execute a function call on a deployed contract in Boa."""
        if not self.contract:
            return CallResult(success=False, error=Exception("Contract not deployed"))

        try:
            # Set up sender if specified
            if call.msg_sender:
                boa.env.set_balance(call.msg_sender, 10**20)  # Give sender some ETH
                with boa.env.prank(call.msg_sender):
                    fn = getattr(self.contract, call.fn_name)
                    result = fn(*call.args, **call.kwargs)
            else:
                fn = getattr(self.contract, call.fn_name)
                result = fn(*call.args, **call.kwargs)

            # Get storage dump after call
            storage_dump = self.contract._storage.dump()

            return CallResult(success=True, output=result, storage_dump=storage_dump)

        except Exception as e:
            return CallResult(success=False, error=e)

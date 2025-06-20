"""
Ivy scenario runner implementation.
"""

from typing import Any, Dict, List, Optional

from ivy.frontend.loader import loads as ivy_loads
from ivy.frontend.env import Env
from ivy.types import Address

from .runner import (
    ScenarioRunner,
    Scenario,
    ScenarioResult,
    DeploymentResult,
    CallResult,
    Call,
)


class IvyRunner(ScenarioRunner):
    """Runner for executing scenarios in Ivy."""

    def __init__(self):
        self.env: Optional[Env] = None
        self.contract: Optional[Any] = None

    def run(self, scenario: Scenario) -> ScenarioResult:
        """Run a complete scenario in Ivy."""
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
        """Deploy a contract from source code in Ivy."""
        self.env = Env()

        try:
            # Set up environment
            deployer_addr = Address("0x0000000000000000000000000000000000000001")
            self.env.set_balance(deployer_addr, 10**21)  # Give deployer some ETH

            # Set deployer as tx origin if available
            if hasattr(self.env, "evm") and hasattr(self.env.evm, "set_tx_origin"):
                self.env.evm.set_tx_origin(deployer_addr)

            # ivy_loads compiles and deploys the contract with constructor args
            value = kwargs.get("value", 0)
            if args:
                self.contract = ivy_loads(
                    source,
                    env=self.env,
                    *args,
                    value=value,
                )
            else:
                self.contract = ivy_loads(
                    source,
                    env=self.env,
                    value=value,
                )

            # Get storage dump after deployment
            storage_dump = self.contract.storage_dump()

            return DeploymentResult(
                success=True, address=self.contract, storage_dump=storage_dump
            )

        except Exception as e:
            return DeploymentResult(success=False, error=e)

    def call(self, contract: Any, call: Call) -> CallResult:
        """Execute a function call on a deployed contract in Ivy."""
        if not self.env or not self.contract:
            return CallResult(success=False, error=Exception("Contract not deployed"))

        try:
            # Set up sender if specified
            if call.msg_sender:
                sender_addr = Address(call.msg_sender)
                self.env.set_balance(sender_addr, 10**20)  # Give sender some ETH
            else:
                sender_addr = None

            # Execute the call
            result = getattr(self.contract, call.fn_name)(
                *call.args, value=call.kwargs.get("value", 0), sender=sender_addr
            )

            # Get storage dump after call
            storage_dump = self.contract.storage_dump()

            return CallResult(success=True, output=result, storage_dump=storage_dump)

        except Exception as e:
            return CallResult(success=False, error=e)

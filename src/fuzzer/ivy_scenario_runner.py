"""
Ivy implementation of the scenario runner.
"""

from typing import Any, Dict, List, Optional

from ivy.frontend.env import Env
from ivy.frontend.loader import loads_from_solc_json
from ivy.types import Address

from .base_scenario_runner import BaseScenarioRunner, ScenarioResult
from .scenario import Scenario


class IvyScenarioRunner(BaseScenarioRunner):
    """Runner for executing scenarios in Ivy."""

    def __init__(self, collect_storage_dumps: bool = False):
        super().__init__(collect_storage_dumps)
        self.env = Env()
        self._original_eoa = None

    def _deploy_from_source(
        self,
        source: str,
        solc_json: Dict[str, Any],
        args: List[Any],
        kwargs: Dict[str, Any],
        sender: Optional[str] = None,
    ) -> Any:
        """Deploy a contract from source in Ivy."""
        # Set sender as eoa if specified
        if sender:
            self._original_eoa = self.env.eoa
            self.env.eoa = Address(sender)
            # Ensure deployer has enough balance
            self.env.set_balance(
                self.env.eoa,
                self.env.get_balance(self.env.eoa) + kwargs.get("value", 0) + 10**18,
            )

        try:
            # Prepare deployment kwargs
            deployment_kwargs = {
                "env": self.env,
                "value": kwargs.get("value", 0),
            }

            # Add constructor args if any
            if args:
                deployment_kwargs["constructor_args"] = {"args": args, "kwargs": {}}

            # Deploy using loads_from_solc_json
            contract = loads_from_solc_json(solc_json, **deployment_kwargs)
            return contract

        finally:
            # Restore original eoa
            if self._original_eoa is not None:
                self.env.eoa = self._original_eoa
                self._original_eoa = None

    def _call_method(
        self,
        contract: Any,
        method_name: str,
        args: List[Any],
        kwargs: Dict[str, Any],
        sender: Optional[str] = None,
    ) -> Any:
        """Call a contract method in Ivy."""
        # Set sender as eoa if specified
        if sender:
            self._original_eoa = self.env.eoa
            self.env.eoa = Address(sender)

        try:
            # Get the method
            method = getattr(contract, method_name)

            # Call with args and kwargs
            result = method(*args, **kwargs)
            return result

        finally:
            # Restore original eoa
            if self._original_eoa is not None:
                self.env.eoa = self._original_eoa
                self._original_eoa = None

    def _set_balance(self, address: str, value: int) -> None:
        """Set balance of an address in Ivy."""
        addr = Address(address)
        self.env.set_balance(addr, value)

    def _get_balance(self, address: str) -> int:
        """Get balance of an address in Ivy."""
        addr = Address(address)
        return self.env.get_balance(addr)

    def _message_call(
        self,
        to_address: str,
        data: bytes,
        value: int = 0,
        sender: Optional[str] = None,
    ) -> bytes:
        """Low-level message call in Ivy."""
        # Set sender as eoa if specified
        if sender:
            self._original_eoa = self.env.eoa
            self.env.eoa = Address(sender)

        try:
            # Perform the message call
            result = self.env.message_call(
                to_address=to_address,
                data=data,
                value=value,
            )
            return result

        finally:
            # Restore original eoa
            if self._original_eoa is not None:
                self.env.eoa = self._original_eoa
                self._original_eoa = None

    def _clear_transient_storage(self) -> None:
        """Clear transient storage in Ivy."""
        if hasattr(self.env, "clear_transient_storage"):
            self.env.clear_transient_storage()
        else:
            # Transient storage might not be implemented
            pass

    def _get_storage_dump(self, contract: Any) -> Optional[Dict[str, Any]]:
        """Get storage dump from a contract in Ivy."""
        if hasattr(contract, "storage_dump"):
            return contract.storage_dump()
        return None

    def run(self, scenario: Scenario) -> ScenarioResult:
        with self.env.anchor():
            return super().run(scenario)

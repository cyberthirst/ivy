"""
Boa implementation of the scenario runner.
"""

from typing import Any, Dict, List, Optional

import boa

from .base_scenario_runner import BaseScenarioRunner, ScenarioResult
from .scenario import Scenario


class BoaScenarioRunner(BaseScenarioRunner):
    """Runner for executing scenarios in Boa."""

    def __init__(self, collect_storage_dumps: bool = False):
        super().__init__(collect_storage_dumps)

    def _deploy_from_source(
        self,
        source: str,
        solc_json: Dict[str, Any],
        args: List[Any],
        kwargs: Dict[str, Any],
        sender: Optional[str] = None,
    ) -> Any:
        """Deploy a contract from source in Boa."""
        # Set sender if specified
        if sender:
            boa.env.set_balance(
                sender, self._get_balance(sender) + kwargs.get("value", 0) + 10**18
            )
            boa.env.eoa = sender

        try:
            # Deploy with provided args and kwargs
            contract = boa.loads(source, *args, **kwargs)
            return contract

        except Exception as e:
            raise e

    def _call_method(
        self,
        contract: Any,
        method_name: str,
        args: List[Any],
        kwargs: Dict[str, Any],
        sender: Optional[str] = None,
    ) -> Any:
        """Call a contract method in Boa."""
        # Set sender if specified
        if sender:
            # Ensure sender has balance
            self._set_balance(
                sender, self._get_balance(sender) + kwargs.get("value", 0) + 10**18
            )
            with boa.env.prank(sender):
                method = getattr(contract, method_name)
                for arg in args:
                    if arg == "0x":
                        pass
                result = method(*args, **kwargs)
        else:
            method = getattr(contract, method_name)
            result = method(*args, **kwargs)

        return result

    def _set_balance(self, address: str, value: int) -> None:
        """Set balance of an address in Boa."""
        boa.env.set_balance(address, value)

    def _get_balance(self, address: str) -> int:
        """Get balance of an address in Boa."""
        return boa.env.get_balance(address)

    def _message_call(
        self,
        to_address: str,
        data: bytes,
        value: int = 0,
        sender: Optional[str] = None,
    ) -> bytes:
        """Low-level message call in Boa."""
        # In boa, we typically don't do raw message calls easily
        # This is a limitation when using calldata mode with boa
        raise NotImplementedError("Raw message calls not fully supported in Boa runner")

    def _clear_transient_storage(self) -> None:
        """Clear transient storage in Boa."""
        # Boa might not have explicit transient storage clearing
        if hasattr(boa.env, "clear_transient_storage"):
            boa.env.clear_transient_storage()

    def _get_storage_dump(self, contract: Any) -> Optional[Dict[str, Any]]:
        return contract._storage.dump()

    def run(self, scenario: Scenario) -> ScenarioResult:
        with boa.env.anchor():
            return super().run(scenario)

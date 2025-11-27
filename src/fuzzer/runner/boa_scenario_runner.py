"""
Boa implementation of the scenario runner.
"""

from typing import Any, Dict, List, Optional

import boa

from .base_scenario_runner import BaseScenarioRunner, ScenarioResult
from .scenario import Scenario


class BoaScenarioRunner(BaseScenarioRunner):
    """Runner for executing scenarios in Boa."""

    def __init__(
        self,
        compiler_args: Optional[Dict[str, Any]] = None,
        collect_storage_dumps: bool = False,
    ):
        super().__init__(boa.env, collect_storage_dumps)
        self.compiler_args = compiler_args or {}

    def _deploy_from_source(
        self,
        source: str,
        solc_json: Dict[str, Any],
        args: List[Any],
        kwargs: Dict[str, Any],
        sender: Optional[str] = None,
    ) -> Any:
        """Deploy a contract from source in Boa."""
        sender = self._get_sender(sender)
        with self.env.prank(sender):
            self.env.set_balance(
                sender, self._get_balance(sender) + kwargs.get("value", 0) + 10**18
            )
            contract = boa.loads(
                source, *args, compiler_args=self.compiler_args, **kwargs
            )
            return contract

    def _call_method(
        self,
        contract: Any,
        method_name: str,
        args: List[Any],
        kwargs: Dict[str, Any],
        sender: Optional[str] = None,
    ) -> Any:
        """Call a contract method in Boa."""
        sender = self._get_sender(sender)
        with self.env.prank(sender):
            self._set_balance(
                sender, self._get_balance(sender) + kwargs.get("value", 0) + 10**18
            )
            method = getattr(contract, method_name)
            result = method(*args, **kwargs)

            return result

    def _set_balance(self, address: str, value: int) -> None:
        self.env.set_balance(address, value)

    def _get_balance(self, address: str) -> int:
        return self.env.get_balance(address)

    def _message_call(
        self,
        to_address: str,
        data: bytes,
        value: int = 0,
        sender: Optional[str] = None,
    ) -> bytes:
        raise NotImplementedError("TODO does boa support this")

    def _clear_transient_storage(self) -> None:
        raise NotImplementedError("TODO does boa support this")

    def _get_storage_dump(self, contract: Any) -> Optional[Dict[str, Any]]:
        return contract._storage.dump()

    def run(self, scenario: Scenario) -> ScenarioResult:
        with self.env.anchor():
            return super().run(scenario)

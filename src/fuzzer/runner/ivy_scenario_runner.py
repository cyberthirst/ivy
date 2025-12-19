"""
Ivy implementation of the scenario runner.
"""

from typing import Any, Dict, List, Optional

import ivy
from ivy.frontend.loader import loads_from_solc_json, loads
from ivy.types import Address

from .base_scenario_runner import BaseScenarioRunner, ScenarioResult
from .scenario import Scenario
from ..trace_types import Env


class IvyScenarioRunner(BaseScenarioRunner):
    """Runner for executing scenarios in Ivy."""

    def __init__(self, collect_storage_dumps: bool = False, no_solc_json: bool = False):
        super().__init__(ivy.env, collect_storage_dumps)
        self._original_eoa = None
        self.no_solc_json = no_solc_json

    def _deploy_from_source(
        self,
        source: str,
        solc_json: Optional[Dict[str, Any]],
        args: List[Any],
        kwargs: Dict[str, Any],
        sender: Optional[str] = None,
        compiler_settings: Optional[Dict[str, Any]] = None,
    ) -> Any:
        sender = self._get_sender(sender)

        with self.env.prank(sender):
            # Ensure deployer has enough balance
            self.env.set_balance(
                self.env.eoa,
                self.env.get_balance(self.env.eoa) + kwargs.get("value", 0) + 10**18,
            )
            if (self.no_solc_json and source) or solc_json is None:
                contract = loads(
                    source, *args, compiler_args=compiler_settings, **kwargs
                )
            else:
                contract = loads_from_solc_json(solc_json, *args, **kwargs)

            return contract

    def _call_method(
        self,
        contract: Any,
        method_name: str,
        args: List[Any],
        kwargs: Dict[str, Any],
        sender: Optional[str] = None,
    ) -> Any:
        sender = self._get_sender(sender)
        with self.env.prank(sender):
            method = getattr(contract, method_name)
            result = method(*args, **kwargs)
            return result

    def _set_balance(self, address: str, value: int) -> None:
        addr = Address(address)
        self.env.set_balance(addr, value)

    def _get_balance(self, address: str) -> int:
        addr = Address(address)
        return self.env.get_balance(addr)

    def _message_call(
        self,
        to_address: str,
        data: bytes,
        value: int = 0,
        sender: Optional[str] = None,
    ) -> bytes:
        sender = self._get_sender(sender)
        with self.env.prank(sender):
            result = self.env.message_call(
                to_address=to_address,
                data=data,
                value=value,
            )
            return result

    def _clear_transient_storage(self) -> None:
        self.env.clear_transient_storage()

    def _get_storage_dump(self, contract: Any) -> Dict[str, Any]:
        return contract.storage_dump()

    def _set_block_env(self, trace_env: Optional[Env]) -> None:
        if trace_env is None:
            return
        self.env.block_number = trace_env.block.number
        self.env.timestamp = trace_env.block.timestamp

    def run(self, scenario: Scenario) -> ScenarioResult:
        with self.env.anchor():
            return super().run(scenario)

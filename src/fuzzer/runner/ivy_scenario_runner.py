"""
Ivy implementation of the scenario runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast

import ivy
from vyper.compiler.phases import CompilerData
from ivy.frontend.loader import loads_from_solc_json
from ivy.frontend.vyper_contract import VyperContract
from ivy.types import Address

from fuzzer.runner.base_scenario_runner import (
    BaseScenarioRunner,
    DeploymentResult,
    ScenarioResult,
    UNPARSABLE_CONTRACT_FINGERPRINT,
)
from fuzzer.runner.scenario import Scenario
from fuzzer.trace_types import DeploymentTrace, Env


@dataclass
class IvyDeploymentPreparation:
    contract_fingerprint: str
    compiled: Optional[CompilerData] = None
    compilation_error: Optional[DeploymentResult] = None


class IvyScenarioRunner(BaseScenarioRunner):
    """Runner for executing scenarios in Ivy."""

    def __init__(
        self,
        collect_storage_dumps: bool = False,
        compiler_settings: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(ivy.env, collect_storage_dumps, compiler_settings)
        self._original_eoa = None

    def _compile_from_solc_json(
        self,
        solc_json: Dict[str, Any],
        compiler_settings: Optional[Dict[str, Any]] = None,
    ) -> CompilerData:
        compiler_data = cast(
            CompilerData,
            loads_from_solc_json(solc_json, get_compiler_data=True),
        )

        # Force initcode compilation to surface compile-time errors here.
        _ = compiler_data.bytecode

        return compiler_data

    def _resolve_contract_fingerprint(self, compiled: CompilerData) -> str:
        try:
            integrity_sum = getattr(compiled, "integrity_sum", None)
        except Exception:
            integrity_sum = None

        if isinstance(integrity_sum, str) and integrity_sum:
            return integrity_sum

        return UNPARSABLE_CONTRACT_FINGERPRINT

    def prepare_deployment_context(
        self,
        trace: DeploymentTrace,
    ) -> IvyDeploymentPreparation:
        merged_settings = self._get_merged_compiler_settings(trace)
        solc_json = trace.solc_json
        if not solc_json:
            return IvyDeploymentPreparation(
                contract_fingerprint=UNPARSABLE_CONTRACT_FINGERPRINT,
                compilation_error=DeploymentResult(
                    success=False,
                    error=ValueError("No solc_json available for deployment"),
                    solc_json=None,
                    error_phase="compile",
                    compiler_settings=merged_settings,
                ),
            )

        try:
            compiled = self._compile_from_solc_json(
                solc_json=solc_json,
                compiler_settings=merged_settings,
            )
        except Exception as e:
            return IvyDeploymentPreparation(
                contract_fingerprint=UNPARSABLE_CONTRACT_FINGERPRINT,
                compilation_error=DeploymentResult(
                    success=False,
                    error=e,
                    solc_json=solc_json,
                    error_phase="compile",
                    compiler_settings=merged_settings,
                ),
            )

        return IvyDeploymentPreparation(
            contract_fingerprint=self._resolve_contract_fingerprint(compiled),
            compiled=compiled,
        )

    def _deploy_compiled(
        self,
        compiled: CompilerData,
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
            filename = getattr(compiled, "contract_path", None)
            if isinstance(filename, str):
                return VyperContract(compiled, *args, filename=filename, **kwargs)
            return VyperContract(compiled, *args, **kwargs)

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
            self.env.set_balance(
                self.env.eoa,
                self.env.get_balance(self.env.eoa) + kwargs.get("value", 0) + 10**18,
            )
            method = getattr(contract, method_name)
            result = method(*args, **kwargs)
            return result

    def _set_balance(self, address: str, value: int) -> None:
        addr = Address(address)
        self.env.set_balance(addr, value)

    def _get_balance(self, address: str) -> int:
        addr = Address(address)
        return self.env.get_balance(addr)

    def _set_nonce(self, address: str, value: int) -> None:
        addr = Address(address)
        self.env.state.get_account(addr).nonce = value

    def _get_nonce(self, address: str) -> int:
        addr = Address(address)
        return self.env.state.get_nonce(addr)

    def _raw_call(
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

    def _get_transient_storage_dump(self, contract: Any) -> Dict[str, Any]:
        return contract.transient_storage_dump()

    def _set_block_env(self, trace_env: Optional[Env]) -> None:
        if trace_env is None or trace_env.block is None:
            return
        self.env.block_number = trace_env.block.number
        self.env.timestamp = trace_env.block.timestamp

    def run(self, scenario: Scenario) -> ScenarioResult:
        with self.env.anchor():
            return super().run(scenario)

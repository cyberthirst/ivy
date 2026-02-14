"""
Boa implementation of the scenario runner.
"""

from typing import Any, Dict, List, Optional, cast

import boa
from boa.contracts.vyper.vyper_contract import VyperDeployer
from boa.util.abi import Address as BoaAddress
from vyper.cli.vyper_json import (
    get_inputs,
    get_output_formats,
    get_search_paths as get_json_search_paths,
)
from vyper.compiler.input_bundle import FileInput, JSONInputBundle
from vyper.compiler.phases import CompilerData
from vyper.compiler.settings import Settings

from fuzzer.runner.base_scenario_runner import BaseScenarioRunner, ScenarioResult
from fuzzer.runner.scenario import Scenario
from fuzzer.trace_types import Env
from fuzzer.coverage.collector import ArcCoverageCollector


def _deployer_from_solc_json(
    solc_json: Dict[str, Any],
    compiler_args: Optional[Dict[str, Any]] = None,
) -> VyperDeployer:
    sources = get_inputs(solc_json)
    output_formats = get_output_formats(solc_json)
    search_paths = get_json_search_paths(solc_json)

    compilation_targets = list(output_formats.keys())
    if not compilation_targets:
        raise ValueError("No compilation targets found in solc_json")

    target = compilation_targets[0]
    input_bundle = JSONInputBundle(sources, search_paths=search_paths)

    if target in input_bundle.input_json:
        file = input_bundle._load_from_path(target, target)
    else:
        file = input_bundle.load_file(target.name)

    if not isinstance(file, FileInput):
        raise ValueError(f"Expected FileInput for {target}, got {type(file)}")

    settings = Settings(**(compiler_args or {}))
    data = CompilerData(file, input_bundle, settings)
    return VyperDeployer(data, filename=str(target))


class BoaScenarioRunner(BaseScenarioRunner):
    """Runner for executing scenarios in Boa."""

    def __init__(
        self,
        collect_storage_dumps: bool = False,
        compiler_settings: Optional[Dict[str, Any]] = None,
        coverage_collector: Optional[ArcCoverageCollector] = None,
        config_name: Optional[str] = None,
    ):
        super().__init__(boa.env, collect_storage_dumps, compiler_settings)
        self.coverage_collector = coverage_collector
        self.config_name = config_name

    def _compile_from_solc_json(
        self,
        solc_json: Dict[str, Any],
        compiler_settings: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Compile source in Boa and return a deployer."""
        if self.coverage_collector is None:
            return _deployer_from_solc_json(solc_json, compiler_args=compiler_settings)

        # Track coverage of Vyper's codegen/IR/venom passes during compilation.
        # VyperDeployer initialization triggers full compilation.
        with self.coverage_collector.collect_compile(config_name=self.config_name):
            return _deployer_from_solc_json(solc_json, compiler_args=compiler_settings)

    def _deploy_compiled(
        self,
        compiled: Any,
        args: List[Any],
        kwargs: Dict[str, Any],
        sender: Optional[str] = None,
        compiler_settings: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Deploy a contract from a compiled deployer in Boa."""
        sender = self._get_sender(sender)

        with self.env.prank(sender):
            self.env.set_balance(
                sender, self._get_balance(sender) + kwargs.get("value", 0) + 10**18
            )
            deployer = cast(VyperDeployer, compiled)
            return deployer.deploy(*args, **kwargs)

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

    def _set_nonce(self, address: str, value: int) -> None:
        addr = BoaAddress(address).canonical_address
        self.env.evm.vm.state.set_nonce(addr, value)

    def _get_nonce(self, address: str) -> int:
        addr = BoaAddress(address).canonical_address
        return self.env.evm.vm.state.get_nonce(addr)

    def _raw_call(
        self,
        to_address: str,
        data: bytes,
        value: int = 0,
        sender: Optional[str] = None,
    ) -> bytes:
        sender = self._get_sender(sender)
        with self.env.prank(sender):
            computation = self.env.raw_call(to_address, value=value, data=data)
            return computation.output

    def _clear_transient_storage(self) -> None:
        self.env.clear_transient_storage()

    def _get_storage_dump(self, contract: Any) -> Optional[Dict[str, Any]]:
        return contract._storage.dump()

    def _get_transient_storage_dump(self, contract: Any) -> Optional[Dict[str, Any]]:
        return contract._transient_storage.dump()

    def _set_block_env(self, trace_env: Optional[Env]) -> None:
        if trace_env is None or trace_env.block is None:
            return
        self.env.evm.patch.block_number = trace_env.block.number
        self.env.evm.patch.timestamp = trace_env.block.timestamp

    def run(self, scenario: Scenario) -> ScenarioResult:
        with self.env.anchor():
            return super().run(scenario)

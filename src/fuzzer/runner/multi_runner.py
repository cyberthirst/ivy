"""
Multi-runner that orchestrates running scenarios across Ivy and multiple Boa configurations.

TODO: Long-term, CompilerConfig list should be loaded from a user config file
rather than hardcoded, allowing users to specify which compiler settings to fuzz.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from vyper.compiler.settings import OptimizationLevel

from .base_scenario_runner import ScenarioResult
from .scenario import Scenario
from .ivy_scenario_runner import IvyScenarioRunner
from .boa_scenario_runner import BoaScenarioRunner
from ..coverage.collector import ArcCoverageCollector


@dataclass
class CompilerConfig:
    """Configuration for a Boa runner with specific compiler settings."""

    name: str
    compiler_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiRunnerResults:
    """Results from running a scenario across all runners."""

    ivy_result: ScenarioResult
    boa_results: Dict[str, tuple[CompilerConfig, ScenarioResult]]


class MultiRunner:
    """Orchestrates running scenarios across Ivy and multiple Boa configurations."""

    # Hardcoded configs for now - will be loaded from user config later
    BOA_CONFIGS = [
        # CompilerConfig("default", {}),
        CompilerConfig("venom", {"experimental_codegen": True}),
        CompilerConfig("codesize", {"optimize": OptimizationLevel.CODESIZE}),
        CompilerConfig("gas", {"optimize": OptimizationLevel.GAS}),
    ]

    def __init__(
        self,
        collect_storage_dumps: bool = False,
        no_solc_json: bool = False,
    ):
        self.boa_configs = self.BOA_CONFIGS
        self.collect_storage_dumps = collect_storage_dumps
        self.no_solc_json = no_solc_json

    def run(
        self,
        scenario: Scenario,
        *,
        coverage_collector: Optional[ArcCoverageCollector] = None,
    ) -> MultiRunnerResults:
        """Run scenario on all runners, creating fresh runner instances."""
        ivy_runner = IvyScenarioRunner(
            collect_storage_dumps=self.collect_storage_dumps,
            no_solc_json=self.no_solc_json,
        )
        ivy_result = ivy_runner.run(scenario)
        boa_results = self._run_boa_configs(scenario, coverage_collector)
        return MultiRunnerResults(ivy_result=ivy_result, boa_results=boa_results)

    def run_boa_only(
        self,
        scenario: Scenario,
        ivy_result: ScenarioResult,
        *,
        coverage_collector: Optional[ArcCoverageCollector] = None,
    ) -> MultiRunnerResults:
        boa_results = self._run_boa_configs(scenario, coverage_collector)

        return MultiRunnerResults(
            ivy_result=ivy_result,
            boa_results=boa_results,
        )

    def _run_boa_configs(
        self,
        scenario: Scenario,
        coverage_collector: Optional[ArcCoverageCollector],
    ) -> Dict[str, tuple[CompilerConfig, ScenarioResult]]:
        boa_results: Dict[str, tuple[CompilerConfig, ScenarioResult]] = {}
        for config in self.boa_configs:
            runner = BoaScenarioRunner(
                compiler_settings=config.compiler_args,
                collect_storage_dumps=self.collect_storage_dumps,
                coverage_collector=coverage_collector,
                config_name=config.name,
            )
            result = runner.run(scenario)
            boa_results[config.name] = (config, result)
        return boa_results

    def get_config_names(self) -> List[str]:
        """Return list of all Boa config names."""
        return [c.name for c in self.boa_configs]

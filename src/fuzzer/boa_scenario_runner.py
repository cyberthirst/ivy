"""
Boa implementation of the scenario runner.
"""

from .scenario_runner import BaseScenarioRunner
from .trace_executor import ExecutionEnvironment
from .boa_env_adapter import BoaEnvAdapter


class BoaScenarioRunner(BaseScenarioRunner):
    """Runner for executing scenarios in Boa."""

    def create_environment(self) -> ExecutionEnvironment:
        """Create a Boa execution environment."""
        return BoaEnvAdapter()

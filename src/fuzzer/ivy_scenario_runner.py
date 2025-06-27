"""
Ivy implementation of the scenario runner.
"""

from .scenario_runner import BaseScenarioRunner
from .trace_executor import ExecutionEnvironment
from .ivy_env_adapter import IvyEnvAdapter
from ivy.frontend.env import Env


class IvyScenarioRunner(BaseScenarioRunner):
    """Runner for executing scenarios in Ivy."""

    def create_environment(self) -> ExecutionEnvironment:
        """Create an Ivy execution environment."""
        env = Env()
        return IvyEnvAdapter(env)

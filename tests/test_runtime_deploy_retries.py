from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fuzzer.runner.base_scenario_runner import (
    BaseScenarioRunner,
    DeploymentResult,
    UNPARSABLE_CONTRACT_FINGERPRINT,
)
from fuzzer.runner.ivy_scenario_runner import IvyDeploymentPreparation
from fuzzer.runner.scenario import Scenario
from fuzzer.runtime_engine.runtime_fuzz_engine import HarnessConfig, RuntimeFuzzEngine
from fuzzer.trace_types import CallTrace, DeploymentTrace, Env, Tx


DEFAULT_SENDER = "0x00000000000000000000000000000000000000AA"
DEFAULT_BALANCE = 10**24


@dataclass
class DummyContract:
    address: str


class DummyState:
    def __init__(self, env: "DummyEnv"):
        self._env = env

    def get_code(self, address: Any) -> object | None:
        runner = self._env.runner
        if runner is None:
            return None
        return object() if str(address) in runner.deployed_contracts else None


class DummyEnv:
    def __init__(self):
        self.runner: Optional[InstrumentedRunner] = None
        self.state = DummyState(self)

    @contextmanager
    def anchor(self):
        yield


class InstrumentedRunner(BaseScenarioRunner):
    def __init__(
        self,
        deployment_outcomes: List[bool],
        *,
        compile_should_fail: bool = False,
    ):
        env = DummyEnv()
        super().__init__(env=env, collect_storage_dumps=False)
        env.runner = self
        self.deployment_outcomes = deployment_outcomes
        self.compile_should_fail = compile_should_fail

        self.compile_calls = 0
        self.deploy_calls = 0
        self.call_calls = 0

        self.deployment_args_seen: List[List[Any]] = []
        self.deployment_kwargs_seen: List[Dict[str, Any]] = []
        self.deployment_start_states: List[tuple[int, int]] = []

        self._balances: Dict[str, int] = {}
        self._nonces: Dict[str, int] = {}

    def _compile_from_solc_json(
        self,
        solc_json: Dict[str, Any],
        compiler_settings: Optional[Dict[str, Any]] = None,
    ) -> Any:
        self.compile_calls += 1
        if self.compile_should_fail:
            raise RuntimeError("compile failure")
        return object()

    def _deploy_compiled(
        self,
        compiled: Any,
        args: List[Any],
        kwargs: Dict[str, Any],
        sender: Optional[str] = None,
        compiler_settings: Optional[Dict[str, Any]] = None,
    ) -> Any:
        del compiled
        del compiler_settings
        sender = self._get_sender(sender)
        self.deployment_start_states.append(
            (self._get_nonce(sender), self._get_balance(sender))
        )

        self.deploy_calls += 1
        self.deployment_args_seen.append(list(args))
        self.deployment_kwargs_seen.append(dict(kwargs))

        # Simulate sender drift that must be rolled back on failed constructor init.
        self._set_nonce(sender, self._get_nonce(sender) + 1)
        self._set_balance(sender, self._get_balance(sender) - kwargs.get("value", 0) - 1)

        idx = self.deploy_calls - 1
        if idx >= len(self.deployment_outcomes) or not self.deployment_outcomes[idx]:
            raise RuntimeError(f"init failure {self.deploy_calls}")

        return DummyContract(address=f"0x{self.deploy_calls:040x}")

    def _call_method(
        self,
        contract: Any,
        method_name: str,
        args: List[Any],
        kwargs: Dict[str, Any],
        sender: Optional[str] = None,
    ) -> Any:
        del contract
        del method_name
        del args
        del kwargs
        del sender
        self.call_calls += 1
        return b""

    def _raw_call(
        self,
        to_address: str,
        data: bytes,
        value: int = 0,
        sender: Optional[str] = None,
    ) -> bytes:
        del to_address
        del data
        del value
        del sender
        self.call_calls += 1
        return b""

    def _set_balance(self, address: str, value: int) -> None:
        self._balances[address] = value

    def _get_balance(self, address: str) -> int:
        if address not in self._balances:
            self._balances[address] = DEFAULT_BALANCE
        return self._balances[address]

    def _set_nonce(self, address: str, value: int) -> None:
        self._nonces[address] = value

    def _get_nonce(self, address: str) -> int:
        return self._nonces.get(address, 0)

    def _clear_transient_storage(self) -> None:
        return None

    def _get_storage_dump(self, contract: Any) -> Optional[Dict[str, Any]]:
        del contract
        return None

    def _get_transient_storage_dump(self, contract: Any) -> Optional[Dict[str, Any]]:
        del contract
        return None

    def _set_block_env(self, trace_env: Optional[Env]) -> None:
        del trace_env
        return None

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
            contract_fingerprint=UNPARSABLE_CONTRACT_FINGERPRINT,
            compiled=compiled,
        )


def _make_deployment_trace(
    *,
    sender: str = DEFAULT_SENDER,
    value: int = 7,
    python_args: Optional[Dict[str, Any]] = None,
) -> DeploymentTrace:
    return DeploymentTrace(
        deployment_type="source",
        calldata=None,
        value=value,
        solc_json={"language": "Vyper", "sources": {"test.vy": {"content": ""}}},
        blueprint_initcode_prefix=None,
        deployed_address="0x0000000000000000000000000000000000000100",
        deployment_succeeded=True,
        env=Env(tx=Tx(origin=sender)),
        python_args=python_args,
    )


def _make_call_trace(*, to_address: str, sender: str = DEFAULT_SENDER) -> CallTrace:
    return CallTrace(
        output=None,
        call_args={"to": to_address, "value": 0},
        call_succeeded=True,
        env=Env(tx=Tx(origin=sender)),
        python_args=None,
        function_name=None,
    )


def _new_engine_with_runner(
    runner: InstrumentedRunner,
    *,
    max_total_calls: int = 0,
    max_replay_calls: int = 0,
    max_deploy_retries: int = 30,
) -> RuntimeFuzzEngine:
    config = HarnessConfig(
        max_total_calls=max_total_calls,
        max_enumeration_calls=0,
        max_replay_calls=max_replay_calls,
        max_fuzz_calls=0,
        max_deploy_retries=max_deploy_retries,
    )
    engine = RuntimeFuzzEngine(config=config, seed=1)
    engine.runner = runner
    return engine


def test_execute_trace_reuses_precompiled_deployment_context():
    runner = InstrumentedRunner([False, False, True])
    trace = _make_deployment_trace(python_args={"args": [1], "kwargs": {}})

    assert trace.solc_json is not None
    merged_settings = {
        **(trace.compiler_settings or {}),
        **runner.compiler_settings,
    }
    prepared = runner._compile_from_solc_json(trace.solc_json, merged_settings)

    result = None
    for i in range(3):
        assert trace.python_args is not None
        trace.python_args["args"] = [i]
        trace.python_args["kwargs"] = {}
        trace.value = i
        result = runner.execute_trace(
            trace=trace,
            trace_index=0,
            use_python_args=True,
            compiled_artifact=prepared,
        )

    assert result is not None
    assert result.result is not None
    assert result.result.success is True
    assert runner.compile_calls == 1
    assert runner.deploy_calls == 3


def test_runtime_retries_stop_at_first_success_and_update_trace():
    runner = InstrumentedRunner([False, False, True, True])
    engine = _new_engine_with_runner(runner, max_deploy_retries=30)
    trace = _make_deployment_trace(python_args={"args": [11], "kwargs": {"salt": 99}})
    initial_value = trace.value
    scenario = Scenario(traces=[trace], dependencies=[], use_python_args=True)

    harness_result = engine.run(scenario)
    deployment_result = harness_result.ivy_result.results[0].result

    assert deployment_result is not None
    assert deployment_result.success is True
    assert runner.compile_calls == 1
    assert runner.deploy_calls == 3
    assert harness_result.stats.deployment_attempts == 3
    assert harness_result.stats.deployment_successes == 1
    assert harness_result.stats.deployment_failures == 2
    assert runner.deployment_args_seen[0] == [11]
    assert runner.deployment_kwargs_seen[0]["value"] == initial_value

    winning_args = runner.deployment_args_seen[2]
    winning_kwargs = runner.deployment_kwargs_seen[2]
    assert trace.python_args is not None
    assert trace.python_args["args"] == winning_args
    assert trace.python_args["kwargs"] == {"salt": 99}
    assert "value" not in trace.python_args["kwargs"]
    assert trace.value == winning_kwargs["value"]


def test_runtime_retries_cap_at_max_attempts():
    runner = InstrumentedRunner([False] * 40)
    engine = _new_engine_with_runner(runner, max_deploy_retries=30)
    trace = _make_deployment_trace(python_args={"args": [1], "kwargs": {}})
    scenario = Scenario(traces=[trace], dependencies=[], use_python_args=True)

    harness_result = engine.run(scenario)
    deployment_result = harness_result.ivy_result.results[0].result

    assert deployment_result is not None
    assert deployment_result.success is False
    assert runner.compile_calls == 1
    assert runner.deploy_calls == 30
    assert harness_result.stats.deployment_attempts == 30
    assert harness_result.stats.deployment_successes == 0
    assert harness_result.stats.deployment_failures == 30
    assert runner.deployment_start_states == [(0, DEFAULT_BALANCE)] * 30

    # Terminal failed attempt is committed (no rollback after final retry).
    sender = runner._get_sender(trace.env.tx.origin)
    final_value = runner.deployment_kwargs_seen[-1]["value"]
    assert runner._get_nonce(sender) == 1
    assert runner._get_balance(sender) == DEFAULT_BALANCE - final_value - 1


def test_runtime_retries_restore_sender_nonce_and_balance():
    runner = InstrumentedRunner([False, False, True])
    engine = _new_engine_with_runner(runner, max_deploy_retries=30)
    trace = _make_deployment_trace(python_args={"args": [3], "kwargs": {}})
    scenario = Scenario(traces=[trace], dependencies=[], use_python_args=True)

    _ = engine.run(scenario)

    sender = runner._get_sender(trace.env.tx.origin)
    assert runner.deployment_start_states == [(0, DEFAULT_BALANCE)] * 3

    winning_value = runner.deployment_kwargs_seen[-1]["value"]
    assert runner._get_nonce(sender) == 1
    assert runner._get_balance(sender) == DEFAULT_BALANCE - winning_value - 1


def test_call_replay_still_runs_after_successful_deployment():
    runner = InstrumentedRunner([True])
    engine = _new_engine_with_runner(
        runner,
        max_total_calls=1,
        max_replay_calls=1,
        max_deploy_retries=30,
    )
    deployment_trace = _make_deployment_trace(python_args={"args": [], "kwargs": {}})
    call_trace = _make_call_trace(to_address=deployment_trace.deployed_address)
    scenario = Scenario(
        traces=[deployment_trace, call_trace],
        dependencies=[],
        use_python_args=True,
    )

    harness_result = engine.run(scenario)

    assert harness_result.stats.replay_calls == 1
    assert harness_result.stats.call_attempts == 1
    assert harness_result.stats.call_successes == 1
    assert harness_result.stats.call_failures == 0
    assert harness_result.stats.calls_to_no_code == 0
    assert runner.call_calls == 1


def test_call_replay_counts_calls_to_no_code_when_target_not_deployed():
    runner = InstrumentedRunner([])
    engine = _new_engine_with_runner(
        runner,
        max_total_calls=1,
        max_replay_calls=1,
        max_deploy_retries=30,
    )
    call_trace = _make_call_trace(
        to_address="0x00000000000000000000000000000000000000BB"
    )
    scenario = Scenario(
        traces=[call_trace],
        dependencies=[],
        use_python_args=True,
    )

    harness_result = engine.run(scenario)

    assert harness_result.stats.replay_calls == 1
    assert harness_result.stats.call_attempts == 1
    assert harness_result.stats.call_successes == 1
    assert harness_result.stats.call_failures == 0
    assert harness_result.stats.calls_to_no_code == 1
    assert runner.call_calls == 1

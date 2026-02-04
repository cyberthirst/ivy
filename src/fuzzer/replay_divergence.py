"""
Replay a saved divergence by reconstructing the exact mutated scenario.

Usage:
  PYTHONPATH=src python -m fuzzer.replay_divergence path/to/divergence.json
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from fuzzer.export_utils import normalize_compiler_settings
from fuzzer.trace_types import DeploymentTrace, CallTrace, Env, Tx, Block
from fuzzer.xfail import XFailExpectation
from fuzzer.runner.scenario import Scenario
from fuzzer.runner.multi_runner import MultiRunner
from fuzzer.divergence_detector import DivergenceDetector


def _load_divergence(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _deserialize_env(data: Dict[str, Any] | None) -> Env | None:
    """Deserialize Env from dict."""
    if data is None:
        return None
    block_data = data.get("block")
    return Env(
        tx=Tx(**data["tx"]),
        block=None if block_data is None else Block(**block_data),
    )


def _deserialize_xfails(data: List[Dict[str, Any]] | None) -> List[XFailExpectation]:
    """Deserialize list of XFailExpectation from dicts."""
    if not data:
        return []
    return [XFailExpectation(**x) for x in data]


def _deserialize_deployment_trace(data: Dict[str, Any]) -> DeploymentTrace:
    env_data = data.get("env")
    if env_data is None:
        raise ValueError("DeploymentTrace requires env data")

    env = _deserialize_env(env_data)
    assert env is not None

    solc_json = data.get("solc_json")
    if data["deployment_type"] == "source" and not solc_json:
        raise ValueError("source deployment trace missing solc_json")

    return DeploymentTrace(
        deployment_type=data["deployment_type"],
        calldata=data.get("calldata"),
        value=data["value"],
        solc_json=solc_json,
        blueprint_initcode_prefix=data.get("blueprint_initcode_prefix"),
        deployed_address=data["deployed_address"],
        deployment_succeeded=data["deployment_succeeded"],
        env=env,
        python_args=data.get("python_args"),
        compiler_settings=normalize_compiler_settings(data.get("compiler_settings")),
        compilation_xfails=_deserialize_xfails(data.get("compilation_xfails")),
        runtime_xfails=_deserialize_xfails(data.get("runtime_xfails")),
    )


def _deserialize_call_trace(data: Dict[str, Any]) -> CallTrace:
    """Deserialize CallTrace from dict."""
    return CallTrace(
        output=data.get("output"),
        call_args=data["call_args"],
        call_succeeded=data.get("call_succeeded"),
        env=_deserialize_env(data.get("env")),
        python_args=data.get("python_args"),
        function_name=data.get("function_name"),
        runtime_xfails=_deserialize_xfails(data.get("runtime_xfails")),
    )


def _build_scenario_from_traces(data: dict) -> Scenario:
    """Build a Scenario directly from divergence trace data."""
    traces_data = data.get("traces", [])

    traces = []
    for trace_data in traces_data:
        # Check for deployment_type to identify DeploymentTrace
        if "deployment_type" in trace_data:
            traces.append(_deserialize_deployment_trace(trace_data))
        # Check for call_args to identify CallTrace
        elif "call_args" in trace_data:
            traces.append(_deserialize_call_trace(trace_data))

    return Scenario(traces=traces, use_python_args=True)


def replay_divergence(divergence_path: Path) -> bool:
    data = _load_divergence(divergence_path)

    if not data.get("traces"):
        raise ValueError("Cannot replay: no traces in divergence file")

    scenario = _build_scenario_from_traces(data)
    return _run_and_check(scenario)


def _run_and_check(scenario: Scenario) -> bool:
    """Run scenario and check for divergences."""
    multi_runner = MultiRunner(collect_storage_dumps=True)
    detector = DivergenceDetector()

    results = multi_runner.run(scenario)

    divergences = detector.compare_all_results(
        results.ivy_result, results.boa_results, scenario
    )
    return len(divergences) > 0


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        print("Usage: python -m fuzzer.replay_divergence path/to/divergence.json")
        raise SystemExit(2)

    path = Path(argv[1])
    ok = replay_divergence(path)
    if ok:
        print("Reproduced divergence")
        raise SystemExit(0)
    else:
        print("No divergence reproduced; Ivy and Boa matched")
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv)

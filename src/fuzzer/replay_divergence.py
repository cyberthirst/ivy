"""
Replay a saved divergence by reconstructing the exact mutated scenario.

Usage:
  PYTHONPATH=src python -m fuzzer.replay_divergence path/to/divergence.json
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from vyper.compiler.settings import OptimizationLevel

from .test_fuzzer import TestFuzzer
from .export_utils import TestFilter
from .trace_types import DeploymentTrace, CallTrace, Env, Tx, Block
from .xfail import XFailExpectation
from .runner.scenario import Scenario
from .runner.multi_runner import MultiRunner
from .divergence_detector import DivergenceDetector


def _load_divergence(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _find_item(fuzzer: TestFuzzer, item_name: str):
    exports = fuzzer.load_filtered_exports(TestFilter(exclude_multi_module=True))
    for export in exports.values():
        if item_name in export.items:
            return export.items[item_name]
    return None


def _deserialize_env(data: Dict[str, Any] | None) -> Env | None:
    """Deserialize Env from dict."""
    if data is None:
        return None
    return Env(
        tx=Tx(**data["tx"]),
        block=Block(**data["block"]),
    )


def _deserialize_xfails(data: List[Dict[str, Any]] | None) -> List[XFailExpectation]:
    """Deserialize list of XFailExpectation from dicts."""
    if not data:
        return []
    return [XFailExpectation(**x) for x in data]


def _deserialize_compiler_settings(
    data: Dict[str, Any] | None,
) -> Dict[str, Any] | None:
    """Deserialize compiler settings, converting string enums back to proper types."""
    if not data:
        return None
    result = dict(data)
    # Convert 'optimize' string back to OptimizationLevel enum
    if "optimize" in result and isinstance(result["optimize"], str):
        opt_str = result["optimize"].upper()
        result["optimize"] = OptimizationLevel[opt_str]
    return result


def _deserialize_deployment_trace(data: Dict[str, Any]) -> DeploymentTrace:
    """Deserialize DeploymentTrace from dict."""
    return DeploymentTrace(
        deployment_type=data["deployment_type"],
        contract_abi=data["contract_abi"],
        initcode=data["initcode"],
        calldata=data.get("calldata"),
        value=data["value"],
        source_code=data.get("source_code"),
        annotated_ast=data.get("annotated_ast"),
        solc_json=data.get("solc_json"),
        raw_ir=data.get("raw_ir"),
        blueprint_initcode_prefix=data.get("blueprint_initcode_prefix"),
        deployed_address=data["deployed_address"],
        runtime_bytecode=data["runtime_bytecode"],
        deployment_succeeded=data["deployment_succeeded"],
        env=_deserialize_env(data.get("env")),
        python_args=data.get("python_args"),
        compiler_settings=_deserialize_compiler_settings(data.get("compiler_settings")),
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

    item_name = data.get("item_name")
    base_seed = data.get("seed")
    scenario_seed = data.get("scenario_seed")

    # Try reconstruction from exports first (original behavior)
    if item_name and base_seed is not None and scenario_seed is not None:
        fuzzer = TestFuzzer(seed=base_seed)
        item = _find_item(fuzzer, item_name)

        if item is not None:
            scenario = fuzzer.create_mutated_scenario(item, scenario_seed=scenario_seed)
            return _run_and_check(scenario)

    # Fallback: build scenario directly from stored traces
    if data.get("traces"):
        scenario = _build_scenario_from_traces(data)
        return _run_and_check(scenario)

    raise ValueError(
        "Cannot replay: no item found in exports and no traces in divergence"
    )


def _run_and_check(scenario: Scenario) -> bool:
    """Run scenario and check for divergences."""
    multi_runner = MultiRunner(collect_storage_dumps=True, no_solc_json=True)
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

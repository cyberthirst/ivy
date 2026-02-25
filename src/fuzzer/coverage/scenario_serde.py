from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fuzzer.runner.scenario import Scenario
from fuzzer.trace_types import (
    CallTrace,
    ClearTransientStorageTrace,
    DeploymentTrace,
    SetBalanceTrace,
    Trace,
)


def serialize_scenario(scenario: Scenario) -> dict[str, Any]:
    return {
        "traces": [_serialize_trace(trace) for trace in scenario.traces],
        "dependencies": [
            serialize_scenario(dep_scenario) for dep_scenario in scenario.dependencies
        ],
        "scenario_id": scenario.scenario_id,
        "use_python_args": scenario.use_python_args,
    }


def deserialize_scenario(data: dict[str, Any]) -> Scenario:
    return Scenario(
        traces=[
            _deserialize_trace(trace_data) for trace_data in data.get("traces", [])
        ],
        dependencies=[
            deserialize_scenario(dep_data) for dep_data in data.get("dependencies", [])
        ],
        scenario_id=data.get("scenario_id"),
        use_python_args=data.get("use_python_args", True),
    )


def _serialize_trace(trace: Trace) -> dict[str, Any]:
    trace_data = asdict(trace)
    if isinstance(trace, DeploymentTrace):
        trace_data["trace_type"] = "deployment"
    elif isinstance(trace, CallTrace):
        trace_data["trace_type"] = "call"
    elif isinstance(trace, SetBalanceTrace):
        trace_data["trace_type"] = "set_balance"
    elif isinstance(trace, ClearTransientStorageTrace):
        trace_data["trace_type"] = "clear_transient_storage"
    else:
        raise ValueError(f"Unsupported trace type: {type(trace)}")
    return trace_data


def _deserialize_deployment_trace(data: dict[str, Any]) -> DeploymentTrace:
    from fuzzer.replay_divergence import (
        _deserialize_deployment_trace as _replay_deserialize_deployment_trace,
    )

    return _replay_deserialize_deployment_trace(data)


def _deserialize_call_trace(data: dict[str, Any]) -> CallTrace:
    from fuzzer.replay_divergence import (
        _deserialize_call_trace as _replay_deserialize_call_trace,
    )

    return _replay_deserialize_call_trace(data)


def _deserialize_trace(trace_data: dict[str, Any]) -> Trace:
    trace_type = trace_data.get("trace_type")

    if trace_type == "deployment":
        return _deserialize_deployment_trace(trace_data)
    if trace_type == "call":
        return _deserialize_call_trace(trace_data)
    if trace_type == "set_balance":
        return SetBalanceTrace(
            address=trace_data["address"],
            value=trace_data["value"],
        )
    if trace_type == "clear_transient_storage":
        return ClearTransientStorageTrace()

    if "deployment_type" in trace_data:
        return _deserialize_deployment_trace(trace_data)
    if "call_args" in trace_data:
        return _deserialize_call_trace(trace_data)
    if "address" in trace_data and "value" in trace_data:
        return SetBalanceTrace(address=trace_data["address"], value=trace_data["value"])

    return ClearTransientStorageTrace()

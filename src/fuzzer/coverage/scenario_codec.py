"""
JSON codec for Scenario serialization with Enum round-tripping.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import asdict, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Type

from ..runner.scenario import Scenario
from ..trace_types import (
    Block,
    CallTrace,
    ClearTransientStorageTrace,
    DeploymentTrace,
    Env,
    SetBalanceTrace,
    Trace,
    Tx,
)
from ..xfail import XFailExpectation

SCHEMA_VERSION = 1

TRACE_TYPE_MAP: Dict[str, Type[Trace]] = {
    "deployment": DeploymentTrace,
    "call": CallTrace,
    "set_balance": SetBalanceTrace,
    "clear_transient_storage": ClearTransientStorageTrace,
}

TRACE_CLASS_TO_TYPE: Dict[Type[Trace], str] = {v: k for k, v in TRACE_TYPE_MAP.items()}

# Fields to exclude from scenario payloads (large, not needed for fuzzing)
# TODO we should preemptively strip those during fuzzer bootstrap
TRIM_DEPLOYMENT_FIELDS = frozenset(
    {
        "annotated_ast",
        "initcode",
        "runtime_bytecode",
        "raw_ir",
    }
)


def _encode_enum(obj: Enum) -> Dict[str, str]:
    return {
        "__enum__": f"{obj.__class__.__module__}.{obj.__class__.__name__}",
        "name": obj.name,
    }


def _decode_enum(d: Dict[str, str]) -> Enum:
    module_class = d["__enum__"]
    module_path, class_name = module_class.rsplit(".", 1)
    module = importlib.import_module(module_path)
    enum_class = getattr(module, class_name)
    return enum_class[d["name"]]


def _is_enum_dict(obj: Any) -> bool:
    return isinstance(obj, dict) and "__enum__" in obj and "name" in obj


class ScenarioEncoder(json.JSONEncoder):
    def __init__(self, *args, trim_large_fields: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.trim_large_fields = trim_large_fields

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Enum):
            return _encode_enum(obj)
        if isinstance(obj, Path):
            return str(obj)
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
        return super().default(obj)

    def encode(self, obj: Any) -> str:
        return super().encode(self._preprocess(obj))

    def _preprocess(self, obj: Any) -> Any:
        if isinstance(obj, Enum):
            return _encode_enum(obj)
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: self._preprocess(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._preprocess(item) for item in obj]
        if is_dataclass(obj) and not isinstance(obj, type):
            return self._preprocess_dataclass(obj)
        return obj

    def _preprocess_dataclass(self, obj: Any) -> Dict[str, Any]:
        result = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            if self.trim_large_fields and isinstance(obj, DeploymentTrace):
                if f.name in TRIM_DEPLOYMENT_FIELDS:
                    continue
                if f.name == "solc_json" and value is not None:
                    value = {
                        k: v for k, v in value.items()
                        if k in ("sources", "settings", "language")
                    }
            result[f.name] = self._preprocess(value)
        return result


def _decode_xfail(d: Dict[str, Any]) -> XFailExpectation:
    return XFailExpectation(
        kind=d.get("kind", ""),
        reason=d.get("reason"),
    )


def _decode_tx(d: Dict[str, Any]) -> Tx:
    return Tx(
        origin=d["origin"],
        gas=d["gas"],
        gas_price=d["gas_price"],
        blob_hashes=d.get("blob_hashes", []),
    )


def _decode_block(d: Dict[str, Any]) -> Block:
    return Block(
        number=d["number"],
        timestamp=d["timestamp"],
        gas_limit=d["gas_limit"],
        excess_blob_gas=d.get("excess_blob_gas"),
    )


def _decode_env(d: Dict[str, Any]) -> Env:
    return Env(
        tx=_decode_tx(d["tx"]),
        block=_decode_block(d["block"]),
    )


def _decode_value(obj: Any) -> Any:
    if _is_enum_dict(obj):
        return _decode_enum(obj)
    if isinstance(obj, dict):
        return {k: _decode_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_value(item) for item in obj]
    return obj


def _decode_deployment_trace(d: Dict[str, Any]) -> DeploymentTrace:
    compilation_xfails = [_decode_xfail(x) for x in d.get("compilation_xfails", [])]
    runtime_xfails = [_decode_xfail(x) for x in d.get("runtime_xfails", [])]
    compiler_settings = _decode_value(d.get("compiler_settings"))

    return DeploymentTrace(
        deployment_type=d["deployment_type"],
        contract_abi=d.get("contract_abi", []),
        initcode=d.get("initcode", ""),
        calldata=d.get("calldata"),
        value=d.get("value", 0),
        source_code=d.get("source_code"),
        annotated_ast=d.get("annotated_ast"),
        solc_json=d.get("solc_json"),
        raw_ir=d.get("raw_ir"),
        blueprint_initcode_prefix=d.get("blueprint_initcode_prefix"),
        deployed_address=d["deployed_address"],
        runtime_bytecode=d.get("runtime_bytecode", ""),
        deployment_succeeded=d.get("deployment_succeeded", True),
        env=_decode_env(d["env"]),
        python_args=d.get("python_args"),
        compiler_settings=compiler_settings,
        compilation_xfails=compilation_xfails,
        runtime_xfails=runtime_xfails,
    )


def _decode_call_trace(d: Dict[str, Any]) -> CallTrace:
    runtime_xfails = [_decode_xfail(x) for x in d.get("runtime_xfails", [])]
    env = _decode_env(d["env"]) if d.get("env") else None

    return CallTrace(
        output=d.get("output"),
        call_args=d.get("call_args", {}),
        call_succeeded=d.get("call_succeeded"),
        env=env,
        python_args=d.get("python_args"),
        function_name=d.get("function_name"),
        runtime_xfails=runtime_xfails,
    )


def _decode_set_balance_trace(d: Dict[str, Any]) -> SetBalanceTrace:
    return SetBalanceTrace(
        address=d["address"],
        value=d["value"],
    )


def _decode_clear_transient_storage_trace(
    d: Dict[str, Any],
) -> ClearTransientStorageTrace:
    return ClearTransientStorageTrace()


def _decode_trace(d: Dict[str, Any]) -> Trace:
    trace_type = d.get("trace_type")
    if trace_type == "deployment":
        return _decode_deployment_trace(d)
    elif trace_type == "call":
        return _decode_call_trace(d)
    elif trace_type == "set_balance":
        return _decode_set_balance_trace(d)
    elif trace_type == "clear_transient_storage":
        return _decode_clear_transient_storage_trace(d)
    else:
        raise ValueError(f"Unknown trace type: {trace_type}")


def _encode_trace(trace: Trace) -> Dict[str, Any]:
    trace_type = TRACE_CLASS_TO_TYPE.get(type(trace))
    if trace_type is None:
        raise ValueError(f"Unknown trace class: {type(trace)}")

    encoder = ScenarioEncoder(trim_large_fields=True)
    d = encoder._preprocess(trace)
    d["trace_type"] = trace_type
    return d


def encode_scenario(
    scenario: Scenario, *, trim_large_fields: bool = True
) -> Dict[str, Any]:
    traces = [_encode_trace(t) for t in scenario.traces]
    dependencies = [
        encode_scenario(dep, trim_large_fields=trim_large_fields)
        for dep in scenario.dependencies
    ]

    return {
        "schema": SCHEMA_VERSION,
        "scenario": {
            "traces": traces,
            "dependencies": dependencies,
            "scenario_id": scenario.scenario_id,
            "use_python_args": scenario.use_python_args,
        },
    }


def decode_scenario(d: Dict[str, Any]) -> Scenario:
    schema = d.get("schema", 1)
    if schema != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema version: {schema}")

    scenario_data = d["scenario"]
    traces = [_decode_trace(t) for t in scenario_data.get("traces", [])]
    dependencies = [
        decode_scenario(dep) for dep in scenario_data.get("dependencies", [])
    ]

    return Scenario(
        traces=traces,
        dependencies=dependencies,
        scenario_id=scenario_data.get("scenario_id"),
        use_python_args=scenario_data.get("use_python_args", True),
    )


def scenario_to_json(scenario: Scenario, *, trim_large_fields: bool = True) -> str:
    data = encode_scenario(scenario, trim_large_fields=trim_large_fields)
    return json.dumps(data, indent=2)


def scenario_from_json(json_str: str) -> Scenario:
    data = json.loads(json_str)
    return decode_scenario(data)

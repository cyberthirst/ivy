"""
Data types for representing Vyper test traces (export format).

Traces match the JSON export format and contain expected outputs for validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeAlias, Union

from fuzzer.xfail import XFailExpectation


@dataclass
class Tx:
    origin: str
    gas: int = 10_000_000_000
    gas_price: int = 0
    blob_hashes: List[str] = field(default_factory=list)


@dataclass
class Block:
    number: int
    timestamp: int
    gas_limit: int
    excess_blob_gas: Optional[int]


@dataclass
class Env:
    """Represents environment data for a trace."""

    tx: Tx
    block: Optional[Block] = None


@dataclass
class DeploymentTrace:
    """Represents a contract deployment trace."""

    deployment_type: str  # "source", "ir", "blueprint", "raw_bytecode"
    contract_abi: List[Dict[str, Any]]
    initcode: str
    calldata: Optional[str]
    value: int
    source_code: Optional[str]
    annotated_ast: Optional[Dict[str, Any]]
    solc_json: Optional[Dict[str, Any]]
    raw_ir: Optional[str]
    blueprint_initcode_prefix: Optional[str]
    deployed_address: str
    runtime_bytecode: str
    deployment_succeeded: bool
    env: Env
    python_args: Optional[Dict[str, Any]] = None  # {"args": [], "kwargs": {}}
    compiler_settings: Optional[Dict[str, Any]] = None
    compilation_xfails: List[XFailExpectation] = field(default_factory=list)
    runtime_xfails: List[XFailExpectation] = field(default_factory=list)

    def to_trace_info(self, index: int) -> Dict[str, Any]:
        info: Dict[str, Any] = {"type": self.__class__.__name__, "index": index}
        if self.source_code:
            info["source_code"] = self.source_code
        if self.python_args:
            info["args"] = self.python_args
        return info


@dataclass
class CallTrace:
    """Represents a function call trace."""

    output: Optional[str]
    call_args: Dict[str, Any]
    call_succeeded: Optional[bool] = None
    env: Optional[Env] = None
    python_args: Optional[Dict[str, Any]] = None  # {"args": [], "kwargs": {}}
    function_name: Optional[str] = None
    runtime_xfails: List[XFailExpectation] = field(default_factory=list)

    def to_trace_info(self, index: int) -> Dict[str, Any]:
        info: Dict[str, Any] = {"type": self.__class__.__name__, "index": index}
        if self.function_name:
            info["function"] = self.function_name
        elif self.python_args and "method" in self.python_args:
            info["function"] = self.python_args["method"]
        if self.python_args:
            info["args"] = self.python_args
        return info


@dataclass
class SetBalanceTrace:
    """Represents a set_balance trace."""

    address: str
    value: int

    def to_trace_info(self, index: int) -> Dict[str, Any]:
        return {"type": self.__class__.__name__, "index": index}


@dataclass
class ClearTransientStorageTrace:
    """Represents a clear_transient_storage trace."""

    # No fields needed for this trace type

    def to_trace_info(self, index: int) -> Dict[str, Any]:
        return {"type": self.__class__.__name__, "index": index}


# Type alias for any trace type
Trace: TypeAlias = Union[
    DeploymentTrace, CallTrace, SetBalanceTrace, ClearTransientStorageTrace
]


@dataclass
class TestItem:
    """Represents a test or fixture with its traces."""

    name: str
    item_type: str  # "test" or "fixture"
    deps: List[str]
    traces: List[Trace]


@dataclass
class TestExport:
    """Container for all test exports from a file."""

    path: Path
    items: Dict[str, TestItem] = field(default_factory=dict)

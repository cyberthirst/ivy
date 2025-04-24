import os
import json
from importlib import resources
from vyper.compiler.phases import CompilerData

DUMP_FLAG_ENV_VAR = "DUMP_CONTRACTS"
DUMP_ENABLED = os.environ.get(DUMP_FLAG_ENV_VAR, "false").lower() in (
    "true",
    "1",
    "yes",
)

# Package and resource name for the bundled JSON database
# Assumes the JSON lives in src/dumper/data/contract_db.json
RESOURCE_PACKAGE = __package__ + ".data"
RESOURCE_NAME = "contract_db.json"


def load_records() -> list:
    try:
        data_str = resources.read_text(RESOURCE_PACKAGE, RESOURCE_NAME)
        return json.loads(data_str)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_records(records: list) -> None:
    db_path = resources.files(RESOURCE_PACKAGE).joinpath(RESOURCE_NAME)
    db_path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def dump_contract(compiler_data: CompilerData, ctor_args: str, calldata_hex: str):
    """
    Dumps contract source, constructor args, and calldata if DUMP_CONTRACTS flag is set, with deduplication.
    """
    if not DUMP_ENABLED:
        return

    # Only handle single-file contracts without constructor args
    # rest is not yet supported
    source_code = compiler_data.source_code
    if len(compiler_data.resolved_imports.seen) != 1:
        return
    if ctor_args:
        return

    records = load_records()

    # Look for a matching entry by source + ctor_args
    for rec in records:
        if rec.get("source") == source_code and rec.get("ctor_args", "") == ctor_args:
            # Append new calldata if not already recorded
            if calldata_hex and calldata_hex not in rec.get("calldatas", []):
                # print(
                #    f"Adding calldata for contract: {calldata_hex[:30]}...",
                #    file=sys.stderr,
                # )
                rec.setdefault("calldatas", []).append(calldata_hex)
                _save_records(records)
            return

    new_entry = {
        "source": source_code,
        "ctor_args": ctor_args,
        "calldatas": [calldata_hex],
    }
    records.append(new_entry)
    _save_records(records)

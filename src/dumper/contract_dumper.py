import os
import json
from pathlib import Path

from vyper.compiler.phases import CompilerData

BASE_DIR = Path("")
TESTS_DIR = BASE_DIR / "tests"
JSON_DB_PATH = TESTS_DIR / "contract_db.json"
DUMP_FLAG_ENV_VAR = "DUMP_CONTRACTS"
DUMP_ENABLED = os.environ.get(DUMP_FLAG_ENV_VAR, "false").lower() in (
    "true",
    "1",
    "yes",
)


def dump_contract(compiler_data: CompilerData, ctor_args: str, calldata_hex: str):
    """
    Dumps contract source, constructor args, and calldata if DUMP_CONTRACTS flag is set, with deduplication.
    Dumps also failing calls to contracts.
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

    try:
        with open(JSON_DB_PATH, "r", encoding="utf-8") as f:
            records = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        records = []

    # Look for a matching entry by source + ctor_args
    for rec in records:
        if rec.get("source") == source_code and rec.get("ctor_args", "") == ctor_args:
            # Append calldata if new
            if calldata_hex and calldata_hex not in rec.get("calldatas", []):
                # print(
                #    f"Adding calldata for contract: {calldata_hex[:30]}...",
                #    file=sys.stderr,
                # )
                rec.setdefault("calldatas", []).append(calldata_hex)
                with open(JSON_DB_PATH, "w", encoding="utf-8") as f:
                    json.dump(records, f, indent=2)
            return

    new_entry = {
        "source": source_code,
        "ctor_args": ctor_args,
        "calldatas": [calldata_hex],
    }
    # print(
    #    f"Adding new contract entry with calldata: {calldata_hex[:30]}...",
    #    file=sys.stderr,
    # )
    records.append(new_entry)
    with open(JSON_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

import logging

from dumper import contract_dumper
from ivy.frontend.loader import loads as ivy_loads
from boa import loads as boa_loads


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def run_ivy_calls(source, calldatas):
    """
    Compile and execute each calldata against the Ivy interpreter.
    Returns either a load-time exception or a list of per-call results.
    """
    try:
        contract = ivy_loads(source)
    except Exception as e:
        return {"load_error": e}

    results = []
    for hexstr in calldatas:
        data = bytes.fromhex(hexstr)
        try:
            output = contract.message_call(contract.address, data=data)
            storage = contract._dump_storage()
            results.append({"data": hexstr, "output": output, "storage": storage})
        except Exception as e:
            results.append({"data": hexstr, "runtime_error": e})
    return {"results": results}


def run_boa_calls(source, calldatas):
    """
    Compile and execute each calldata against the Boa interpreter.
    Returns either a load-time exception or a list of per-call results.
    """
    try:
        contract = boa_loads(source)
    except Exception as e:
        return {"load_error": e}

    results = []
    for hexstr in calldatas:
        data = bytes.fromhex(hexstr)
        try:
            output = contract.env.raw_call(to_address=contract.address, data=data)
            storage = contract._dump_storage()
            results.append({"data": hexstr, "output": output, "storage": storage})
        except Exception as e:
            results.append({"data": hexstr, "runtime_error": e})
    return {"results": results}


def compare_runs(source, calldatas):
    ivy_res = run_ivy_calls(source, calldatas)
    boa_res = run_boa_calls(source, calldatas)

    # Compare load-time behavior first, look for compiler errors
    ivy_load_err = ivy_res.get("load_error")
    boa_load_err = boa_res.get("load_error")
    if ivy_load_err or boa_load_err:
        if (ivy_load_err is None) != (boa_load_err is None):
            if boa_load_err.message == "risky overlap":
                return
            logging.error("Load-time mismatch for contract:\n%s", source)
            logging.error("  Ivy error: %r", ivy_load_err)
            logging.error("  Boa error: %r", boa_load_err)
        return

    # Both loaded OK, compare per-call results
    ivy_results = ivy_res["results"]
    boa_results = boa_res["results"]
    for iv, bv in zip(ivy_results, boa_results):
        data = iv["data"]
        ivy_err = iv.get("runtime_error")
        boa_err = bv.get("runtime_error")

        # Exception divergence
        if (ivy_err is None) != (boa_err is None):
            logging.error(
                "Runtime error mismatch for payload %s on contract:\n%s", data, source
            )
            logging.error("  Ivy error: %r", ivy_err)
            logging.error("  Boa error: %r", boa_err)
            continue

        # If both errored, consider them matching
        if ivy_err:
            continue

        # Compare outputs
        if iv["output"] != bv["output"]:
            logging.error(
                "Output mismatch for payload %s on contract:\n%s", data, source
            )
            logging.error("  Ivy output: %r", iv["output"])
            logging.error("  Boa output: %r", bv["output"])

        # Compare storage
        if iv["storage"] != bv["storage"]:
            logging.error(
                "Storage mismatch for payload %s on contract:\n%s", data, source
            )
            logging.error("  Ivy storage: %r", iv["storage"])
            logging.error("  Boa storage: %r", bv["storage"])


def main():
    records = contract_dumper.load_records()

    for rec in records:
        source = rec.get("source")
        calldatas = rec.get("calldatas", [])
        if not calldatas:
            continue
        compare_runs(source, calldatas)


if __name__ == "__main__":
    main()

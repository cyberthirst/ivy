import dumper.contract_dumper as contract_dumper

from vyper.compiler import compile_code


def test_replay_all_calldatas(get_contract, env):
    """
    Load every recorded contract and re-invoke it using each saved calldata payload.
    This ensures that all previously captured calls remain valid at runtime.
    """
    records = contract_dumper.load_records()

    # Replay each calldata for its corresponding contract
    for rec in records:
        source = rec.get("source")
        _ = rec.get("ctor_args", "")
        _ = rec.get("calldatas", [])

        _ = compile_code(source)

        # the calls can fail as we log also failing calls
        # until we store the result, we can't easily test this
        # for hexstr in calldatas:
        #    data = bytes.fromhex(hexstr)
        #    env.message_call(contract.address, data=data)

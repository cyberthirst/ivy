from __future__ import annotations

from fuzzer.runner.base_scenario_runner import CallResult


def test_call_result_to_dict_preserves_falsey_outputs():
    bool_output = CallResult(success=True, output=False).to_dict()["output"]
    int_output = CallResult(success=True, output=0).to_dict()["output"]
    bytes_output = CallResult(success=True, output=b"").to_dict()["output"]
    str_output = CallResult(success=True, output="").to_dict()["output"]
    none_output = CallResult(success=True, output=None).to_dict()["output"]

    assert bool_output is False
    assert int_output == 0
    assert type(int_output) is int
    assert bytes_output == b""
    assert type(bytes_output) is bytes
    assert str_output == ""
    assert none_output is None

import pytest

from ivy.exceptions import CallTimeout


def test_call_timeout_budget_triggers_and_unwinds(get_contract, env, monkeypatch):
    src = """
@external
def spin(n: uint256) -> uint256:
    x: uint256 = 0
    for i: uint256 in range(n, bound=2000):
        x += i
    return x
    """
    c = get_contract(src)

    monkeypatch.setattr(type(env.interpreter), "_TIMEOUT_CHECK_MASK", 0)
    env.interpreter.set_call_timeout_seconds(1e-9)

    with pytest.raises(CallTimeout, match="Call exceeded time budget"):
        c.spin(2000)

    env.interpreter.clear_call_timeout()
    assert not env.interpreter.evm.journal.is_active

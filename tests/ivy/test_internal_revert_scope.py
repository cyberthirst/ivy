from ivy.exceptions import Assert


def test_revert_in_internal_loop_does_not_corrupt_scope(get_contract, tx_failed):
    src = """
@internal
def _boom(x: uint256):
    assert x > 0

@external
def foo(x: uint256):
    for i: uint256 in range(1):
        self._boom(x)

@external
def ok() -> uint256:
    return 123
    """
    c = get_contract(src)

    with tx_failed(Assert):
        c.foo(0)

    assert c.ok() == 123

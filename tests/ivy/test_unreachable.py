from ivy.exceptions import Invalid


def test_assert_unreachable(get_contract, tx_failed):
    src = """
@external
def foo(x: uint256):
    assert x > 10, UNREACHABLE
    """
    c = get_contract(src)

    # Should succeed when condition is True
    c.foo(15)

    # Should raise Invalid when condition is False
    with tx_failed(Invalid):
        c.foo(5)


def test_raise_unreachable(get_contract, tx_failed):
    src = """
@external
def foo(x: uint256):
    if x == 0:
        raise UNREACHABLE
    """
    c = get_contract(src)

    # Should succeed when x != 0
    c.foo(5)

    # Should raise Invalid when x == 0
    with tx_failed(Invalid):
        c.foo(0)


def test_assert_unreachable_in_if_branch(get_contract, tx_failed):
    src = """
@external
def foo(x: uint256) -> uint256:
    if x > 100:
        # Use a dynamic condition that can't be statically analyzed
        assert x < 50, UNREACHABLE
    return x + 1
    """
    c = get_contract(src)

    # Normal path works
    assert c.foo(50) == 51

    # UNREACHABLE path raises Invalid (x=200 > 100, so enters if; x=200 >= 50, so assert fails)
    with tx_failed(Invalid):
        c.foo(200)

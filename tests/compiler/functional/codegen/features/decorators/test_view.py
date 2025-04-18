import pytest


def test_constant_test(get_contract):
    constant_test = """
@external
@view
def foo() -> int128:
    return 5
    """

    c = get_contract(constant_test)
    assert c.foo() == 5


def test_transient_test(get_contract):
    code = """
x: transient(uint256)

@external
@view
def foo() -> uint256:
    return self.x
    """
    c = get_contract(code)
    assert c.foo() == 0

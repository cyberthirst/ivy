import pytest


def test_convert(get_contract):
    src = """
@external
def foo(a: uint256 ) -> uint128:
    return convert(a, uint128)
    """

    c = get_contract(src)

    c.foo(1)

from decimal import Decimal

import pytest

from ivy.abi import EncodeError


def test_decimal_requires_int168(get_contract):
    src = """
@external
def foo(x: uint256) -> uint256:
    return x
    """
    c = get_contract(src)

    with pytest.raises(EncodeError, match="Decimal values are only supported for int168"):
        c.foo(Decimal("1.0"))


def test_decimal_precision_too_high(get_contract):
    src = """
@external
def foo(x: decimal) -> decimal:
    return x
    """
    c = get_contract(src)

    with pytest.raises(EncodeError, match="Precision of value is greater than allowed"):
        c.foo(Decimal("0.00000000001"))

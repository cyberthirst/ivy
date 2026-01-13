from decimal import Decimal

import pytest

from ivy.frontend.loader import loads
from ivy.exceptions import Assert


def test_as_wei_value_positive_int():
    """Test normal positive integer values work correctly."""
    src = """
@external
def foo(x: uint256) -> uint256:
    return as_wei_value(x, "ether")
    """
    c = loads(src)
    # 1 ether = 10**18 wei
    assert c.foo(1) == 10**18
    assert c.foo(0) == 0
    assert c.foo(100) == 100 * 10**18


def test_as_wei_value_positive_decimal():
    """Test normal positive decimal values work correctly."""
    src = """
@external
def foo(x: decimal) -> uint256:
    return as_wei_value(x, "ether")
    """
    c = loads(src)
    # 1.5 ether = 1.5 * 10**18 wei
    assert c.foo(Decimal("1.5")) == 15 * 10**17
    assert c.foo(Decimal("0.0")) == 0


def test_as_wei_value_different_denoms():
    """Test various wei denominations work correctly."""
    src = """
@external
def test_wei(x: uint256) -> uint256:
    return as_wei_value(x, "wei")

@external
def test_gwei(x: uint256) -> uint256:
    return as_wei_value(x, "gwei")

@external
def test_ether(x: uint256) -> uint256:
    return as_wei_value(x, "ether")
    """
    c = loads(src)
    assert c.test_wei(1) == 1
    assert c.test_gwei(1) == 10**9
    assert c.test_ether(1) == 10**18


def test_as_wei_value_negative_int_reverts(tx_failed):
    """Test that negative int256 value reverts with Assert."""
    src = """
@external
def foo(x: int256) -> uint256:
    return as_wei_value(x, "ether")
    """
    c = loads(src)
    # Negative value should revert
    with tx_failed(Assert):
        c.foo(-1)


def test_as_wei_value_negative_decimal_reverts(tx_failed):
    """Test that negative decimal value reverts with Assert."""
    src = """
@external
def foo(x: decimal) -> uint256:
    return as_wei_value(x, "ether")
    """
    c = loads(src)
    # Negative decimal value should revert
    with tx_failed(Assert):
        c.foo(Decimal("-1.0"))


def test_as_wei_value_overflow_reverts(tx_failed):
    """Test that overflow with large uint256 value reverts with Assert."""
    src = """
@external
def foo(x: uint256) -> uint256:
    return as_wei_value(x, "ether")
    """
    c = loads(src)
    # max_uint256 * 10**18 should overflow
    max_uint256 = 2**256 - 1
    with tx_failed(Assert):
        c.foo(max_uint256)


def test_as_wei_value_max_safe_succeeds():
    """Test that maximum safe value (no overflow) succeeds."""
    src = """
@external
def foo(x: uint256) -> uint256:
    return as_wei_value(x, "ether")
    """
    c = loads(src)
    # Max safe value for ether: floor(max_uint256 / 10**18)
    # This value should not overflow when multiplied by 10**18
    max_safe = (2**256 - 1) // (10**18)
    result = c.foo(max_safe)
    assert result == max_safe * 10**18


def test_as_wei_value_just_above_max_safe_reverts(tx_failed):
    """Test that value just above max safe reverts."""
    src = """
@external
def foo(x: uint256) -> uint256:
    return as_wei_value(x, "ether")
    """
    c = loads(src)
    # Value just above max safe should overflow
    max_safe = (2**256 - 1) // (10**18)
    with tx_failed(Assert):
        c.foo(max_safe + 1)


def test_as_wei_value_zero_value_no_overflow_check():
    """Test that zero value doesn't trigger false overflow (division by zero guard)."""
    src = """
@external
def foo(x: uint256) -> uint256:
    return as_wei_value(x, "ether")
    """
    c = loads(src)
    # Zero should work and not trigger overflow check
    assert c.foo(0) == 0

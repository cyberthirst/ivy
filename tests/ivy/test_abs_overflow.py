"""Tests for abs() overflow checking with MIN_INT256."""

from ivy.exceptions import Assert


def test_abs_min_int256_reverts(get_contract, tx_failed):
    """abs(MIN_INT256) should revert because 2^255 exceeds MAX_INT256."""
    src = """
@external
def foo(x: int256) -> int256:
    return abs(x)
    """
    c = get_contract(src)
    MIN_INT256 = -(2**255)

    with tx_failed(Assert):
        c.foo(MIN_INT256)


def test_abs_min_int256_plus_one_works(get_contract):
    """abs(MIN_INT256 + 1) should return MAX_INT256."""
    src = """
@external
def foo(x: int256) -> int256:
    return abs(x)
    """
    c = get_contract(src)
    MIN_INT256 = -(2**255)
    MAX_INT256 = 2**255 - 1

    result = c.foo(MIN_INT256 + 1)
    assert result == MAX_INT256


def test_abs_zero(get_contract):
    """abs(0) should return 0."""
    src = """
@external
def foo(x: int256) -> int256:
    return abs(x)
    """
    c = get_contract(src)

    assert c.foo(0) == 0


def test_abs_positive_values(get_contract):
    """abs(positive value) should return the same value."""
    src = """
@external
def foo(x: int256) -> int256:
    return abs(x)
    """
    c = get_contract(src)

    assert c.foo(1) == 1
    assert c.foo(42) == 42
    assert c.foo(2**255 - 1) == 2**255 - 1  # MAX_INT256


def test_abs_negative_values(get_contract):
    """abs(negative value) should return the positive value."""
    src = """
@external
def foo(x: int256) -> int256:
    return abs(x)
    """
    c = get_contract(src)

    assert c.foo(-1) == 1
    assert c.foo(-42) == 42
    assert c.foo(-(2**255 - 1)) == 2**255 - 1  # abs(-MAX_INT256) = MAX_INT256


def test_abs_in_expression(get_contract, tx_failed):
    """abs() in an expression should still revert on MIN_INT256."""
    src = """
@external
def foo(x: int256) -> bool:
    a: int256 = abs(x)
    return a > 0
    """
    c = get_contract(src)
    MIN_INT256 = -(2**255)

    with tx_failed(Assert):
        c.foo(MIN_INT256)


def test_abs_with_literal(get_contract):
    """abs() with a literal should work correctly."""
    src = """
@external
def foo() -> int256:
    return abs(-42)
    """
    c = get_contract(src)

    assert c.foo() == 42

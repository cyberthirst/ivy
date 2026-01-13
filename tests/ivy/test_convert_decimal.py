def test_convert_decimal_to_uint256_truncates_positive(get_contract):
    """convert(1.9, uint256) should return 1 (truncate toward zero)."""
    src = """
@external
def foo() -> uint256:
    d: decimal = 1.9
    return convert(d, uint256)
"""
    c = get_contract(src)
    assert c.foo() == 1


def test_convert_decimal_to_int256_truncates_negative(get_contract):
    """convert(-1.5, int256) should return -1 (truncate toward zero)."""
    src = """
@external
def foo() -> int256:
    d: decimal = -1.5
    return convert(d, int256)
"""
    c = get_contract(src)
    assert c.foo() == -1


def test_convert_decimal_to_uint256_truncates_small_positive(get_contract):
    """convert(0.5, uint256) should return 0 (truncate toward zero)."""
    src = """
@external
def foo() -> uint256:
    d: decimal = 0.5
    return convert(d, uint256)
"""
    c = get_contract(src)
    assert c.foo() == 0


def test_convert_decimal_to_int256_truncates_small_negative(get_contract):
    """convert(-0.5, int256) should return 0 (truncate toward zero)."""
    src = """
@external
def foo() -> int256:
    d: decimal = -0.5
    return convert(d, int256)
"""
    c = get_contract(src)
    assert c.foo() == 0


def test_convert_decimal_whole_number_to_uint256(get_contract):
    """convert(42.0, uint256) should return 42."""
    src = """
@external
def foo() -> uint256:
    d: decimal = 42.0
    return convert(d, uint256)
"""
    c = get_contract(src)
    assert c.foo() == 42


def test_convert_decimal_whole_number_to_int256(get_contract):
    """convert(-42.0, int256) should return -42."""
    src = """
@external
def foo() -> int256:
    d: decimal = -42.0
    return convert(d, int256)
"""
    c = get_contract(src)
    assert c.foo() == -42


def test_convert_decimal_to_uint8_boundary_whole(get_contract):
    """convert(255.0, uint8) should return 255 (whole number at boundary)."""
    src = """
@external
def foo() -> uint8:
    d: decimal = 255.0
    return convert(d, uint8)
"""
    c = get_contract(src)
    assert c.foo() == 255


def test_convert_decimal_to_uint8_fractional_exceeds_bounds_reverts(get_contract, tx_failed):
    """convert(255.9, uint8) should revert (255.9 > 255, checked before truncation)."""
    src = """
@external
def foo() -> uint8:
    d: decimal = 255.9
    return convert(d, uint8)
"""
    c = get_contract(src)
    with tx_failed():
        c.foo()


def test_convert_decimal_in_bounds_with_fraction_uint8(get_contract):
    """convert(100.9, uint8) should return 100 (100.9 < 255, truncates to 100)."""
    src = """
@external
def foo() -> uint8:
    d: decimal = 100.9
    return convert(d, uint8)
"""
    c = get_contract(src)
    assert c.foo() == 100


def test_convert_negative_decimal_to_uint256_reverts(get_contract, tx_failed):
    """convert(-1.0, uint256) should revert (negative to unsigned)."""
    src = """
@external
def foo() -> uint256:
    d: decimal = -1.0
    return convert(d, uint256)
"""
    c = get_contract(src)
    with tx_failed():
        c.foo()


def test_convert_negative_fractional_to_uint256_reverts(get_contract, tx_failed):
    """convert(-0.5, uint256) should revert (negative value, even if truncation yields 0)."""
    src = """
@external
def foo() -> uint256:
    d: decimal = -0.5
    return convert(d, uint256)
"""
    c = get_contract(src)
    with tx_failed():
        c.foo()


def test_convert_decimal_overflow_uint8_reverts(get_contract, tx_failed):
    """convert(256.0, uint8) should revert (value exceeds bounds)."""
    src = """
@external
def foo() -> uint8:
    d: decimal = 256.0
    return convert(d, uint8)
"""
    c = get_contract(src)
    with tx_failed():
        c.foo()


def test_convert_decimal_large_value_to_int8_reverts(get_contract, tx_failed):
    """convert(128.0, int8) should revert (128 > max int8 which is 127)."""
    src = """
@external
def foo() -> int8:
    d: decimal = 128.0
    return convert(d, int8)
"""
    c = get_contract(src)
    with tx_failed():
        c.foo()


def test_convert_decimal_negative_overflow_int8_reverts(get_contract, tx_failed):
    """convert(-129.0, int8) should revert (-129 < min int8 which is -128)."""
    src = """
@external
def foo() -> int8:
    d: decimal = -129.0
    return convert(d, int8)
"""
    c = get_contract(src)
    with tx_failed():
        c.foo()


def test_convert_decimal_boundary_positive_int8_whole(get_contract):
    """convert(127.0, int8) should return 127 (whole number at boundary)."""
    src = """
@external
def foo() -> int8:
    d: decimal = 127.0
    return convert(d, int8)
"""
    c = get_contract(src)
    assert c.foo() == 127


def test_convert_decimal_boundary_positive_int8_fractional_reverts(get_contract, tx_failed):
    """convert(127.9, int8) should revert (127.9 > 127, checked before truncation)."""
    src = """
@external
def foo() -> int8:
    d: decimal = 127.9
    return convert(d, int8)
"""
    c = get_contract(src)
    with tx_failed():
        c.foo()


def test_convert_decimal_boundary_negative_int8_whole(get_contract):
    """convert(-128.0, int8) should return -128 (whole number at boundary)."""
    src = """
@external
def foo() -> int8:
    d: decimal = -128.0
    return convert(d, int8)
"""
    c = get_contract(src)
    assert c.foo() == -128


def test_convert_decimal_boundary_negative_int8_fractional_reverts(get_contract, tx_failed):
    """convert(-128.9, int8) should revert (-128.9 < -128, checked before truncation)."""
    src = """
@external
def foo() -> int8:
    d: decimal = -128.9
    return convert(d, int8)
"""
    c = get_contract(src)
    with tx_failed():
        c.foo()


def test_convert_decimal_in_bounds_with_negative_fraction_int8(get_contract):
    """convert(-100.9, int8) should return -100 (-128 < -100.9 < 127, truncates to -100)."""
    src = """
@external
def foo() -> int8:
    d: decimal = -100.9
    return convert(d, int8)
"""
    c = get_contract(src)
    assert c.foo() == -100


def test_convert_decimal_zero_to_uint256(get_contract):
    """convert(0.0, uint256) should return 0."""
    src = """
@external
def foo() -> uint256:
    d: decimal = 0.0
    return convert(d, uint256)
"""
    c = get_contract(src)
    assert c.foo() == 0


def test_convert_decimal_returns_int_not_decimal(get_contract):
    """Verify the return type is int, not VyperDecimal (ABI encoding check)."""
    src = """
@external
def foo() -> uint256:
    d: decimal = 1.9
    return convert(d, uint256)

@external
def bar() -> uint256:
    d: decimal = 1.0
    return convert(d, uint256)
"""
    c = get_contract(src)
    # If VyperDecimal was returned, ABI encoding would give 10000000000 (1.0 * 10^10)
    # or 19000000000 (1.9 * 10^10) - we should get 1 instead
    assert c.foo() == 1
    assert c.bar() == 1

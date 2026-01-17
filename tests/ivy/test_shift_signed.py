from ivy.frontend.loader import loads


def test_shift_int256_left_negative_one():
    """Regression test: shift(-1, 1) with int256 should return -2."""
    src = """
@external
def foo() -> int256:
    x: int256 = -1
    return shift(x, 1)
    """
    c = loads(src)
    result = c.foo()
    assert result == -2, f"Expected -2, got {result}"


def test_shift_int256_right_negative_two():
    """Regression test: shift(-2, -1) with int256 should return -1."""
    src = """
@external
def foo() -> int256:
    x: int256 = -2
    return shift(x, -1)
    """
    c = loads(src)
    result = c.foo()
    assert result == -1, f"Expected -1, got {result}"


def test_shift_uint256_unchanged():
    """Ensure uint256 shift still works correctly (unchanged behavior)."""
    src = """
@external
def foo() -> uint256:
    return shift(1, 8)
    """
    c = loads(src)
    result = c.foo()
    assert result == 256, f"Expected 256, got {result}"


def test_shift_uint256_large_value():
    """Ensure uint256 shift with large values works correctly."""
    src = """
@external
def foo() -> uint256:
    return shift(max_value(uint256), -1)
    """
    c = loads(src)
    result = c.foo()
    expected = (2**256 - 1) >> 1
    assert result == expected, f"Expected {expected}, got {result}"


def test_shift_int256_boundary_max():
    """Test shift at boundary: max_value(int256) left shift by 1 wraps to negative."""
    src = """
@external
def foo() -> int256:
    # max_value(int256) = 2^255 - 1
    # shift by 1 left should wrap and become negative
    x: int256 = max_value(int256)
    return shift(x, 1)
    """
    c = loads(src)
    result = c.foo()
    # (2^255 - 1) << 1 = 2^256 - 2, which is >= 2^255, so becomes 2^256 - 2 - 2^256 = -2
    expected = -2
    assert result == expected, f"Expected {expected}, got {result}"


def test_shift_int256_min_value_right():
    """Test shift min_value(int256) right by 1."""
    src = """
@external
def foo() -> int256:
    # min_value(int256) = -2^255
    # shift right by 1: -2^255 >> 1 = -2^254
    x: int256 = min_value(int256)
    return shift(x, -1)
    """
    c = loads(src)
    result = c.foo()
    expected = -(2**254)
    assert result == expected, f"Expected {expected}, got {result}"


def test_shift_int256_positive_stays_positive():
    """Test shifting positive int256 stays positive when result < 2^255."""
    src = """
@external
def foo() -> int256:
    # 2^254 - 1 << 1 = 2^255 - 2, which is < 2^255 so stays positive
    x: int256 = 2**254 - 1
    return shift(x, 1)
    """
    c = loads(src)
    result = c.foo()
    expected = 2**255 - 2
    assert result == expected, f"Expected {expected}, got {result}"


def test_shift_int256_zero():
    """Test shift with zero value."""
    src = """
@external
def foo() -> int256:
    x: int256 = 0
    return shift(x, 10)
    """
    c = loads(src)
    result = c.foo()
    assert result == 0, f"Expected 0, got {result}"


def test_shift_int256_left_then_right():
    """Test consistency: shift left then right should recover (for small shifts)."""
    src = """
@external
def foo() -> int256:
    x: int256 = 12345
    y: int256 = shift(x, 4)  # left by 4
    z: int256 = shift(y, -4)  # right by 4
    return z
    """
    c = loads(src)
    result = c.foo()
    assert result == 12345, f"Expected 12345, got {result}"


def test_shift_negative_right_preserves_sign():
    """Right shift of negative int256 should preserve negative (arithmetic shift)."""
    src = """
@external
def foo() -> int256:
    x: int256 = -128
    return shift(x, -3)
    """
    c = loads(src)
    result = c.foo()
    # -128 >> 3 = -16 (arithmetic right shift)
    expected = -16
    assert result == expected, f"Expected {expected}, got {result}"


def test_shift_by_zero_int256():
    """Shift by 0 should return the original value for int256."""
    src = """
@external
def foo() -> int256:
    x: int256 = -42
    return shift(x, 0)
    """
    c = loads(src)
    result = c.foo()
    assert result == -42, f"Expected -42, got {result}"


def test_shift_by_zero_uint256():
    """Shift by 0 should return the original value for uint256."""
    src = """
@external
def foo() -> uint256:
    x: uint256 = 42
    return shift(x, 0)
    """
    c = loads(src)
    result = c.foo()
    assert result == 42, f"Expected 42, got {result}"


def test_shift_int256_large_left_shift():
    """Test shifting int256 left by a large amount (>= 256 returns 0)."""
    src = """
@external
def foo() -> int256:
    x: int256 = -1
    return shift(x, 256)
    """
    c = loads(src)
    result = c.foo()
    # Shifting left by 256 or more bits should result in 0
    assert result == 0, f"Expected 0, got {result}"


def test_shift_uint256_large_left_shift():
    """Test shifting uint256 left by a large amount (>= 256 returns 0)."""
    src = """
@external
def foo() -> uint256:
    x: uint256 = max_value(uint256)
    return shift(x, 256)
    """
    c = loads(src)
    result = c.foo()
    # Shifting left by 256 or more bits should result in 0
    assert result == 0, f"Expected 0, got {result}"


def test_shift_int256_large_right_shift_negative():
    """Test shifting negative int256 right by a large amount."""
    src = """
@external
def foo() -> int256:
    x: int256 = -1
    return shift(x, -256)
    """
    c = loads(src)
    result = c.foo()
    # -1 >> 256 = -1 (arithmetic right shift preserves sign bit)
    assert result == -1, f"Expected -1, got {result}"


def test_shift_int256_large_right_shift_positive():
    """Test shifting positive int256 right by a large amount."""
    src = """
@external
def foo() -> int256:
    x: int256 = max_value(int256)
    return shift(x, -256)
    """
    c = loads(src)
    result = c.foo()
    # max_value(int256) >> 256 = 0
    assert result == 0, f"Expected 0, got {result}"


def test_shift_int256_dynamic_args():
    """Test shift with dynamic (runtime) arguments for int256."""
    src = """
@external
def foo(x: int256, s: int256) -> int256:
    return shift(x, s)
    """
    c = loads(src)
    # Test left shift of negative
    assert c.foo(-1, 1) == -2
    # Test right shift of negative
    assert c.foo(-2, -1) == -1
    # Test shift by 0
    assert c.foo(-42, 0) == -42

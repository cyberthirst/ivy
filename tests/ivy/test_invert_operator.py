from ivy.frontend.loader import loads


def test_invert_bytes32_zeros():
    """Regression test: ~bytes32(0) should return all 0xff bytes."""
    src = """
@external
def foo() -> bytes32:
    x: bytes32 = 0x0000000000000000000000000000000000000000000000000000000000000000
    return ~x
    """
    c = loads(src)
    result = c.foo()
    expected = b"\xff" * 32
    assert result == expected, f"Expected all 0xff, got {result.hex()}"


def test_invert_bytes32_ones():
    """Regression test: ~bytes32(0xff..ff) should return all 0x00 bytes."""
    src = """
@external
def foo() -> bytes32:
    x: bytes32 = 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
    return ~x
    """
    c = loads(src)
    result = c.foo()
    expected = b"\x00" * 32
    assert result == expected, f"Expected all 0x00, got {result.hex()}"


def test_invert_bytes32_pattern():
    """Test invert with a mixed pattern."""
    src = """
@external
def foo() -> bytes32:
    x: bytes32 = 0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0
    return ~x
    """
    c = loads(src)
    result = c.foo()
    expected = b"\x0f" * 32
    assert result == expected, f"Expected all 0x0f, got {result.hex()}"


def test_invert_bytes32_double():
    """Test that ~~x == x for bytes32."""
    src = """
@external
def foo() -> bytes32:
    x: bytes32 = 0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef
    return ~~x
    """
    c = loads(src)
    result = c.foo()
    expected = bytes.fromhex("deadbeef" * 8)
    assert result == expected, f"Expected original value, got {result.hex()}"


def test_invert_uint256_zero():
    """Test ~uint256(0) returns max_value."""
    src = """
@external
def foo() -> uint256:
    x: uint256 = 0
    return ~x
    """
    c = loads(src)
    result = c.foo()
    expected = 2**256 - 1
    assert result == expected, f"Expected {expected}, got {result}"


def test_invert_uint256_max():
    """Test ~max_value(uint256) returns 0."""
    src = """
@external
def foo() -> uint256:
    x: uint256 = max_value(uint256)
    return ~x
    """
    c = loads(src)
    result = c.foo()
    assert result == 0, f"Expected 0, got {result}"


def test_invert_uint256_double():
    """Test that ~~x == x for uint256."""
    src = """
@external
def foo() -> uint256:
    x: uint256 = 12345678901234567890
    return ~~x
    """
    c = loads(src)
    result = c.foo()
    assert result == 12345678901234567890, (
        f"Expected 12345678901234567890, got {result}"
    )

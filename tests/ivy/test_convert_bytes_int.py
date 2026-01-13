from ivy.frontend.loader import loads


def test_convert_bytes_single_byte_to_uint256():
    """convert(b'\\x01', uint256) from Bytes[32] should equal 1, not 1 << 248"""
    src = """
@external
def convert_bytes(b: Bytes[32]) -> uint256:
    return convert(b, uint256)
    """
    c = loads(src)
    # Single byte 0x01 should convert to 1
    assert c.convert_bytes(b"\x01") == 1


def test_convert_bytes_single_byte_to_int256_sign_extension():
    """convert(b'\\xff', int256) from Bytes[32] should equal -1 (sign extended)"""
    src = """
@external
def convert_bytes(b: Bytes[32]) -> int256:
    return convert(b, int256)
    """
    c = loads(src)
    # Single byte 0xff should sign extend to -1
    assert c.convert_bytes(b"\xff") == -1


def test_convert_bytes_two_bytes_to_uint256():
    """convert(b'\\x00\\x01', uint256) from Bytes[32] should equal 1"""
    src = """
@external
def convert_bytes(b: Bytes[32]) -> uint256:
    return convert(b, uint256)
    """
    c = loads(src)
    # Two bytes 0x00 0x01 should convert to 1 (big-endian)
    assert c.convert_bytes(b"\x00\x01") == 1


def test_convert_bytes_two_bytes_to_int256_sign_extension():
    """convert(b'\\xff\\xff', int256) from Bytes[32] should equal -1"""
    src = """
@external
def convert_bytes(b: Bytes[32]) -> int256:
    return convert(b, int256)
    """
    c = loads(src)
    # Two bytes 0xff 0xff should sign extend to -1
    assert c.convert_bytes(b"\xff\xff") == -1


def test_convert_bytes_full_32_bytes_to_uint256():
    """Full 32-byte value should convert correctly"""
    src = """
@external
def convert_bytes(b: Bytes[32]) -> uint256:
    return convert(b, uint256)
    """
    c = loads(src)
    # 32 bytes with value 256 (0x100)
    val_bytes = b"\x00" * 30 + b"\x01\x00"
    assert c.convert_bytes(val_bytes) == 256


def test_convert_bytes_full_32_bytes_to_int256():
    """Full 32-byte negative value should convert correctly"""
    src = """
@external
def convert_bytes(b: Bytes[32]) -> int256:
    return convert(b, int256)
    """
    c = loads(src)
    # 32 bytes representing -1 (all 0xff)
    val_bytes = b"\xff" * 32
    assert c.convert_bytes(val_bytes) == -1


def test_convert_bytes_larger_value_to_uint256():
    """Test conversion of multi-byte values"""
    src = """
@external
def convert_bytes(b: Bytes[32]) -> uint256:
    return convert(b, uint256)
    """
    c = loads(src)
    # 0x1234 = 4660
    assert c.convert_bytes(b"\x12\x34") == 0x1234
    # 0x123456 = 1193046
    assert c.convert_bytes(b"\x12\x34\x56") == 0x123456


def test_convert_bytes_negative_sign_extension():
    """Test sign extension for various negative values"""
    src = """
@external
def convert_bytes(b: Bytes[32]) -> int256:
    return convert(b, int256)
    """
    c = loads(src)
    # 0x80 = -128 when sign extended from 1 byte
    assert c.convert_bytes(b"\x80") == -128
    # 0x8000 = -32768 when sign extended from 2 bytes
    assert c.convert_bytes(b"\x80\x00") == -32768


def test_convert_bytes_empty_to_uint256():
    """Empty bytes should convert to 0"""
    src = """
@external
def convert_bytes(b: Bytes[32]) -> uint256:
    return convert(b, uint256)
    """
    c = loads(src)
    assert c.convert_bytes(b"") == 0


def test_convert_bytes_positive_in_signed():
    """Positive values in signed conversion should remain positive"""
    src = """
@external
def convert_bytes(b: Bytes[32]) -> int256:
    return convert(b, int256)
    """
    c = loads(src)
    # 0x7f = 127 (max positive for 1 byte signed)
    assert c.convert_bytes(b"\x7f") == 127
    # 0x7fff = 32767 (max positive for 2 bytes signed)
    assert c.convert_bytes(b"\x7f\xff") == 32767


def test_convert_bytes_empty_to_int256():
    """Empty bytes should convert to 0 for signed type too"""
    src = """
@external
def convert_bytes(b: Bytes[32]) -> int256:
    return convert(b, int256)
    """
    c = loads(src)
    assert c.convert_bytes(b"") == 0


def test_convert_bytes_32_byte_signed_boundaries():
    """Test 32-byte boundary values for int256"""
    src = """
@external
def convert_bytes(b: Bytes[32]) -> int256:
    return convert(b, int256)
    """
    c = loads(src)
    # Max positive: 2**255 - 1 (0x7f followed by 31 0xff bytes)
    max_positive = b"\x7f" + b"\xff" * 31
    assert c.convert_bytes(max_positive) == 2**255 - 1
    # Min negative: -2**255 (0x80 followed by 31 0x00 bytes)
    min_negative = b"\x80" + b"\x00" * 31
    assert c.convert_bytes(min_negative) == -(2**255)

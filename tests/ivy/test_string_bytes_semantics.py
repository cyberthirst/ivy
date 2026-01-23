"""
Regression tests for VyperString byte semantics.

Vyper strings are UTF-8 byte sequences internally:
- len() returns byte count, not character count
- slice() operates on byte offsets, not character offsets
- String[N] means max N bytes, not N characters
- At runtime, any bytes 0-255 are allowed (no ASCII validation)

These tests verify Ivy matches Vyper/Boa behavior for runtime strings
containing non-ASCII bytes.
"""

import pytest
import boa
from ivy.frontend.loader import loads


class TestStringLenByteSemantics:
    """Test that len() returns byte count for strings."""

    def test_len_ascii_string(self):
        """ASCII strings: byte count == character count."""
        src = """
@external
def get_len(s: String[100]) -> uint256:
    return len(s)
"""
        ivy_c = loads(src)
        boa_c = boa.loads(src)

        assert ivy_c.get_len("hello") == boa_c.get_len("hello") == 5
        assert ivy_c.get_len("") == boa_c.get_len("") == 0
        assert ivy_c.get_len("a") == boa_c.get_len("a") == 1

    def test_len_euro_sign(self):
        """Euro sign (€) is 1 character but 3 bytes in UTF-8."""
        src = """
@external
def get_len(s: String[100]) -> uint256:
    return len(s)
"""
        ivy_c = loads(src)
        boa_c = boa.loads(src)

        # € is 3 bytes in UTF-8: 0xE2 0x82 0xAC
        result_ivy = ivy_c.get_len("€")
        result_boa = boa_c.get_len("€")

        assert result_ivy == result_boa == 3, (
            f"len('€') should be 3 (byte count), got ivy={result_ivy}, boa={result_boa}"
        )

    def test_len_chinese_character(self):
        """Chinese character (中) is 1 character but 3 bytes in UTF-8."""
        src = """
@external
def get_len(s: String[100]) -> uint256:
    return len(s)
"""
        ivy_c = loads(src)
        boa_c = boa.loads(src)

        # 中 is 3 bytes in UTF-8
        result_ivy = ivy_c.get_len("中")
        result_boa = boa_c.get_len("中")

        assert result_ivy == result_boa == 3

    def test_len_mixed_ascii_utf8(self):
        """Mixed ASCII and UTF-8 characters."""
        src = """
@external
def get_len(s: String[100]) -> uint256:
    return len(s)
"""
        ivy_c = loads(src)
        boa_c = boa.loads(src)

        # "hello€" = 5 ASCII bytes + 3 UTF-8 bytes = 8 bytes total
        test_string = "hello€"
        result_ivy = ivy_c.get_len(test_string)
        result_boa = boa_c.get_len(test_string)

        assert result_ivy == result_boa == 8

    def test_len_emoji(self):
        """Emoji (🎉) is 1 character but 4 bytes in UTF-8."""
        src = """
@external
def get_len(s: String[100]) -> uint256:
    return len(s)
"""
        ivy_c = loads(src)
        boa_c = boa.loads(src)

        # 🎉 is 4 bytes in UTF-8
        result_ivy = ivy_c.get_len("🎉")
        result_boa = boa_c.get_len("🎉")

        assert result_ivy == result_boa == 4


class TestStringSliceByteSemantics:
    """Test that slice() operates on byte offsets."""

    def test_slice_ascii(self):
        """ASCII strings: byte and character offsets are the same."""
        src = """
@external
def do_slice(s: String[100], start: uint256, length: uint256) -> String[100]:
    return slice(s, start, length)
"""
        ivy_c = loads(src)
        boa_c = boa.loads(src)

        assert ivy_c.do_slice("hello", 0, 3) == boa_c.do_slice("hello", 0, 3)
        assert ivy_c.do_slice("hello", 2, 2) == boa_c.do_slice("hello", 2, 2)

    def test_slice_utf8_full(self):
        """Slicing full UTF-8 character."""
        src = """
@external
def do_slice(s: String[100], start: uint256, length: uint256) -> String[100]:
    return slice(s, start, length)
"""
        ivy_c = loads(src)
        boa_c = boa.loads(src)

        # € is 3 bytes, so slice(s, 0, 3) should return the full euro sign
        result_ivy = ivy_c.do_slice("€", 0, 3)
        result_boa = boa_c.do_slice("€", 0, 3)

        # Both should return something representing the € (may be surrogates or actual char)
        assert len(result_ivy.encode("utf-8") if isinstance(result_ivy, str) else result_ivy) == 3
        assert result_ivy == result_boa

    def test_slice_utf8_partial(self):
        """Slicing partial UTF-8 character returns raw bytes."""
        src = """
@external
def do_slice(s: String[100], start: uint256, length: uint256) -> String[100]:
    return slice(s, start, length)
"""
        ivy_c = loads(src)
        boa_c = boa.loads(src)

        # "A€B" = A(1 byte) + €(3 bytes) + B(1 byte) = 5 bytes
        # slice(s, 1, 3) should get bytes 1-3, which is the 3 bytes of €
        test_str = "A€B"
        result_ivy = ivy_c.do_slice(test_str, 1, 3)
        result_boa = boa_c.do_slice(test_str, 1, 3)

        assert result_ivy == result_boa

    def test_slice_bounds_check_uses_bytes(self):
        """Slice bounds are checked against byte length, not char length."""
        src = """
@external
def do_slice(s: String[100], start: uint256, length: uint256) -> String[100]:
    return slice(s, start, length)
"""
        ivy_c = loads(src)
        boa_c = boa.loads(src)

        # € is 3 bytes, so slice(s, 0, 3) should work
        # If bounds were checked against char count (1), this would fail
        result_ivy = ivy_c.do_slice("€", 0, 3)
        result_boa = boa_c.do_slice("€", 0, 3)

        assert result_ivy == result_boa


class TestStringConvertByteSemantics:
    """Test convert() between Bytes and String uses byte semantics."""

    def test_convert_bytes_to_string_len(self):
        """convert(Bytes, String) preserves byte count."""
        src = """
@external
def bytes_to_string_len(b: Bytes[100]) -> uint256:
    s: String[100] = convert(b, String[100])
    return len(s)
"""
        ivy_c = loads(src)
        boa_c = boa.loads(src)

        # € in UTF-8 bytes
        euro_bytes = bytes([0xE2, 0x82, 0xAC])

        result_ivy = ivy_c.bytes_to_string_len(euro_bytes)
        result_boa = boa_c.bytes_to_string_len(euro_bytes)

        assert result_ivy == result_boa == 3

    def test_convert_string_to_bytes_len(self):
        """convert(String, Bytes) preserves byte count."""
        src = """
@external
def string_to_bytes_len(s: String[100]) -> uint256:
    b: Bytes[100] = convert(s, Bytes[100])
    return len(b)
"""
        ivy_c = loads(src)
        boa_c = boa.loads(src)

        result_ivy = ivy_c.string_to_bytes_len("€")
        result_boa = boa_c.string_to_bytes_len("€")

        assert result_ivy == result_boa == 3


class TestStringConcatByteSemantics:
    """Test concat() with strings uses byte semantics."""

    def test_concat_utf8_strings_len(self):
        """concat() result length is sum of byte lengths."""
        src = """
@external
def concat_len(a: String[100], b: String[100]) -> uint256:
    return len(concat(a, b))
"""
        ivy_c = loads(src)
        boa_c = boa.loads(src)

        # "€" (3 bytes) + "€" (3 bytes) = 6 bytes
        result_ivy = ivy_c.concat_len("€", "€")
        result_boa = boa_c.concat_len("€", "€")

        assert result_ivy == result_boa == 6


class TestStringKeccak256ByteSemantics:
    """Test keccak256() on strings uses byte representation."""

    def test_keccak256_utf8_string(self):
        """keccak256() hashes the UTF-8 bytes, not characters."""
        src = """
@external
def hash_string(s: String[100]) -> bytes32:
    return keccak256(s)
"""
        ivy_c = loads(src)
        boa_c = boa.loads(src)

        result_ivy = ivy_c.hash_string("€")
        result_boa = boa_c.hash_string("€")

        assert result_ivy == result_boa

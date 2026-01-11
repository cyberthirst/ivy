from ivy.frontend.loader import loads
from ivy.types import Address

# Test vector (verified with coincurve)
H = bytes.fromhex(
    "6c9c5e133b8aafb4d3c4a1f2e5e628f9da0c4e827b25bb3e1ec5bf0c7adc7055"
)
V = 28
R = 78616903610863619048312090996582975134846548711908124330623869860028714000524
S = 37668412685023658579322665379076573648491476554461132093879548014047408844408
EXPECTED_ADDR = "0xEf4aB0dac7136E1F44ADD67C6288a71629B2bb87"

SECP256K1N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


def u256(x: int) -> bytes:
    return x.to_bytes(32, "big")


def test_ecrecover_builtin_u8_b32():
    src = """
@external
@view
def recover(h: bytes32, v: uint8, r: bytes32, s: bytes32) -> address:
    return ecrecover(h, v, r, s)
"""
    c = loads(src)
    result = c.recover(H, V, u256(R), u256(S))
    assert result == EXPECTED_ADDR


def test_ecrecover_builtin_u256_u256():
    src = """
@external
@view
def recover(h: bytes32, v: uint256, r: uint256, s: uint256) -> address:
    return ecrecover(h, v, r, s)
"""
    c = loads(src)
    result = c.recover(H, V, R, S)
    assert result == EXPECTED_ADDR


def test_ecrecover_builtin_u256_b32():
    src = """
@external
@view
def recover(h: bytes32, v: uint256, r: bytes32, s: bytes32) -> address:
    return ecrecover(h, v, r, s)
"""
    c = loads(src)
    result = c.recover(H, V, u256(R), u256(S))
    assert result == EXPECTED_ADDR


def test_ecrecover_builtin_u8_u256():
    src = """
@external
@view
def recover(h: bytes32, v: uint8, r: uint256, s: uint256) -> address:
    return ecrecover(h, v, r, s)
"""
    c = loads(src)
    result = c.recover(H, V, R, S)
    assert result == EXPECTED_ADDR


# Failure cases - builtin must return zero address


def test_ecrecover_invalid_v_returns_zero():
    src = """
@external
@view
def recover(h: bytes32, v: uint256, r: uint256, s: uint256) -> address:
    return ecrecover(h, v, r, s)
"""
    c = loads(src)
    # v = 0 is invalid (must be 27 or 28)
    result = c.recover(H, 0, R, S)
    assert result == Address(0)

    # v = 29 is also invalid
    result = c.recover(H, 29, R, S)
    assert result == Address(0)


def test_ecrecover_zero_r_returns_zero():
    src = """
@external
@view
def recover(h: bytes32, v: uint256, r: uint256, s: uint256) -> address:
    return ecrecover(h, v, r, s)
"""
    c = loads(src)
    result = c.recover(H, V, 0, S)
    assert result == Address(0)


def test_ecrecover_zero_s_returns_zero():
    src = """
@external
@view
def recover(h: bytes32, v: uint256, r: uint256, s: uint256) -> address:
    return ecrecover(h, v, r, s)
"""
    c = loads(src)
    result = c.recover(H, V, R, 0)
    assert result == Address(0)


def test_ecrecover_r_equals_n_returns_zero():
    src = """
@external
@view
def recover(h: bytes32, v: uint256, r: uint256, s: uint256) -> address:
    return ecrecover(h, v, r, s)
"""
    c = loads(src)
    result = c.recover(H, V, SECP256K1N, S)
    assert result == Address(0)


def test_ecrecover_s_equals_n_returns_zero():
    src = """
@external
@view
def recover(h: bytes32, v: uint256, r: uint256, s: uint256) -> address:
    return ecrecover(h, v, r, s)
"""
    c = loads(src)
    result = c.recover(H, V, R, SECP256K1N)
    assert result == Address(0)


def test_ecrecover_x_not_on_curve_returns_zero():
    src = """
@external
@view
def recover(h: bytes32, v: uint256, r: uint256, s: uint256) -> address:
    return ecrecover(h, v, r, s)
"""
    c = loads(src)
    # r=5 has no valid y on secp256k1 curve
    result = c.recover(H, 27, 5, 1)
    assert result == Address(0)


# Precompile tests via raw_call


def test_precompile_success():
    src = """
@external
@view
def call_ec(data: Bytes[256]) -> Bytes[32]:
    return raw_call(
        0x0000000000000000000000000000000000000001,
        data,
        max_outsize=32,
        is_static_call=True,
    )
"""
    c = loads(src)
    calldata = H + u256(V) + u256(R) + u256(S)
    result = c.call_ec(calldata)
    expected = b"\x00" * 12 + bytes.fromhex("ef4ab0dac7136e1f44add67c6288a71629b2bb87")
    assert result == expected


def test_precompile_invalid_v_returns_empty():
    src = """
@external
@view
def call_ec(data: Bytes[256]) -> Bytes[32]:
    success: bool = False
    result: Bytes[32] = b""
    success, result = raw_call(
        0x0000000000000000000000000000000000000001,
        data,
        max_outsize=32,
        is_static_call=True,
        revert_on_failure=False,
    )
    return result
"""
    c = loads(src)
    # v = 29 is invalid
    calldata = H + u256(29) + u256(R) + u256(S)
    result = c.call_ec(calldata)
    assert result == b""


def test_precompile_zero_r_returns_empty():
    src = """
@external
@view
def call_ec(data: Bytes[256]) -> Bytes[32]:
    success: bool = False
    result: Bytes[32] = b""
    success, result = raw_call(
        0x0000000000000000000000000000000000000001,
        data,
        max_outsize=32,
        is_static_call=True,
        revert_on_failure=False,
    )
    return result
"""
    c = loads(src)
    calldata = H + u256(V) + u256(0) + u256(S)
    result = c.call_ec(calldata)
    assert result == b""


def test_precompile_x_not_on_curve_returns_empty():
    src = """
@external
@view
def call_ec(data: Bytes[256]) -> Bytes[32]:
    success: bool = False
    result: Bytes[32] = b""
    success, result = raw_call(
        0x0000000000000000000000000000000000000001,
        data,
        max_outsize=32,
        is_static_call=True,
        revert_on_failure=False,
    )
    return result
"""
    c = loads(src)
    # r=5, s=1, v=27 - r is not a valid x-coordinate on the curve
    calldata = H + u256(27) + u256(5) + u256(1)
    result = c.call_ec(calldata)
    assert result == b""


def test_precompile_short_input_pads_with_zeros():
    src = """
@external
@view
def call_ec(data: Bytes[256]) -> Bytes[32]:
    success: bool = False
    result: Bytes[32] = b""
    success, result = raw_call(
        0x0000000000000000000000000000000000000001,
        data,
        max_outsize=32,
        is_static_call=True,
        revert_on_failure=False,
    )
    return result
"""
    c = loads(src)
    # Only pass the hash - v/r/s will be zero-padded, making them invalid
    calldata = H
    result = c.call_ec(calldata)
    # v=0 is invalid, should return empty
    assert result == b""

import pytest

from ivy.frontend.loader import loads


@pytest.mark.parametrize("typ", ["bytes1", "bytes3", "bytes31", "bytes32"])
@pytest.mark.parametrize("op", ["&", "|", "^"])
def test_bytesm_bitwise_binary_ops(typ, op):
    src = f"""
@external
def do_op(a: {typ}, b: {typ}) -> {typ}:
    return a {op} b
    """
    c = loads(src)

    length = int(typ.removeprefix("bytes"))
    left = bytes((i + 1) % 256 for i in range(length))
    right = bytes((255 - i) % 256 for i in range(length))

    res = c.do_op(left, right)
    if op == "&":
        expected = bytes(l & r for l, r in zip(left, right))
    elif op == "|":
        expected = bytes(l | r for l, r in zip(left, right))
    else:
        assert op == "^"
        expected = bytes(l ^ r for l, r in zip(left, right))

    assert res == expected


def test_bytes32_shift_left():
    src = """
@external
def do_shift(a: bytes32, b: uint256) -> bytes32:
    return a << b
    """
    c = loads(src)

    value = bytes.fromhex(
        "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff"
    )
    val_int = int.from_bytes(value, "big")
    mask = (1 << 256) - 1

    for shift in (0, 1, 8, 31, 32, 63, 128, 255, 256, 257):
        res = c.do_shift(value, shift)
        expected = ((val_int << shift) & mask).to_bytes(32, "big")
        assert res == expected


def test_bytes32_shift_right():
    src = """
@external
def do_shift(a: bytes32, b: uint256) -> bytes32:
    return a >> b
    """
    c = loads(src)

    value = bytes.fromhex(
        "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff"
    )
    val_int = int.from_bytes(value, "big")

    for shift in (0, 1, 8, 31, 32, 63, 128, 255, 256, 257):
        res = c.do_shift(value, shift)
        expected = (val_int >> shift).to_bytes(32, "big")
        assert res == expected

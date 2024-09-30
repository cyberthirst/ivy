# NOTE: tests from https://github.com/vyperlang/vyper/blob/e21f3e8b1f7d5a5b5b40fb73ab4c6a2946e878ef/tests/functional/builtins/codegen/test_abi_decode.py
# adapted for the interpreter

import pytest

from vyper.utils import method_id

from ivy.frontend.loader import loads
from ivy.abi.abi_decoder import DecodeError


def _abi_payload_from_tuple(payload: tuple[int | bytes, ...]) -> bytes:
    return b"".join(p.to_bytes(32, "big") if isinstance(p, int) else p for p in payload)


def _replicate(value: int, count: int) -> tuple[int, ...]:
    return (value,) * count


@pytest.mark.skip(reason="low-level calls are not yet supported")
def test_abi_decode_arithmetic_overflow():
    # test based on GHSA-9p8r-4xp4-gw5w:
    # https://github.com/vyperlang/vyper/security/advisories/GHSA-9p8r-4xp4-gw5w#advisory-comment-91841
    # buf + head causes arithmetic overflow
    code = """
@external
def f(x: Bytes[32 * 3]):
    a: Bytes[32] = b"foo"
    y: Bytes[32 * 3] = x

    decoded_y1: Bytes[32] = _abi_decode(y, Bytes[32])
    a = b"bar"
    decoded_y2: Bytes[32] = _abi_decode(y, Bytes[32])
    # original POC:
    # assert decoded_y1 != decoded_y2
    """
    c = loads(code)

    data = method_id("f(bytes)")
    payload = (
        0x20,  # tuple head
        0x60,  # parent array length
        # parent payload - this word will be considered as the head of the abi-encoded inner array
        # and it will be added to base ptr leading to an arithmetic overflow
        2**256 - 0x60,
    )
    data += _abi_payload_from_tuple(payload)

    with pytest.raises(DecodeError):
        # env.message_call(c.address, data=data)
        pass


@pytest.mark.skip(reason="low-level calls are not yet supported")
def test_abi_decode_nonstrict_head():
    # data isn't strictly encoded - head is 0x21 instead of 0x20
    # but the head + length is still within runtime bounds of the parent buffer
    code = """
@external
def f(x: Bytes[32 * 5]):
    y: Bytes[32 * 5] = x
    a: Bytes[32] = b"a"
    decoded_y1: DynArray[uint256, 3] = _abi_decode(y, DynArray[uint256, 3])
    a = b"aaaa"
    decoded_y1 = _abi_decode(y, DynArray[uint256, 3])
    """
    c = loads(code)

    data = method_id("f(bytes)")

    payload = (
        0x20,  # tuple head
        0xA0,  # parent array length
        # head should be 0x20 but is 0x21 thus the data isn't strictly encoded
        0x21,
        # we don't want to revert on invalid length, so set this to 0
        # the first byte of payload will be considered as the length
        0x00,
        (0x01).to_bytes(1, "big"),  # will be considered as the length=1
        (0x00).to_bytes(31, "big"),
        *_replicate(0x03, 2),
    )

    data += _abi_payload_from_tuple(payload)

    # env.message_call(c.address, data=data)


def test_abi_decode_child_head_points_to_parent():
    # data isn't strictly encoded and the head for the inner array
    # skipts the corresponding payload and points to other valid section of the parent buffer
    code = """
@external
def run(x: Bytes[14 * 32]):
    y: Bytes[14 * 32] = x
    decoded_y1: DynArray[DynArray[DynArray[uint256, 2], 1], 2] = _abi_decode(
        y,
        DynArray[DynArray[DynArray[uint256, 2], 1], 2]
    )
    """
    c = loads(code)
    # encode [[[1, 1]], [[2, 2]]] and modify the head for [1, 1]
    # to actually point to [2, 2]
    payload = (
        0x20,  # top-level array head
        0x02,  # top-level array length
        0x40,  # head of DAr[DAr[DAr, uint256]]][0]
        0xE0,  # head of DAr[DAr[DAr, uint256]]][1]
        0x01,  # DAr[DAr[DAr, uint256]]][0] length
        # head of DAr[DAr[DAr, uint256]]][0][0]
        # points to DAr[DAr[DAr, uint256]]][1][0]
        0x20 * 6,
        0x02,  # DAr[DAr[DAr, uint256]]][0][0] length
        0x01,  # DAr[DAr[DAr, uint256]]][0][0][0]
        0x01,  # DAr[DAr[DAr, uint256]]][0][0][1]
        0x01,  # DAr[DAr[DAr, uint256]]][1] length
        0x20,  # DAr[DAr[DAr, uint256]]][1][0] head
        0x02,  # DAr[DAr[DAr, uint256]]][1][0] length
        0x02,  # DAr[DAr[DAr, uint256]]][1][0][0]
        0x02,  # DAr[DAr[DAr, uint256]]][1][0][1]
    )

    data = _abi_payload_from_tuple(payload)

    c.run(data)


def test_abi_decode_nonstrict_head_oob():
    # data isn't strictly encoded and (non_strict_head + len(DynArray[..][2])) > parent_static_sz
    # thus decoding the data pointed to by the head would cause an OOB read
    # non_strict_head + length == parent + parent_static_sz + 1
    code = """
@external
def run(x: Bytes[2 * 32 + 3 * 32  + 3 * 32 * 4]):
    y: Bytes[2 * 32 + 3 * 32 + 3 * 32 * 4] = x
    decoded_y1: DynArray[Bytes[32 * 3], 3] = _abi_decode(y,  DynArray[Bytes[32 * 3], 3])
    """
    c = loads(code)

    payload = (
        0x20,  # DynArray head
        0x03,  # DynArray length
        # non_strict_head - if the length pointed to by this head is 0x60 (which is valid
        # length for the Bytes[32*3] buffer), the decoding function  would decode
        # 1 byte over the end of the buffer
        # we define the non_strict_head as: skip the remaining heads, 1st and 2nd tail
        # to the third tail + 1B
        0x20 * 8 + 0x20 * 3 + 0x01,  # inner array0 head
        0x20 * 4 + 0x20 * 3,  # inner array1 head
        0x20 * 8 + 0x20 * 3,  # inner array2 head
        0x60,  # DynArray[Bytes[96], 3][0] length
        *_replicate(0x01, 3),  # DynArray[Bytes[96], 3][0] data
        0x60,  # DynArray[Bytes[96], 3][1] length
        *_replicate(0x01, 3),  # DynArray[Bytes[96], 3][1]  data
        # the invalid head points here + 1B (thus the length is 0x60)
        # we don't revert because of invalid length, but because head+length is OOB
        0x00,  # DynArray[Bytes[96], 3][2] length
        (0x60).to_bytes(1, "big"),
        (0x00).to_bytes(31, "big"),
        *_replicate(0x03, 2),
    )

    data = _abi_payload_from_tuple(payload)

    with pytest.raises(DecodeError):
        c.run(data)


def test_abi_decode_nonstrict_head_oob2():
    # same principle as in Test_abi_decode_nonstrict_head_oob
    # but adapted for dynarrays
    code = """
@external
def run(x: Bytes[2 * 32 + 3 * 32  + 3 * 32 * 4]):
    y: Bytes[2 * 32 + 3 * 32 + 3 * 32 * 4] = x
    decoded_y1: DynArray[DynArray[uint256, 3], 3] = _abi_decode(
        y,
        DynArray[DynArray[uint256, 3], 3]
    )
    """
    c = loads(code)

    payload = (
        0x20,  # DynArray head
        0x03,  # DynArray length
        (0x20 * 8 + 0x20 * 3 + 0x01),  # inner array0 head
        (0x20 * 4 + 0x20 * 3),  # inner array1 head
        (0x20 * 8 + 0x20 * 3),  # inner array2 head
        0x03,  # DynArray[..][0] length
        *_replicate(0x01, 3),  # DynArray[..][0] data
        0x03,  # DynArray[..][1] length
        *_replicate(0x01, 3),  # DynArray[..][1] data
        0x00,  # DynArray[..][2] length
        (0x03).to_bytes(1, "big"),
        (0x00).to_bytes(31, "big"),
        *_replicate(0x01, 2),  # DynArray[..][2] data
    )

    data = _abi_payload_from_tuple(payload)

    with pytest.raises(DecodeError):
        c.run(data)


def test_abi_decode_head_pointing_outside_buffer():
    # the head points completely outside the buffer
    code = """
@external
def run(x: Bytes[3 * 32]):
    y: Bytes[3 * 32] = x
    decoded_y1: Bytes[32] = _abi_decode(y, Bytes[32])
    """
    c = loads(code)

    payload = (0x80, 0x20, 0x01)
    data = _abi_payload_from_tuple(payload)

    with pytest.raises(DecodeError):
        c.run(data)


def test_abi_decode_bytearray_clamp():
    # data has valid encoding, but the length of DynArray[Bytes[96], 3][0] is set to 0x61
    # and thus the decoding should fail on bytestring clamp
    code = """
@external
def run(x: Bytes[2 * 32 + 3 * 32  + 3 * 32 * 4]):
    y: Bytes[2 * 32 + 3 * 32 + 3 * 32 * 4] = x
    decoded_y1: DynArray[Bytes[32 * 3], 3] = _abi_decode(y,  DynArray[Bytes[32 * 3], 3])
    """
    c = loads(code)

    payload = (
        0x20,  # DynArray head
        0x03,  # DynArray length
        0x20 * 3,  # inner array0 head
        0x20 * 4 + 0x20 * 3,  # inner array1 head
        0x20 * 8 + 0x20 * 3,  # inner array2 head
        # invalid length - should revert on bytestring clamp
        0x61,  # DynArray[Bytes[96], 3][0] length
        *_replicate(0x01, 3),  # DynArray[Bytes[96], 3][0] data
        0x60,  # DynArray[Bytes[96], 3][1] length
        *_replicate(0x01, 3),  # DynArray[Bytes[96], 3][1] data
        0x60,  # DynArray[Bytes[96], 3][2] length
        *_replicate(0x01, 3),  # DynArray[Bytes[96], 3][2] data
    )

    data = _abi_payload_from_tuple(payload)

    with pytest.raises(DecodeError):
        c.run(data)


@pytest.mark.skip(reason="low-level calls are not yet supported")
def test_abi_decode_runtimesz_oob():
    # provide enough data, but set the runtime size to be smaller than the actual size
    # so after y: [..] = x, y will have the incorrect size set and only part of the
    # original data will be copied. This will cause oob read outside the
    # runtime sz (but still within static size of the buffer)
    code = """
@external
def f(x: Bytes[2 * 32 + 3 * 32  + 3 * 32 * 4]):
    y: Bytes[2 * 32 + 3 * 32 + 3 * 32 * 4] = x
    decoded_y1: DynArray[Bytes[32 * 3], 3] = _abi_decode(y,  DynArray[Bytes[32 * 3], 3])
    """
    c = loads(code)

    data = method_id("f(bytes)")

    payload = (
        0x20,  # tuple head
        # the correct size is 0x220 (2*32+3*32+4*3*32)
        # therefore we will decode after the end of runtime size (but still within the buffer)
        0x01E4,  # top-level bytes array length
        0x20,  # DynArray head
        0x03,  # DynArray length
        0x20 * 3,  # inner array0 head
        0x20 * 4 + 0x20 * 3,  # inner array1 head
        0x20 * 8 + 0x20 * 3,  # inner array2 head
        0x60,  # DynArray[Bytes[96], 3][0] length
        *_replicate(0x01, 3),  # DynArray[Bytes[96], 3][0] data
        0x60,  # DynArray[Bytes[96], 3][1] length
        *_replicate(0x01, 3),  # DynArray[Bytes[96], 3][1] data
        0x60,  # DynArray[Bytes[96], 3][2] length
        *_replicate(0x01, 3),  # DynArray[Bytes[96], 3][2] data
    )

    data += _abi_payload_from_tuple(payload)

    with pytest.raises(DecodeError):
        # env.message_call(c.address, data=data)
        pass


@pytest.mark.skip(reason="low-level calls are not yet supported")
def test_abi_decode_runtimesz_oob2():
    # same principle as in test_abi_decode_runtimesz_oob
    # but adapted for dynarrays
    code = """
@external
def f(x: Bytes[2 * 32 + 3 * 32  + 3 * 32 * 4]):
    y: Bytes[2 * 32 + 3 * 32 + 3 * 32 * 4] = x
    decoded_y1: DynArray[DynArray[uint256, 3], 3] = _abi_decode(
        y,
        DynArray[DynArray[uint256, 3], 3]
    )
    """
    c = loads(code)

    data = method_id("f(bytes)")

    payload = (
        0x20,  # tuple head
        0x01E4,  # top-level bytes array length
        0x20,  # DynArray head
        0x03,  # DynArray length
        0x20 * 3,  # inner array0 head
        0x20 * 4 + 0x20 * 3,  # inner array1 head
        0x20 * 8 + 0x20 * 3,  # inner array2 head
        0x03,  # DynArray[..][0] length
        *_replicate(0x01, 3),  # DynArray[..][0] data
        0x03,  # DynArray[..][1] length
        *_replicate(0x01, 3),  # DynArray[..][1] data
        0x03,  # DynArray[..][2] length
        *_replicate(0x01, 3),  # DynArray[..][2] data
    )

    data += _abi_payload_from_tuple(payload)

    with pytest.raises(DecodeError):
        # env.message_call(c.address, data=data)
        pass


def test_abi_decode_head_roundtrip():
    # top-level head in the y2 buffer points to the y1 buffer
    # and y1 contains intermediate heads pointing to the inner arrays
    # which are in turn in the y2 buffer
    # NOTE: the test is memory allocator dependent - we assume that y1 and y2
    # have the 800 & 960 addresses respectively
    code = """
@external
def run(x1: Bytes[4 * 32], x2: Bytes[2 * 32 + 3 * 32  + 3 * 32 * 4]):
    y1: Bytes[4*32] = x1 # addr: 800
    y2: Bytes[2 * 32 + 3 * 32 + 3 * 32 * 4] = x2 # addr: 960
    decoded_y1: DynArray[DynArray[uint256, 3], 3] = _abi_decode(
        y2,
        DynArray[DynArray[uint256, 3], 3]
    )
    """
    c = loads(code)

    payload = (
        0x03,  # DynArray length
        # distance to y2 from y1 is 160
        160 + 0x20 + 0x20 * 3,  # points to DynArray[..][0] length
        160 + 0x20 + 0x20 * 4 + 0x20 * 3,  # points to DynArray[..][1] length
        160 + 0x20 + 0x20 * 8 + 0x20 * 3,  # points to DynArray[..][2] length
    )

    data1 = _abi_payload_from_tuple(payload)

    payload = (
        # (960 + (2**256 - 160)) % 2**256 == 800, ie will roundtrip to y1
        2**256 - 160,  # points to y1
        0x03,  # DynArray length (not used)
        0x20 * 3,  # inner array0 head
        0x20 * 4 + 0x20 * 3,  # inner array1 head
        0x20 * 8 + 0x20 * 3,  # inner array2 head
        0x03,  # DynArray[..][0] length
        *_replicate(0x01, 3),  # DynArray[..][0] data
        0x03,  # DynArray[..][1] length
        *_replicate(0x02, 3),  # DynArray[..][1] data
        0x03,  # DynArray[..][2] length
        *_replicate(0x03, 3),  # DynArray[..][2] data
    )

    data2 = _abi_payload_from_tuple(payload)

    with pytest.raises(DecodeError):
        c.run(data1, data2)


def test_abi_decode_merge_head_and_length():
    # compress head and length into 33B
    code = """
@external
def run(x: Bytes[32 * 2 + 8 * 32]) -> uint256:
    y: Bytes[32 * 2 + 8 * 32] = x
    decoded_y1: Bytes[256] = _abi_decode(y, Bytes[256])
    return len(decoded_y1)
    """
    c = loads(code)

    payload = (0x01, (0x00).to_bytes(1, "big"), *_replicate(0x00, 8))

    data = _abi_payload_from_tuple(payload)

    length = c.run(data)

    assert length == 256


def test_abi_decode_extcall_invalid_head():
    # the head returned from the extcall is set to invalid value of 480
    code = """
@external
def bar() -> (uint256, uint256, uint256):
    return (480, 0, 0)

interface A:
    def bar() -> String[32]: nonpayable

@external
def foo():
    x:String[32] = extcall A(self).bar()
    """
    c = loads(code)
    with pytest.raises(DecodeError):
        c.foo()


def test_abi_decode_extcall_oob():
    # the head returned from the extcall is 1 byte bigger than expected
    # thus we'll take the last 31 0-bytes from tuple[1] and the 1st byte from tuple[2]
    # and consider this the length - thus the length is 2**5
    # and thus we'll read 1B over the buffer end (33 + 32 + 32)
    code = """
@external
def bar() -> (uint256, uint256, uint256):
    return (33, 0, 2**(5+248))

interface A:
    def bar() -> String[32]: nonpayable

@external
def foo():
    x:String[32] = extcall A(self).bar()
    """
    c = loads(code)
    with pytest.raises(DecodeError):
        c.foo()


def test_abi_decode_extcall_runtimesz_oob():
    # the runtime size (33) is bigger than the actual payload (32 bytes)
    # thus we'll read 1B over the runtime size - but still within the static size of the buffer
    code = """
@external
def bar() -> (uint256, uint256, uint256):
    return (32, 33, 0)

interface A:
    def bar() -> String[64]: nonpayable

@external
def foo():
    x:String[64] = extcall A(self).bar()
    """
    c = loads(code)
    with pytest.raises(DecodeError):
        c.foo()


def test_abi_decode_extcall_truncate_returndata():
    # return more data than expected
    # the truncated data is still valid
    code = """
@external
def bar() -> (uint256, uint256, uint256, uint256):
    return (32, 32, 36, 36)

interface A:
    def bar() -> Bytes[32]: nonpayable

@external
def foo():
    x:Bytes[32] = extcall A(self).bar()
    """
    c = loads(code)
    c.foo()


def test_abi_decode_extcall_truncate_returndata2():
    # return more data than expected
    # after truncation the data is invalid because the length is too big
    # wrt to the static size of the buffer
    code = """
@external
def bar() -> (uint256, uint256, uint256, uint256):
    return (32, 33, 36, 36)

interface A:
    def bar() -> Bytes[32]: nonpayable

@external
def foo():
    x:Bytes[32] = extcall A(self).bar()
    """
    c = loads(code)
    with pytest.raises(DecodeError):
        c.foo()


def test_abi_decode_extcall_return_nodata():
    code = """
@external
def bar():
    return

interface A:
    def bar() -> Bytes[32]: nonpayable

@external
def foo():
    x:Bytes[32] = extcall A(self).bar()
    """
    c = loads(code)
    with pytest.raises(DecodeError):
        c.foo()


def test_abi_decode_extcall_array_oob():
    # same as in test_abi_decode_extcall_oob
    # DynArray[..][1] head isn't strict and points 1B over
    # thus the 1st B of 2**(5+248) is considered as the length (32)
    # thus we try to decode 1B over the buffer end
    code = """
@external
def bar() -> (uint256, uint256, uint256, uint256, uint256, uint256, uint256, uint256):
    return (
        32, # DynArray head
        2,  # DynArray length
        32 * 2,  # DynArray[..][0] head
        32 * 2 + 32 * 2 + 1, # DynArray[..][1] head
        32, # DynArray[..][0] length
        0,  # DynArray[..][0] data
        0,  # DynArray[..][1] length
        2**(5+248) # DynArray[..][1] length (and data)
    )

interface A:
    def bar() -> DynArray[Bytes[32], 2]: nonpayable

@external
def run():
    x: DynArray[Bytes[32], 2] = extcall A(self).bar()
    """
    c = loads(code)

    with pytest.raises(DecodeError):
        c.run()


def test_abi_decode_extcall_array_oob_with_truncate():
    # same as in test_abi_decode_extcall_oob but we also return more data than expected
    # DynArray[..][1] head isn't strict and points 1B over
    # thus the 1st B of 2**(5+248) is considered as the length (32)
    # thus we try to decode 1B over the buffer end
    code = """
@external
def bar() -> (uint256, uint256, uint256, uint256, uint256, uint256, uint256, uint256, uint256):
    return (
        32, # DynArray head
        2,  # DynArray length
        32 * 2,  # DynArray[..][0] head
        32 * 2 + 32 * 2 + 1, # DynArray[..][1] head
        32, # DynArray[..][0] length
        0,  # DynArray[..][0] data
        0,  # DynArray[..][1] length
        2**(5+248), # DynArray[..][1] length (and data)
        0   # extra data
    )

interface A:
    def bar() -> DynArray[Bytes[32], 2]: nonpayable

@external
def run():
    x: DynArray[Bytes[32], 2] = extcall A(self).bar()
    """
    c = loads(code)

    with pytest.raises(DecodeError):
        c.run()


def test_abi_decode_extcall_empty_array():
    code = """
@external
def bar() -> (uint256, uint256):
    return 32, 0

interface A:
    def bar() -> DynArray[Bytes[32], 2]: nonpayable

@external
def run():
    x: DynArray[Bytes[32], 2] = extcall A(self).bar()
    """
    c = loads(code)

    c.run()


@pytest.mark.skip(reason="structs not yet supported")
def test_abi_decode_extcall_complex_empty_dynarray():
    # 5th word of the payload points to the last word of the payload
    # which is considered the length of the Point.y array
    # because the length is 0, the decoding should succeed
    code = """
struct Point:
    x: uint256
    y: DynArray[uint256, 2]
    z: uint256

@external
def bar() -> (uint256, uint256, uint256, uint256, uint256, uint256):
    return 32, 1, 32, 1, 64, 0

interface A:
    def bar() -> DynArray[Point, 2]: nonpayable

@external
def run():
    x: DynArray[Point, 2] = extcall A(self).bar()
    assert len(x) == 1 and len(x[0].y) == 0
    """
    c = loads(code)

    c.run()


def test_abi_decode_extcall_complex_empty_dynarray2():
    # top-level head points 1B over the runtime buffer end
    # thus the decoding should fail although the length is 0
    code = """
struct Point:
    x: uint256
    y: DynArray[uint256, 2]
    z: uint256

@external
def bar() -> (uint256, uint256):
    return 33, 0

interface A:
    def bar() -> DynArray[Point, 2]: nonpayable

@external
def run():
    x: DynArray[Point, 2] = extcall A(self).bar()
    """
    c = loads(code)

    with pytest.raises(DecodeError):
        c.run()


def test_abi_decode_extcall_zero_len_array2():
    code = """
@external
def bar() -> (uint256, uint256):
    return 0, 0

interface A:
    def bar() -> DynArray[Bytes[32], 2]: nonpayable

@external
def run() -> uint256:
    x: DynArray[Bytes[32], 2] = extcall A(self).bar()
    return len(x)
    """
    c = loads(code)

    length = c.run()

    assert length == 0


def test_abi_decode_top_level_head_oob():
    code = """
@external
def run(x: Bytes[256], y: uint256):
    player_lost: bool = empty(bool)

    if y == 1:
        player_lost = True

    decoded: DynArray[Bytes[1], 2] = empty(DynArray[Bytes[1], 2])
    decoded = _abi_decode(x, DynArray[Bytes[1], 2])
    """
    c = loads(code)

    # head points over the buffer end
    payload = (0x0100, *_replicate(0x00, 7))

    data = _abi_payload_from_tuple(payload)

    with pytest.raises(DecodeError):
        c.run(data, 1)

    with pytest.raises(DecodeError):
        c.run(data, 0)


def test_abi_decode_dynarray_complex_insufficient_data():
    code = """
struct Point:
    x: uint256
    y: uint256

@external
def run(x: Bytes[32 * 8]):
    y: Bytes[32 * 8] = x
    decoded_y1: DynArray[Point, 3] = _abi_decode(y, DynArray[Point, 3])
    """
    c = loads(code)

    # runtime buffer has insufficient size - we decode 3 points, but provide only
    # 3 * 32B of payload
    payload = (0x20, 0x03, *_replicate(0x03, 3))

    data = _abi_payload_from_tuple(payload)

    with pytest.raises(DecodeError):
        c.run(data)


def test_abi_decode_dynarray_complex2():
    # point head to the 1st 0x01 word (ie the length)
    # but size of the point is 3 * 32B, thus we'd decode 2B over the buffer end
    code = """
struct Point:
    x: uint256
    y: uint256
    z: uint256


@external
def run(x: Bytes[32 * 8]):
    y: Bytes[32 * 11] = x
    decoded_y1: DynArray[Point, 2] = _abi_decode(y, DynArray[Point, 2])
    """
    c = loads(code)

    payload = (
        0xC0,  # points to the 1st 0x01 word (ie the length)
        *_replicate(0x03, 5),
        *_replicate(0x01, 2),
    )

    data = _abi_payload_from_tuple(payload)

    with pytest.raises(DecodeError):
        c.run(data)


@pytest.mark.skip(reason="structs not yet supported")
def test_abi_decode_complex_empty_dynarray():
    # point head to the last word of the payload
    # this will be the length, but because it's set to 0, the decoding should succeed
    code = """
struct Point:
    x: uint256
    y: DynArray[uint256, 2]
    z: uint256


@external
def run(x: Bytes[32 * 16]):
    y: Bytes[32 * 16] = x
    decoded_y1: DynArray[Point, 2] = _abi_decode(y, DynArray[Point, 2])
    assert len(decoded_y1) == 1 and len(decoded_y1[0].y) == 0
    """
    c = loads(code)

    payload = (
        0x20,
        0x01,
        0x20,
        0x01,
        0xA0,  # points to the last word of the payload
        0x04,
        0x02,
        0x02,
        0x00,  # length is 0, so decoding should succeed
    )

    data = _abi_payload_from_tuple(payload)

    c.run(data)


def test_abi_decode_complex_arithmetic_overflow():
    # inner head roundtrips due to arithmetic overflow
    code = """
struct Point:
    x: uint256
    y: DynArray[uint256, 2]
    z: uint256


@external
def run(x: Bytes[32 * 16]):
    y: Bytes[32 * 16] = x
    decoded_y1: DynArray[Point, 2] = _abi_decode(y, DynArray[Point, 2])
    """
    c = loads(code)

    payload = (
        0x20,
        0x01,
        0x20,
        0x01,  # both Point.x and Point.y length
        2**256 - 0x20,  # points to the "previous" word of the payload
        0x04,
        0x02,
        0x02,
        0x00,
    )

    data = _abi_payload_from_tuple(payload)

    with pytest.raises(DecodeError):
        c.run(data)


def test_abi_decode_empty_toplevel_dynarray():
    code = """
@external
def run(x: Bytes[2 * 32 + 3 * 32  + 3 * 32 * 4]):
    y: Bytes[2 * 32 + 3 * 32 + 3 * 32 * 4] = x
    assert len(y) == 2 * 32
    decoded_y1: DynArray[DynArray[uint256, 3], 3] = _abi_decode(
        y,
        DynArray[DynArray[uint256, 3], 3]
    )
    assert len(decoded_y1) == 0
    """
    c = loads(code)

    payload = (0x20, 0x00)  # DynArray head, DynArray length

    data = _abi_payload_from_tuple(payload)

    c.run(data)


def test_abi_decode_invalid_toplevel_dynarray_head():
    # head points 1B over the bounds of the runtime buffer
    code = """
@external
def run(x: Bytes[2 * 32 + 3 * 32  + 3 * 32 * 4]):
    y: Bytes[2 * 32 + 3 * 32 + 3 * 32 * 4] = x
    decoded_y1: DynArray[DynArray[uint256, 3], 3] = _abi_decode(
        y,
        DynArray[DynArray[uint256, 3], 3]
    )
    """
    c = loads(code)

    # head points 1B over the bounds of the runtime buffer
    payload = (0x21, 0x00)  # DynArray head, DynArray length

    data = _abi_payload_from_tuple(payload)

    with pytest.raises(DecodeError):
        c.run(data)


def test_nested_invalid_dynarray_head():
    code = """
@nonpayable
@external
def foo(x:Bytes[320]):
    if True:
        a: Bytes[320-32] = b''

        # make the word following the buffer x_mem dirty to make a potential
        # OOB revert
        fake_head: uint256 = 32
    x_mem: Bytes[320] = x

    y: DynArray[DynArray[uint256, 2], 2] = _abi_decode(x_mem,DynArray[DynArray[uint256, 2], 2])

@nonpayable
@external
def bar(x:Bytes[320]):
    x_mem: Bytes[320] = x

    y:DynArray[DynArray[uint256, 2], 2] = _abi_decode(x_mem,DynArray[DynArray[uint256, 2], 2])
    """
    c = loads(code)

    encoded = (0x20, 0x02)  # head of the dynarray  # len of outer
    inner = (
        0x0,  # head1
        # 0x0,  # head2
    )

    encoded = _abi_payload_from_tuple(encoded + inner)
    with pytest.raises(DecodeError):
        c.foo(encoded)  # revert
    with pytest.raises(DecodeError):
        c.bar(encoded)  # return [[],[]]


def test_static_outer_type_invalid_heads():
    code = """
@nonpayable
@external
def foo(x:Bytes[320]):
    x_mem: Bytes[320] = x
    y:DynArray[uint256, 2][2] = _abi_decode(x_mem,DynArray[uint256, 2][2])

@nonpayable
@external
def bar(x:Bytes[320]):
    if True:
        a: Bytes[160] = b''
        # write stuff here to make the call revert in case decode do
        # an out of bound access:
        fake_head: uint256 = 32
    x_mem: Bytes[320] = x
    y:DynArray[uint256, 2][2] = _abi_decode(x_mem,DynArray[uint256, 2][2])
    """
    c = loads(code)

    encoded = (0x20,)  # head of the static array
    inner = (
        0x00,  # head of the first dynarray
        # 0x00,  # head of the second dynarray
    )

    encoded = _abi_payload_from_tuple(encoded + inner)

    with pytest.raises(DecodeError):
        c.foo(encoded)
    with pytest.raises(DecodeError):
        c.bar(encoded)


def test_abi_decode_max_size():
    # test case where the payload is "too large" than the max size
    # of abi encoding the type. this can happen when the payload is
    # "sparse" and has garbage bytes in between the static and dynamic
    # sections
    code = """
@external
def foo(a:Bytes[1000]):
    v: DynArray[uint256, 1] = _abi_decode(a,DynArray[uint256, 1])
    """
    c = loads(code)

    payload = (
        0xA0,  # head
        0x00,  # garbage
        0x00,  # garbage
        0x00,  # garbage
        0x00,  # garbage
        0x01,  # len
        0x12,  # elem1
    )

    with pytest.raises(DecodeError):
        c.foo(_abi_payload_from_tuple(payload))


# returndatasize check for uint256
def test_returndatasize_check():
    code = """
@external
def bar():
    pass

interface A:
    def bar() -> uint256: nonpayable

@external
def run() -> uint256:
    return extcall A(self).bar()
    """
    c = loads(code)

    with pytest.raises(DecodeError):
        c.run()

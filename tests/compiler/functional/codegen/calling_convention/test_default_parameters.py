import pytest

from tests.utils import ZERO_ADDRESS
from vyper.compiler import compile_code
from vyper.exceptions import (
    InvalidLiteral,
    NonPayableViolation,
    StateAccessViolation,
    TypeMismatch,
    UndeclaredDefinition,
)
from vyper.utils import method_id


def test_default_param_abi(get_contract):
    code = """
@external
@payable
def safeTransferFrom(_data: Bytes[100] = b"test", _b: int128 = 1):
    pass
    """
    abi = get_contract(code).abi

    assert len(abi) == 3
    assert set([fdef["name"] for fdef in abi]) == {"safeTransferFrom"}
    assert abi[0]["inputs"] == []
    assert abi[1]["inputs"] == [{"type": "bytes", "name": "_data"}]
    assert abi[2]["inputs"] == [
        {"type": "bytes", "name": "_data"},
        {"type": "int128", "name": "_b"},
    ]


def test_basic_default_param_passthrough(get_contract):
    code = """
@external
def fooBar(_data: Bytes[100] = b"test", _b: int128 = 1) -> int128:
    return 12321
    """

    c = get_contract(code)

    assert c.fooBar() == 12321
    assert c.fooBar(b"drum drum") == 12321
    assert c.fooBar(b"drum drum", 2) == 12321


def test_basic_default_param_set(get_contract):
    code = """
@external
def fooBar(a:int128, b: uint256 = 333) -> (int128, uint256):
    return a, b
    """

    c = get_contract(code)
    assert c.fooBar(456, 444) == (456, 444)
    assert c.fooBar(456) == (456, 333)


def test_basic_default_param_set_2args(get_contract):
    code = """
@external
def fooBar(a:int128, b: uint256 = 999, c: address = 0x0000000000000000000000000000000000000001) -> (int128, uint256, address):  # noqa: E501
    return a, b, c
    """

    c = get_contract(code)
    c_default_value = "0x0000000000000000000000000000000000000001"
    b_default_value = 999
    addr2 = "0x1000000000000000000000000000000000004321"

    # b default value, c default value
    assert c.fooBar(123) == (123, b_default_value, c_default_value)
    # c default_value, b set from param
    assert c.fooBar(456, 444) == (456, 444, c_default_value)
    # no default values
    assert c.fooBar(6789, 4567, addr2) == (6789, 4567, addr2)


def test_default_param_bytes(get_contract):
    code = """
@external
def fooBar(a: Bytes[100], b: int128, c: Bytes[100] = b"testing", d: uint256 = 999) -> (Bytes[100], int128, Bytes[100], uint256):  # noqa: E501
    return a, b, c, d
    """
    c = get_contract(code)
    c_default = b"testing"
    d_default = 999

    # c set, 7d default value
    assert c.fooBar(b"booo", 12321, b"woo") == (b"booo", 12321, b"woo", d_default)
    # d set, c set
    assert c.fooBar(b"booo", 12321, b"lucky", 777) == (b"booo", 12321, b"lucky", 777)
    # no default values
    assert c.fooBar(b"booo", 12321) == (b"booo", 12321, c_default, d_default)


def test_default_param_array(get_contract):
    code = """
@external
def fooBar(a: Bytes[100], b: uint256[2], c: Bytes[6] = b"hello", d: int128[3] = [6, 7, 8]) -> (Bytes[100], uint256, Bytes[6], int128):  # noqa: E501
    return a, b[1], c, d[2]
    """
    c = get_contract(code)
    c_default = b"hello"
    d_default = 8

    # c set, d default value
    assert c.fooBar(b"booo", [99, 88], b"woo") == (b"booo", 88, b"woo", d_default)
    # d set, c set
    assert c.fooBar(b"booo", [22, 11], b"lucky", [24, 25, 26]) == (
        b"booo",
        11,
        b"lucky",
        26,
    )
    # no default values
    assert c.fooBar(b"booo", [55, 66]) == (b"booo", 66, c_default, d_default)


def test_default_param_interface(get_contract):
    code = """
interface Foo:
    def bar(): payable

FOO: constant(Foo) = Foo(0xFFfFfFffFFfffFFfFFfFFFFFffFFFffffFfFFFfF)

@external
def bar(a: uint256, b: Foo = Foo(0xF5D4020dCA6a62bB1efFcC9212AAF3c9819E30D7)) -> Foo:
    return b

@external
def baz(a: uint256, b: Foo = Foo(empty(address))) -> Foo:
    return b

@external
def faz(a: uint256, b: Foo = FOO) -> Foo:
    return b
    """
    c = get_contract(code)

    addr1 = "0xFFfFfFffFFfffFFfFFfFFFFFffFFFffffFfFFFfF"
    addr2 = "0xF5D4020dCA6a62bB1efFcC9212AAF3c9819E30D7"

    assert c.bar(1) == addr2
    assert c.bar(1, addr1) == addr1
    assert c.baz(1) == ZERO_ADDRESS
    assert c.baz(1, ZERO_ADDRESS) == ZERO_ADDRESS
    assert c.faz(1) == addr1
    assert c.faz(1, addr1) == addr1


def test_default_param_internal_function(get_contract):
    code = """
@internal
@view
def _foo(a: int128[3] = [1, 2, 3]) -> int128[3]:
    b: int128[3] = a
    return b


@external
@view
def foo() -> int128[3]:
    return self._foo([4, 5, 6])

@external
@view
def foo2() -> int128[3]:
    return self._foo()
    """
    c = get_contract(code)

    assert c.foo() == [4, 5, 6]
    assert c.foo2() == [1, 2, 3]


def test_default_param_external_function(get_contract):
    code = """
@external
@view
def foo(a: int128[3] = [1, 2, 3]) -> int128[3]:
    b: int128[3] = a
    return b
    """
    c = get_contract(code)

    assert c.foo([4, 5, 6]) == [4, 5, 6]
    assert c.foo() == [1, 2, 3]


def test_default_param_clamp(env, get_contract, tx_failed):
    code = """
@external
def bar(a: int128, b: int128 = -1) -> (int128, int128):  # noqa: E501
    return a, b
    """

    c = get_contract(code)

    assert c.bar(-123) == (-123, -1)
    assert c.bar(100, 100) == (100, 100)

    method = method_id("bar(int128,int128)")
    good_data = (200).to_bytes(32, "big") + (2**127 - 1).to_bytes(32, "big")
    bad_data = (200).to_bytes(32, "big") + (2**127).to_bytes(32, "big")

    assert env.message_call(c.address, data=method + good_data) == good_data
    with tx_failed():
        env.message_call(c.address, data=method + bad_data)


def test_default_param_private(get_contract):
    code = """
@internal
def fooBar(a: Bytes[100], b: uint256, c: Bytes[20] = b"crazy") -> (Bytes[100], uint256, Bytes[20]):
    return a, b, c

@external
def callMe() -> (Bytes[100], uint256, Bytes[20]):
    return self.fooBar(b'I just met you', 123456)

@external
def callMeMaybe() -> (Bytes[100], uint256, Bytes[20]):
    # return self.fooBar(b'here is my number', 555123456, b'baby')
    a: Bytes[100] = b""
    b: uint256 = 0
    c: Bytes[20] = b""
    a, b, c = self.fooBar(b'here is my number', 555123456, b'baby')
    return a, b, c
    """

    c = get_contract(code)

    # assert c.callMe() == (b'hello there', 123456, b'crazy')
    assert c.callMeMaybe() == (b"here is my number", 555123456, b"baby")


def test_environment_vars_as_default(get_contract, env):
    code = """
xx: uint256

@external
@payable
def foo(a: uint256 = msg.value) -> bool:
    self.xx += a
    return True

@external
def bar() -> uint256:
    return self.xx

@external
def get_balance() -> uint256:
    return self.balance
    """
    c = get_contract(code)
    env.set_balance(env.deployer, 31337 + 9001)
    c.foo(value=31337)
    assert c.bar() == 31337
    c.foo(666, value=9001)
    assert c.bar() == 31337 + 666
    assert c.get_balance() == 31337 + 9001
    assert env.get_balance(env.deployer) == 0

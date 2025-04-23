import pytest

from vyper.exceptions import StateAccessViolation


def test_pure_operation(get_contract):
    code = """
@pure
@external
def foo() -> int128:
    return 5
    """
    c = get_contract(code)
    assert c.foo() == 5


def test_pure_call(get_contract):
    code = """
@pure
@internal
def _foo() -> int128:
    return 5

@pure
@external
def foo() -> int128:
    return self._foo()
    """
    c = get_contract(code)
    assert c.foo() == 5


def test_pure_interface(get_contract):
    code1 = """
@pure
@external
def foo() -> int128:
    return 5
    """
    code2 = """
interface Foo:
    def foo() -> int128: pure

@pure
@external
def foo(a: address) -> int128:
    return staticcall Foo(a).foo()
    """
    c1 = get_contract(code1)
    c2 = get_contract(code2)
    assert c2.foo(c1.address) == 5


def test_type_in_pure(get_contract):
    code = """
@pure
@external
def _convert(x: bytes32) -> uint256:
    return convert(x, uint256)
    """
    c = get_contract(code)
    x = 123456
    bs = x.to_bytes(32, "big")
    assert x == c._convert(bs)

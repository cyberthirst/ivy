import pytest

from vyper.exceptions import StructureException


def test_continue1(get_contract):
    code = """
@external
def foo() -> bool:
    for i: uint256 in range(2):
        continue
        return False
    return True
"""
    c = get_contract(code)
    assert c.foo()


def test_continue2(get_contract):
    code = """
@external
def foo() -> int128:
    x: int128 = 0
    for i: int128 in range(3):
        x += 1
        continue
        x -= 1
    return x
"""
    c = get_contract(code)
    assert c.foo() == 3


def test_continue3(get_contract):
    code = """
@external
def foo() -> int128:
    x: int128 = 0
    for i: int128 in range(3):
        x += i
        continue
    return x
"""
    c = get_contract(code)
    assert c.foo() == 3


def test_continue4(get_contract):
    code = """
@external
def foo() -> int128:
    x: int128 = 0
    for i: int128 in range(6):
        if i % 2 == 0:
            continue
        x += 1
    return x
"""
    c = get_contract(code)
    assert c.foo() == 3

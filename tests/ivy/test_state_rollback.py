import pytest

from dataclasses import dataclass
from typing import Any

from tests.conftest import get_contract


@pytest.mark.parametrize("reverts", [True, False])
def test_storage_rollback(reverts, get_contract):
    src = """
c: public(uint256)

@external
def bar(fail: bool) -> uint256:
    self.c = 10
    assert fail == False
    return 66

@external
def foo(fail: bool) -> bool:
    b: Bytes[32] = b""
    s: bool = False
    s, b = raw_call(self, abi_encode(fail, method_id=method_id("bar(bool)")), max_outsize=32, revert_on_failure=False)
    return s
    """

    c = get_contract(src)
    success = c.foo(reverts)
    assert success == (not reverts)
    assert c.c() == (10 if success else 0)


@pytest.mark.parametrize("reverts", [True, False])
def test_storage_rollback2(reverts, get_contract):
    src = """
c: public(uint256)

@external
def foobar(u: uint256) -> uint256:
    self.c = 10
    return 66

@external
def bar(fail: bool) -> uint256:
    b: Bytes[32] = b""
    s: bool = False
    s, b = raw_call(self, abi_encode(self.c, method_id=method_id("foobar(uint256)")), max_outsize=32, revert_on_failure=False)
    assert fail == False
    return 66

@external
def foo(fail: bool) -> bool:
    b: Bytes[32] = b""
    s: bool = False
    s, b = raw_call(self, abi_encode(fail, method_id=method_id("bar(bool)")), max_outsize=32, revert_on_failure=False)
    return s
    """

    c = get_contract(src)
    success = c.foo(reverts)
    assert success == (not reverts)
    assert c.c() == (10 if success else 0)


@pytest.mark.parametrize("reverts", [True, False])
def test_storage_rollback3(reverts, get_contract):
    src = """
c: public(uint256)

@external
def foobar(u: uint256) -> uint256:
    self.c = 10
    return 66

@external
def bar(fail: bool) -> uint256:
    self.c = 20
    b: Bytes[32] = b""
    s: bool = False
    s, b = raw_call(self, abi_encode(self.c, method_id=method_id("foobar(uint256)")), max_outsize=32, revert_on_failure=False)
    assert fail == False
    return 66

@external
def foo(fail: bool) -> bool:
    b: Bytes[32] = b""
    s: bool = False
    s, b = raw_call(self, abi_encode(fail, method_id=method_id("bar(bool)")), max_outsize=32, revert_on_failure=False)
    return s
    """

    c = get_contract(src)
    success = c.foo(reverts)
    assert success == (not reverts)
    assert c.c() == (10 if success else 20)


@pytest.mark.parametrize("reverts", [True, False])
def test_storage_rollback4(get_contract, reverts):
    src = """
c: public(DynArray[uint256, 10])

@external
def bar(fail: bool) -> uint256:
    self.c[0] = 2
    assert fail == False
    return 66

@external
def foo(fail: bool) -> bool:
    self.c = [1]
    b: Bytes[32] = b""
    s: bool = False
    s, b = raw_call(self, abi_encode(fail, method_id=method_id("bar(bool)")), max_outsize=32, revert_on_failure=False)
    return s
    """

    c = get_contract(src)
    success = c.foo(reverts)
    assert success == (not reverts)
    assert c.c(0) == (2 if success else 1)


@pytest.mark.parametrize("reverts", [True, False])
def test_storage_rollback5(get_contract, reverts):
    src = """
struct C:
    a: uint256

c: public(C)

@external
def bar(fail: bool) -> uint256:
    self.c.a = 2
    assert fail == False
    return 66

@external
def foo(fail: bool) -> (bool, uint256):
    self.c.a = 1
    b: Bytes[32] = b""
    s: bool = False
    s, b = raw_call(self, abi_encode(fail, method_id=method_id("bar(bool)")), max_outsize=32, revert_on_failure=False)
    return s, self.c.a
    """

    c = get_contract(src)
    success, c_a = c.foo(reverts)
    assert success == (not reverts)
    assert c_a == 2 if success else 1


@pytest.mark.parametrize("reverts", [True, False])
def test_storage_rollback6(get_contract, reverts):
    src = """
struct C:
    a: uint256[10]

c: public(C)

@external
def bar(fail: bool) -> uint256:
    self.c.a[0] = 2
    assert fail == False
    return 66

@external
def foo(fail: bool) -> (bool, C):
    b: Bytes[32] = b""
    s: bool = False
    s, b = raw_call(self, abi_encode(fail, method_id=method_id("bar(bool)")), max_outsize=32, revert_on_failure=False)
    return s, self.c
    """

    c = get_contract(src)
    success, res = c.foo(reverts)
    assert success == (not reverts)
    expected = [2] + [0 for _ in range(9)] if success else [0 for _ in range(10)]
    expected = (expected,)
    assert res == expected


def test_storage_rollback7(get_contract):
    src = """
struct C:
    a: uint256[10]

c: public(C)

interface Foobar:
    def foobar(fail: bool): payable

@external
def foobar(fail: bool):
    self.c.a[0] = 11
    assert fail == False

@external
def bar(fail: bool) -> uint256:
    self.c.a[0] = 2
    extcall Foobar(self).foobar(fail)
    return 66

@external
def foo() -> uint256:
    b: Bytes[32] = b""
    s: bool = False
    fail: bool = True
    s, b = raw_call(self, abi_encode(fail, method_id=method_id("bar(bool)")), max_outsize=32, revert_on_failure=False)
    return self.c.a[0]
    """

    c = get_contract(src)
    res = c.foo()
    assert res == 0


@pytest.mark.parametrize("reverts", [True, False])
def test_storage_rollback8(get_contract, reverts):
    success_array = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    src = f"""
struct C:
    a: uint256[10]

c: public(C)

@external
def bar(fail: bool) -> uint256:
    self.c.a[0] = 2
    self.c.a = {success_array}
    assert fail == False
    return 66

@external
def foo(fail: bool) -> (bool, C):
    b: Bytes[32] = b""
    s: bool = False
    s, b = raw_call(self, abi_encode(fail, method_id=method_id("bar(bool)")), max_outsize=32, revert_on_failure=False)
    return s, self.c
    """

    c = get_contract(src)
    success, res = c.foo(reverts)
    assert success == (not reverts)
    expected = success_array if success else [0 for _ in range(10)]
    expected = (expected,)
    assert res == expected


@dataclass
class StorageTestCase:
    name: str
    declaration: str
    initial_value: str
    level_values: list[str]
    check_expr: str
    expected_initial: Any


TEST_CASES = [
    StorageTestCase(
        name="uint256",
        declaration="x: public(uint256)",
        initial_value="self.x = 1",
        level_values=["self.x = 100", "self.x = 200", "self.x = 300", "self.x = 400"],
        check_expr="x()",
        expected_initial=1,
    ),
    StorageTestCase(
        name="struct",
        declaration="""
        struct Point:
            x: uint256
            y: uint256
        p: public(Point)
        """,
        initial_value="self.p = Point({x: 1, y: 2})",
        level_values=[
            "self.p = Point({x: 100, y: 101})",
            "self.p = Point({x: 200, y: 201})",
            "self.p = Point({x: 300, y: 301})",
            "self.p = Point({x: 400, y: 401})",
        ],
        check_expr="p()",
        expected_initial=(1, 2),
    ),
    StorageTestCase(
        name="nested_struct",
        declaration="""
        struct Inner:
            x: uint256
        struct Outer:
            inner: Inner
            y: uint256
        p: public(Outer)
        """,
        initial_value="self.p = Outer({inner: Inner({x: 1}), y: 2})",
        level_values=[
            "self.p = Outer({inner: Inner({x: 100}), y: 101})",
            "self.p = Outer({inner: Inner({x: 200}), y: 201})",
            "self.p = Outer({inner: Inner({x: 300}), y: 301})",
            "self.p = Outer({inner: Inner({x: 400}), y: 401})",
        ],
        check_expr="p()",
        expected_initial=((1,), 2),
    ),
    StorageTestCase(
        name="dynarray",
        declaration="x: public(DynArray[uint256, 4])",
        initial_value="self.x = [1, 2, 3, 4]",
        level_values=[
            "self.x = [100, 101, 102, 103]",
            "self.x = [200, 201, 202, 203]",
            "self.x = [300, 301, 302, 303]",
            "self.x = [400, 401, 402, 403]",
        ],
        check_expr="x",
        expected_initial=[1, 2, 3, 4],
    ),
]


# TODO finish this test
@pytest.mark.parametrize("test_case", TEST_CASES)
@pytest.mark.parametrize("fail_depth", range(4))
def _deep_nested_rollbacks(fail_depth, test_case, get_contract):
    src = f"""
{test_case.declaration}

@external
def level3() -> uint256:
    {test_case.level_values[3]}
    assert {fail_depth != 3}
    return 1

@external
def level2() -> uint256:
    {test_case.level_values[2]}
    assert {fail_depth != 2}
    b: Bytes[32] = b""
    s: bool = False
    s, b = raw_call(self, method_id("level3()"), max_outsize=32, revert_on_failure=False)
    return 1

@external
def level1() -> uint256:
    {test_case.level_values[1]}
    assert {fail_depth != 1}
    b: Bytes[32] = b""
    s: bool = False
    s, b = raw_call(self, method_id("level2()"), max_outsize=32, revert_on_failure=False)
    return 1

@external
def level0() -> uint256:
    {test_case.level_values[0]}
    assert {fail_depth != 0}
    b: Bytes[32] = b""
    s: bool = False
    s, b = raw_call(self, method_id("level1()"), max_outsize=32, revert_on_failure=False)
    return 1

@external
def main() -> (bool, {test_case.name}):
    {test_case.initial_value}
    b: Bytes[32] = b""
    s: bool = False
    s, b = raw_call(self, method_id("level0()"), max_outsize=32, revert_on_failure=False)
    return s, self.{test_case.check_expr}
    """

    c = get_contract(src)
    success, final_value = c.main()
    assert not success
    assert final_value == test_case.expected_initial

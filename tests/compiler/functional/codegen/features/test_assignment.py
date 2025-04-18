import pytest

from vyper.evm.opcodes import version_check
from vyper.exceptions import CodegenPanic, ImmutableViolation, InvalidType, TypeMismatch


def test_augassign(get_contract):
    augassign_test = """
@external
def augadd(x: int128, y: int128) -> int128:
    z: int128 = x
    z += y
    return z

@external
def augmul(x: int128, y: int128) -> int128:
    z: int128 = x
    z *= y
    return z

@external
def augsub(x: int128, y: int128) -> int128:
    z: int128 = x
    z -= y
    return z

@external
def augmod(x: int128, y: int128) -> int128:
    z: int128 = x
    z %= y
    return z
    """

    c = get_contract(augassign_test)

    assert c.augadd(5, 12) == 17
    assert c.augmul(5, 12) == 60
    assert c.augsub(5, 12) == -7
    assert c.augmod(5, 12) == 5
    print("Passed aug-assignment test")


@pytest.mark.parametrize(
    "source",
    [
        """
@external
def entry() -> DynArray[uint256, 2]:
    a: DynArray[uint256, 2] = [1, 1]
    a[1] += a[1]
    return a
    """,
        """
@external
def entry() -> DynArray[uint256, 2]:
    a: uint256 = 1
    a += a
    b: DynArray[uint256, 2] = [a, a]
    b[0] -= b[0]
    b[0] += b[1] // 2
    return b
    """,
        """
a: DynArray[uint256, 2]

def read() -> uint256:
    return self.a[1]

@external
def entry() -> DynArray[uint256, 2]:
    self.a = [1, 1]
    self.a[1] += self.read()
    return self.a
    """,
        """
interface Foo:
    def foo() -> uint256: nonpayable

@external
def foo() -> uint256:
    return 1

@external
def entry() -> DynArray[uint256, 2]:
    # memory variable, can't be overwritten by extcall, so there
    # is no panic
    a: DynArray[uint256, 2] = [1, 1]
    a[1] += extcall Foo(self).foo()
    return a
    """,
        """
interface Foo:
    def foo() -> uint256: nonpayable

def get_foo() -> uint256:
    return extcall Foo(self).foo()

@external
def foo() -> uint256:
    return 1

@external
def entry() -> DynArray[uint256, 2]:
    # memory variable, can't be overwritten by extcall, so there
    # is no panic
    a: DynArray[uint256, 2] = [1, 1]
    # extcall hidden inside internal function
    a[1] += self.get_foo()
    return a
    """,
        """
a: public(DynArray[uint256, 2])

interface Foo:
    def foo() -> uint256: view

@external
def foo() -> uint256:
    return self.a[1]

@external
def entry() -> DynArray[uint256, 2]:
    self.a = [1, 1]
    self.a[1] += staticcall Foo(self).foo()
    return self.a
    """,
    ],
)
def test_augassign_rhs_references_lhs2(get_contract, source):
    c = get_contract(source)
    assert c.entry() == [1, 2]


def test_augassign_rhs_references_lhs_transient(get_contract):
    source = """
x: transient(DynArray[uint256, 2])

def read() -> uint256:
    return self.x[0]

@external
def entry() -> DynArray[uint256, 2]:
    self.x = [1, 1]
    # test augassign with state read hidden behind function call
    self.x[0] += self.read()
    # augassign with direct state read
    self.x[1] += self.x[0]
    return self.x
    """
    c = get_contract(source)

    assert c.entry() == [2, 3]


@pytest.mark.parametrize(
    "typ,in_val,out_val",
    [
        ("uint256", 77, 123),
        ("uint256[3]", [1, 2, 3], [4, 5, 6]),
        ("DynArray[uint256, 3]", [1, 2, 3], [4, 5, 6]),
        ("Bytes[5]", b"vyper", b"conda"),
    ],
)
def test_internal_assign(get_contract, typ, in_val, out_val):
    code = f"""
@internal
def foo(x: {typ}) -> {typ}:
    x = {out_val}
    return x

@external
def bar(x: {typ}) -> {typ}:
    return self.foo(x)
    """
    c = get_contract(code)

    assert c.bar(in_val) == out_val


def test_internal_assign_struct(get_contract):
    code = """
flag Bar:
    BAD
    BAK
    BAZ

struct Foo:
    a: uint256
    b: DynArray[Bar, 3]
    c: String[5]

@internal
def foo(x: Foo) -> Foo:
    x = Foo(a=789, b=[Bar.BAZ, Bar.BAK, Bar.BAD], c=\"conda\")
    return x

@external
def bar(x: Foo) -> Foo:
    return self.foo(x)
    """
    c = get_contract(code)

    assert c.bar((123, [1, 2, 4], "vyper")) == (789, [4, 2, 1], "conda")


def test_internal_assign_struct_member(get_contract):
    code = """
flag Bar:
    BAD
    BAK
    BAZ

struct Foo:
    a: uint256
    b: DynArray[Bar, 3]
    c: String[5]

@internal
def foo(x: Foo) -> Foo:
    x.a = 789
    x.b.pop()
    return x

@external
def bar(x: Foo) -> Foo:
    return self.foo(x)
    """
    c = get_contract(code)

    assert c.bar((123, [1, 2, 4], "vyper")) == (789, [1, 2], "vyper")


def test_internal_augassign(get_contract):
    code = """
@internal
def foo(x: int128) -> int128:
    x += 77
    return x

@external
def bar(x: int128) -> int128:
    return self.foo(x)
    """
    c = get_contract(code)

    assert c.bar(123) == 200


@pytest.mark.parametrize("typ", ["DynArray[uint256, 3]", "uint256[3]"])
def test_internal_augassign_arrays(get_contract, typ):
    code = f"""
@internal
def foo(x: {typ}) -> {typ}:
    x[1] += 77
    return x

@external
def bar(x: {typ}) -> {typ}:
    return self.foo(x)
    """
    c = get_contract(code)

    assert c.bar([1, 2, 3]) == [1, 79, 3]


def test_valid_literal_increment(get_contract):
    code = """
storx: uint256

@external
def foo1() -> int128:
    x: int128 = 122
    x += 1
    return x

@external
def foo2() -> uint256:
    x: uint256 = 122
    x += 1
    return x

@external
def foo3(y: uint256) -> uint256:
    self.storx = y
    self.storx += 1
    return self.storx
"""
    c = get_contract(code)

    assert c.foo1() == 123
    assert c.foo2() == 123
    assert c.foo3(11) == 12


def test_invalid_uint256_assignment_calculate_literals(get_contract):
    code = """
storx: uint256

@external
def foo2() -> uint256:
    x: uint256 = 0
    x = 3 * 4 // 2 + 1 - 2
    return x
"""
    c = get_contract(code)

    assert c.foo2() == 5


# See #838. Confirm that nested keys and structs work properly.
def test_nested_map_key_works(get_contract):
    code = """
struct X:
    a: int128
    b: int128
struct Y:
    c: int128
    d: int128
test_map1: HashMap[int128, X]
test_map2: HashMap[int128, Y]

@external
def set():
    self.test_map1[1].a = 333
    self.test_map2[333].c = 111


@external
def get(i: int128) -> int128:
    idx: int128 = self.test_map1[i].a
    return self.test_map2[idx].c
    """
    c = get_contract(code)
    c.set()
    assert c.get(1) == 111


def test_nested_map_key_problem(get_contract):
    code = """
struct X:
    a: int128
    b: int128
struct Y:
    c: int128
    d: int128
test_map1: HashMap[int128, X]
test_map2: HashMap[int128, Y]

@external
def set():
    self.test_map1[1].a = 333
    self.test_map2[333].c = 111


@external
def get() -> int128:
    return self.test_map2[self.test_map1[1].a].c
    """
    c = get_contract(code)
    c.set()
    assert c.get() == 111


overlap_codes = [
    """
@external
def bug(xs: uint256[2]) -> uint256[2]:
    # Initial value
    ys: uint256[2] = xs
    ys = [ys[1], ys[0]]
    return ys
    """,
    """
foo: uint256[2]
@external
def bug(xs: uint256[2]) -> uint256[2]:
    # Initial value
    self.foo = xs
    self.foo = [self.foo[1], self.foo[0]]
    return self.foo
    """,
    # TODO add transient tests when it's available
]


@pytest.mark.parametrize("code", overlap_codes)
def test_assign_rhs_lhs_overlap(get_contract, code):
    c = get_contract(code)

    assert c.bug([1, 2]) == [2, 1]


def test_assign_rhs_lhs_partial_overlap(get_contract):
    # GH issue 2418, generalize when lhs is not only dependency of rhs.
    code = """
@external
def bug(xs: uint256[2]) -> uint256[2]:
    # Initial value
    ys: uint256[2] = xs
    ys = [xs[1], ys[0]]
    return ys
    """
    c = get_contract(code)

    assert c.bug([1, 2]) == [2, 1]


def test_assign_rhs_lhs_overlap_dynarray(get_contract):
    # GH issue 2418, generalize to dynarrays
    code = """
@external
def bug(xs: DynArray[uint256, 2]) -> DynArray[uint256, 2]:
    ys: DynArray[uint256, 2] = xs
    ys = [ys[1], ys[0]]
    return ys
    """
    c = get_contract(code)
    assert c.bug([1, 2]) == [2, 1]


def test_assign_rhs_lhs_overlap_struct(get_contract):
    # GH issue 2418, generalize to structs
    code = """
struct Point:
    x: uint256
    y: uint256

@external
def bug(p: Point) -> Point:
    t: Point = p
    t = Point(x=t.y, y=t.x)
    return t
    """
    c = get_contract(code)
    assert c.bug((1, 2)) == (2, 1)


mload_merge_codes = [
    (
        """
@external
def foo() -> uint256[4]:
    # copy "backwards"
    xs: uint256[4] = [1, 2, 3, 4]

# dst < src
    xs[0] = xs[1]
    xs[1] = xs[2]
    xs[2] = xs[3]

    return xs
    """,
        [2, 3, 4, 4],
    ),
    (
        """
@external
def foo() -> uint256[4]:
    # copy "forwards"
    xs: uint256[4] = [1, 2, 3, 4]

# src < dst
    xs[1] = xs[0]
    xs[2] = xs[1]
    xs[3] = xs[2]

    return xs
    """,
        [1, 1, 1, 1],
    ),
    (
        """
@external
def foo() -> uint256[5]:
    # partial "forward" copy
    xs: uint256[5] = [1, 2, 3, 4, 5]

# src < dst
    xs[2] = xs[0]
    xs[3] = xs[1]
    xs[4] = xs[2]

    return xs
    """,
        [1, 2, 1, 2, 1],
    ),
]


# functional test that mload merging does not occur when source and dest
# buffers overlap. (note: mload merging only applies after cancun)
@pytest.mark.parametrize("code,expected_result", mload_merge_codes)
def test_mcopy_overlap(get_contract, code, expected_result):
    c = get_contract(code)
    assert c.foo() == expected_result

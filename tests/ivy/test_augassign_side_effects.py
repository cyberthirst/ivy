"""
Tests for augmented assignment with side effects in target expressions.

Bug: AugAssign evaluates subscript/attribute indices twice - once for read, once for write.
Expected: Index should be evaluated once, same location used for both read and write.
"""

from ivy.frontend.loader import loads


def test_augassign_subscript_side_effect_index():
    """
    Test that subscript index with side effects is only evaluated once.

    Bug: arr[inc()] += 1 calls inc() twice (once for read, once for write).
    Expected: inc() should be called once, both read and write use same index.
    """
    src = """
counter: uint256
arr: DynArray[uint256, 10]

@internal
def inc() -> uint256:
    old: uint256 = self.counter
    self.counter += 1
    return old

@external
def setup():
    self.arr = [100, 200, 300]
    self.counter = 0

@external
def aug_with_side_effect() -> (uint256, DynArray[uint256, 10]):
    # inc() should be called ONCE, returning 0
    # arr[0] should become 101 (100 + 1)
    self.arr[self.inc()] += 1
    return (self.counter, self.arr)
    """
    c = loads(src)
    c.setup()
    counter, arr = c.aug_with_side_effect()

    # Expected (Vyper behavior): counter=1, arr=[101, 200, 300]
    # Bug (Ivy): counter=2, arr=[100, 201, 300]
    assert counter == 1, f"inc() called {counter} times, expected 1"
    assert arr == [101, 200, 300], f"arr={arr}, expected [101, 200, 300]"


def test_augassign_subscript_side_effect_mul():
    """
    Test *= with side-effecting index.
    """
    src = """
counter: uint256
arr: DynArray[uint256, 10]

@internal
def inc() -> uint256:
    old: uint256 = self.counter
    self.counter += 1
    return old

@external
def setup():
    self.arr = [10, 20, 30]
    self.counter = 0

@external
def mul_with_side_effect() -> (uint256, DynArray[uint256, 10]):
    self.arr[self.inc()] *= 2
    return (self.counter, self.arr)
    """
    c = loads(src)
    c.setup()
    counter, arr = c.mul_with_side_effect()

    assert counter == 1, f"inc() called {counter} times, expected 1"
    assert arr == [20, 20, 30], f"arr={arr}, expected [20, 20, 30]"


def test_augassign_subscript_side_effect_sub():
    """
    Test -= with side-effecting index.
    """
    src = """
counter: uint256
arr: DynArray[uint256, 10]

@internal
def inc() -> uint256:
    old: uint256 = self.counter
    self.counter += 1
    return old

@external
def setup():
    self.arr = [100, 200, 300]
    self.counter = 0

@external
def sub_with_side_effect() -> (uint256, DynArray[uint256, 10]):
    self.arr[self.inc()] -= 50
    return (self.counter, self.arr)
    """
    c = loads(src)
    c.setup()
    counter, arr = c.sub_with_side_effect()

    assert counter == 1, f"inc() called {counter} times, expected 1"
    assert arr == [50, 200, 300], f"arr={arr}, expected [50, 200, 300]"


def test_augassign_nested_subscript_side_effect():
    """
    Test nested subscript with side-effecting outer index.
    arr2d[inc()][0] += 1
    """
    src = """
counter: uint256
arr2d: DynArray[DynArray[uint256, 5], 5]

@internal
def inc() -> uint256:
    old: uint256 = self.counter
    self.counter += 1
    return old

@external
def setup():
    self.arr2d = [[10, 11], [20, 21], [30, 31]]
    self.counter = 0

@external
def nested_side_effect() -> (uint256, DynArray[DynArray[uint256, 5], 5]):
    self.arr2d[self.inc()][0] += 1
    return (self.counter, self.arr2d)
    """
    c = loads(src)
    c.setup()
    counter, arr2d = c.nested_side_effect()

    assert counter == 1, f"inc() called {counter} times, expected 1"
    assert arr2d[0][0] == 11, f"arr2d[0][0]={arr2d[0][0]}, expected 11"


def test_augassign_hashmap_side_effect():
    """
    Test HashMap with side-effecting key.
    """
    src = """
counter: uint256
balances: HashMap[uint256, uint256]

@internal
def inc() -> uint256:
    old: uint256 = self.counter
    self.counter += 1
    return old

@external
def setup():
    self.balances[0] = 100
    self.balances[1] = 200
    self.counter = 0

@external
def hashmap_side_effect() -> (uint256, uint256, uint256):
    self.balances[self.inc()] += 50
    return (self.counter, self.balances[0], self.balances[1])
    """
    c = loads(src)
    c.setup()
    counter, bal0, bal1 = c.hashmap_side_effect()

    assert counter == 1, f"inc() called {counter} times, expected 1"
    assert bal0 == 150, f"balances[0]={bal0}, expected 150"
    assert bal1 == 200, f"balances[1]={bal1}, expected 200"


def test_augassign_struct_attribute_no_side_effect():
    """
    Test struct attribute augmented assignment (no side effects expected).

    This should work correctly since accessing self.my_struct doesn't
    involve function calls.
    """
    src = """
struct Point:
    x: uint256
    y: uint256

point: Point

@external
def setup():
    self.point = Point(x=10, y=20)

@external
def aug_struct_attr() -> (uint256, uint256):
    self.point.x += 5
    return (self.point.x, self.point.y)
    """
    c = loads(src)
    c.setup()
    x, y = c.aug_struct_attr()

    assert x == 15, f"x={x}, expected 15"
    assert y == 20, f"y={y}, expected 20"


def test_augassign_nested_struct_attribute():
    """
    Test nested struct attribute augmented assignment.
    """
    src = """
struct Inner:
    value: uint256

struct Outer:
    inner: Inner
    count: uint256

data: Outer

@external
def setup():
    self.data = Outer(inner=Inner(value=100), count=1)

@external
def aug_nested_attr() -> (uint256, uint256):
    self.data.inner.value += 10
    self.data.count += 1
    return (self.data.inner.value, self.data.count)
    """
    c = loads(src)
    c.setup()
    value, count = c.aug_nested_attr()

    assert value == 110, f"value={value}, expected 110"
    assert count == 2, f"count={count}, expected 2"


def test_augassign_struct_in_array_side_effect():
    """
    Test struct in array with side-effecting index.
    arr[inc()].value += 1
    """
    src = """
struct Item:
    value: uint256
    name: uint256

counter: uint256
items: DynArray[Item, 10]

@internal
def inc() -> uint256:
    old: uint256 = self.counter
    self.counter += 1
    return old

@external
def setup():
    self.items = [Item(value=100, name=1), Item(value=200, name=2)]
    self.counter = 0

@external
def struct_array_side_effect() -> (uint256, uint256, uint256):
    self.items[self.inc()].value += 50
    return (self.counter, self.items[0].value, self.items[1].value)
    """
    c = loads(src)
    c.setup()
    counter, val0, val1 = c.struct_array_side_effect()

    # Expected: counter=1, items[0].value=150, items[1].value=200
    assert counter == 1, f"inc() called {counter} times, expected 1"
    assert val0 == 150, f"items[0].value={val0}, expected 150"
    assert val1 == 200, f"items[1].value={val1}, expected 200"


def test_augassign_local_array_side_effect():
    """
    Test local (memory) array with side-effecting index.
    """
    src = """
counter: uint256

@internal
def inc() -> uint256:
    old: uint256 = self.counter
    self.counter += 1
    return old

@external
def local_array_side_effect() -> (uint256, DynArray[uint256, 10]):
    self.counter = 0
    arr: DynArray[uint256, 10] = [10, 20, 30]
    arr[self.inc()] += 5
    return (self.counter, arr)
    """
    c = loads(src)
    counter, arr = c.local_array_side_effect()

    assert counter == 1, f"inc() called {counter} times, expected 1"
    assert arr == [15, 20, 30], f"arr={arr}, expected [15, 20, 30]"


def test_augassign_nested_subscript_outer_side_effect():
    """
    Test nested subscript with side effect in outer index only.
    arr2d[inc()][1] += 1
    Note: Vyper disallows side effects in BOTH indices (risky overlap).
    """
    src = """
counter: uint256
arr2d: DynArray[DynArray[uint256, 5], 5]

@internal
def inc() -> uint256:
    old: uint256 = self.counter
    self.counter += 1
    return old

@external
def setup():
    self.arr2d = [[10, 11, 12], [20, 21, 22], [30, 31, 32]]
    self.counter = 0

@external
def nested_outer_side_effect() -> (uint256, DynArray[DynArray[uint256, 5], 5]):
    # inc() returns 0, so arr2d[0][1] = 11 -> 12
    self.arr2d[self.inc()][1] += 1
    return (self.counter, self.arr2d)
    """
    c = loads(src)
    c.setup()
    counter, arr2d = c.nested_outer_side_effect()

    assert counter == 1, f"inc() called {counter} times, expected 1"
    assert arr2d[0][1] == 12, f"arr2d[0][1]={arr2d[0][1]}, expected 12"
    # Other elements unchanged
    assert arr2d[0][0] == 10
    assert arr2d[1][0] == 20


def test_augassign_triple_nested_subscript():
    """
    Test triple nested subscript with side effect in outermost index.
    arr3d[inc()][0][0] += 1
    """
    src = """
counter: uint256
arr3d: DynArray[DynArray[DynArray[uint256, 3], 3], 3]

@internal
def inc() -> uint256:
    old: uint256 = self.counter
    self.counter += 1
    return old

@external
def setup():
    self.arr3d = [
        [[100, 101], [110, 111]],
        [[200, 201], [210, 211]],
    ]
    self.counter = 0

@external
def triple_nested() -> (uint256, uint256, uint256):
    self.arr3d[self.inc()][0][0] += 5
    return (self.counter, self.arr3d[0][0][0], self.arr3d[1][0][0])
    """
    c = loads(src)
    c.setup()
    counter, val0, val1 = c.triple_nested()

    assert counter == 1, f"inc() called {counter} times, expected 1"
    assert val0 == 105, f"arr3d[0][0][0]={val0}, expected 105"
    assert val1 == 200, f"arr3d[1][0][0]={val1}, expected 200 (unchanged)"


def test_augassign_deeply_nested_struct():
    """
    Test deeply nested struct attribute access.
    self.outer.middle.inner.value += 1
    """
    src = """
struct Inner:
    value: uint256

struct Middle:
    inner: Inner

struct Outer:
    middle: Middle

data: Outer

@external
def setup():
    self.data = Outer(middle=Middle(inner=Inner(value=100)))

@external
def deep_nested_attr() -> uint256:
    self.data.middle.inner.value += 7
    return self.data.middle.inner.value
    """
    c = loads(src)
    c.setup()
    result = c.deep_nested_attr()

    assert result == 107, f"value={result}, expected 107"


def test_augassign_struct_array_in_struct():
    """
    Test struct containing array, with side-effecting index.
    self.container.items[inc()].value += 1
    """
    src = """
struct Item:
    value: uint256

struct Container:
    items: DynArray[Item, 10]
    count: uint256

counter: uint256
container: Container

@internal
def inc() -> uint256:
    old: uint256 = self.counter
    self.counter += 1
    return old

@external
def setup():
    self.container = Container(
        items=[Item(value=100), Item(value=200), Item(value=300)],
        count=3
    )
    self.counter = 0

@external
def struct_array_in_struct() -> (uint256, uint256, uint256, uint256):
    self.container.items[self.inc()].value += 50
    return (
        self.counter,
        self.container.items[0].value,
        self.container.items[1].value,
        self.container.items[2].value
    )
    """
    c = loads(src)
    c.setup()
    counter, v0, v1, v2 = c.struct_array_in_struct()

    assert counter == 1, f"inc() called {counter} times, expected 1"
    assert v0 == 150, f"items[0].value={v0}, expected 150"
    assert v1 == 200, f"items[1].value={v1}, expected 200"
    assert v2 == 300, f"items[2].value={v2}, expected 300"


def test_augassign_array_inner_index_side_effect():
    """
    Test 2D array with side effect in inner index only.
    Note: Vyper disallows side effects in BOTH indices (risky overlap).
    """
    src = """
counter: uint256
arr2d: DynArray[DynArray[uint256, 5], 5]

@internal
def inc() -> uint256:
    old: uint256 = self.counter
    self.counter += 1
    return old

@external
def setup():
    self.arr2d = [[10, 11, 12], [20, 21, 22], [30, 31, 32]]
    self.counter = 0

@external
def inner_side_effect() -> (uint256, DynArray[DynArray[uint256, 5], 5]):
    # inc() returns 0, so arr2d[1][0] = 20 -> 120
    self.arr2d[1][self.inc()] += 100
    return (self.counter, self.arr2d)
    """
    c = loads(src)
    c.setup()
    counter, arr2d = c.inner_side_effect()

    assert counter == 1, f"counter={counter}, expected 1"
    assert arr2d[1][0] == 120, f"arr2d[1][0]={arr2d[1][0]}, expected 120"
    assert arr2d[0][0] == 10, "arr2d[0][0] should be unchanged"


def test_augassign_hashmap_nested_side_effect():
    """
    Test nested HashMap with side-effecting key.
    map[inc()][key] += value
    """
    src = """
counter: uint256
nested_map: HashMap[uint256, HashMap[uint256, uint256]]

@internal
def inc() -> uint256:
    old: uint256 = self.counter
    self.counter += 1
    return old

@external
def setup():
    self.nested_map[0][0] = 100
    self.nested_map[0][1] = 200
    self.nested_map[1][0] = 300
    self.counter = 0

@external
def nested_hashmap() -> (uint256, uint256, uint256):
    self.nested_map[self.inc()][0] += 50
    return (self.counter, self.nested_map[0][0], self.nested_map[1][0])
    """
    c = loads(src)
    c.setup()
    counter, v00, v10 = c.nested_hashmap()

    assert counter == 1, f"inc() called {counter} times, expected 1"
    assert v00 == 150, f"nested_map[0][0]={v00}, expected 150"
    assert v10 == 300, f"nested_map[1][0]={v10}, expected 300"


def test_augassign_multiple_augassigns_sequence():
    """
    Test multiple augmented assignments in sequence with same side-effect function.
    """
    src = """
counter: uint256
arr: DynArray[uint256, 10]

@internal
def inc() -> uint256:
    old: uint256 = self.counter
    self.counter += 1
    return old

@external
def setup():
    self.arr = [10, 20, 30, 40, 50]
    self.counter = 0

@external
def multiple_augassigns() -> (uint256, DynArray[uint256, 10]):
    # First: inc() returns 0, arr[0] = 10 -> 11
    self.arr[self.inc()] += 1
    # Second: inc() returns 1, arr[1] = 20 -> 22
    self.arr[self.inc()] += 2
    # Third: inc() returns 2, arr[2] = 30 -> 33
    self.arr[self.inc()] += 3
    return (self.counter, self.arr)
    """
    c = loads(src)
    c.setup()
    counter, arr = c.multiple_augassigns()

    assert counter == 3, f"counter={counter}, expected 3"
    assert arr == [11, 22, 33, 40, 50], f"arr={arr}, expected [11, 22, 33, 40, 50]"


def test_augassign_static_array_side_effect():
    """
    Test static array (fixed size) with side-effecting index.
    """
    src = """
counter: uint256
arr: uint256[5]

@internal
def inc() -> uint256:
    old: uint256 = self.counter
    self.counter += 1
    return old

@external
def setup():
    self.arr[0] = 100
    self.arr[1] = 200
    self.arr[2] = 300
    self.counter = 0

@external
def static_array_side_effect() -> (uint256, uint256[5]):
    self.arr[self.inc()] += 50
    return (self.counter, self.arr)
    """
    c = loads(src)
    c.setup()
    counter, arr = c.static_array_side_effect()

    assert counter == 1, f"inc() called {counter} times, expected 1"
    assert arr[0] == 150, f"arr[0]={arr[0]}, expected 150"
    assert arr[1] == 200, f"arr[1]={arr[1]}, expected 200"


def test_augassign_mixed_operators_same_target():
    """
    Test different augmented operators on same array with side effects.
    """
    src = """
counter: uint256
arr: DynArray[uint256, 10]

@internal
def get_idx() -> uint256:
    old: uint256 = self.counter
    self.counter += 1
    return old % 3  # Cycles through 0, 1, 2

@external
def setup():
    self.arr = [100, 100, 100]
    self.counter = 0

@external
def mixed_operators() -> (uint256, DynArray[uint256, 10]):
    self.arr[self.get_idx()] += 10   # idx=0, arr[0] = 110
    self.arr[self.get_idx()] -= 5    # idx=1, arr[1] = 95
    self.arr[self.get_idx()] *= 2    # idx=2, arr[2] = 200
    return (self.counter, self.arr)
    """
    c = loads(src)
    c.setup()
    counter, arr = c.mixed_operators()

    assert counter == 3, f"counter={counter}, expected 3"
    assert arr == [110, 95, 200], f"arr={arr}, expected [110, 95, 200]"


def test_augassign_struct_in_nested_arrays():
    """
    Test struct inside nested arrays with side effects.
    arr[inc()][idx].field += value
    """
    src = """
struct Point:
    x: uint256
    y: uint256

counter: uint256
grid: DynArray[DynArray[Point, 5], 5]

@internal
def inc() -> uint256:
    old: uint256 = self.counter
    self.counter += 1
    return old

@external
def setup():
    self.grid = [
        [Point(x=0, y=0), Point(x=1, y=1)],
        [Point(x=10, y=10), Point(x=11, y=11)],
    ]
    self.counter = 0

@external
def struct_in_nested() -> (uint256, uint256, uint256):
    # inc() returns 0, so grid[0][1].x is modified
    self.grid[self.inc()][1].x += 100
    return (self.counter, self.grid[0][1].x, self.grid[1][1].x)
    """
    c = loads(src)
    c.setup()
    counter, v01, v11 = c.struct_in_nested()

    assert counter == 1, f"inc() called {counter} times, expected 1"
    assert v01 == 101, f"grid[0][1].x={v01}, expected 101"
    assert v11 == 11, f"grid[1][1].x={v11}, expected 11 (unchanged)"


# ============================================================================
# Tests exploring Vyper's "risky overlap" boundary
#
# Vyper's read_write_overlap() prevents patterns where:
# - The container expression reads variables that the index expression writes
# - This catches: arr[inc()][inc()] where inc() reads AND writes counter
#
# These tests use SEPARATE counters for each index to avoid overlap.
# ============================================================================


def test_augassign_nested_subscript_separate_counters():
    """
    Test nested subscript with SEPARATE counters for each index.
    arr2d[inc_a()][inc_b()] += 1
    This is allowed because inc_a only touches counter_a, inc_b only touches counter_b.
    No read-write overlap.
    """
    src = """
counter_a: uint256
counter_b: uint256
arr2d: DynArray[DynArray[uint256, 5], 5]

@internal
def inc_a() -> uint256:
    old: uint256 = self.counter_a
    self.counter_a += 1
    return old

@internal
def inc_b() -> uint256:
    old: uint256 = self.counter_b
    self.counter_b += 1
    return old

@external
def setup():
    self.arr2d = [[10, 11, 12], [20, 21, 22], [30, 31, 32]]
    self.counter_a = 0
    self.counter_b = 1

@external
def nested_separate_counters() -> (uint256, uint256, DynArray[DynArray[uint256, 5], 5]):
    # inc_a() returns 0, inc_b() returns 1
    # arr2d[0][1] = 11 -> 111
    self.arr2d[self.inc_a()][self.inc_b()] += 100
    return (self.counter_a, self.counter_b, self.arr2d)
    """
    c = loads(src)
    c.setup()
    ca, cb, arr2d = c.nested_separate_counters()

    assert ca == 1, f"counter_a={ca}, expected 1"
    assert cb == 2, f"counter_b={cb}, expected 2"
    assert arr2d[0][1] == 111, f"arr2d[0][1]={arr2d[0][1]}, expected 111"


def test_augassign_triple_nested_separate_counters():
    """
    Test triple nested subscript with separate counters for each index.
    arr3d[inc_a()][inc_b()][inc_c()] += value
    """
    src = """
counter_a: uint256
counter_b: uint256
counter_c: uint256
arr3d: DynArray[DynArray[DynArray[uint256, 3], 3], 3]

@internal
def inc_a() -> uint256:
    old: uint256 = self.counter_a
    self.counter_a += 1
    return old

@internal
def inc_b() -> uint256:
    old: uint256 = self.counter_b
    self.counter_b += 1
    return old

@internal
def inc_c() -> uint256:
    old: uint256 = self.counter_c
    self.counter_c += 1
    return old

@external
def setup():
    self.arr3d = [
        [[100, 101, 102], [110, 111, 112]],
        [[200, 201, 202], [210, 211, 212]],
    ]
    self.counter_a = 0
    self.counter_b = 1
    self.counter_c = 0

@external
def triple_nested() -> (uint256, uint256, uint256, uint256):
    # inc_a()=0, inc_b()=1, inc_c()=0
    # arr3d[0][1][0] = 110 -> 115
    self.arr3d[self.inc_a()][self.inc_b()][self.inc_c()] += 5
    return (self.counter_a, self.counter_b, self.counter_c, self.arr3d[0][1][0])
    """
    c = loads(src)
    c.setup()
    ca, cb, cc, val = c.triple_nested()

    assert ca == 1, f"counter_a={ca}, expected 1"
    assert cb == 2, f"counter_b={cb}, expected 2"
    assert cc == 1, f"counter_c={cc}, expected 1"
    assert val == 115, f"arr3d[0][1][0]={val}, expected 115"


def test_augassign_nested_hashmap_separate_counters():
    """
    Test nested HashMap with separate counters.
    map[inc_a()][inc_b()] += value
    """
    src = """
counter_a: uint256
counter_b: uint256
nested_map: HashMap[uint256, HashMap[uint256, uint256]]

@internal
def inc_a() -> uint256:
    old: uint256 = self.counter_a
    self.counter_a += 1
    return old

@internal
def inc_b() -> uint256:
    old: uint256 = self.counter_b
    self.counter_b += 1
    return old

@external
def setup():
    self.nested_map[0][1] = 100
    self.nested_map[1][0] = 200
    self.counter_a = 0
    self.counter_b = 1

@external
def nested_hashmap() -> (uint256, uint256, uint256, uint256):
    # inc_a()=0, inc_b()=1 -> nested_map[0][1] = 100 -> 150
    self.nested_map[self.inc_a()][self.inc_b()] += 50
    return (self.counter_a, self.counter_b, self.nested_map[0][1], self.nested_map[1][0])
    """
    c = loads(src)
    c.setup()
    ca, cb, v01, v10 = c.nested_hashmap()

    assert ca == 1, f"counter_a={ca}, expected 1"
    assert cb == 2, f"counter_b={cb}, expected 2"
    assert v01 == 150, f"nested_map[0][1]={v01}, expected 150"
    assert v10 == 200, f"nested_map[1][0]={v10}, expected 200 (unchanged)"


def test_augassign_struct_array_nested_separate_counters():
    """
    Test struct in nested array with separate counters.
    arr[inc_a()][inc_b()].field += value
    """
    src = """
struct Item:
    value: uint256

counter_a: uint256
counter_b: uint256
grid: DynArray[DynArray[Item, 5], 5]

@internal
def inc_a() -> uint256:
    old: uint256 = self.counter_a
    self.counter_a += 1
    return old

@internal
def inc_b() -> uint256:
    old: uint256 = self.counter_b
    self.counter_b += 1
    return old

@external
def setup():
    self.grid = [
        [Item(value=10), Item(value=11)],
        [Item(value=20), Item(value=21)],
    ]
    self.counter_a = 1
    self.counter_b = 0

@external
def struct_nested() -> (uint256, uint256, uint256, uint256):
    # inc_a()=1, inc_b()=0 -> grid[1][0].value = 20 -> 70
    self.grid[self.inc_a()][self.inc_b()].value += 50
    return (self.counter_a, self.counter_b, self.grid[0][0].value, self.grid[1][0].value)
    """
    c = loads(src)
    c.setup()
    ca, cb, v00, v10 = c.struct_nested()

    assert ca == 2, f"counter_a={ca}, expected 2"
    assert cb == 1, f"counter_b={cb}, expected 1"
    assert v00 == 10, f"grid[0][0].value={v00}, expected 10 (unchanged)"
    assert v10 == 70, f"grid[1][0].value={v10}, expected 70"


def test_augassign_pure_function_index():
    """
    Test with a pure function (no state access) as index.
    Pure functions don't write to any storage, so no overlap is possible.
    """
    src = """
arr: DynArray[uint256, 10]

@pure
@internal
def get_index(x: uint256) -> uint256:
    return x % 3

@external
def setup():
    self.arr = [100, 200, 300]

@external
def pure_index(x: uint256) -> DynArray[uint256, 10]:
    self.arr[self.get_index(x)] += 10
    return self.arr
    """
    c = loads(src)
    c.setup()

    arr = c.pure_index(0)  # get_index(0) = 0
    assert arr == [110, 200, 300], f"arr={arr}, expected [110, 200, 300]"

    arr = c.pure_index(1)  # get_index(1) = 1
    assert arr == [110, 210, 300], f"arr={arr}, expected [110, 210, 300]"


def test_augassign_view_function_index():
    """
    Test with a view function (reads but doesn't write) as index.
    View functions don't write to storage, so no overlap with themselves.
    """
    src = """
current_index: uint256
arr: DynArray[uint256, 10]

@view
@internal
def get_current() -> uint256:
    return self.current_index

@external
def setup():
    self.arr = [100, 200, 300]
    self.current_index = 0

@external
def set_index(idx: uint256):
    self.current_index = idx

@external
def view_index() -> DynArray[uint256, 10]:
    self.arr[self.get_current()] += 5
    return self.arr
    """
    c = loads(src)
    c.setup()

    arr = c.view_index()
    assert arr == [105, 200, 300], f"arr={arr}, expected [105, 200, 300]"

    c.set_index(2)
    arr = c.view_index()
    assert arr == [105, 200, 305], f"arr={arr}, expected [105, 200, 305]"


def test_augassign_complex_nested_with_attribute():
    """
    Test complex nested pattern: arr[inc()][idx].struct_field.nested += value
    Uses a deeply nested struct accessed via array with side effect.
    """
    src = """
struct Inner:
    value: uint256

struct Outer:
    inner: Inner
    count: uint256

counter: uint256
items: DynArray[DynArray[Outer, 3], 3]

@internal
def inc() -> uint256:
    old: uint256 = self.counter
    self.counter += 1
    return old

@external
def setup():
    self.items = [
        [Outer(inner=Inner(value=10), count=1), Outer(inner=Inner(value=11), count=1)],
        [Outer(inner=Inner(value=20), count=2), Outer(inner=Inner(value=21), count=2)],
    ]
    self.counter = 0

@external
def complex_nested() -> (uint256, uint256, uint256):
    # inc() returns 0, so items[0][1].inner.value is modified
    self.items[self.inc()][1].inner.value += 100
    return (self.counter, self.items[0][1].inner.value, self.items[1][1].inner.value)
    """
    c = loads(src)
    c.setup()
    counter, v01, v11 = c.complex_nested()

    assert counter == 1, f"counter={counter}, expected 1"
    assert v01 == 111, f"items[0][1].inner.value={v01}, expected 111"
    assert v11 == 21, f"items[1][1].inner.value={v11}, expected 21 (unchanged)"

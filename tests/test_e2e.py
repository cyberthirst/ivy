from ivy.frontend.loader import loads


def test_if_control_flow():
    src = """
@external
def foo() -> uint256:
    a: uint256 = 1
    if a == 1:
        a = 2
    else:
        a = 3
    return a + 42
    """

    c = loads(src)
    assert c.foo() == 44


def test_for_control_flow():
    src = """
@external
def foo() -> uint256:
    a: DynArray[uint256, 10] = [1, 2, 3]
    counter: uint256 = 0
    for i: uint256 in a:
        counter += i
    return counter
    """

    c = loads(src)
    assert c.foo() == 6


def test_array_assign():
    src = """
@external
def foo() -> uint256:
    bar: DynArray[uint256, 10] = [1, 2]
    bar[0] = 3
    return bar[0] + bar[1] + 42
    """

    c = loads(src)
    assert c.foo() == 47


def test_internal_call():
    src = """
    @internal
    def bar() -> uint256:
        a: DynArray[uint256, 10] = [1, 2, 3]
        counter: uint256 = 0
        for i: uint256 in a:
            counter += i
        return counter


    @external
    def foo() -> uint256:
        a: DynArray[uint256, 10] = [1, 2, 3]
        counter: uint256 = 0
        for i: uint256 in a:
            counter += i
        return counter + self.bar()
    """
    c = loads(src)
    assert c.foo() == 12


def test_internal_call_with_args():
    src = """
    @internal
    def baz(a: uint256) -> uint256:
        return a 
    
    @internal
    def bar(a: uint256) -> uint256:
        return a + self.baz(3)

    @external
    def foo() -> uint256:
        a: uint256 = 1
        return a + self.bar(2)
    """

    c = loads(src)
    assert c.foo() == 6


def test_storage_variables():
    src = """
d: uint256

@external
def foo() -> uint256:
    a: uint256 = 1
    self.d = a
    if a == 1:
        a = 2
    else:
        a = 3
    return self.d + 42
    """

    c = loads(src)
    assert c.foo() == 43


def test_storage_variables2():
    src = """
d: uint256
k: uint256

@external
def foo() -> uint256:
    self.k = 1
    self.d = self.k
    self.d += self.k
    return self.d + self.k
    """

    c = loads(src)
    assert c.foo() == 3


def test_storage_variables3():
    src = """
d: uint256

@internal
def bar():
    a: DynArray[uint256, 10] = [1, 2, 3]
    for i: uint256 in a:
        self.d += i


@external
def foo() -> uint256:
    a: DynArray[uint256, 10] = [1, 2, 3]
    counter: uint256 = 0
    for i: uint256 in a:
        self.d += i
    self.bar()
    return self.d
    """
    c = loads(src)
    assert c.foo() == 12


def test_tstorage_variables0():
    src = """
d: transient(uint256)
k: transient(uint256)

@external
def foo() -> uint256:
    self.k = 1
    self.d = self.k
    self.d += self.k
    return self.d + self.k
    """

    c = loads(src)
    assert c.foo() == 3


def test_tstorage_variables2():
    src = """
d: transient(uint256)
k: transient(uint256)

@external
def foo() -> uint256:
    if self.k == 0:
        self.k = 1
    self.d = self.k
    self.d += self.k
    return self.d + self.k
    """

    c = loads(src)
    assert c.foo() == 3


def test_default_storage_values():
    src = """
struct S:
    a: uint256 
    
a: uint256
b: uint256
c: DynArray[uint256, 10]
#d: S
e: Bytes[10]
f: String[10]

@external
def foo() -> uint256:
    assert self.a == 0
    assert self.b == 0
    assert len(self.c) == 0
    #assert self.d.a == 0
    assert len(self.e) == 0
    assert len(self.f) == 0
    return 1
    """

    c = loads(src)
    assert c.foo() == 1


def test_range_builtin():
    src = """
a: uint256

@external
def foo() -> uint256:
    for i: uint256 in range(10):
        self.a += i 
    return self.a
    """

    c = loads(src)
    expected = 0
    for i in range(10):
        expected += i
    assert c.foo() == expected


def test_range_builtin2():
    src = """
a: uint256

@external
def foo() -> uint256:
    k: uint256 = 5
    for i: uint256 in range(k, bound=5):
        self.a += i 
    return self.a
    """

    c = loads(src)
    expected = 0
    for i in range(5):
        expected += i
    assert c.foo() == expected


def test_range_builtin3():
    src = """
a: uint256

@external
def foo() -> uint256:
    for i: uint256 in range(1, 5):
        self.a += i 
    return self.a
    """

    c = loads(src)
    expected = 0
    for i in range(1, 5):
        expected += i
    assert c.foo() == expected


def test_range_builtin4():
    src = """
a: uint256

@external
def foo() -> uint256:
    k: uint256 = 1
    for i: uint256 in range(k, 5, bound=4):
        self.a += i 
    return self.a
    """

    c = loads(src)
    expected = 0
    for i in range(1, 5):
        expected += i
    assert c.foo() == expected


def test_len_builtin():
    src = """
@external
def foo() -> uint256:
    a: DynArray[uint256, 3] = [1, 2, 3]
    return len(a)
    """
    c = loads(src)
    assert c.foo() == 3


def test_len_builtin2():
    src = """
d: DynArray[uint256, 3]
@external
def foo() -> uint256:
    return len(self.d)
    """
    c = loads(src)
    assert c.foo() == 0


def test_len_builtin3():
    src = """
s: String[10]
@external
def foo() -> uint256:
    self.s = "hello"
    return len(self.s)
    """
    c = loads(src)
    assert c.foo() == 5


def test_len_builtin4():
    src = """
s: Bytes[10]
@external
def foo() -> uint256:
    self.s = b"hello"
    return len(self.s)
    """
    c = loads(src)
    assert c.foo() == 5


def test_return_abi_encode():
    src = """
@external
def foo() -> String[32]:
    return "hello"
    """

    c = loads(src)
    assert c.foo() == "hello"


def test_return_abi_encode2():
    src = """
@external
def foo() -> DynArray[uint256, 3]:
    a: DynArray[uint256, 3] = [1, 2, 3]
    return a
    """

    c = loads(src)
    assert c.foo() == [1, 2, 3]


def test_return_abi_encode3():
    src = """
@external
def foo() -> (uint256, uint256):
    return 666, 666
    """

    c = loads(src)
    assert c.foo() == (666, 666)


def test_self_call():
    src = """
interface Foo:
    def bar() -> uint256: payable

@external
def bar() -> uint256:
    return 1

@external
def foo() -> uint256:
    a: uint256 = 0
    a = extcall Foo(self).bar()
    return a
    """

    c = loads(src)
    assert c.foo() == 1


def test_self_call2():
    src = """
interface Foo:
    def bar() -> String[32]: payable
    
a: uint256

@external
def bar() -> String[32]:
    self.a += 42
    return "hello"

@external
def foo() -> uint256:
    self.a = 10 
    s: String[32] = extcall Foo(self).bar()
    return self.a
    """

    c = loads(src)
    assert c.foo() == 52


def test_default_arg_value():
    src = """
    @internal
    def baz(a: uint256, b:uint256=10) -> uint256:
        return a + b

    @internal
    def bar(a: uint256, b: uint256=20) -> uint256:
        return a + b + self.baz(3)

    @external
    def foo() -> uint256:
        a: uint256 = 1
        return a + self.bar(2)
    """

    c = loads(src)
    assert c.foo() == 36


def test_default_arg_value2():
    src = """
    @internal
    def baz(a:uint256=3, b:uint256=4) -> uint256:
        return a + b

    @internal
    def bar(a:uint256=1, b: uint256=2) -> uint256:
        return a + b + self.baz()

    @external
    def foo() -> uint256:
        a: uint256 = 0
        return a + self.bar()
    """

    c = loads(src)
    assert c.foo() == 0 + 1 + 2 + 3 + 4


def test_default_arg_value3():
    src = """
@external
def foo(a: uint256=42) -> uint256:
    return a
    """

    c = loads(src)
    assert c.foo() == 42


def test_default_arg_value4():
    src = """
@external
def foo(a: uint256=34, b: uint256=48) -> uint256:
    return a + b
    """

    c = loads(src)
    assert c.foo(42) == 90


def test_default_arg_value5():
    src = """
interface Foo:
    def bar(s: String[32]="hello") -> String[32]: payable

@external
def bar(s: String[32]="hello") -> String[32]:
    return "hello"

@external
def foo(a:uint256=10) -> uint256:
    s: String[32] = extcall Foo(self).bar()
    return a + len(s)
    """

    c = loads(src)
    assert c.foo() == 15


def test_external_func_arg():
    src = """
@external
def foo(a: uint256) -> uint256:
    return a
    """

    c = loads(src)
    assert c.foo(42) == 42


def test_external_func_arg2():
    src = """
@external
def foo(a: DynArray[uint256, 10], s: String[100]) -> (DynArray[uint256, 10], String[100]):
    return a, s
    """

    c = loads(src)
    assert c.foo([1, 2, 3], "hello") == ([1, 2, 3], "hello")


def test_external_func_arg3():
    dynarray_t = "DynArray[DynArray[uint256, 10], 10]"
    src = f"""
@external
def foo(a: DynArray[uint256, 10], s: String[100], b: {dynarray_t}) -> (DynArray[uint256, 10], String[100], {dynarray_t}):
    return a, s, b
    """

    c = loads(src)
    complex_array = [[4, 5, 6], [7, 8, 9, 10, 11], [], [12]]
    assert c.foo([1, 2, 3], "hello", complex_array) == (
        [1, 2, 3],
        "hello",
        complex_array,
    )


def test_external_func_arg4():
    tuple_t = "(String[93], DynArray[DynArray[uint256, 10], 10])"
    src = f"""
@external
def foo(a: DynArray[uint256, 10], s: String[100], b: {tuple_t}) -> (DynArray[uint256, 10], String[100], {tuple_t}):
    return a, s, b
    """

    c = loads(src)
    complex_tuple = ("apollo", [[4, 5, 6], [7, 8, 9, 10, 11], [], [12]])
    assert c.foo([1, 2, 3], "hello", complex_tuple) == (
        [1, 2, 3],
        "hello",
        complex_tuple,
    )

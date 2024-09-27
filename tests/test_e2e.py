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

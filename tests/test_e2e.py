import pytest

from ivy.frontend.loader import loads
from ivy.exceptions import StaticCallViolation


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


def test_internal_call_without_return():
    src = """
a: uint256
    
@internal
def bar():
    self.a = 42

@external
def foo() -> uint256:
    self.bar()
    return self.a
    """
    c = loads(src)
    assert c.foo() == 42


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


def test_internal_call_with_args2():
    src = """
@internal
def baz(a: uint256) -> uint256:
    return a 

@internal
def bar(a: uint256) -> uint256:
    return self.baz(3) + a

@external
def foo() -> uint256:
    a: uint256 = 1
    return self.bar(2) + a
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


def test_statefulness_of_storage():
    src = """
d: uint256

@external
def foo() -> uint256:
    self.d += 1
    return self.d
    """

    c = loads(src)
    for i in range(5):
        assert c.foo() == i + 1


def test_statefulness_of_storage2():
    src = """
d: uint256

@external
def foo() -> uint256:
    self.d += 1
    return self.d
    
@external
def bar() -> uint256:
    self.d += 1
    return self.d
    """

    c = loads(src)
    for i in range(5):
        assert c.foo() == i * 2 + 1
        assert c.bar() == i * 2 + 2


def test_statefulness_of_tstorage():
    src = """
d: transient(uint256)

interface Bar:
    def bar() -> uint256: payable

@external
def foo() -> uint256:
    self.d += 1
    return extcall Bar(self).bar()

@external
def bar() -> uint256:
    self.d += 1
    return self.d
    """

    c = loads(src)
    for i in range(3):
        assert c.foo() == 2


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
d: S
e: Bytes[10]
f: String[10]

@external
def foo() -> uint256:
    assert self.a == 0
    assert self.b == 0
    assert len(self.c) == 0
    assert self.d.a == 0
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


def test_empty_builtin():
    src = """
@external
def foo() -> uint256:
    return empty(uint256)
    """

    c = loads(src)
    assert c.foo() == 0


def test_empty_builtin2():
    src = """
@external
def foo() -> String[56]:
    return empty(String[56])
    """

    c = loads(src)
    assert c.foo() == ""


def test_empty_builtin3():
    src = """
@external
def foo() -> DynArray[String[32], 10]:
    return empty(DynArray[String[32], 10])
    """

    c = loads(src)
    assert c.foo() == []


def test_raw_call_builtin():
    src = """
@external
def bar() -> uint256:
    return 66

@external
def foo() -> uint256:
    b: Bytes[32] = raw_call(self, abi_encode(method_id=method_id("bar()")), max_outsize=32)
    return abi_decode(b, uint256)
    """

    c = loads(src)
    assert c.foo() == 66


def test_raw_call_builtin2():
    src = """
@external
def bar(a: uint256) -> uint256:
    return a

@external
def foo(foo: uint256) -> uint256:
    b: Bytes[32] = raw_call(self, abi_encode(foo, method_id=method_id("bar(uint256)")), max_outsize=32)
    return abi_decode(b, uint256)
    """

    c = loads(src)
    assert c.foo(66) == 66


def test_raw_call_builtin3():
    value = 66
    src = f"""
@external
def bar(a: uint256) -> uint256:
    return a

@external
def foo(target: address) -> uint256:
    arg: uint256 = {value}
    b: Bytes[32] = raw_call(target, abi_encode(arg, method_id=method_id("bar(uint256)")), max_outsize=32)
    return abi_decode(b, uint256)
    """

    c = loads(src)
    c2 = loads(src)
    assert c.foo(c2) == value


def test_raw_call_revert_on_failure():
    src = """
@external
def bar(u: uint256) -> uint256:
    assert u == 666
    return 66

@external
def foo() -> bool:
    b: Bytes[32] = b""
    s: bool = False
    u: uint256 = 0
    s, b = raw_call(self, abi_encode(u, method_id=method_id("bar(uint256)")), max_outsize=32, revert_on_failure=False)
    return s
    """

    c = loads(src)
    assert c.foo() == False


def test_raw_call_delegate():
    value = 66
    src = f"""
c: uint256 
    
@external
def bar(a: uint256):
    self.c = a

@external
def foo(target: address) -> uint256:
    arg: uint256 = {value}
    raw_call(target, abi_encode(arg, method_id=method_id("bar(uint256)")), is_delegate_call=True)
    return self.c
    """

    c = loads(src)
    c2 = loads(src)
    assert c.foo(c2) == value


def test_raw_call_delegate2():
    value = 66
    src = f"""
c: uint256 
a: DynArray[uint256, 10]

@external
def foo(target: address) -> (uint256, DynArray[uint256, 10]):
    arg: uint256 = {value}
    raw_call(target, abi_encode(arg, method_id=method_id("bar(uint256)")), is_delegate_call=True)
    return self.c, self.a
    """

    src2 = """
c: uint256 
a: DynArray[uint256, 10]

@external
def bar(a: uint256):
    self.c = a
    self.a = [a, a, a]
    """

    c = loads(src)
    c2 = loads(src2)
    assert c.foo(c2) == (value, [value, value, value])


def test_raw_call_static():
    value = 66
    src = f"""
c: uint256 

@external
def foo(target: address) -> uint256:
    arg: uint256 = {value}
    raw_call(target, abi_encode(arg, method_id=method_id("bar(uint256)")), is_static_call=True)
    return self.c
    """

    src2 = """
c: uint256 

@external
def bar(a: uint256):
    self.c = a
    """

    c = loads(src)
    c2 = loads(src2)
    with pytest.raises(StaticCallViolation):
        c.foo(c2)


def test_raw_call_static2():
    value = 66
    src = f"""
c: uint256 

@external
def foo(target: address) -> uint256:
    arg: uint256 = {value}
    res: Bytes[32] = raw_call(target, abi_encode(arg, method_id=method_id("bar(uint256)")), max_outsize=32, is_static_call=True)
    return abi_decode(res, uint256)
    """

    src2 = """
c: uint256 

@external
def bar(a: uint256) -> uint256:
    return self.c + a
    """

    c = loads(src)
    c2 = loads(src2)
    assert c.foo(c2) == value


def test_external_static_call():
    value = 66
    src = f"""
interface Bar:
    def bar(a: uint256) -> uint256: view
    
c: uint256 

@external
def foo(target: address) -> uint256:
    arg: uint256 = {value}
    return staticcall Bar(target).bar(arg)
    """

    src2 = """
c: uint256 

@external
def bar(a: uint256) -> uint256:
    return self.c + a
    """

    c = loads(src)
    c2 = loads(src2)
    assert c.foo(c2) == value


def test_external_static_call2():
    value = 66
    src = f"""
interface Bar:
    def bar(a: uint256) -> uint256: view

c: uint256 

@external
def foo(target: address) -> uint256:
    arg: uint256 = {value}
    return staticcall Bar(target).bar(arg)
    """

    src2 = """
c: uint256 

@external
def bar(a: uint256) -> uint256:
    self.c = a
    return self.c + a
    """

    c = loads(src)
    c2 = loads(src2)
    with pytest.raises(StaticCallViolation):
        c.foo(c2)


def test_abi_encode_builtin():
    src = """
@external
def foo(foo: uint256) -> uint256:
    return abi_decode(abi_encode(foo), uint256)
    """

    c = loads(src)
    assert c.foo(66) == 66


def test_abi_encode_builtin2():
    typ = "DynArray[uint256, 10]"
    src = f"""
@external
def foo(foo: {typ}) -> {typ}:
    return abi_decode(abi_encode(foo), {typ})
    """

    c = loads(src)
    for i in range(10):
        arr = [i for i in range(i)]
        assert c.foo(arr) == arr


def test_interface_call():
    src = """
interface Foo:
    def bar() -> uint256: payable
    def foobar() -> uint256: view

@external
def bar() -> uint256:
    return 1
    
@external
def foobar() -> uint256:
    return 2

@external
def foo() -> uint256:
    a: uint256 = 0
    i: Foo = Foo(self)
    a = extcall i.bar()
    a += staticcall i.foobar()
    return a
    """

    c = loads(src)
    assert c.foo() == 3


def test_interface_call2():
    src = """
interface Foo:
    def bar() -> uint256: payable
    def foobar() -> uint256: view
    
i: public(Foo)

@external
def bar() -> uint256:
    return 1
    
@external
def foobar() -> uint256:
    return 2

@external
def foo() -> uint256:
    a: uint256 = 0
    self.i = Foo(self)
    a = extcall self.i.bar()
    a += staticcall self.i.foobar()
    return a
    """

    c = loads(src)
    assert c.foo() == 3


def test_struct():
    src = """
struct S:
    a: uint256
    b: uint256

@external
def foo() -> uint256:
    s: S = S(a=1, b=2)
    return s.a
    """

    c = loads(src)
    assert c.foo() == 1


def test_struct2():
    src = """
struct S:
    a: uint256
    b: uint256
    
struct T:
    s: S
    c: uint256

@external
def foo() -> uint256:
    s: S = S(a=1, b=2)
    t: T = T(s=s, c=3)
    return t.s.a + t.s.b + t.c
    """

    c = loads(src)
    assert c.foo() == 6


def test_struct3():
    length = 3

    src = f"""
struct S:
    a: uint256
    b: uint256

d: DynArray[S, {length}]

@external
def foo() -> uint256:
    self.d = [S(a=0, b=1), S(a=2, b=3), S(a=4, b=5)]
        
    acc: uint256 = 0
    for s: S in self.d:
        acc += s.a + s.b
    return acc
    """

    c = loads(src)

    expected = 1 + 2 + 3 + 4 + 5

    assert c.foo() == expected


def test_tstorage_clearing():
    src = """
    
t: transient(uint256)

@external
def foo() -> uint256:
    self.t = 42
    return self.t
    
@external
def bar() -> uint256:
    return self.t
    """

    c = loads(src)
    assert c.foo() == 42
    assert c.bar() == 0
    assert c.foo() == 42


def test_tstorage_clearing2():
    src = """
struct S:
    a: uint256 

a: transient(uint256)
b: transient(uint256)
c: transient(DynArray[uint256, 10])
d: transient(S)
e: transient(Bytes[10])
f: transient(String[10])

@external
def foo():
    assert self.a == 0
    assert self.b == 0
    assert len(self.c) == 0
    assert self.d.a == 0
    assert len(self.e) == 0
    assert len(self.f) == 0

@external
def bar():
    self.a = 1
    self.b = 1
    self.c = [1, 2, 3]
    self.d.a = 1
    self.e = b"hello"
    self.f = "hello"
    """

    c = loads(src)
    c.foo()
    c.bar()
    c.foo()


def test_storage_rollback():
    src = """
c: public(uint256)
    
@external
def bar(u: uint256) -> uint256:
    self.c = 10
    assert False
    return 66

@external
def foo() -> bool:
    b: Bytes[32] = b""
    s: bool = False
    u: uint256 = 0
    s, b = raw_call(self, abi_encode(u, method_id=method_id("bar(uint256)")), max_outsize=32, revert_on_failure=False)
    return s
    """

    c = loads(src)
    assert c.foo() == False
    assert c.c() == 0


def test_storage_rollback2():
    src = """
c: public(uint256)

@external
def foobar(u: uint256) -> uint256:
    self.c = 10
    return 66

@external
def bar(u: uint256) -> uint256:
    b: Bytes[32] = b""
    s: bool = False
    s, b = raw_call(self, abi_encode(self.c, method_id=method_id("foobar(uint256)")), max_outsize=32, revert_on_failure=False)
    assert False
    return 66

@external
def foo() -> bool:
    b: Bytes[32] = b""
    s: bool = False
    u: uint256 = 0
    s, b = raw_call(self, abi_encode(u, method_id=method_id("bar(uint256)")), max_outsize=32, revert_on_failure=False)
    return s
    """

    c = loads(src)
    assert c.foo() == False
    assert c.c() == 0


def test_storage_rollback3():
    src = """
c: public(uint256)

@external
def foobar(u: uint256) -> uint256:
    self.c = 10
    return 66

@external
def bar(u: uint256) -> uint256:
    self.c = 20
    b: Bytes[32] = b""
    s: bool = False
    s, b = raw_call(self, abi_encode(self.c, method_id=method_id("foobar(uint256)")), max_outsize=32, revert_on_failure=False)
    assert False
    return 66

@external
def foo() -> bool:
    b: Bytes[32] = b""
    s: bool = False
    u: uint256 = 0
    s, b = raw_call(self, abi_encode(u, method_id=method_id("bar(uint256)")), max_outsize=32, revert_on_failure=False)
    return s
    """

    c = loads(src)
    assert c.foo() == False
    assert c.c() == 20


def test_hash_map():
    src = f"""

var: public(HashMap[uint256, uint256])

@external
def foo() -> uint256:
    self.var[0] = 42
    return self.var[0] + self.var[1] 
        """

    c = loads(src)
    assert c.foo() == 42
    assert c.var(0) == 42
    assert c.var(1) == 0


@pytest.mark.parametrize(
    "public,typ,value",
    [
        (True, "uint256", 42),
        (False, "uint256", 42),
        (True, "DynArray[uint256, 10]", [1, 2, 3]),
        (False, "DynArray[uint256, 10]", [1, 2, 3]),
        (True, "String[10]", "hello"),
        (False, "String[10]", "hello"),
        (True, "Bytes[10]", b"hello"),
        (False, "Bytes[10]", b"hello"),
    ],
)
def test_public_var_getter(public, typ, value):
    src = f"""
    var: {"public(" + typ + ")" if public else typ}

    @external
    def foo():
        self.var = {repr(value)}
        """

    c = loads(src)
    c.foo()

    if public:
        if isinstance(value, list):
            for i, v in enumerate(value):
                assert c.var(i) == v
        else:
            assert c.var() == value
    else:
        with pytest.raises(AttributeError):
            c.var()


def test_encode_address():
    src = """
    @external
    def foo() -> address:
        return self
        """

    c = loads(src)
    assert c.foo() == c.address


def test_init(get_contract):
    src = """
d: public(uint256)

@deploy
def __init__(a: uint256):
    self.d = a 
    """

    c = get_contract(src, 42)
    assert c.d() == 42


def test_init2(get_contract):
    src = """
d: public(uint256)

@deploy
def __init__(a: uint256):
    self.bar()
   
   
def bar():
    self.d = self.foo() 
    
def foo() -> uint256:
    return 42
    """

    c = get_contract(src, 42)
    assert c.d() == 42


def test_init3(get_contract):
    src = """
d: public(uint256)

@deploy
def __init__():
    assert self.is_contract == False
    """

    _ = get_contract(src)


def test_init4(get_contract):
    src = """
interface C:
    def foo(a: uint256): nonpayable

@deploy
def __init__(callback: address, a: uint256):
    extcall C(callback).foo(a) 
    """

    callback = """
d: public(uint256)

@external
def foo(a: uint256):
    self.d = a
"""

    callback = get_contract(callback)
    _ = get_contract(src, callback, 42)
    assert callback.d() == 42

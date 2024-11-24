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


def test_storage_array_assign():
    src = """
a: DynArray[uint256, 10]

@external
def foo() -> uint256:
    self.a = [1, 2]
    self.a[0] = 3
    return self.a[0] + self.a[1]
    """

    c = loads(src)
    assert c.foo() == 5


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
    assert not c.foo()


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


def test_struct4():
    src = """
struct S:
    a: uint256
    b: uint256

s: S

@external
def foo() -> uint256:
    self.s = S(a=1, b=2)
    self.s.a = 3
    return self.s.a + self.s.b
    """

    c = loads(src)

    assert c.foo() == 3 + 2


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


def test_abi_encode_struct(get_contract):
    src = """
struct C:
    a: uint256 

c: public(C)

@external
def foo() -> C:
    return self.c 
    """

    c = get_contract(src)
    assert c.foo() == (0,)


def test_hash_map():
    src = """

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


def test_library_storage(get_contract, make_input_bundle):
    src = """
import lib1

initializes: lib1

exports: lib1.d

@external
def foo():
    lib1.bar()
    """

    lib1 = """
d: public(uint256)

def bar():
    self.d = 1
"""

    input_bundle = make_input_bundle({"lib1.vy": lib1})

    c = get_contract(src, input_bundle=input_bundle)
    c.foo()
    assert c.d() == 1


def test_library_storage2(get_contract, make_input_bundle):
    src = """
import lib1

initializes: lib1

exports: lib1.d

e: public(uint256)

@external
def foo():
    self.e = 1
    lib1.bar()
    """

    lib1 = """
d: public(uint256)
e: uint256

def bar():
    self.d = 1
    self.e = 2
"""

    input_bundle = make_input_bundle({"lib1.vy": lib1})

    c = get_contract(src, input_bundle=input_bundle)
    c.foo()
    assert c.d() == 1


def test_library_storage3(get_contract, make_input_bundle):
    src = """
import lib1

initializes: lib1

exports: lib1.d

@external
def foo():
    lib1.bar()
    lib1.d = 2
    """

    lib1 = """
d: public(uint256)

def bar():
    self.d = 1
"""

    input_bundle = make_input_bundle({"lib1.vy": lib1})

    c = get_contract(src, input_bundle=input_bundle)
    c.foo()
    assert c.d() == 2


def test_library_storage4(get_contract, make_input_bundle):
    src = """
import lib1

initializes: lib1

exports: lib1.d
exports: lib1.lib2.d2

@external
def foo():
    lib1.bar()
    lib1.d = 4
    lib1.lib2.d2 = 5
    """

    lib1 = """
import lib2

initializes: lib2

d: public(uint256)

def bar():
    self.d = 1
"""
    lib2 = """
d2: public(uint256)

def bar():
    self.d2 = 1
    """

    input_bundle = make_input_bundle({"lib1.vy": lib1, "lib2.vy": lib2})

    c = get_contract(src, input_bundle=input_bundle)
    c.foo()
    assert c.d() == 4
    assert c.d2() == 5


def test_library_storage5(get_contract, make_input_bundle):
    src = """
import lib1

initializes: lib1

d: public(uint256)

@external
def foo():
    self.d = 4
    lib1.bar()
    """

    lib1 = """
d: public(uint256)

def bar():
    self.d = 1
"""

    input_bundle = make_input_bundle({"lib1.vy": lib1})

    c = get_contract(src, input_bundle=input_bundle)
    c.foo()
    assert c.d() == 4


def test_library_storage6(get_contract, make_input_bundle):
    src = """
import lib1

initializes: lib1
exports: lib1.d

@external
def foo():
    lib1.bar()
    lib1.d.s.a[0] = 0
    lib1.d.b[2] = 7
    """

    lib1 = """
struct S:
    a: DynArray[uint256, 10]
    
struct S2:
    s: S
    b: DynArray[uint256, 10]
    
d: public(S2)

def bar():
    self.d = S2(s=S(a=[1, 2, 3]), b=[4, 5, 6])
"""

    input_bundle = make_input_bundle({"lib1.vy": lib1})

    c = get_contract(src, input_bundle=input_bundle)
    c.foo()

    assert c.d() == (([0, 2, 3],), [4, 5, 7])


def test_library_storage7(get_contract, make_input_bundle):
    src = """
import lib1

initializes: lib1

exports: lib1.lib2.d

@external
def foo():
    lib1.bar()
    lib1.lib2.d.s.a[2] = 66
    lib1.lib2.d.b[1] = 77
    """

    lib1 = """
import lib2

initializes: lib2

d: public(uint256)

def bar():
    lib2.d = lib2.S2(s=lib2.S(a=[1, 2, 3]), b=[4, 5, 6])

"""
    lib2 = """
struct S:
    a: DynArray[uint256, 10]
    
struct S2:
    s: S
    b: DynArray[uint256, 10]
    
d: public(S2)
    """

    input_bundle = make_input_bundle({"lib1.vy": lib1, "lib2.vy": lib2})

    c = get_contract(src, input_bundle=input_bundle)
    c.foo()
    assert c.d() == (([1, 2, 66],), [4, 77, 6])


def test_module_struct_attribute(get_contract):
    src = """
struct S:
    a: uint256
    
a: public(DynArray[S, 10])

@external
def foo():
    self.a = [S(a=1), S(a=2)]
    self.a[0] = S(a=3)
    self.a[1].a = 4
    """

    c = get_contract(src)
    c.foo()
    assert c.a(0) == (3,)
    assert c.a(1) == (4,)


def test_darray_append(get_contract):
    src = """
a: public(DynArray[uint256, 10])

@external
def foo() -> uint256:
    self.a = []
    self.a.append(1)
    return self.a[0]
    """

    c = get_contract(src)
    assert c.foo() == 1


def test_darray_append2(get_contract):
    src = """
a: public(DynArray[uint256, 1])

@external
def foo() -> uint256:
    self.a = []
    self.a.append(1)
    self.a.append(1)
    return self.a[0]
    """

    c = get_contract(src)
    with pytest.raises(ValueError) as e:
        c.foo()

    assert "Cannot exceed maximum length 1" in str(e.value)


def test_darray_append3(get_contract):
    src = """
a: public(DynArray[DynArray[uint256, 1], 2])

@external
def foo() -> DynArray[uint256, 10]:
    self.a = []
    self.a.append([1])
    self.a.append([2])
    return self.a[0]
    """

    c = get_contract(src)
    assert c.foo() == [1]


def test_darray_pop(get_contract):
    src = """
a: public(DynArray[DynArray[uint256, 1], 2])

def bar() -> DynArray[uint256, 1]:
    return []

@external
def foo() -> uint256:
    self.a = []
    self.a.append([1]) 
    u: uint256 = self.a.pop()[0]
    return u
    """

    c = get_contract(src)
    assert c.foo() == 1


def test_darray_pop2(get_contract):
    src = """
a: public(DynArray[DynArray[uint256, 1], 2])

def bar() -> DynArray[uint256, 1]:
    return []

@external
def foo() -> uint256:
    self.a = []
    self.a.append([1]) 
    u: DynArray[uint256, 10] = self.a.pop()
    return u.pop()
    """

    c = get_contract(src)
    assert c.foo() == 1


def test_darray_pop3(get_contract):
    src = """
a: public(DynArray[DynArray[uint256, 1], 2])

def bar() -> DynArray[uint256, 1]:
    return []

@external
def foo() -> uint256:
    self.a = []
    self.a.append([1]) 
    self.a.append([2]) 
    u: DynArray[uint256, 10] = self.a.pop()
    return u.pop() + self.a.pop()[0]
    """

    c = get_contract(src)
    assert c.foo() == 3


def test_pass_by_value(get_contract):
    src = """
a: public(DynArray[uint256, 2])

def bar(d: DynArray[uint256, 2]):
    d[0] = 0

@external
def foo() -> DynArray[uint256, 2]:
    self.a = [1, 1]
    self.bar(self.a)
    return self.a
    """

    c = get_contract(src)
    assert c.foo() == [1, 1]


def test_pass_by_value2(get_contract):
    src = """
def bar(d: DynArray[uint256, 2]):
    d[0] = 0

@external
def foo() -> DynArray[uint256, 2]:
    d: DynArray[uint256, 2] = [1, 1]
    self.bar(d)
    return d
    """

    c = get_contract(src)
    assert c.foo() == [1, 1]


def test_pass_by_value3(get_contract):
    src = """
@external
def foo() -> DynArray[uint256, 2]:
    d: DynArray[uint256, 2] = [1, 1]
    d2: DynArray[uint256, 2] = d
    d2[0] = 0
    return d
    """

    c = get_contract(src)
    assert c.foo() == [1, 1]


def test_pass_by_value4(get_contract):
    src = """
@external
def foo() -> DynArray[uint256, 2]:
    d: DynArray[uint256, 2] = [1, 1]
    d2: DynArray[uint256, 2] = d
    d2[0] += 1
    return d
    """

    c = get_contract(src)
    assert c.foo() == [1, 1]


def test_pass_by_value5(get_contract):
    src = """
a: public(DynArray[uint256, 2])

def bar() -> DynArray[uint256, 2]:
    return self.a

@external
def foo() -> DynArray[uint256, 2]:
    self.a = [1, 1]
    d: DynArray[uint256, 2] = self.bar()
    d[0] = 0
    return self.a
    """

    c = get_contract(src)
    assert c.foo() == [1, 1]


def test_pass_by_value6(get_contract):
    src = """
struct Foo:
    a: uint256
    b: DynArray[uint256, 2] 
    
f: Foo

def bar() -> Foo:
    return self.f

@external
def foo() -> Foo:
    self.f = Foo(a=1, b=[1, 1])
    d: Foo = self.bar()
    d.a = 0
    d.b[0] = 0
    return self.f
    """

    c = get_contract(src)
    assert c.foo() == (1, [1, 1])


def test_pass_by_value7(get_contract):
    src = """
h: HashMap[uint256, DynArray[uint256, 2]]

@external
def foo() -> DynArray[uint256, 2]:
    self.h[0] = [1, 1]
    d: DynArray[uint256, 2] = self.h[0]
    d[0] = 0
    return self.h[0]
    """

    c = get_contract(src)
    assert c.foo() == [1, 1]


def test_pass_by_value8(get_contract):
    src = """
struct Foo:
    a: uint256
    b: DynArray[uint256, 2] 

@external
def foo() -> Foo:
    f: Foo = Foo(a=1, b=[1, 1])
    d: DynArray[uint256, 2] = f.b
    d[0] = 0
    return f
    """

    c = get_contract(src)
    assert c.foo() == (1, [1, 1])


def test_max_builtin(get_contract):
    src = """
@external
def foo() -> uint256:
    return max_value(uint256)
    """

    c = get_contract(src)
    _ = c.foo()


def test_return_constant(get_contract):
    src = """
u: constant(uint256) = 1
    
@external
def foo() -> uint256:
    return u
    """

    c = get_contract(src)
    assert c.foo() == 1


def test_create_minimal_proxy_state(get_contract):
    src = """
c: uint256

interface Self:
    def side_effect() -> uint256: nonpayable
    
@external
def side_effect() -> uint256:
    self.c += 1
    return self.c

@external
def foo() -> (uint256, uint256):
    proxy: address = create_minimal_proxy_to(self) 
    res: uint256 = extcall Self(proxy).side_effect() 
    return res, self.c 
    """

    c = get_contract(src)

    assert c.foo() == (1, 0)


def test_create_minimal_proxy_msg_sender(get_contract):
    src = """
interface Self:
    def return_sender() -> address: nonpayable

@external
def return_sender() -> address:
    return msg.sender

@external
def foo() -> address:
    proxy: address = create_minimal_proxy_to(self) 
    res: address = extcall Self(proxy).return_sender() 
    return res
    """

    c = get_contract(src)

    assert c.foo() == c.address


def test_create_minimal_proxy_extcall(get_contract):
    src = """
contract: immutable(address) 
c: uint256
    
interface Bar:
    def bar() -> uint256: nonpayable
    
interface Proxy:
    def call_3rd_contract() -> uint256: nonpayable
    
@deploy
def __init__(addr: address):
    contract = addr
    
@external
def call_3rd_contract() -> uint256:
    return extcall Bar(contract).bar()

@external
def foo() -> (uint256, uint256):
    proxy: address = create_minimal_proxy_to(self) 
    res: uint256 = extcall Proxy(proxy).call_3rd_contract() 
    return res, self.c
    """

    src2 = """
c: uint256 
    
@external
def bar() -> uint256:
    self.c = 66
    return self.c
    """

    c2 = get_contract(src2)
    c = get_contract(src, c2.address)

    assert c.foo() == (66, 0)


def test_create_minimal_proxy_raw_call(get_contract):
    src = """
c: uint256

interface Self:
    def side_effect() -> uint256: nonpayable

@external
def side_effect() -> uint256:
    self.c += 1
    return self.c

@external
def foo() -> (uint256, uint256):
    proxy: address = create_minimal_proxy_to(self)
    res: Bytes[32] = raw_call(proxy, method_id("side_effect()"), max_outsize=32)
    return abi_decode(res, uint256), self.c
    """

    c = get_contract(src)

    assert c.foo() == (1, 0)


def test_create_minimal_proxy_state_rollback(get_contract):
    src = """
c: public(uint256)

interface Self:
    def side_effect() -> uint256: nonpayable
    def c() -> uint256: view

@external
def side_effect(u: uint256) -> uint256:
    self.c += 1
    assert u == 1
    return self.c

@external
def foo() -> (uint256, uint256):
    proxy: address = create_minimal_proxy_to(self)
    b: Bytes[32] = b""
    s: bool = False
    u: uint256 = 0
    s, b = raw_call(proxy, abi_encode(u, method_id=method_id("side_effect(uint256)")), max_outsize=32, revert_on_failure=False)
    assert s == False
    return staticcall Self(proxy).c(), self.c
    """

    c = get_contract(src)

    assert c.foo() == (0, 0)


def test_create_copy_of(get_contract):
    src = """
a: public(uint256)
    
interface Foo:
    def a() -> uint256: view
    
@external
def foo() -> (uint256, uint256):
    self.a = 18 
    copy: address = create_copy_of(self)
    return staticcall Foo(copy).a(), self.a
    """

    c = get_contract(src)

    assert c.foo() == (0, 18)


def test_create_copy_of_with_constant(get_contract):
    src = """
a: public(constant(uint256)) = 144

interface Foo:
    def a() -> uint256: view

@external
def foo() -> uint256:
    copy: address = create_copy_of(self)
    return staticcall Foo(copy).a()
    """

    c = get_contract(src)

    assert c.foo() == 144


def test_create_copy_of_with_immutable(get_contract):
    src = """
a: public(immutable(uint256))

@deploy
def __init__():
    a = 144

interface Foo:
    def a() -> uint256: view

@external
def foo() -> uint256:
    copy: address = create_copy_of(self)
    return staticcall Foo(copy).a()
    """

    c = get_contract(src)

    assert c.foo() == 144


def test_create_copy_of_3rd_contract(get_contract):
    src = """
interface Foo:
    def a() -> uint256: view

@external
def foo(target: address) -> uint256:
    copy: address = create_copy_of(target)
    return staticcall Foo(copy).a()
    """

    src2 = """
@external
def a() -> uint256:
    return 144 
    """

    target = get_contract(src2)
    c = get_contract(src)

    assert c.foo(target) == 144


def test_create_copy_of_3rd_contract_state_not_coppied(get_contract):
    src = """
interface Foo:
    def a() -> uint256: view

@external
def foo(target: address) -> uint256:
    copy: address = create_copy_of(target)
    return staticcall Foo(copy).a()
    """

    src2 = """
u: uint256
str: String[32] 
arr: DynArray[uint256, 10]
    
@deploy 
def __init__():
    self.u = 144
    self.str = "wise man"
    self.arr = [1, 2, 3] 
    
@external
def a() -> uint256:
    return self.u + len(self.str) + len(self.arr) 
    """

    target = get_contract(src2)
    c = get_contract(src)

    assert c.foo(target) == 0

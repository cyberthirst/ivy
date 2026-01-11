import pytest

from ivy.frontend.loader import loads
from ivy.exceptions import StaticCallViolation, Assert, Raise, Revert
from vyper.utils import method_id


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
        assert c.foo(transact=True) == 2


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
    b: Bytes[32] = raw_call(self, abi_encode(b'', method_id=method_id("bar()")), max_outsize=32)
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


def test_log_in_static_context():
    """LOG operations should fail in static context per EVM spec."""
    src = """
event MyEvent:
    value: uint256

@external
def emit_event(val: uint256) -> uint256:
    log MyEvent(value=val)
    return val
    """

    src2 = """
interface Emitter:
    def emit_event(val: uint256) -> uint256: view

@external
def foo(target: address) -> uint256:
    return staticcall Emitter(target).emit_event(42)
    """

    emitter = loads(src)
    caller = loads(src2)
    with pytest.raises(StaticCallViolation):
        caller.foo(emitter)


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
    assert c.foo(transact=True) == 42
    assert c.bar(transact=True) == 0
    assert c.foo(transact=True) == 42


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
    c.foo(transact=True)
    c.bar(transact=True)
    c.foo(transact=True)


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


def test_log(get_contract):
    src = """
event Foo:
    a: uint256
    
@external
def foo():
    log Foo(1)
    """

    c = get_contract(src)

    c.foo()


def test_encode_static_array(get_contract):
    src = """

a: uint256[3]

@external
def foo() -> uint256[3]:
    self.a[0] = 1
    self.a[1] = 2
    return self.a
    """

    c = get_contract(src)

    res = c.foo()

    assert res == [1, 2, 0]


def test_storage_dump(get_contract):
    src = """
i: uint256
k: String[32]
j: Bytes[10]
s: uint256[3]
d: DynArray[DynArray[uint256, 3], 3]
h: HashMap[uint256, String[32]]
bm: bytes2
bytesm_list: bytes1[1]

struct S:
    a: uint256
    b: DynArray[uint256, 3]
    
v: S

@external
def foo():
    self.i = 1
    self.k = "hello"
    self.j = b"hello"
    self.s[0] = 1
    self.s[1] = 2
    self.d = [[1], [2, 3, 4], [5, 6]]
    for i: uint8 in range(3):
        self.h[convert(i, uint256)] = uint2str(i)
    self.v = S(a=1, b=self.d[1])
    """

    c = get_contract(src)
    c.foo()
    dump = c.storage_dump()
    print(dump)
    assert dump["i"] == 1
    assert dump["k"] == "hello"
    assert dump["j"] == b"hello"
    assert dump["s"] == [1, 2, 0]
    assert dump["d"] == [[1], [2, 3, 4], [5, 6]]
    assert dump["h"] == {0: "0", 1: "1", 2: "2"}
    assert dump["bm"] == b"\x00\x00"
    assert dump["bytesm_list"] == [b"\x00"]


def test_elif_condition(get_contract):
    src = """
@external
def foo(a: uint256) -> uint256:
    if a == 10:
        return 1
    elif a == 11:
        return 2
    elif a == 12:
        return 3
    else:
        return 4
    """

    c = get_contract(src)

    assert c.foo(10) == 1
    assert c.foo(11) == 2
    assert c.foo(12) == 3
    assert c.foo(66) == 4


def test_assert_passes(get_contract):
    src = """
@external
def foo():
    assert True
    """

    c = get_contract(src)
    c.foo()


def test_assert_passes2(get_contract):
    src = """
@external
def foo(a: uint256):
    assert True and a > 10
    assert a == 25 or (True and a == 29)
    """

    c = get_contract(src)
    c.foo(29)
    c.foo(25)


def test_assert_fails(get_contract):
    src = """
@external
def foo(a: uint256):
    assert True and a > 10
    assert a == 25 or (True and a == 29)
    """

    c = get_contract(src)
    for i in [11, 26, 30]:
        with pytest.raises(Assert):
            c.foo(i)


def test_assert_fails_with_message(get_contract):
    src = """
@external
def foo(a: uint256):
    assert True and a > 10
    assert a == 25 or (True and a == 29), "assertion failed"
    """

    c = get_contract(src)
    for i in [11, 26, 30]:
        with pytest.raises(Assert) as e:
            c.foo(i)
        assert str(e.value) == "assertion failed"


def test_raise_raises(get_contract):
    src = """
@external
def foo():
    raise "you shall not pass"
    """

    c = get_contract(src)
    with pytest.raises(Raise) as e:
        c.foo()
    assert str(e.value) == "you shall not pass"


def test_raw_revert(get_contract):
    reverty_code = """
@external
def foo():
    data: Bytes[4] = method_id("NoFives()")
    raw_revert(data)
    """

    revert_bytes = method_id("NoFives()")

    with pytest.raises(Revert) as e:
        get_contract(reverty_code).foo()

    assert e.value.data == revert_bytes


# raw_call into an account without code shouldn't revert
# as there's no code to revert
def test_raw_call_into_acc_without_code(get_contract):
    src = """
@external
def foo() -> bool:
    success: bool = False
    response: Bytes[32] = empty(Bytes[32])
    success, response = raw_call(
    empty(address),
    method_id("foo()"),
    max_outsize=32,
    revert_on_failure=False
    )
    return success
    """

    c = get_contract(src)
    assert c.foo() == True


def test_raw_call_into_acc_without_code2(get_contract):
    src = """
@external
def foo() -> Bytes[32]:
    response: Bytes[32] = raw_call(
        empty(address),
        method_id("foo()"),
        max_outsize=32
    )
    return response
    """

    c = get_contract(src)
    assert c.foo() == b""


# don't use skip_contract_check and thus force revert
def test_extcall_into_acc_without_code(get_contract):
    src = """
interface Foo:
    def foo(): nonpayable
    
@external
def foo() -> bool:
    zero_addess: address = empty(address)
    extcall Foo(zero_addess).foo()
    return True
    """

    c = get_contract(src)

    with pytest.raises(Revert) as e:
        c.foo()

    assert str(e.value) == f"Account at {'0x' + 20 * '00'} does not have code"


# use skip_contract_check and thus don't force
def test_extcall_into_acc_without_code2(get_contract):
    src = """
interface Foo:
    def foo(): nonpayable

@external
def foo() -> bool:
    zero_addess: address = empty(address)
    extcall Foo(zero_addess).foo(skip_contract_check=True)
    return True
    """

    c = get_contract(src)

    assert c.foo() == True


# use skip_contract_check with a random address
def test_extcall_into_random_acc_without_code(get_contract):
    src = """
interface Foo:
    def foo(): nonpayable

@external
def foo() -> bool:
    random_addr: address = 0xDeaDbeefdEAdbeefdEadbEEFdeadbeEFdEaDbeeF
    extcall Foo(random_addr).foo(skip_contract_check=True)
    return True
    """

    c = get_contract(src)

    assert c.foo() == True


IDENTITY_ADDRESS = "0x" + 19 * "00" + "04"


def test_identity_precompile(get_contract):
    src = f"""
@external
def foo(input: Bytes[32]) -> Bytes[32]:
    output: Bytes[32] = raw_call(
        {IDENTITY_ADDRESS},
        input,
        max_outsize=32
    )
    return output
    """

    c = get_contract(src)

    input_data = b"Hello, World!"
    assert c.foo(input_data) == b"Hello, World!"


def test_identity_precompile2(get_contract):
    src = f"""
@external
def foo(input: Bytes[64]) -> Bytes[32]:
    output: Bytes[32] = raw_call(
        {IDENTITY_ADDRESS},
        input,
        max_outsize=32
    )
    return output
    """

    c = get_contract(src)

    input_data = (64 * "a").encode("utf8")
    assert c.foo(input_data) == input_data[:32]


@pytest.mark.xfail(reason="Convert failing")
def test_blah(get_contract):
    src = """
@external
def foo(a: uint256) -> uint256:
    b: Bytes[32] = b''
    return convert(b, uint256)
    """

    c = get_contract(src)
    assert c.foo(0) == 0


def test_raw_call_with_revert_on_failure(get_contract):
    src = """
    
x_BOOL_0: public(bool)
C_INT_0: constant(uint8) = 0

@external
def func_1():
    self.x_BOOL_0 = raw_call(0x0000000000000000000000000000000000000000, b"0", revert_on_failure=False)
    """

    c = get_contract(src)
    c.func_1()
    assert c.x_BOOL_0() == True


def test_convert(get_contract):
    src = """
@external
def foo() -> Bytes[32]:
    s: String[32] = ""
    return convert(s, Bytes[32])
"""

    c = get_contract(src)
    assert c.foo() == b""


def tst_convert_bytes_to_adddress(get_contract):
    src = """
   @external
   def foo(b: Bytes[32]) -> address:
       return convert(b, address)
   """

    c = get_contract(src)
    i = b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01"
    assert c.foo(i) == "0x0000000000000000000000000000000000000001"


def test_unsafe_add(get_contract):
    src = """
    @external
    @view
    def foo(x: uint8, y: uint8) -> uint8:
        return unsafe_add(x, y)
        
    @external
    @view
    def bar(x: int8, y: int8) -> int8:
        return unsafe_add(x, y)
    """
    c = get_contract(src)

    assert c.foo(1, 1) == 2

    assert c.foo(255, 255) == 254

    assert c.bar(127, 127) == -2


def test_unsafe_sub(get_contract):
    src = """
    @external
    @view
    def foo(x: uint8, y: uint8) -> uint8:
        return unsafe_sub(x, y)

    @external
    @view
    def bar(x: int8, y: int8) -> int8:
        return unsafe_sub(x, y)
    """
    c = get_contract(src)

    assert c.foo(4, 3) == 1

    assert c.foo(0, 1) == 255

    assert c.bar(-128, 1) == 127


def test_unsafe_mul(get_contract):
    src = """
    @external
    @view
    def foo(x: uint8, y: uint8) -> uint8:
        return unsafe_mul(x, y)

    @external
    @view
    def bar(x: int8, y: int8) -> int8:
        return unsafe_mul(x, y)
    """
    c = get_contract(src)

    assert c.foo(1, 1) == 1

    assert c.foo(255, 255) == 1

    assert c.bar(-128, -128) == 0

    assert c.bar(127, -128) == -128


def test_unsafe_div(get_contract):
    src = """
    @external
    @view
    def foo(x: uint8, y: uint8) -> uint8:
        return unsafe_div(x, y)

    @external
    @view
    def bar(x: int8, y: int8) -> int8:
        return unsafe_div(x, y)
    """
    c = get_contract(src)

    assert c.foo(1, 1) == 1

    assert c.foo(1, 0) == 0

    assert c.bar(-128, -1) == -128


def test_boolop():
    src = """
@external
def foo(a: bool, b: bool, c: bool) -> bool:
	return a or b and c
    """

    c = loads(src)
    assert c.foo(False, True, True) == True


@pytest.mark.xfail(reason="vyper.exceptions.CompilerPanic: risky overlap")
def test_array_index_overlap(get_contract):
    code = """
struct Foo:
    d: DynArray[DynArray[uint256, 5], 5]
    
a: Foo
    
@external
def foo() -> uint256:
    self.a.d.append([1, 2, 3])
    return self.a.d[self.bar()][0]


@internal
def bar() -> uint256:
    self.a.d.pop()
    self.a.d.append([4, 5, 6])
    return 0
    """
    c = get_contract(code)
    assert c.foo() == 4


@pytest.mark.xfail(reason="vyper.exceptions.CompilerPanic: risky overlap")
def test_array_index_overlap2(get_contract, tx_failed):
    code = """
struct Foo:
    d: DynArray[DynArray[uint256, 5], 5]

a: Foo

@external
def foo() -> uint256:
    self.a.d.append([1, 2, 3])
    return self.a.d[self.bar()][0]


@internal
def bar() -> uint256:
    self.a = Foo(d=[[4]])
    return 0
    """
    c = get_contract(code)
    assert c.foo() == 4


# Nested DynArray pops & appends inside the index expression
@pytest.mark.xfail(reason="vyper.exceptions.CompilerPanic: risky overlap")
def test_refresh_nested_pop_append(get_contract):
    code = """
struct Child:
    v: DynArray[uint256, 5]
struct Parent:
    c: DynArray[Child, 5]

p: Parent

@internal
def bump() -> uint256:
    self.p.c.pop()
    self.p.c.append(Child(v=[7, 8, 9]))
    return 0              # <-- used as index

@external
def foo() -> uint256:
    self.p.c.append(Child(v=[1, 2, 3]))
    return self.p.c[self.bump()].v[0]
    """
    c = get_contract(code)
    assert c.foo() == 7


# Mapping replaced inside the index expression
def test_refresh_mapping_replaced(get_contract):
    code = """
h: HashMap[uint256, DynArray[uint256, 5]]

@internal
def mutate() -> uint256:
    self.h[0] = [5, 5, 5]
    return 0

@external
def foo() -> uint256:
    self.h[0] = [1, 2, 3]
    return self.h[self.mutate()][0]
    """
    c = get_contract(code)
    assert c.foo() == 5


# Module‑level DynArray changed by a library function used in slice
@pytest.mark.xfail(reason="vyper.exceptions.CompilerPanic: risky overlap")
def test_refresh_module_array(get_contract, make_input_bundle):
    lib1 = """
d: public(DynArray[uint256, 5])

interface Foo:
    def touch () -> uint256: nonpayable

@external
def touch() -> uint256:
    self.d = [4]
    return 0
"""
    main = """
import lib1
initializes: lib1

exports: lib1.touch

@external
def foo() -> uint256:
    lib1.d = [1]
    return lib1.d[extcall lib1.Foo(self).touch()]
    """
    c = get_contract(main, input_bundle=make_input_bundle({"lib1.vy": lib1}))
    assert c.foo() == 4


# Entire struct replaced between container & slice evaluation
@pytest.mark.xfail(reason="vyper.exceptions.CompilerPanic: risky overlap")
def test_refresh_struct_replaced(get_contract):
    code = """
struct S:
    v: DynArray[uint256, 5]
s: S

def mutate() -> uint256:
    self.s = S(v=[42])
    return 0

@external
def foo() -> uint256:
    self.s = S(v=[1])
    return self.s.v[self.mutate()]
    """
    c = get_contract(code)
    assert c.foo() == 42


# Attribute‑subscript chain where parent struct mutates in slice
@pytest.mark.xfail(reason="vyper.exceptions.CompilerPanic: risky overlap")
def test_refresh_attr_then_subscript(get_contract):
    code = """
struct Inner:
    b: DynArray[uint256, 5]
struct Outer:
    a: Inner
o: Outer

@internal
def tweak() -> uint256:
    self.o = Outer(a=Inner(b=[9, 8, 7]))
    return 0

@external
def foo() -> uint256:
    self.o = Outer(a=Inner(b=[1, 2, 3]))
    return self.o.a.b[self.tweak()]
    """
    c = get_contract(code)
    assert c.foo() == 9


# Static array completely replaced before subscript
@pytest.mark.xfail(reason="vyper.exceptions.CompilerPanic: risky overlap")
def test_refresh_static_array_replaced(get_contract):
    code = """
a: uint256[3]

@internal
def poke() -> uint256:
    self.a = [9, 9, 9]
    return 1

@external
def foo() -> uint256:
    self.a = [1, 2, 3]
    return self.a[self.poke()]
    """
    c = get_contract(code)
    assert c.foo() == 9


# Module variable, base NOT evaluated in visitor but must refresh
@pytest.mark.xfail(reason="vyper.exceptions.CompilerPanic: risky overlap")
def test_refresh_module_base_lazy(get_contract, make_input_bundle):
    lib1 = """
arr: DynArray[uint256, 5]

interface Foo:
    def swap() -> uint256: nonpayable

@external
def swap() -> uint256:
    self.arr = [11]
    return 0
"""
    main = """
import lib1
initializes: lib1

exports: lib1.swap

@external
def foo() -> uint256:
    lib1.arr = [1]
    return lib1.arr[extcall lib1.Foo(self).swap()]
    """
    c = get_contract(main, input_bundle=make_input_bundle({"lib1.vy": lib1}))
    assert c.foo() == 11


# Deep library hierarchy, mutation in grand‑child module
@pytest.mark.xfail(reason="vyper.exceptions.CompilerPanic: risky overlap")
def test_refresh_grandchild_module(get_contract, make_input_bundle):
    lib2 = """
val: DynArray[uint256, 5]
"""
    lib1 = """
import lib2
initializes: lib2

def ping() -> uint256:
    lib2.val = [33]
    return 0
"""
    main = """
import lib1
import lib2

initializes: lib1
uses: lib2

@external
def foo() -> uint256:
    lib2.val = [1]
    return lib2.val[lib1.ping()]
    """
    input_bundle = make_input_bundle({"lib1.vy": lib1, "lib2.vy": lib2})
    c = get_contract(main, input_bundle=input_bundle)
    assert c.foo() == 33


# Pop‑then‑append on outer DynArray while reading inner DynArray
@pytest.mark.xfail(reason="vyper.exceptions.CompilerPanic: risky overlap")
def test_refresh_pop_append_outer(get_contract):
    code = """
d: DynArray[DynArray[uint256, 2], 2]

@internal
def juggle() -> uint256:
    self.d.pop()
    self.d.append([77, 88])
    return 0

@external
def foo() -> uint256:
    self.d = [[11, 22]]
    return self.d[self.juggle()][1]
    """
    c = get_contract(code)
    assert c.foo() == 88


# Mapping‑of‑struct with inner DynArray mutated in slice
@pytest.mark.xfail(reason="vyper.exceptions.CompilerPanic: risky overlap")
def test_refresh_mapping_struct_inner(get_contract):
    code = """
struct S:
    x: DynArray[uint256, 5]
m: HashMap[uint256, S]

@internal
def touch() -> uint256:
    self.m[0] = S(x=[55])
    return 0

@external
def foo() -> uint256:
    self.m[0] = S(x=[3])
    return self.m[0].x[self.touch()]
    """
    c = get_contract(code)
    assert c.foo() == 55


# ---------------------------------------------------------------------
# 12. Two‑level subscript, both indices have side effects
# ---------------------------------------------------------------------
@pytest.mark.xfail(reason="vyper.exceptions.CompilerPanic: risky overlap")
def test_refresh_double_index_side_effect(get_contract):
    code = """
nested: DynArray[DynArray[uint256, 2], 2]

@internal
def first() -> uint256:
    self.nested.pop()           # remove outer[1]
    self.nested.append([9, 9])  # push new outer[1]
    return 1

@internal
def second() -> uint256:
    self.nested[1][0] = 42
    return 0

@external
def foo() -> uint256:
    self.nested = [[1, 1], [2, 2]]
    return self.nested[self.first()][self.second()]
    """
    c = get_contract(code)
    assert c.foo() == 42


def test_modify_array_in_struct_constructor(get_contract):
    code = """

d: DynArray[uint256, 10]

struct Foo:
    a: DynArray[uint256, 10]
    b: uint256

@external
def foo() -> Foo:
    self.d = [1, 2]
    return Foo(a=self.d, b=self.d.pop())
    """
    c = get_contract(code)

    assert c.foo() == ([1, 2], 2)


def test_modify_array_in_struct_constructor2(get_contract):
    code = """

d: DynArray[uint256, 10]

struct Foo:
  a: DynArray[uint256, 10]
  b: uint256
  c: DynArray[uint256, 10]


@external
def foo() -> Foo:
    self.d = [1, 2]
    return Foo(
          a=self.d,
          b=self.d.pop(),
          c=self.d
      )

    """
    c = get_contract(code)

    assert c.foo() == ([1, 2], 2, [1])


def test_modify_array_in_func_call(get_contract):
    code = """

d: DynArray[uint256, 10]

def process(a: DynArray[uint256, 10], b: uint256, c: DynArray[uint256, 10]) -> uint256:
  return len(a) + b + len(c)

@external
def foo() -> uint256:
    self.d = [1, 2]
    return self.process(self.d, self.d.pop(), self.d)
    """
    c = get_contract(code)

    assert c.foo() == 5


def test_constant_assignment_to_other_constant(get_contract):
    val = [20769187434139310514121985316880385, 36028797018963969, 524287]
    typ = "uint152[3]"

    code = f"""
a: constant({typ}) = {val}

b: public(constant({typ})) = a

@external
def foo() -> {typ}:
    return b
    """

    c = get_contract(code)

    assert c.foo() == val


def test_internal_pure_accessing_flag(get_contract, make_input_bundle):
    """Test flag accesses in internal pure functions"""
    code = """
import lib1

@pure
def bar() -> lib1.Action:
    return lib1.Action.BUY

@pure
@external
def foo() -> lib1.Action:
    return self.bar()

    """
    lib1 = """
flag Action:
    BUY
    SELL
    """

    input_bundle = make_input_bundle({"lib1.vy": lib1})
    c = get_contract(code, input_bundle=input_bundle)
    assert c.foo() == 1  # BUY


def test_struct_storage_dump(get_contract):
    src = """
struct MyStruct:
    member1: uint256
    member2: DynArray[String[10], 5]

my_struct: MyStruct

@external
def setup():
    self.my_struct.member1 = 42
    self.my_struct.member2 = ["hello", "world", "test"]

@external
def get_struct() -> MyStruct:
    return self.my_struct
    """

    c = get_contract(src)
    c.setup()

    # Test that struct is returned correctly
    result = c.get_struct()
    assert result == (42, ["hello", "world", "test"])

    # Test storage dump returns struct as dict
    dump = c.storage_dump()
    assert "my_struct" in dump
    assert isinstance(dump["my_struct"], dict)
    assert dump["my_struct"]["member1"] == 42
    assert dump["my_struct"]["member2"] == ["hello", "world", "test"]


def test_tuple_indexing(get_contract):
    src = """
t: (uint256, uint256)

def bar() -> (uint256, uint256):
    return 1, 1

@external
def foo() -> uint256:
    self.t = self.bar()
    self.t[0] = 2
    return self.t[0]
    """
    c = get_contract(src)
    assert c.foo() == 2


def test_tuple_storage_copy_on_assign(get_contract):
    src = """
t: (uint256, uint256)
t2: (uint256, uint256)

@external
def foo() -> (uint256, uint256):
    self.t[0] = 10
    self.t[1] = 1
    self.t2 = self.t
    self.t[0] = 42
    return self.t2
    """
    c = get_contract(src)
    assert c.foo() == (10, 1)


def test_tuple_destructuring_from_storage(get_contract):
    src = """
t: (uint256, uint256)

@external
def foo() -> uint256:
    self.t[0] = 1
    self.t[1] = 2
    a: uint256 = 0
    b: uint256 = 0
    a, b = self.t
    return a + b
    """
    c = get_contract(src)
    assert c.foo() == 3


def test_tuple_destructuring_from_internal_return(get_contract):
    src = """
@internal
def bar() -> (uint256, uint256):
    return 5, 7

@external
def foo() -> uint256:
    a: uint256 = 0
    b: uint256 = 0
    a, b = self.bar()
    return a * b
    """
    c = get_contract(src)
    assert c.foo() == 35


@pytest.mark.xfail(reason="bad resolution order, we might need topsort")
def test_assigning_constants_to_each_other(get_contract):
    src = """
gen_var2: public(constant(address)) = gen_var1
gen_var1: constant(address) = 0x0000000000000000000000000000000000000003
    """
    c = get_contract(src)

    assert c.gen_var2() == "0x0000000000000000000000000000000000000003"


def test_convert_bytes_array_to_bytesm(get_contract):
    src = """
@external
def foo() -> bytes32:
    b: Bytes[32] = b''
    return convert(b, bytes32)

@external
def bar() -> bytes32:
    b: Bytes[32] = b'hello'
    return convert(b, bytes32)

@external
def baz() -> bytes4:
    b: Bytes[4] = b'ab'
    return convert(b, bytes4)

@external
def empty_bytes4() -> bytes4:
    b: Bytes[4] = b''
    return convert(b, bytes4)

@external
def empty_bytes8() -> bytes8:
    b: Bytes[8] = b''
    return convert(b, bytes8)

@external
def bytes8_partial() -> bytes8:
    b: Bytes[8] = b'abc'
    return convert(b, bytes8)

@external
def bytes16_full() -> bytes16:
    b: Bytes[16] = b'0123456789abcdef'
    return convert(b, bytes16)

@external
def bytes1_test() -> bytes1:
    b: Bytes[1] = b'x'
    return convert(b, bytes1)

@external
def bytes1_empty() -> bytes1:
    b: Bytes[1] = b''
    return convert(b, bytes1)
"""
    c = get_contract(src)
    assert c.foo() == b"\x00" * 32
    assert c.bar() == b"hello" + b"\x00" * 27
    assert c.baz() == b"ab\x00\x00"
    assert c.empty_bytes4() == b"\x00" * 4
    assert c.empty_bytes8() == b"\x00" * 8
    assert c.bytes8_partial() == b"abc" + b"\x00" * 5
    assert c.bytes16_full() == b"0123456789abcdef"
    assert c.bytes1_test() == b"x"
    assert c.bytes1_empty() == b"\x00"


def test_extract32_bytes4_valid(get_contract):
    src = """
@external
def foo(inp: Bytes[32]) -> bytes4:
    return extract32(inp, 0, output_type=bytes4)
"""
    c = get_contract(src)
    # Valid: 4 bytes of data + 28 zero bytes
    inp = b"\xde\xad\xbe\xef" + b"\x00" * 28
    assert c.foo(inp) == b"\xde\xad\xbe\xef"


def test_extract32_bytes16_valid(get_contract):
    src = """
@external
def foo(inp: Bytes[32]) -> bytes16:
    return extract32(inp, 0, output_type=bytes16)
"""
    c = get_contract(src)
    # Valid: 16 bytes of data + 16 zero bytes
    inp = b"0123456789abcdef" + b"\x00" * 16
    assert c.foo(inp) == b"0123456789abcdef"


def test_extract32_bytes4_invalid(get_contract):
    src = """
@external
def foo(inp: Bytes[32]) -> bytes4:
    return extract32(inp, 0, output_type=bytes4)
"""
    c = get_contract(src)
    # Invalid: 4 bytes of data + non-zero byte in tail
    inp = b"\xde\xad\xbe\xef" + b"\x01" + b"\x00" * 27
    with pytest.raises(Revert):
        c.foo(inp)


def test_extract32_bytes16_invalid(get_contract):
    src = """
@external
def foo(inp: Bytes[32]) -> bytes16:
    return extract32(inp, 0, output_type=bytes16)
"""
    c = get_contract(src)
    # Invalid: 16 bytes of data + non-zero bytes in tail
    inp = b"0123456789abcdef" + b"\xff" * 16
    with pytest.raises(Revert):
        c.foo(inp)


def test_deploy_with_value(get_contract, env):
    """Test that deploying a contract with value credits the contract's balance."""
    src = """
@deploy
@payable
def __init__():
    pass

@external
def get_balance() -> uint256:
    return self.balance
    """

    env.set_balance(env.deployer, 100)
    c = get_contract(src, value=10)

    # Contract should have received the 10 wei
    assert env.get_balance(c.address) == 10
    assert c.get_balance() == 10


def test_create_from_constructor_with_value(get_contract, env):
    """Test that creating a contract from within a constructor uses correct caller."""
    blueprint_src = """
@deploy
@payable
def __init__():
    pass

@external
def get_balance() -> uint256:
    return self.balance
    """

    deployer_src = """
child: public(address)

@deploy
@payable
def __init__(blueprint: address):
    # Create a child contract with value=5 from within constructor
    self.child = create_from_blueprint(blueprint, value=5)
    """

    # Deploy blueprint first
    blueprint = get_contract(blueprint_src)

    # Set deployer balance
    env.set_balance(env.deployer, 100)

    # Deploy the deployer contract with value=10
    # It should create a child with value=5, keeping 5 for itself
    deployer = get_contract(deployer_src, blueprint, value=10)

    # The deployer should have 5 (10 - 5 sent to child)
    assert env.get_balance(deployer.address) == 5

    # The child should have 5
    child_addr = deployer.child()
    assert env.get_balance(child_addr) == 5


def test_create_from_constructor_caller(get_contract):
    """Test that msg.sender is the deploying contract when creating from constructor."""
    blueprint_src = """
creator: public(address)

@deploy
def __init__():
    self.creator = msg.sender
    """

    deployer_src = """
interface Child:
    def creator() -> address: view

child: public(address)

@deploy
def __init__(blueprint: address):
    self.child = create_from_blueprint(blueprint)

@external
def get_child_creator() -> address:
    return staticcall Child(self.child).creator()
    """

    blueprint = get_contract(blueprint_src)
    deployer = get_contract(deployer_src, blueprint)

    # The child's creator (msg.sender during its constructor) should be
    # the deployer contract's address, not b"" or the EOA
    assert deployer.get_child_creator() == deployer.address


def test_tx_origin_is_sender(get_contract, env):
    """Test that tx.origin is set to the transaction sender, not the 'to' address."""
    src = """
@external
def get_origin() -> address:
    return tx.origin

@external
def get_sender() -> address:
    return msg.sender
    """

    c = get_contract(src)

    # tx.origin should be the deployer (EOA that initiated the transaction)
    # msg.sender should also be the deployer for a direct call
    assert c.get_origin() == env.deployer
    assert c.get_sender() == env.deployer


def test_tx_origin_through_subcall(get_contract, env):
    """Test that tx.origin remains the EOA even through contract-to-contract calls."""
    callee_src = """
@external
def get_origin() -> address:
    return tx.origin

@external
def get_sender() -> address:
    return msg.sender
    """

    caller_src = """
interface Callee:
    def get_origin() -> address: view
    def get_sender() -> address: view

@external
def call_get_origin(target: address) -> address:
    return staticcall Callee(target).get_origin()

@external
def call_get_sender(target: address) -> address:
    return staticcall Callee(target).get_sender()
    """

    callee = get_contract(callee_src)
    caller = get_contract(caller_src)

    # tx.origin should still be the deployer (original EOA)
    assert caller.call_get_origin(callee) == env.deployer

    # msg.sender should be the caller contract
    assert caller.call_get_sender(callee) == caller.address


def test_darray_append_rollback(get_contract):
    src = """
trace: DynArray[uint256, 3]

@external
def test(should_fail: bool):
    self.trace.append(1)
    assert not should_fail

@external
def get_trace() -> DynArray[uint256, 3]:
    return self.trace
    """

    c = get_contract(src)
    assert c.get_trace() == []

    with pytest.raises(Assert):
        c.test(True)

    assert c.get_trace() == []


def test_darray_append_rollback2(get_contract):
    src = """
trace: DynArray[uint256, 3]

@external
def test(should_fail: bool):
    self.trace.append(1)
    self.trace.append(2)
    self.trace.append(3)
    assert not should_fail

@external
def get_trace() -> DynArray[uint256, 3]:
    return self.trace
    """

    c = get_contract(src)
    assert c.get_trace() == []

    with pytest.raises(Assert):
        c.test(True)

    assert c.get_trace() == []


def test_darray_pop_rollback(get_contract):
    src = """
trace: DynArray[uint256, 3]

@deploy
def __init__():
    self.trace = [1, 2, 3]

@external
def test(should_fail: bool):
    self.trace.pop()
    assert not should_fail

@external
def get_trace() -> DynArray[uint256, 3]:
    return self.trace
    """

    c = get_contract(src)
    assert c.get_trace() == [1, 2, 3]

    with pytest.raises(Assert):
        c.test(True)

    assert c.get_trace() == [1, 2, 3]


def test_darray_pop_rollback2(get_contract):
    src = """
trace: DynArray[uint256, 3]

@deploy
def __init__():
    self.trace = [1, 2, 3]

@external
def test(should_fail: bool):
    self.trace.pop()
    self.trace.pop()
    self.trace.pop()
    assert not should_fail

@external
def get_trace() -> DynArray[uint256, 3]:
    return self.trace
    """

    c = get_contract(src)
    assert c.get_trace() == [1, 2, 3]

    with pytest.raises(Assert):
        c.test(True)

    assert c.get_trace() == [1, 2, 3]


def test_darray_mixed_rollback(get_contract):
    src = """
trace: DynArray[uint256, 5]

@deploy
def __init__():
    self.trace = [1, 2]

@external
def test(should_fail: bool):
    self.trace.append(3)
    self.trace.pop()
    self.trace.append(4)
    self.trace.append(5)
    assert not should_fail

@external
def get_trace() -> DynArray[uint256, 5]:
    return self.trace
    """

    c = get_contract(src)
    assert c.get_trace() == [1, 2]

    with pytest.raises(Assert):
        c.test(True)

    assert c.get_trace() == [1, 2]


def test_darray_append_rollback_overflow(get_contract):
    src = """
x: uint256
trace: DynArray[uint256, 3]

@external
def test():
    for i: uint256 in [self.usesideeffect(), self.usesideeffect(), self.usesideeffect()]:
        self.x += max_value(uint256)
        self.trace.append(i)

@view
def usesideeffect() -> uint256:
    return self.x

@external
def get_trace() -> DynArray[uint256, 3]:
    return self.trace

@external
def get_x() -> uint256:
    return self.x
    """

    c = get_contract(src)
    assert c.get_trace() == []
    assert c.get_x() == 0

    with pytest.raises(ValueError):
        c.test()

    assert c.get_trace() == []
    assert c.get_x() == 0


def test_darray_nested_call_rollback(get_contract):
    src = """
interface Self:
    def child_append(): nonpayable

trace: DynArray[uint256, 10]

@external
def test(should_fail: bool):
    self.trace.append(1)
    extcall Self(self).child_append()
    self.trace.append(4)
    assert not should_fail

@external
def child_append():
    self.trace.append(2)
    self.trace.append(3)

@external
def get_trace() -> DynArray[uint256, 10]:
    return self.trace
    """

    c = get_contract(src)
    assert c.get_trace() == []

    with pytest.raises(Assert):
        c.test(True)

    assert c.get_trace() == []


def test_darray_nested_call_child_reverts(get_contract):
    src = """
interface Self:
    def child_append(should_fail: bool): nonpayable

trace: DynArray[uint256, 10]

@external
def test():
    self.trace.append(1)
    success: bool = False
    response: Bytes[32] = b""
    success, response = raw_call(
        self,
        abi_encode(True, method_id=method_id("child_append(bool)")),
        max_outsize=32,
        revert_on_failure=False
    )
    self.trace.append(4)

@external
def child_append(should_fail: bool):
    self.trace.append(2)
    self.trace.append(3)
    assert not should_fail

@external
def get_trace() -> DynArray[uint256, 10]:
    return self.trace
    """

    c = get_contract(src)
    assert c.get_trace() == []

    c.test()

    assert c.get_trace() == [1, 4]


def test_darray_nested_call_pop_rollback(get_contract):
    src = """
interface Self:
    def child_pop(): nonpayable

trace: DynArray[uint256, 10]

@deploy
def __init__():
    self.trace = [1, 2, 3, 4, 5]

@external
def test(should_fail: bool):
    self.trace.pop()
    extcall Self(self).child_pop()
    self.trace.pop()
    assert not should_fail

@external
def child_pop():
    self.trace.pop()
    self.trace.pop()

@external
def get_trace() -> DynArray[uint256, 10]:
    return self.trace
    """

    c = get_contract(src)
    assert c.get_trace() == [1, 2, 3, 4, 5]

    with pytest.raises(Assert):
        c.test(True)

    assert c.get_trace() == [1, 2, 3, 4, 5]


def test_is_contract(get_contract, env):
    """Test is_contract behavior for various address types."""
    src = """
@external
def check_is_contract(addr: address) -> bool:
    return addr.is_contract

@external
def check_self_is_contract() -> bool:
    return self.is_contract
    """

    c = get_contract(src)

    # After deployment, self.is_contract should be True
    assert c.check_self_is_contract() is True

    # Deployed contract address should return True
    assert c.check_is_contract(c.address) is True

    # EOA address (the caller) should return False
    caller = env.eoa
    assert c.check_is_contract(caller) is False

    # Empty/unused address should return False
    empty_addr = "0x" + "00" * 20
    assert c.check_is_contract(empty_addr) is False

    # Arbitrary unused address should return False
    arbitrary = "0x" + "de" * 20
    assert c.check_is_contract(arbitrary) is False


def test_block_prevrandao(get_contract, env):
    src = """
@external
def get_prevrandao() -> bytes32:
    return block.prevrandao
    """

    c = get_contract(src)

    # Test with default value (all zeros)
    default_result = c.get_prevrandao()
    assert default_result == b"\x00" * 32

    # Test with a specific value
    test_value = b"\x42" + b"\x00" * 31
    env.prev_randao = test_value
    result = c.get_prevrandao()
    assert result == test_value

    # Test with another specific value
    test_value2 = bytes.fromhex(
        "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
    )
    env.prev_randao = test_value2
    result2 = c.get_prevrandao()
    assert result2 == test_value2


def test_block_difficulty(get_contract, env):
    src = """
@external
def get_difficulty() -> uint256:
    return block.difficulty
    """

    c = get_contract(src)

    # Reset to all zeros first
    env.prev_randao = b"\x00" * 32

    # Test with zero value
    default_result = c.get_difficulty()
    assert default_result == 0

    # Test with a specific value - bytes32 interpreted as big-endian uint256
    # 0x42 followed by 31 zeros = 0x4200...00 as big-endian = very large number
    test_value = b"\x42" + b"\x00" * 31
    env.prev_randao = test_value
    result = c.get_difficulty()
    expected = int.from_bytes(test_value, "big")
    assert result == expected

    # Test with a small value at the end - should result in small uint256
    test_value2 = b"\x00" * 31 + b"\x42"
    env.prev_randao = test_value2
    result2 = c.get_difficulty()
    assert result2 == 0x42


def test_block_prevhash(get_contract, env):
    """Test that block.prevhash returns the hash of the previous block."""
    src = """
@external
def get_prevhash() -> bytes32:
    return block.prevhash
    """
    # Set up block_hashes with a known value
    prev_hash = b"\xab" * 32  # 32-byte hash for the previous block
    env._env.block_hashes = [prev_hash]
    env._env.block_number = 1

    c = get_contract(src)
    result = c.get_prevhash()

    assert result == prev_hash


def test_block_prevhash_empty(get_contract, env):
    """Test that block.prevhash returns zero when block_hashes is empty."""
    src = """
@external
def get_prevhash() -> bytes32:
    return block.prevhash
    """
    # Empty block_hashes
    env._env.block_hashes = []
    env._env.block_number = 1

    c = get_contract(src)
    result = c.get_prevhash()

    assert result == b"\x00" * 32


def test_block_prevhash_block_zero(get_contract, env):
    """Test that block.prevhash returns zero when block_number is 0."""
    src = """
@external
def get_prevhash() -> bytes32:
    return block.prevhash
    """
    # Block 0 has no previous block
    env._env.block_hashes = [b"\xab" * 32]
    env._env.block_number = 0

    c = get_contract(src)
    result = c.get_prevhash()

    assert result == b"\x00" * 32


def test_call_depth_limit(get_contract):
    """Test that call depth limit (1024) is enforced per EVM spec.

    When call depth exceeds 1024, nested calls should fail gracefully
    (return failure status) without reverting the caller.
    """
    src = """
max_depth_reached: public(uint256)

@external
def recursive_call(depth: uint256) -> (bool, uint256):
    self.max_depth_reached = depth

    if depth >= 1030:  # Stop before hitting Python's recursion limit
        return True, depth

    # Try to call ourselves recursively using raw_call
    success: bool = False
    response: Bytes[64] = b""
    success, response = raw_call(
        self,
        abi_encode(depth + 1, method_id=method_id("recursive_call(uint256)")),
        max_outsize=64,
        revert_on_failure=False
    )

    if not success:
        # Call failed due to depth limit - this is expected at depth 1024
        return False, depth

    # Call succeeded, decode the result
    child_success: bool = False
    child_depth: uint256 = 0
    child_success, child_depth = abi_decode(response, (bool, uint256))
    return child_success, child_depth
    """

    c = get_contract(src)

    # Start the recursive call chain at depth 1 (top-level call has msg.depth=0)
    # The depth parameter tracks our iteration, while msg.depth is the actual EVM depth
    # When depth param = N, msg.depth = N-1
    success, final_depth = c.recursive_call(1)

    # The call at msg.depth 1024 (depth param 1025) should succeed and set max_depth_reached=1025
    # The raw_call FROM that depth trying to create msg.depth 1025 should fail (1025 > 1024)
    # So max_depth_reached = 1025 (last value set before the child call failed)
    assert c.max_depth_reached() == 1025
    assert success == False  # The chain stopped due to depth limit
    assert final_depth == 1025  # Last successfully reached depth (where the child call failed)


def test_call_depth_tracking(get_contract):
    """Test that call depth is correctly tracked through nested calls."""
    src = """
interface Self:
    def inner() -> uint256: nonpayable

depth_at_inner: public(uint256)

@external
def outer() -> uint256:
    # Call inner which will check its depth
    return extcall Self(self).inner()

@external
def inner() -> uint256:
    # Return some value indicating we reached here
    return 42
    """

    c = get_contract(src)

    # outer() at depth 0 calls inner() at depth 1
    result = c.outer()
    assert result == 42


def test_codesize(get_contract, env):
    """Test codesize returns the bytecode size for various address types."""
    src = """
@external
def get_codesize(addr: address) -> uint256:
    return addr.codesize

@external
def get_self_codesize() -> uint256:
    return self.codesize
    """

    c = get_contract(src)

    # Deployed contract should have non-zero codesize
    self_codesize = c.get_self_codesize()
    assert self_codesize > 0

    # Querying own address should match self.codesize
    assert c.get_codesize(c.address) == self_codesize

    # EOA address should have zero codesize
    eoa = env.eoa
    assert c.get_codesize(eoa) == 0

    # Empty/unused address should have zero codesize
    empty_addr = "0x" + "00" * 20
    assert c.get_codesize(empty_addr) == 0

    # Arbitrary unused address should have zero codesize
    arbitrary = "0x" + "de" * 20
    assert c.get_codesize(arbitrary) == 0


def test_codesize_another_contract(get_contract, env):
    """Test codesize works for querying another deployed contract."""
    src1 = """
@external
def foo() -> uint256:
    return 42
    """

    src2 = """
@external
def get_codesize(addr: address) -> uint256:
    return addr.codesize
    """

    c1 = get_contract(src1)
    c2 = get_contract(src2)

    # c2 should be able to get c1's codesize
    c1_codesize = c2.get_codesize(c1.address)
    assert c1_codesize > 0


def test_codehash(get_contract, env, keccak):
    """Test codehash returns the keccak256 hash of the bytecode."""
    src = """
@external
def get_codehash(addr: address) -> bytes32:
    return addr.codehash

@external
def get_self_codehash() -> bytes32:
    return self.codehash
    """

    c = get_contract(src)

    # Deployed contract should have non-empty codehash
    self_codehash = c.get_self_codehash()
    assert self_codehash != b"\x00" * 32

    # Querying own address should match self.codehash
    assert c.get_codehash(c.address) == self_codehash

    # EOA address should return keccak256 of empty bytes
    # (EOA is an account that exists - has been used - but has no code)
    eoa = env.eoa
    eoa_codehash = c.get_codehash(eoa)
    assert eoa_codehash == keccak(b"")

    # Non-existent account (never touched, no balance/nonce/code) returns 0
    # per EIP-1052
    empty_addr = "0x" + "00" * 20
    assert c.get_codehash(empty_addr) == b"\x00" * 32

    # Another arbitrary unused address should also return 0
    arbitrary = "0x" + "de" * 20
    assert c.get_codehash(arbitrary) == b"\x00" * 32


def test_codehash_another_contract(get_contract, env):
    """Test codehash works for querying another deployed contract."""
    src1 = """
@external
def foo() -> uint256:
    return 42
    """

    src2 = """
@external
def get_codehash(addr: address) -> bytes32:
    return addr.codehash
    """

    c1 = get_contract(src1)
    c2 = get_contract(src2)

    # c2 should be able to get c1's codehash
    c1_codehash = c2.get_codehash(c1.address)
    assert c1_codehash != b"\x00" * 32

    # Two different contracts should have different codehashes
    c2_self_codehash = c2.get_codehash(c2.address)
    assert c1_codehash != c2_self_codehash


def test_address_code(get_contract, env):
    """Test address.code returns the runtime bytecode (via slice)."""
    src = """
@external
def get_code_16(addr: address) -> Bytes[16]:
    return slice(addr.code, 0, 16)

@external
def get_self_code_16() -> Bytes[16]:
    return slice(self.code, 0, 16)

@external
def get_codesize(addr: address) -> uint256:
    return addr.codesize

@external
def get_self_codesize() -> uint256:
    return self.codesize
    """

    c = get_contract(src)

    # Deployed contract should have non-empty code
    self_code = c.get_self_code_16()
    assert len(self_code) == 16

    # Querying own address should return same code
    assert c.get_code_16(c.address) == self_code

    # EOA address has zero codesize (can't slice from it)
    eoa = env.eoa
    assert c.get_codesize(eoa) == 0


def test_address_code_slice(get_contract, env):
    """Test address.code via slice with specific lengths."""
    src = """
@external
def get_code_slice(addr: address) -> Bytes[32]:
    return slice(addr.code, 0, 32)

@external
def get_self_code_slice() -> Bytes[32]:
    return slice(self.code, 0, 32)
    """

    c = get_contract(src)

    # Deployed contract should have at least 32 bytes of code
    self_code = c.get_self_code_slice()
    assert len(self_code) == 32

    # Querying own address should match self.code
    assert c.get_code_slice(c.address) == self_code


def test_address_code_another_contract(get_contract, env):
    """Test address.code works for querying another deployed contract."""
    src1 = """
@external
def foo() -> uint256:
    return 42
    """

    src2 = """
@external
def get_code_slice(addr: address) -> Bytes[16]:
    return slice(addr.code, 0, 16)
    """

    c1 = get_contract(src1)
    c2 = get_contract(src2)

    # c2 should be able to get c1's code (first 16 bytes)
    c1_code = c2.get_code_slice(c1.address)
    assert len(c1_code) == 16

    # Two different contracts should have different code
    c2_self_code = c2.get_code_slice(c2.address)
    assert c1_code != c2_self_code


def test_code_codesize_consistency(get_contract, env):
    """Test that codesize works correctly."""
    src = """
@external
def get_codesize(addr: address) -> uint256:
    return addr.codesize

@external
def get_self_codesize() -> uint256:
    return self.codesize

@external
def get_code_slice(addr: address) -> Bytes[100]:
    return slice(addr.code, 0, 100)

@external
def get_self_code_slice() -> Bytes[100]:
    return slice(self.code, 0, 100)
    """

    c = get_contract(src)

    # Self codesize should be > 0 for deployed contract
    self_codesize = c.get_self_codesize()
    assert self_codesize > 0

    # Contract address codesize should match
    assert c.get_codesize(c.address) == self_codesize

    # EOA codesize should be 0
    eoa = env.eoa
    assert c.get_codesize(eoa) == 0


def test_codehash_consistency(get_contract, env, keccak):
    """Test that codehash returns correct values."""
    src = """
@external
def get_codehash(addr: address) -> bytes32:
    return addr.codehash

@external
def get_self_codehash() -> bytes32:
    return self.codehash
    """

    c = get_contract(src)

    # Self codehash should not be zero for deployed contract
    self_codehash = c.get_self_codehash()
    assert self_codehash != b"\x00" * 32

    # Contract address codehash should match
    assert c.get_codehash(c.address) == self_codehash

    # EOA codehash should be keccak of empty bytes (exists but no code)
    eoa = env.eoa
    eoa_codehash = c.get_codehash(eoa)
    assert eoa_codehash == keccak(b"")

    # Non-existent account should return 0 per EIP-1052
    empty_addr = "0x" + "ab" * 20
    assert c.get_codehash(empty_addr) == b"\x00" * 32


def test_static_call_with_value_raises_before_balance_check(get_contract, env):
    """Test that call with value in static context raises StaticCallViolation.

    Per EVM specs, the check for value != 0 in static context must happen
    BEFORE balance checks. This test verifies we get StaticCallViolation
    rather than 'Insufficient balance for transfer' even when balance is zero.

    Setup: ContractA calls ContractB via staticcall, ContractB then tries
    to call ContractC with value - this should raise StaticCallViolation.
    """
    # Contract C - just receives ether
    target_src = """
@external
@payable
def receive():
    pass
    """

    # Contract B - called via staticcall, tries to forward with value
    middle_src = """
@external
def forward_with_value(target: address) -> bool:
    # This runs in static context when called via staticcall
    # Trying to send value should raise StaticCallViolation
    raw_call(target, b"", value=1)
    return True
    """

    # Contract A - initiates the static call chain
    outer_src = """
interface Middle:
    def forward_with_value(target: address) -> bool: view

@external
def test_static_value(middle: address, target: address) -> bool:
    # staticcall to middle, which then tries raw_call with value
    return staticcall Middle(middle).forward_with_value(target)
    """

    target = get_contract(target_src)
    middle = get_contract(middle_src)
    outer = get_contract(outer_src)

    # Set the middle contract's balance to 0 to ensure we test
    # that static check happens BEFORE balance check
    env.set_balance(middle.address, 0)

    with pytest.raises(StaticCallViolation):
        outer.test_static_value(middle, target)

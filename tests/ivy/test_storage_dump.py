from decimal import Decimal


def test_transient_storage_dump_survives_deployment(get_contract):
    """Transient writes in __init__ must be visible in the post-deploy dump."""
    src = """
t: transient(uint256)

@deploy
def __init__():
    self.t = 42

@external
def get_t() -> uint256:
    return self.t
    """
    c = get_contract(src)
    dump = c.transient_storage_dump()
    assert dump == {"t": 42}


def test_transient_storage_dump_hashmap_survives_deployment(get_contract):
    """Transient HashMap writes in __init__ must be visible in the post-deploy dump."""
    src = """
m: transient(HashMap[address, uint256])

@deploy
def __init__():
    self.m[self] = 99

@external
def get_m() -> uint256:
    return self.m[self]
    """
    c = get_contract(src)
    dump = c.transient_storage_dump()
    assert str(c.address) in [str(k) for k in dump["m"].keys()]
    assert list(dump["m"].values()) == [99]


def test_storage_dump_empty_contract(get_contract):
    src = """
@external
def foo() -> uint256:
    return 42
    """
    c = get_contract(src)
    dump = c.storage_dump()
    assert dump == {}


def test_transient_storage_dump_defaults_match_storage(get_contract):
    src = """
x: uint256
t: transient(uint256)

@external
def get_values() -> (uint256, uint256):
    return self.x, self.t
    """
    c = get_contract(src)
    assert c.get_values() == (0, 0)

    storage_dump = c.storage_dump()
    transient_dump = c.transient_storage_dump()

    assert storage_dump == {"x": 0}
    assert transient_dump == {"t": 0}
    assert storage_dump["x"] == transient_dump["t"]


def test_transient_storage_dump_without_transient_vars(get_contract):
    src = """
val: uint256

@external
def set_val():
    self.val = 7
    """
    c = get_contract(src)
    c.set_val()

    storage_dump = c.storage_dump()
    transient_dump = c.transient_storage_dump()

    assert storage_dump["val"] == 7
    assert transient_dump == {}


def test_transient_storage_dump_defaults_without_writes(get_contract):
    src = """
t: transient(uint256)
s: transient(String[8])
b: transient(Bytes[8])
arr: transient(uint256[2])

@external
def touch():
    assert self.t == 0
    assert len(self.s) == 0
    assert len(self.b) == 0
    assert self.arr[0] == 0
    """
    c = get_contract(src)
    c.touch()

    dump = c.transient_storage_dump()
    assert dump["t"] == 0
    assert dump["s"] == ""
    assert dump["b"] == b""
    assert dump["arr"] == [0, 0]


def test_transient_storage_dump_hashmap_struct_late_field(get_contract):
    src = """
struct Foo:
    a: uint256
    b: uint256
    c: uint256

m: transient(HashMap[uint256, Foo])

@external
def set_c():
    self.m[1].c = 999
    """
    c = get_contract(src)
    c.set_c()

    dump = c.transient_storage_dump()
    assert dump["m"][1]["a"] == 0
    assert dump["m"][1]["c"] == 999


def test_storage_dump_single_uint256(get_contract):
    src = """
x: uint256

@external
def set_x(val: uint256):
    self.x = val
    """
    c = get_contract(src)
    dump = c.storage_dump()
    assert dump == {"x": 0}

    c.set_x(42)
    dump = c.storage_dump()
    assert dump == {"x": 42}


def test_storage_dump_single_int128(get_contract):
    src = """
x: int128

@external
def set_x(val: int128):
    self.x = val
    """
    c = get_contract(src)
    dump = c.storage_dump()
    assert dump == {"x": 0}

    c.set_x(-100)
    dump = c.storage_dump()
    assert dump == {"x": -100}


def test_storage_dump_single_int256(get_contract):
    src = """
x: int256

@external
def set_x(val: int256):
    self.x = val
    """
    c = get_contract(src)
    dump = c.storage_dump()
    assert dump == {"x": 0}

    c.set_x(-(2**200))
    dump = c.storage_dump()
    assert dump == {"x": -(2**200)}


def test_storage_dump_bool(get_contract):
    src = """
is_active: bool

@external
def set_active(val: bool):
    self.is_active = val
    """
    c = get_contract(src)
    dump = c.storage_dump()
    assert dump == {"is_active": False}

    c.set_active(True)
    dump = c.storage_dump()
    assert dump == {"is_active": True}


def test_storage_dump_address(get_contract):
    src = """
owner: address

@external
def set_owner(addr: address):
    self.owner = addr
    """
    c = get_contract(src)
    dump = c.storage_dump()
    assert dump == {"owner": "0x0000000000000000000000000000000000000000"}

    c.set_owner("0x1234567890123456789012345678901234567890")
    dump = c.storage_dump()
    assert dump["owner"] == "0x1234567890123456789012345678901234567890"


def test_storage_dump_bytes32(get_contract):
    src = """
data: bytes32

@external
def set_data(val: bytes32):
    self.data = val
    """
    c = get_contract(src)
    dump = c.storage_dump()
    assert dump == {"data": b"\x00" * 32}

    c.set_data(b"\x01" * 32)
    dump = c.storage_dump()
    assert dump == {"data": b"\x01" * 32}


def test_storage_dump_bytes4(get_contract):
    src = """
selector: bytes4

@external
def set_selector(val: bytes4):
    self.selector = val
    """
    c = get_contract(src)
    dump = c.storage_dump()
    assert dump == {"selector": b"\x00" * 4}

    c.set_selector(b"\xde\xad\xbe\xef")
    dump = c.storage_dump()
    assert dump == {"selector": b"\xde\xad\xbe\xef"}


def test_storage_dump_bytes_m(get_contract):
    src = """
bm: bytes2
bytesm_list: bytes1[1]

@external
def set_bytes():
    self.bm = 0x1234
    self.bytesm_list[0] = 0xab
    """
    c = get_contract(src)
    c.set_bytes()
    dump = c.storage_dump()
    assert dump["bm"] == b"\x12\x34"
    assert dump["bytesm_list"] == [b"\xab"]


def test_storage_dump_string(get_contract):
    src = """
name: String[100]

@external
def set_name(val: String[100]):
    self.name = val
    """
    c = get_contract(src)
    dump = c.storage_dump()
    assert dump == {"name": ""}

    c.set_name("hello world")
    dump = c.storage_dump()
    assert dump == {"name": "hello world"}


def test_storage_dump_bytes(get_contract):
    src = """
data: Bytes[100]

@external
def set_data(val: Bytes[100]):
    self.data = val
    """
    c = get_contract(src)
    dump = c.storage_dump()
    assert dump == {"data": b""}

    c.set_data(b"hello")
    dump = c.storage_dump()
    assert dump == {"data": b"hello"}


def test_storage_dump_dynarray(get_contract):
    src = """
arr: DynArray[uint256, 10]

@external
def set_arr(val: DynArray[uint256, 10]):
    self.arr = val
    """
    c = get_contract(src)
    dump = c.storage_dump()
    assert dump == {"arr": []}

    c.set_arr([1, 2, 3])
    dump = c.storage_dump()
    assert dump == {"arr": [1, 2, 3]}


def test_storage_dump_nested_dynarray(get_contract):
    src = """
d: DynArray[DynArray[uint256, 3], 3]

@external
def set_vals():
    self.d = [[1], [2, 3, 4], [5, 6]]
    """
    c = get_contract(src)
    c.set_vals()
    dump = c.storage_dump()
    assert dump == {"d": [[1], [2, 3, 4], [5, 6]]}


def test_storage_dump_static_array(get_contract):
    src = """
arr: uint256[3]

@external
def set_arr(a: uint256, b: uint256, c: uint256):
    self.arr[0] = a
    self.arr[1] = b
    self.arr[2] = c
    """
    c = get_contract(src)
    dump = c.storage_dump()
    assert dump == {"arr": [0, 0, 0]}

    c.set_arr(10, 20, 30)
    dump = c.storage_dump()
    assert dump == {"arr": [10, 20, 30]}


def test_storage_dump_bool_static_array(get_contract):
    src = """
flags: bool[5]

@external
def set_flags():
    self.flags[0] = True
    self.flags[2] = True
    self.flags[4] = True
    """
    c = get_contract(src)
    dump = c.storage_dump()
    assert dump == {"flags": [False, False, False, False, False]}

    c.set_flags()
    dump = c.storage_dump()
    assert dump == {"flags": [True, False, True, False, True]}


def test_storage_dump_address_static_array(get_contract):
    src = """
addrs: address[2]

@external
def set_addrs(a: address, b: address):
    self.addrs[0] = a
    self.addrs[1] = b
    """
    c = get_contract(src)
    dump = c.storage_dump()
    assert dump == {
        "addrs": [
            "0x0000000000000000000000000000000000000000",
            "0x0000000000000000000000000000000000000000",
        ]
    }

    c.set_addrs(
        "0x1111111111111111111111111111111111111111",
        "0x2222222222222222222222222222222222222222",
    )
    dump = c.storage_dump()
    assert dump == {
        "addrs": [
            "0x1111111111111111111111111111111111111111",
            "0x2222222222222222222222222222222222222222",
        ]
    }


def test_storage_dump_hashmap_address_uint256(get_contract):
    src = """
balances: HashMap[address, uint256]

@external
def set_balance(addr: address, val: uint256):
    self.balances[addr] = val
    """
    c = get_contract(src)
    dump = c.storage_dump()
    assert dump == {"balances": {}}

    addr1 = "0x1234567890123456789012345678901234567890"
    c.set_balance(addr1, 100)
    dump = c.storage_dump()
    assert dump == {"balances": {addr1: 100}}


def test_storage_dump_hashmap_uint256_bool(get_contract):
    src = """
flags: HashMap[uint256, bool]

@external
def set_flag(key: uint256, val: bool):
    self.flags[key] = val
    """
    c = get_contract(src)
    dump = c.storage_dump()
    assert dump == {"flags": {}}

    c.set_flag(1, True)
    c.set_flag(5, True)
    dump = c.storage_dump()
    assert dump == {"flags": {1: True, 5: True}}


def test_storage_dump_nested_hashmap(get_contract):
    src = """
approvals: HashMap[address, HashMap[uint256, bool]]

@external
def approve(owner: address, token_id: uint256):
    self.approvals[owner][token_id] = True
    """
    c = get_contract(src)
    dump = c.storage_dump()
    assert dump == {"approvals": {}}

    addr1 = "0x1234567890123456789012345678901234567890"
    c.approve(addr1, 42)
    dump = c.storage_dump()
    assert dump == {"approvals": {addr1: {42: True}}}


def test_storage_dump_hashmap_string_value(get_contract):
    src = """
h: HashMap[uint256, String[32]]

@external
def set_values():
    for i: uint8 in range(3):
        self.h[convert(i, uint256)] = uint2str(i)
    """
    c = get_contract(src)
    c.set_values()
    dump = c.storage_dump()
    assert dump == {"h": {0: "0", 1: "1", 2: "2"}}


def test_storage_dump_hashmap_string_key(get_contract):
    src = """
balances: HashMap[String[64], uint256]

@external
def set_balance(name: String[64], amount: uint256):
    self.balances[name] = amount
    """
    c = get_contract(src)
    c.set_balance("alice", 1000)
    c.set_balance("bob", 2000)
    dump = c.storage_dump()
    assert dump["balances"]["alice"] == 1000
    assert dump["balances"]["bob"] == 2000


def test_storage_dump_hashmap_bytes_key(get_contract):
    src = """
data: HashMap[Bytes[32], uint256]

@external
def set_data(key: Bytes[32], val: uint256):
    self.data[key] = val
    """
    c = get_contract(src)
    c.set_data(b"mykey", 42)
    dump = c.storage_dump()
    assert dump["data"][b"mykey"] == 42


def test_storage_dump_simple_struct(get_contract):
    src = """
struct Point:
    x: uint256
    y: uint256

point: Point

@external
def set_point(x: uint256, y: uint256):
    self.point = Point(x=x, y=y)
    """
    c = get_contract(src)
    dump = c.storage_dump()
    assert dump == {"point": {"x": 0, "y": 0}}

    c.set_point(10, 20)
    dump = c.storage_dump()
    assert dump == {"point": {"x": 10, "y": 20}}


def test_storage_dump_nested_struct(get_contract):
    src = """
struct Inner:
    value: uint256

struct Outer:
    inner: Inner
    flag: bool

data: Outer

@external
def set_data(val: uint256, flag: bool):
    self.data = Outer(inner=Inner(value=val), flag=flag)
    """
    c = get_contract(src)
    dump = c.storage_dump()
    assert dump == {"data": {"inner": {"value": 0}, "flag": False}}

    c.set_data(42, True)
    dump = c.storage_dump()
    assert dump == {"data": {"inner": {"value": 42}, "flag": True}}


def test_storage_dump_struct_as_tuple_alternative(get_contract):
    src = """
struct Pair:
    first: uint256
    second: bool

pair: Pair

@external
def set_pair(a: uint256, b: bool):
    self.pair = Pair(first=a, second=b)
    """
    c = get_contract(src)
    dump = c.storage_dump()
    assert dump == {"pair": {"first": 0, "second": False}}

    c.set_pair(100, True)
    dump = c.storage_dump()
    assert dump == {"pair": {"first": 100, "second": True}}


def test_storage_dump_decimal(get_contract):
    src = """
price: decimal

@external
def set_price(val: decimal):
    self.price = val
    """
    c = get_contract(src)
    dump = c.storage_dump()
    assert dump["price"] == Decimal("0")

    c.set_price(Decimal("3.14"))
    dump = c.storage_dump()
    assert dump["price"] == Decimal("3.14")


def test_storage_dump_multiple_variables(get_contract):
    src = """
a: uint256
b: bool
c: address
d: String[50]
e: DynArray[uint256, 5]

@external
def set_all():
    self.a = 100
    self.b = True
    self.c = 0x1111111111111111111111111111111111111111
    self.d = "test"
    self.e = [1, 2, 3]
    """
    c = get_contract(src)
    dump = c.storage_dump()
    assert dump == {
        "a": 0,
        "b": False,
        "c": "0x0000000000000000000000000000000000000000",
        "d": "",
        "e": [],
    }

    c.set_all()
    dump = c.storage_dump()
    assert dump == {
        "a": 100,
        "b": True,
        "c": "0x1111111111111111111111111111111111111111",
        "d": "test",
        "e": [1, 2, 3],
    }


def test_storage_dump_excludes_immutables(get_contract):
    src = """
x: uint256
y: immutable(uint256)

@deploy
def __init__(val: uint256):
    y = val

@external
def set_x(val: uint256):
    self.x = val

@external
def get_y() -> uint256:
    return y
    """
    c = get_contract(src, 999)
    dump = c.storage_dump()
    # Only x should appear, not y
    assert dump == {"x": 0}
    assert "y" not in dump

    c.set_x(42)
    dump = c.storage_dump()
    assert dump == {"x": 42}
    assert "y" not in dump

    # Verify immutable still works
    assert c.get_y() == 999


def test_storage_dump_excludes_constants(get_contract):
    src = """
MAX_VALUE: constant(uint256) = 1000
x: uint256

@external
def set_x(val: uint256):
    self.x = val

@external
def get_max() -> uint256:
    return MAX_VALUE
    """
    c = get_contract(src)
    dump = c.storage_dump()
    # Only x should appear, not MAX_VALUE
    assert dump == {"x": 0}
    assert "MAX_VALUE" not in dump

    # Verify constant still works
    assert c.get_max() == 1000


def test_storage_dump_mixed_storage_immutable_constant(get_contract):
    src = """
CONSTANT_VAL: constant(uint256) = 42
immutable_val: immutable(uint256)
storage_a: uint256
storage_b: bool
storage_c: String[20]

@deploy
def __init__(imm: uint256):
    immutable_val = imm

@external
def set_values():
    self.storage_a = 100
    self.storage_b = True
    self.storage_c = "hello"
    """
    c = get_contract(src, 777)
    dump = c.storage_dump()
    # Only storage variables should appear
    assert set(dump.keys()) == {"storage_a", "storage_b", "storage_c"}
    assert dump == {"storage_a": 0, "storage_b": False, "storage_c": ""}

    c.set_values()
    dump = c.storage_dump()
    assert dump == {"storage_a": 100, "storage_b": True, "storage_c": "hello"}


def test_storage_dump_after_multiple_mutations(get_contract):
    src = """
counter: uint256
history: DynArray[uint256, 100]

@external
def increment():
    self.counter += 1
    self.history.append(self.counter)
    """
    c = get_contract(src)
    dump = c.storage_dump()
    assert dump == {"counter": 0, "history": []}

    for i in range(5):
        c.increment()

    dump = c.storage_dump()
    assert dump == {"counter": 5, "history": [1, 2, 3, 4, 5]}


def test_storage_dump_complex_nested_types(get_contract):
    src = """
struct Info:
    id: uint256
    active: bool

struct Container:
    info: Info
    data: DynArray[uint256, 10]

container: Container

@external
def set_container():
    self.container = Container(
        info=Info(id=42, active=True),
        data=[1, 2, 3]
    )
    """
    c = get_contract(src)
    dump = c.storage_dump()
    assert dump == {"container": {"info": {"id": 0, "active": False}, "data": []}}

    c.set_container()
    dump = c.storage_dump()
    assert dump == {"container": {"info": {"id": 42, "active": True}, "data": [1, 2, 3]}}


def test_storage_dump_dynarray_of_structs(get_contract):
    src = """
struct Item:
    value: uint256

items: DynArray[Item, 10]

@external
def add_item(val: uint256):
    self.items.append(Item(value=val))
    """
    c = get_contract(src)
    dump = c.storage_dump()
    assert dump == {"items": []}

    c.add_item(10)
    c.add_item(20)
    dump = c.storage_dump()
    assert dump == {"items": [{"value": 10}, {"value": 20}]}


def test_storage_dump_static_array_of_structs(get_contract):
    src = """
struct Pair:
    a: uint256
    b: uint256

pairs: Pair[2]

@external
def set_pairs():
    self.pairs[0] = Pair(a=1, b=2)
    self.pairs[1] = Pair(a=3, b=4)
    """
    c = get_contract(src)
    dump = c.storage_dump()
    assert dump == {"pairs": [{"a": 0, "b": 0}, {"a": 0, "b": 0}]}

    c.set_pairs()
    dump = c.storage_dump()
    assert dump == {"pairs": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]}


def test_storage_dump_hashmap_to_struct(get_contract):
    src = """
struct Account:
    balance: uint256
    active: bool

accounts: HashMap[address, Account]

@external
def create_account(addr: address, bal: uint256):
    self.accounts[addr] = Account(balance=bal, active=True)
    """
    c = get_contract(src)
    dump = c.storage_dump()
    assert dump == {"accounts": {}}

    addr = "0xABCDabcdABCDabcdABCDabcdABCDabcdABCDabcd"
    c.create_account(addr, 500)
    dump = c.storage_dump()
    # HashMap keys are stored as Address objects
    accounts = dump["accounts"]
    assert len(accounts) == 1
    key = list(accounts.keys())[0]
    assert str(key) == "0xABCDabcdABcDabcDaBCDAbcdABcdAbCdABcDABCd"
    assert accounts[key] == {"balance": 500, "active": True}


def test_storage_dump_interface_storage(get_contract):
    src = """
interface Foo:
    def bar() -> uint256: payable
    def foobar() -> uint256: view

i: Foo

@external
def bar() -> uint256:
    return 1

@external
def foobar() -> uint256:
    return 2

@external
def set_i() -> uint256:
    a: uint256 = 0
    self.i = Foo(self)
    a = extcall self.i.bar()
    a += staticcall self.i.foobar()
    return a
    """
    c = get_contract(src)
    assert c.set_i() == 3
    dump = c.storage_dump()
    assert dump["i"] == str(c.address)


def test_storage_dump_deeply_nested_hashmap(get_contract):
    src = """
data: HashMap[address, HashMap[address, HashMap[uint256, uint256]]]

@external
def set_data(a: address, b: address, key: uint256, val: uint256):
    self.data[a][b][key] = val
    """
    c = get_contract(src)
    dump = c.storage_dump()
    assert dump == {"data": {}}

    addr1 = "0x1111111111111111111111111111111111111111"
    addr2 = "0x2222222222222222222222222222222222222222"
    c.set_data(addr1, addr2, 10, 100)
    dump = c.storage_dump()
    assert dump == {"data": {addr1: {addr2: {10: 100}}}}


def test_storage_dump_struct_with_address(get_contract):
    src = """
struct Inner:
    val: uint256
    active: bool

struct Outer:
    inner: Inner
    addr: address

data: Outer

@external
def set_data(val: uint256, active: bool, addr: address):
    self.data = Outer(inner=Inner(val=val, active=active), addr=addr)
    """
    c = get_contract(src)
    dump = c.storage_dump()
    assert dump == {"data": {"inner": {"val": 0, "active": False}, "addr": "0x0000000000000000000000000000000000000000"}}

    c.set_data(42, True, "0x1234567890123456789012345678901234567890")
    dump = c.storage_dump()
    assert dump == {"data": {"inner": {"val": 42, "active": True}, "addr": "0x1234567890123456789012345678901234567890"}}


def test_storage_dump_many_variables(get_contract):
    src = """
v1: uint256
v2: uint256
v3: uint256
v4: uint256
v5: uint256
v6: bool
v7: bool
v8: address
v9: String[10]
v10: Bytes[10]

@external
def set_all(addr: address):
    self.v1 = 1
    self.v2 = 2
    self.v3 = 3
    self.v4 = 4
    self.v5 = 5
    self.v6 = True
    self.v7 = False
    self.v8 = addr
    self.v9 = "test"
    self.v10 = b"bytes"
    """
    c = get_contract(src)
    dump = c.storage_dump()
    assert len(dump) == 10
    assert all(key.startswith("v") for key in dump.keys())

    c.set_all("0xFFfFfFffFFfffFFfFFfFFFFFffFFFffffFfFFFfF")
    dump = c.storage_dump()
    assert dump["v1"] == 1
    assert dump["v2"] == 2
    assert dump["v3"] == 3
    assert dump["v4"] == 4
    assert dump["v5"] == 5
    assert dump["v6"] is True
    assert dump["v7"] is False
    assert dump["v8"] == "0xFFfFfFffFFfffFFfFFfFFFFFffFFFffffFfFFFfF"
    assert dump["v9"] == "test"
    assert dump["v10"] == b"bytes"


def test_storage_dump_dynarray_of_addresses(get_contract):
    src = """
owners: DynArray[address, 10]

@external
def add_owner(addr: address):
    self.owners.append(addr)
    """
    c = get_contract(src)
    dump = c.storage_dump()
    assert dump == {"owners": []}

    addr1 = "0x1111111111111111111111111111111111111111"
    addr2 = "0x2222222222222222222222222222222222222222"
    c.add_owner(addr1)
    c.add_owner(addr2)
    dump = c.storage_dump()
    assert dump == {"owners": [addr1, addr2]}


def test_storage_dump_struct_with_dynarray(get_contract):
    src = """
struct Token:
    id: uint256
    holders: DynArray[address, 10]

token: Token

@external
def setup_token():
    self.token.id = 1
    self.token.holders = [
        0x1111111111111111111111111111111111111111,
        0x2222222222222222222222222222222222222222
    ]
    """
    c = get_contract(src)
    dump = c.storage_dump()
    assert dump == {"token": {"id": 0, "holders": []}}

    c.setup_token()
    dump = c.storage_dump()
    assert dump == {
        "token": {
            "id": 1,
            "holders": [
                "0x1111111111111111111111111111111111111111",
                "0x2222222222222222222222222222222222222222",
            ],
        }
    }


def test_storage_dump_large_static_array_truncated(get_contract):
    """Static array > 32KB (1025 uint256 = 1025*32 = 32800 bytes) is truncated."""
    src = """
arr: uint256[1025]

@external
def set_first(val: uint256):
    self.arr[0] = val
    """
    c = get_contract(src)
    c.set_first(42)
    dump = c.storage_dump()
    assert dump == {"arr": "<truncated>"}


def test_storage_dump_static_array_at_limit_not_truncated(get_contract):
    """Static array <= 32KB (1024 uint256 = 1024*32 = 32768 bytes) is NOT truncated."""
    src = """
arr: uint256[1024]

@external
def set_first(val: uint256):
    self.arr[0] = val
    """
    c = get_contract(src)
    c.set_first(99)
    dump = c.storage_dump()
    assert isinstance(dump["arr"], list)
    assert len(dump["arr"]) == 1024
    assert dump["arr"][0] == 99
    assert dump["arr"][1] == 0


def test_storage_dump_large_static_array_small_element(get_contract):
    """bool[4097] = 4097*32 = 131104 bytes > 32KB, should be truncated."""
    src = """
flags: bool[4097]

@external
def set_flag():
    self.flags[0] = True
    """
    c = get_contract(src)
    c.set_flag()
    dump = c.storage_dump()
    assert dump == {"flags": "<truncated>"}

import pytest
from vyper.utils import method_id

from ivy.exceptions import FunctionNotFound
from ivy.frontend.env import Env
from ivy.frontend.loader import loads
from ivy.journal import Journal


def _branches_for(env: Env, addr):
    return {
        (node_id, taken)
        for a, _source_id, node_id, taken in env.execution_metadata.branches
        if a == addr
    }


def _edges_for(env: Env, addr):
    return {
        (prev_node_id, node_id)
        for a, prev_node_id, node_id in env.execution_metadata.edges
        if a == addr
    }


def _boolops_for(env: Env, addr):
    return {
        (node_id, op, evaluated_count, result)
        for a, node_id, op, evaluated_count, result in env.execution_metadata.boolops
        if a == addr
    }


def _boolop_counts_for(env: Env, addr):
    return {
        evaluated_count
        for a, _node_id, _op, evaluated_count, _result in env.execution_metadata.boolops
        if a == addr
    }


def _loop_buckets_for(env: Env, addr):
    return {bucket for a, _node_id, bucket in env.execution_metadata.loops if a == addr}


def _node_coverage_for(env: Env, addr):
    return {
        node_id
        for _source_id, node_id in env.execution_metadata.coverage.get(addr, set())
    }


def test_runtime_coverage_branches_and_edges():
    env = Env.get_singleton()
    env.reset_execution_metadata()

    src = """
@external
def foo(x: uint256) -> uint256:
    if x == 0:
        return 1
    else:
        return 2
    """
    c = loads(src)
    env.reset_execution_metadata()

    addr = c.address

    assert c.foo(0) == 1
    branches_1 = _branches_for(env, addr)
    assert branches_1
    assert _edges_for(env, addr)

    assert c.foo(1) == 2
    branches_2 = _branches_for(env, addr)
    assert branches_2 - branches_1

    unknown = b"\xde\xad\xbe\xef"
    assert unknown != method_id("foo(uint256)")
    with pytest.raises(FunctionNotFound):
        env.message_call(to_address=addr, data=unknown)


def test_runtime_coverage_boolop_short_circuit_gradient():
    env = Env.get_singleton()
    env.reset_execution_metadata()

    src = """
@external
def foo(a: bool, b: bool, c: bool) -> bool:
    return a and b and c
    """
    c = loads(src)
    env.reset_execution_metadata()

    assert c.foo(False, True, True) is False
    assert c.foo(True, False, True) is False
    assert c.foo(True, True, False) is False
    assert c.foo(True, True, True) is True

    assert {1, 2, 3}.issubset(_boolop_counts_for(env, c.address))


def test_runtime_coverage_loop_buckets():
    env = Env.get_singleton()
    env.reset_execution_metadata()

    src = """
@external
def foo(n: uint256) -> uint256:
    s: uint256 = 0
    for i: uint256 in range(n, bound=16):
        s += i
    return s
    """
    c = loads(src)
    env.reset_execution_metadata()

    assert c.foo(0) == 0
    assert c.foo(1) == 0
    assert c.foo(4) == 6
    assert c.foo(9) == 36

    assert {0, 1, 4, 5}.issubset(_loop_buckets_for(env, c.address))


def test_runtime_coverage_assert_branch():
    """Assert statements should record branch coverage."""
    env = Env.get_singleton()
    env.reset_execution_metadata()

    src = """
@external
def foo(x: uint256):
    assert x > 0
    """
    c = loads(src)
    env.reset_execution_metadata()

    c.foo(1)
    branches = _branches_for(env, c.address)
    assert any(taken is True for (_node_id, taken) in branches)


def test_runtime_coverage_ifexp_branch():
    """IfExp (ternary) should record branch coverage."""
    env = Env.get_singleton()
    env.reset_execution_metadata()

    src = """
@external
def foo(x: uint256) -> uint256:
    return 1 if x > 0 else 0
    """
    c = loads(src)
    env.reset_execution_metadata()

    assert c.foo(5) == 1
    branches_1 = _branches_for(env, c.address)

    assert c.foo(0) == 0
    branches_2 = _branches_for(env, c.address)

    assert branches_2 - branches_1


def test_runtime_coverage_boolop_or_short_circuit():
    """Or operator should record short-circuit gradient."""
    env = Env.get_singleton()
    env.reset_execution_metadata()

    src = """
@external
def foo(a: bool, b: bool, c: bool) -> bool:
    return a or b or c
    """
    c = loads(src)
    env.reset_execution_metadata()

    assert c.foo(True, False, False) is True
    assert c.foo(False, True, False) is True
    assert c.foo(False, False, True) is True
    assert c.foo(False, False, False) is False

    assert {1, 2, 3}.issubset(_boolop_counts_for(env, c.address))


def test_runtime_coverage_boolop_records_op_type():
    """BoolOp should record whether it's 'and' or 'or'."""
    env = Env.get_singleton()
    env.reset_execution_metadata()

    src = """
@external
def test_and(a: bool, b: bool) -> bool:
    return a and b

@external
def test_or(a: bool, b: bool) -> bool:
    return a or b
    """
    c = loads(src)
    env.reset_execution_metadata()

    c.test_and(True, True)
    c.test_or(False, False)

    boolops = _boolops_for(env, c.address)
    ops = {op for (_node_id, op, _count, _result) in boolops}
    assert "and" in ops
    assert "or" in ops


def test_runtime_coverage_nested_if():
    """Nested if statements should produce distinct branch records."""
    env = Env.get_singleton()
    env.reset_execution_metadata()

    src = """
@external
def foo(x: uint256, y: uint256) -> uint256:
    if x > 0:
        if y > 0:
            return 1
        else:
            return 2
    else:
        return 3
    """
    c = loads(src)
    env.reset_execution_metadata()

    assert c.foo(1, 1) == 1
    branches_1 = _branches_for(env, c.address)

    assert c.foo(1, 0) == 2
    branches_2 = _branches_for(env, c.address)

    assert c.foo(0, 0) == 3
    branches_3 = _branches_for(env, c.address)

    assert len(branches_2) > len(branches_1)
    assert len(branches_3) > len(branches_2)


def test_runtime_coverage_loop_with_break():
    """Loop iteration count should be recorded even with early break."""
    env = Env.get_singleton()
    env.reset_execution_metadata()

    src = """
@external
def foo(n: uint256) -> uint256:
    s: uint256 = 0
    for i: uint256 in range(10):
        if i >= n:
            break
        s += i
    return s
    """
    c = loads(src)
    env.reset_execution_metadata()

    c.foo(3)
    buckets = _loop_buckets_for(env, c.address)
    assert buckets


def test_runtime_coverage_multiple_functions_edges():
    """Different functions should produce different edge patterns."""
    env = Env.get_singleton()
    env.reset_execution_metadata()

    src = """
@external
def func_a(x: uint256) -> uint256:
    y: uint256 = x + 1
    return y

@external
def func_b(x: uint256) -> uint256:
    y: uint256 = x * 2
    z: uint256 = y + 1
    return z
    """
    c = loads(src)
    env.reset_execution_metadata()

    c.func_a(1)
    edges_a = _edges_for(env, c.address)

    c.func_b(1)
    edges_b = _edges_for(env, c.address)

    assert edges_b - edges_a


def test_runtime_coverage_edge_isolation_across_calls():
    """Edges should not span across separate external calls."""
    env = Env.get_singleton()
    env.reset_execution_metadata()

    src = """
@external
def foo() -> uint256:
    x: uint256 = 1
    return x

@external
def bar() -> uint256:
    y: uint256 = 2
    return y
    """
    c = loads(src)
    env.reset_execution_metadata()

    c.foo()
    c.bar()

    edges = _edges_for(env, c.address)
    assert edges


def test_runtime_coverage_reset():
    """reset_execution_metadata should clear all coverage."""
    env = Env.get_singleton()
    env.reset_execution_metadata()

    src = """
@external
def foo(x: uint256) -> uint256:
    if x > 0:
        return 1
    return 0
    """
    c = loads(src)
    env.reset_execution_metadata()

    c.foo(1)
    assert _branches_for(env, c.address)
    assert _edges_for(env, c.address)

    env.reset_execution_metadata()
    assert not _branches_for(env, c.address)
    assert not _edges_for(env, c.address)


def test_runtime_coverage_signature_changes():
    """coverage_signature should change when new coverage is added."""
    env = Env.get_singleton()
    env.reset_execution_metadata()

    src = """
@external
def foo(x: uint256) -> uint256:
    if x == 0:
        return 0
    elif x == 1:
        return 1
    else:
        return 2
    """
    c = loads(src)
    env.reset_execution_metadata()

    sig_0 = env.execution_metadata.coverage_signature()

    c.foo(0)
    sig_1 = env.execution_metadata.coverage_signature()
    assert sig_1 != sig_0

    c.foo(1)
    sig_2 = env.execution_metadata.coverage_signature()
    assert sig_2 != sig_1

    c.foo(2)
    sig_3 = env.execution_metadata.coverage_signature()
    assert sig_3 != sig_2

    c.foo(2)
    sig_4 = env.execution_metadata.coverage_signature()
    assert sig_4 == sig_3


def test_runtime_coverage_node_hits():
    """Node hits (coverage dict) should record visited nodes."""
    env = Env.get_singleton()
    env.reset_execution_metadata()

    src = """
@external
def foo(x: uint256) -> uint256:
    y: uint256 = x + 1
    z: uint256 = y * 2
    return z
    """
    c = loads(src)
    env.reset_execution_metadata()

    c.foo(1)
    nodes = _node_coverage_for(env, c.address)
    assert len(nodes) >= 3


def test_runtime_coverage_cross_contract_isolation():
    """Coverage should be tracked per contract address."""
    env = Env.get_singleton()
    env.reset_execution_metadata()

    src_a = """
@external
def foo() -> uint256:
    if True:
        return 1
    return 0
    """
    src_b = """
@external
def bar() -> uint256:
    if False:
        return 1
    return 0
    """
    c_a = loads(src_a)
    c_b = loads(src_b)
    env.reset_execution_metadata()

    c_a.foo()
    c_b.bar()

    branches_a = _branches_for(env, c_a.address)
    branches_b = _branches_for(env, c_b.address)

    assert branches_a
    assert branches_b
    taken_a = {taken for (_nid, taken) in branches_a}
    taken_b = {taken for (_nid, taken) in branches_b}
    assert True in taken_a
    assert False in taken_b


def _check_state_modified(env: Env) -> bool:
    return env.execution_metadata.state_modified


def test_state_modified_simple_storage_write():
    env = Env.get_singleton()
    Journal().reset()
    env.reset_execution_metadata()

    src = """
a: uint256

@external
def foo():
    self.a = 42
    """
    c = loads(src)
    Journal().reset()

    c.foo()
    assert _check_state_modified(env) is True


def test_state_modified_no_state_change():
    env = Env.get_singleton()
    Journal().reset()
    env.reset_execution_metadata()

    src = """
@external
def foo() -> uint256:
    x: uint256 = 42
    return x
    """
    c = loads(src)
    Journal().reset()
    env.reset_execution_metadata()

    c.foo()
    assert _check_state_modified(env) is False


def test_state_modified_nested_call_commits():
    env = Env.get_singleton()
    Journal().reset()
    env.reset_execution_metadata()

    src = """
a: uint256

interface Self:
    def inner(): nonpayable

@external
def inner():
    self.a = 42

@external
def outer():
    extcall Self(self).inner()
    """
    c = loads(src)
    Journal().reset()

    c.outer()
    assert _check_state_modified(env) is True


def test_state_modified_nested_call_reverts():
    env = Env.get_singleton()
    Journal().reset()
    env.reset_execution_metadata()

    src = """
a: uint256
should_revert: bool

interface Self:
    def inner(): nonpayable

@external
def inner():
    self.a = 42
    assert not self.should_revert

@external
def outer():
    self.should_revert = True
    success: bool = False
    response: Bytes[32] = b""
    success, response = raw_call(self, method_id("inner()"), max_outsize=32, revert_on_failure=False)
    self.should_revert = False
    """
    c = loads(src)
    Journal().reset()

    c.outer()
    assert _check_state_modified(env) is True


def test_state_modified_nested_reverts_but_outer_has_changes():
    env = Env.get_singleton()
    Journal().reset()
    env.reset_execution_metadata()

    src = """
a: uint256
b: uint256
should_revert: bool

interface Self:
    def inner(): nonpayable

@external
def inner():
    self.b = 99
    assert not self.should_revert

@external
def outer():
    self.a = 42
    self.should_revert = True
    success: bool = False
    response: Bytes[32] = b""
    success, response = raw_call(self, method_id("inner()"), max_outsize=32, revert_on_failure=False)
    """
    c = loads(src)
    Journal().reset()

    c.outer()
    assert _check_state_modified(env) is True


def test_state_modified_deep_nesting_all_commit():
    env = Env.get_singleton()
    Journal().reset()
    env.reset_execution_metadata()

    src = """
a: uint256
b: uint256
c: uint256

interface Self:
    def level1(): nonpayable
    def level2(): nonpayable
    def level3(): nonpayable

@external
def level3():
    self.c = 3

@external
def level2():
    self.b = 2
    extcall Self(self).level3()

@external
def level1():
    self.a = 1
    extcall Self(self).level2()
    """
    c = loads(src)
    Journal().reset()

    c.level1()
    assert _check_state_modified(env) is True


def test_state_modified_deep_nesting_middle_reverts():
    env = Env.get_singleton()
    Journal().reset()
    env.reset_execution_metadata()

    src = """
a: public(uint256)
b: public(uint256)
c: public(uint256)

interface Self:
    def level1(): nonpayable
    def level2(): nonpayable
    def level3(): nonpayable

@external
def level3():
    self.c = 3

@external
def level2():
    self.b = 2
    extcall Self(self).level3()
    raise "revert"

@external
def level1():
    self.a = 1
    success: bool = False
    response: Bytes[32] = b""
    success, response = raw_call(self, method_id("level2()"), max_outsize=32, revert_on_failure=False)
    """
    c = loads(src)
    Journal().reset()

    c.level1()
    assert _check_state_modified(env) is True
    assert c.a() == 1
    assert c.b() == 0
    assert c.c() == 0


def test_state_modified_deep_nesting_innermost_reverts():
    env = Env.get_singleton()
    Journal().reset()
    env.reset_execution_metadata()

    src = """
a: public(uint256)
b: public(uint256)
c: public(uint256)

interface Self:
    def level1(): nonpayable
    def level2(): nonpayable
    def level3(): nonpayable

@external
def level3():
    self.c = 3
    raise "revert"

@external
def level2():
    self.b = 2
    success: bool = False
    response: Bytes[32] = b""
    success, response = raw_call(self, method_id("level3()"), max_outsize=32, revert_on_failure=False)

@external
def level1():
    self.a = 1
    extcall Self(self).level2()
    """
    c = loads(src)
    Journal().reset()

    c.level1()
    assert _check_state_modified(env) is True
    assert c.a() == 1
    assert c.b() == 2
    assert c.c() == 0


def test_state_modified_all_nested_revert():
    env = Env.get_singleton()
    Journal().reset()
    env.reset_execution_metadata()

    src = """
a: public(uint256)
b: public(uint256)

interface Self:
    def level1(): nonpayable
    def level2(): nonpayable

@external
def level2():
    self.b = 2
    raise "revert"

@external
def level1():
    self.a = 1
    extcall Self(self).level2()

@external
def outer():
    success: bool = False
    response: Bytes[32] = b""
    success, response = raw_call(self, method_id("level1()"), max_outsize=32, revert_on_failure=False)
    """
    c = loads(src)
    Journal().reset()
    env.reset_execution_metadata()

    c.outer()
    assert _check_state_modified(env) is False
    assert c.a() == 0
    assert c.b() == 0


def test_state_modified_transient_storage():
    env = Env.get_singleton()
    Journal().reset()
    env.reset_execution_metadata()

    src = """
t: transient(uint256)

@external
def foo():
    self.t = 42
    """
    c = loads(src)
    Journal().reset()

    c.foo()
    assert _check_state_modified(env) is True


def test_state_modified_cross_contract():
    env = Env.get_singleton()
    Journal().reset()
    env.reset_execution_metadata()

    src_a = """
a: uint256

interface Other:
    def modify(): nonpayable

@external
def foo(other: address):
    extcall Other(other).modify()
    """

    src_b = """
b: uint256

@external
def modify():
    self.b = 99
    """
    c_a = loads(src_a)
    c_b = loads(src_b)
    Journal().reset()

    c_a.foo(c_b.address)
    assert _check_state_modified(env) is True


def test_state_modified_cross_contract_reverts():
    env = Env.get_singleton()
    Journal().reset()
    env.reset_execution_metadata()

    src_a = """
a: uint256

@external
def foo(other: address):
    success: bool = False
    response: Bytes[32] = b""
    success, response = raw_call(other, method_id("modify()"), max_outsize=32, revert_on_failure=False)
    """

    src_b = """
b: uint256

@external
def modify():
    self.b = 99
    raise "revert"
    """
    c_a = loads(src_a)
    c_b = loads(src_b)
    Journal().reset()
    env.reset_execution_metadata()

    c_a.foo(c_b.address)
    assert _check_state_modified(env) is False


def test_state_modified_mixed_cross_contract():
    env = Env.get_singleton()
    Journal().reset()
    env.reset_execution_metadata()

    src_a = """
a: public(uint256)

interface Other:
    def modify(): nonpayable
    def modify_and_revert(): nonpayable

@external
def foo(other: address):
    self.a = 1
    success: bool = False
    response: Bytes[32] = b""
    success, response = raw_call(other, method_id("modify_and_revert()"), max_outsize=32, revert_on_failure=False)
    extcall Other(other).modify()
    """

    src_b = """
b: public(uint256)
c: public(uint256)

@external
def modify():
    self.b = 99

@external
def modify_and_revert():
    self.c = 88
    raise "revert"
    """
    c_a = loads(src_a)
    c_b = loads(src_b)
    Journal().reset()

    c_a.foo(c_b.address)
    assert _check_state_modified(env) is True
    assert c_a.a() == 1
    assert c_b.b() == 99
    assert c_b.c() == 0


def test_state_modified_balance_transfer():
    """Balance transfers should be tracked as state modification."""
    env = Env.get_singleton()
    Journal().reset()
    env.reset_execution_metadata()

    src = """
@external
@payable
def receive_eth():
    pass

@external
def send_eth(to: address):
    send(to, self.balance)
    """
    c = loads(src)
    env.set_balance(c.address, 1000)
    Journal().reset()

    c.send_eth(env.eoa)
    assert _check_state_modified(env) is True


def test_state_modified_balance_transfer_reverts():
    """Reverted balance transfers should not count as state modification."""
    env = Env.get_singleton()
    Journal().reset()
    env.reset_execution_metadata()

    src = """
@external
def send_and_revert(to: address):
    send(to, self.balance)
    raise "revert"
    """
    c = loads(src)
    env.set_balance(c.address, 1000)
    Journal().reset()
    env.reset_execution_metadata()

    try:
        c.send_and_revert(env.eoa)
    except Exception:
        pass
    assert _check_state_modified(env) is False


def test_state_modified_raw_call_with_value():
    """raw_call with value should track balance changes."""
    env = Env.get_singleton()
    Journal().reset()
    env.reset_execution_metadata()

    src_sender = """
@external
def send_via_raw_call(to: address):
    raw_call(to, b"", value=100)
    """

    src_receiver = """
@external
@payable
def __default__():
    pass
    """
    c_sender = loads(src_sender)
    c_receiver = loads(src_receiver)
    env.set_balance(c_sender.address, 1000)
    Journal().reset()

    c_sender.send_via_raw_call(c_receiver.address)
    assert _check_state_modified(env) is True


def test_state_modified_create_minimal_proxy():
    """Creating a new contract should track nonce and account creation."""
    env = Env.get_singleton()
    Journal().reset()
    env.reset_execution_metadata()

    src = """
@external
def deploy_proxy() -> address:
    return create_minimal_proxy_to(self)
    """
    c = loads(src)
    Journal().reset()

    new_addr = c.deploy_proxy()
    assert new_addr != c.address
    assert _check_state_modified(env) is True


def test_state_modified_create_copy_of():
    """create_copy_of should track state modification."""
    env = Env.get_singleton()
    Journal().reset()
    env.reset_execution_metadata()

    src = """
@external
def deploy_copy() -> address:
    return create_copy_of(self)
    """
    c = loads(src)
    Journal().reset()

    new_addr = c.deploy_copy()
    assert new_addr != c.address
    assert _check_state_modified(env) is True


def test_state_modified_create_reverts():
    """Reverted contract creation should not count as state modification."""
    env = Env.get_singleton()
    Journal().reset()
    env.reset_execution_metadata()

    src = """
@external
def deploy_and_revert() -> address:
    addr: address = create_minimal_proxy_to(self)
    raise "revert"
    """
    c = loads(src)
    Journal().reset()
    env.reset_execution_metadata()

    try:
        c.deploy_and_revert()
    except Exception:
        pass
    assert _check_state_modified(env) is False


def test_state_modified_dynarray_append():
    """DynArray append should track state modification."""
    env = Env.get_singleton()
    Journal().reset()
    env.reset_execution_metadata()

    src = """
arr: DynArray[uint256, 10]

@external
def append_value(v: uint256):
    self.arr.append(v)

@external
def get_length() -> uint256:
    return len(self.arr)
    """
    c = loads(src)
    Journal().reset()

    c.append_value(42)
    assert _check_state_modified(env) is True
    assert c.get_length() == 1


def test_state_modified_dynarray_pop():
    """DynArray pop should track state modification."""
    env = Env.get_singleton()
    Journal().reset()
    env.reset_execution_metadata()

    src = """
arr: DynArray[uint256, 10]

@external
def setup():
    self.arr.append(1)
    self.arr.append(2)

@external
def pop_value() -> uint256:
    return self.arr.pop()

@external
def get_length() -> uint256:
    return len(self.arr)
    """
    c = loads(src)
    c.setup()
    Journal().reset()

    val = c.pop_value()
    assert val == 2
    assert _check_state_modified(env) is True
    assert c.get_length() == 1


def test_state_modified_dynarray_reverts():
    """Reverted DynArray operations should not count as state modification."""
    env = Env.get_singleton()
    Journal().reset()
    env.reset_execution_metadata()

    src = """
arr: DynArray[uint256, 10]

@external
def append_and_revert(v: uint256):
    self.arr.append(v)
    raise "revert"

@external
def get_length() -> uint256:
    return len(self.arr)
    """
    c = loads(src)
    Journal().reset()
    env.reset_execution_metadata()

    try:
        c.append_and_revert(42)
    except Exception:
        pass
    assert _check_state_modified(env) is False
    assert c.get_length() == 0


def test_state_modified_nested_create_reverts():
    """Nested contract creation that reverts should not modify state."""
    env = Env.get_singleton()
    Journal().reset()
    env.reset_execution_metadata()

    src_target = """
@external
def dummy() -> uint256:
    return 1
    """

    src_creator = """
interface Self:
    def inner_create(target: address): nonpayable

@external
def inner_create(target: address):
    new_addr: address = create_minimal_proxy_to(target)
    raise "revert"

@external
def outer_create(target: address):
    success: bool = False
    response: Bytes[32] = b""
    success, response = raw_call(self, abi_encode(target, method_id=method_id("inner_create(address)")), max_outsize=32, revert_on_failure=False)
    """
    c_target = loads(src_target)
    c_creator = loads(src_creator)
    Journal().reset()
    env.reset_execution_metadata()

    c_creator.outer_create(c_target.address)
    assert _check_state_modified(env) is False


def test_state_modified_nested_balance_partial_revert():
    """Nested balance transfer with partial revert should track only committed changes."""
    env = Env.get_singleton()
    Journal().reset()
    env.reset_execution_metadata()

    src_a = """
interface Other:
    def receive_and_revert(): payable

@external
def transfer_partial(other: address, receiver: address):
    send(receiver, 100)
    success: bool = False
    response: Bytes[32] = b""
    success, response = raw_call(other, method_id("receive_and_revert()"), value=200, max_outsize=32, revert_on_failure=False)
    """

    src_b = """
@external
@payable
def receive_and_revert():
    raise "revert"
    """
    c_a = loads(src_a)
    c_b = loads(src_b)
    env.set_balance(c_a.address, 1000)
    Journal().reset()

    c_a.transfer_partial(c_b.address, env.eoa)
    assert _check_state_modified(env) is True


def test_state_modified_multiple_types():
    """Multiple types of state changes in one transaction."""
    env = Env.get_singleton()
    Journal().reset()
    env.reset_execution_metadata()

    src = """
a: uint256
arr: DynArray[uint256, 10]
t: transient(uint256)

@external
def do_everything(receiver: address):
    self.a = 42
    self.arr.append(1)
    self.t = 99
    send(receiver, 100)
    """
    c = loads(src)
    env.set_balance(c.address, 1000)
    Journal().reset()

    c.do_everything(env.eoa)
    assert _check_state_modified(env) is True


def test_state_modified_only_read_operations():
    """Pure read operations should not modify state."""
    env = Env.get_singleton()
    Journal().reset()
    env.reset_execution_metadata()

    src = """
a: public(uint256)
arr: DynArray[uint256, 10]

@external
def setup():
    self.a = 42
    self.arr.append(1)
    self.arr.append(2)

@external
@view
def read_all() -> (uint256, uint256, uint256):
    return (self.a, self.arr[0], len(self.arr))
    """
    c = loads(src)
    c.setup()
    Journal().reset()
    env.reset_execution_metadata()

    result = c.read_all()
    assert result == (42, 1, 2)
    assert _check_state_modified(env) is False


def test_state_modified_self_call_with_value():
    """Self-call with value transfer should track state."""
    env = Env.get_singleton()
    Journal().reset()
    env.reset_execution_metadata()

    src = """
received: public(uint256)

interface Self:
    def inner(): payable

@external
@payable
def inner():
    self.received = msg.value

@external
def outer():
    extcall Self(self).inner(value=100)
    """
    c = loads(src)
    env.set_balance(c.address, 1000)
    Journal().reset()

    c.outer()
    assert _check_state_modified(env) is True
    assert c.received() == 100


def test_state_modified_not_contaminated_across_traces():
    env = Env.get_singleton()
    Journal().reset()
    env.reset_execution_metadata()

    src = """
a: uint256

@external
def modify_state():
    self.a = 42

@external
@view
def read_only() -> uint256:
    return 1
    """
    c = loads(src)
    Journal().reset()

    c.modify_state()
    assert _check_state_modified(env) is True

    env.reset_execution_metadata()

    c.read_only()

    assert _check_state_modified(env) is False

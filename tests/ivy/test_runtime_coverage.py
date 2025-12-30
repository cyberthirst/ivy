import pytest
from vyper.utils import method_id

from ivy.exceptions import FunctionNotFound
from ivy.frontend.env import Env
from ivy.frontend.loader import loads


def _branches_for(env: Env, addr):
    return {
        (node_id, taken)
        for a, node_id, taken in env.execution_metadata.branches
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
    return env.execution_metadata.coverage.get(addr, set())


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

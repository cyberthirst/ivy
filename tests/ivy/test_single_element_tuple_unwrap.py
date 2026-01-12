from ivy.frontend.loader import loads


def test_single_element_tuple_return_python_interface():
    """Regression test: single-element tuple returns should be unwrapped correctly."""
    src = """
@external
def bar() -> (uint256,):
    x: uint256 = 42
    return (x,)
    """
    c = loads(src)
    result = c.bar()
    # Should return (42,), not ((42,),)
    assert result == (42,), f"Expected (42,), got {result}"


def test_single_element_tuple_extcall():
    """Regression test: extcall with single-element tuple return works correctly."""
    src = """
@external
def bar() -> (uint256,):
    x: uint256 = 42
    return (x,)

interface __Callee__:
    def bar() -> (uint256,): nonpayable

@external
def foo() -> uint256:
    result: (uint256,) = extcall __Callee__(self).bar()
    return result[0]
    """
    c = loads(src)
    # Internal extcall path
    assert c.foo() == 42
    # Python interface path
    assert c.bar() == (42,)


def test_abi_decode_single_element_tuple():
    """Regression test: abi_decode with single-element tuple unwraps correctly."""
    src = """
@external
def test_decode() -> (uint256,):
    x: uint256 = 123
    encoded: Bytes[64] = abi_encode(x)
    return abi_decode(encoded, (uint256,))
    """
    c = loads(src)
    result = c.test_decode()
    assert result == (123,), f"Expected (123,), got {result}"


def test_multi_element_tuple_not_affected():
    """Ensure multi-element tuples still work correctly (not double-wrapped)."""
    src = """
@external
def bar() -> (uint256, uint256):
    return (1, 2)
    """
    c = loads(src)
    result = c.bar()
    assert result == (1, 2), f"Expected (1, 2), got {result}"


def test_non_tuple_return_not_affected():
    """Ensure non-tuple returns still work correctly."""
    src = """
@external
def bar() -> uint256:
    return 42
    """
    c = loads(src)
    result = c.bar()
    assert result == 42, f"Expected 42, got {result}"

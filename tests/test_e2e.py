from ivy.loader import loads


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

from tests.utils import decimal_to_int


def test_break_test(get_contract):
    break_test = """
@external
def foo(n: decimal) -> int128:
    c: decimal = n * 1.0
    output: int128 = 0
    for i: int128 in range(400):
        c = c / 1.2589
        if c < 1.0:
            output = i
            break
    return output
    """

    c = get_contract(break_test)

    assert c.foo(decimal_to_int("1")) == 0
    assert c.foo(decimal_to_int("2")) == 3
    assert c.foo(decimal_to_int("10")) == 10
    assert c.foo(decimal_to_int("200")) == 23

    print("Passed for-loop break test")


def test_break_test_2(get_contract):
    break_test_2 = """
@external
def foo(n: decimal) -> int128:
    c: decimal = n * 1.0
    output: int128 = 0
    for i: int128 in range(40):
        if c < 10.0:
            output = i * 10
            break
        c = c / 10.0
    for i: int128 in range(10):
        c = c / 1.2589
        if c < 1.0:
            output = output + i
            break
    return output
    """

    c = get_contract(break_test_2)
    assert c.foo(decimal_to_int("1")) == 0
    assert c.foo(decimal_to_int("2")) == 3
    assert c.foo(decimal_to_int("10")) == 10
    assert c.foo(decimal_to_int("200")) == 23
    assert c.foo(decimal_to_int("4000000")) == 66
    print("Passed for-loop break test 2")


def test_break_test_3(get_contract):
    break_test_3 = """
@external
def foo(n: int128) -> int128:
    c: decimal = convert(n, decimal)
    output: int128 = 0
    for i: int128 in range(40):
        if c < 10.0:
            output = i * 10
            break
        c /= 10.0
    for i: int128 in range(10):
        c /= 1.2589
        if c < 1.0:
            output = output + i
            break
    return output
    """

    c = get_contract(break_test_3)
    assert c.foo(1) == 0
    assert c.foo(2) == 3
    assert c.foo(10) == 10
    assert c.foo(200) == 23
    assert c.foo(4000000) == 66
    print("Passed aug-assignment break composite test")

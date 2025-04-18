import pytest

from vyper.exceptions import CallViolation

from ivy.exceptions import PayabilityViolation

nonpayable_code = [
    """
# single function, nonpayable
@external
def foo() -> bool:
    return True
    """,
    """
# multiple functions, one is payable
@external
def foo() -> bool:
    return True

@payable
@external
def bar() -> bool:
    return True
    """,
    """
# multiple functions, nonpayable
@external
def foo() -> bool:
    return True

@external
def bar() -> bool:
    return True
    """,
    """
# multiple functions and default func, nonpayable
@external
def foo() -> bool:
    return True

@external
def bar() -> bool:
    return True

@external
def __default__():
    pass
    """,
    """
    # multiple functions and default func, payable
@external
def foo() -> bool:
    return True

@external
def bar() -> bool:
    return True

@external
@payable
def __default__():
    pass
    """,
    """
# multiple functions, nonpayable (view)
@external
def foo() -> bool:
    return True

@view
@external
def bar() -> bool:
    return True
    """,
    """
# payable init function
@deploy
@payable
def __init__():
    a: int128 = 1

@external
def foo() -> bool:
    return True
    """,
    """
# payable default function
@external
@payable
def __default__():
    a: int128 = 1

@external
def foo() -> bool:
    return True
    """,
    """
# payable default function and other function
@external
@payable
def __default__():
    a: int128 = 1

@external
def foo() -> bool:
    return True

@external
@payable
def bar() -> bool:
    return True
    """,
    """
# several functions, one payable
@external
def foo() -> bool:
    return True

@payable
@external
def bar() -> bool:
    return True

@external
def baz() -> bool:
    return True
    """,
]


@pytest.mark.parametrize("code", nonpayable_code)
def test_nonpayable_runtime_assertion(env, keccak, tx_failed, get_contract, code):
    c = get_contract(code)
    env.set_balance(env.deployer, 10**18)

    c.foo(value=0)
    sig = keccak("foo()".encode()).hex()[:10]
    with tx_failed():
        env.message_call(c.address, data=sig, value=10**18)


payable_code = [
    """
# single function, payable
@payable
@external
def foo() -> bool:
    return True
    """,
    """
# two functions, one is payable
@payable
@external
def foo() -> bool:
    return True

@external
def bar() -> bool:
    return True
    """,
    """
# two functions, payable
@payable
@external
def foo() -> bool:
    return True

@payable
@external
def bar() -> bool:
    return True
    """,
    """
# two functions, one nonpayable (view)
@payable
@external
def foo() -> bool:
    return True

@view
@external
def bar() -> bool:
    return True
    """,
    """
# several functions, all payable
@payable
@external
def foo() -> bool:
    return True

@payable
@external
def bar() -> bool:
    return True

@payable
@external
def baz() -> bool:
    return True
    """,
    """
# several functions, one payable
@payable
@external
def foo() -> bool:
    return True

@external
def bar() -> bool:
    return True

@external
def baz() -> bool:
    return True
    """,
    """
# several functions, two payable
@payable
@external
def foo() -> bool:
    return True

@external
def bar() -> bool:
    return True

@payable
@external
def baz() -> bool:
    return True
    """,
    """
# init function
@deploy
def __init__():
    a: int128 = 1

@payable
@external
def foo() -> bool:
    return True
    """,
    """
# default function
@external
def __default__():
    a: int128 = 1

@external
@payable
def foo() -> bool:
    return True
    """,
    """
# payable default function
@external
@payable
def __default__():
    a: int128 = 1

@external
@payable
def foo() -> bool:
    return True
    """,
    """
# payable default function and nonpayable other function
@external
@payable
def __default__():
    a: int128 = 1

@external
@payable
def foo() -> bool:
    return True

@external
def bar() -> bool:
    return True
    """,
]


@pytest.mark.parametrize("code", payable_code)
def test_payable_runtime_assertion(env, get_contract, code):
    c = get_contract(code)
    env.set_balance(env.deployer, 10**18)
    c.foo(value=10**18)
    c.foo(value=0)


def test_payable_default_func_invalid_calldata(get_contract, env):
    code = """
@external
def foo() -> bool:
    return True

@payable
@external
def __default__():
    pass
    """
    c = get_contract(code)
    env.set_balance(env.deployer, 100)
    env.message_call(c.address, value=100, data="0x12345678")


def test_nonpayable_default_func_invalid_calldata(get_contract, env, tx_failed):
    code = """
@external
@payable
def foo() -> bool:
    return True

@external
def __default__():
    pass
    """

    c = get_contract(code)
    env.message_call(c.address, value=0, data="0x12345678")
    env.set_balance(env.deployer, 100)
    with tx_failed(PayabilityViolation, text="Function __default__ is not payable"):
        env.message_call(c.address, value=100, data="0x12345678")


def test_batch_nonpayable(get_contract, env, tx_failed):
    code = """
@external
def foo() -> bool:
    return True

@external
def __default__():
    pass
    """

    c = get_contract(code)
    env.message_call(c.address, value=0, data="0x12345678")
    data = bytes([1, 2, 3, 4])
    for i in range(5):
        calldata = "0x" + data[:i].hex()
        env.set_balance(env.deployer, 100)
        with tx_failed(PayabilityViolation, text="Function __default__ is not payable"):
            env.message_call(c.address, value=100, data=calldata)

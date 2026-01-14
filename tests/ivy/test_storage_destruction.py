"""
Regression tests for CREATE storage collisions.

Under Prague, CREATE must fail if the target address has pre-existing storage.
"""

import pytest

from ivy.exceptions import EVMException
from ivy.frontend.env import Env
from ivy.frontend.loader import loads
from ivy.utils import compute_contract_address
from ivy.types import Address


@pytest.fixture
def env():
    environment = Env().get_singleton()
    environment.clear_state()
    return environment


def test_create_rejects_preexisting_storage(env):
    """
    CREATE should fail if the target address already has storage.
    """
    sender = env.eoa
    sender_nonce = env.state.get_nonce(sender)
    predicted_address = Address(
        compute_contract_address(sender.canonical_address, sender_nonce)
    )

    raw_state = env.interpreter.state._state
    account = raw_state.state[predicted_address]
    account.storage[0] = 0xDEADBEEF
    account.storage[1] = 0xCAFEBABE
    account.storage[100] = 0x12345678

    src = """
value: uint256

@external
def get_value() -> uint256:
    return self.value

@external
def set_value(x: uint256):
    self.value = x
    """

    with pytest.raises(EVMException):
        loads(src)

    account = raw_state.state[predicted_address]
    assert account.storage[0] == 0xDEADBEEF
    assert account.storage[1] == 0xCAFEBABE
    assert account.storage[100] == 0x12345678


def test_create_collision_skips_init(env):
    """
    CREATE collision should prevent __init__ from running.
    """
    sender = env.eoa
    sender_nonce = env.state.get_nonce(sender)
    predicted_address = Address(
        compute_contract_address(sender.canonical_address, sender_nonce)
    )

    raw_state = env.interpreter.state._state
    account = raw_state.state[predicted_address]
    account.storage[0] = 999999

    src = """
value: uint256

@deploy
def __init__():
    self.value = 42

@external
def get_value() -> uint256:
    return self.value
    """

    with pytest.raises(EVMException):
        loads(src)

    account = raw_state.state[predicted_address]
    assert account.storage[0] == 999999


def test_empty_storage_no_op(env):
    """
    CREATE should succeed when there is no pre-existing storage.
    """
    src = """
value: uint256

@deploy
def __init__():
    self.value = 123

@external
def get_value() -> uint256:
    return self.value
    """

    contract = loads(src)
    assert contract.get_value() == 123

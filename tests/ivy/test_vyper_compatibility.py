import pytest
from ivy.frontend.env import Env
from ivy.frontend.loader import loads
from ivy.types import Address


def test_message_call_nonce_behavior(get_contract):
    """Test that message calls (default) don't increment nonce."""
    env = Env.get_singleton()

    # Create a test account
    sender = Address("0xa234567890123456789012345678901234567890")
    env.set_balance(sender, 10**20)
    env.eoa = sender

    # Deploy a contract
    src = """
counter: uint256

@external
def increment():
    self.counter += 1

@external
def get_counter() -> uint256:
    return self.counter
    """
    c = get_contract(src)

    initial_nonce = env.state.get_nonce(sender)

    # Call contract method multiple times WITHOUT transact flag (message calls)
    c.increment()  # message call - no nonce increment
    assert env.state.get_nonce(sender) == initial_nonce

    c.increment()  # message call - no nonce increment
    assert env.state.get_nonce(sender) == initial_nonce

    c.increment()  # message call - no nonce increment
    assert env.state.get_nonce(sender) == initial_nonce

    # Verify contract state is correct
    assert c.get_counter() == 3


def test_transaction_nonce_behavior(get_contract):
    """Test that transactions (transact=True) increment nonce."""
    env = Env.get_singleton()

    # Create a test account
    sender = Address("0xb234567890123456789012345678901234567890")
    env.set_balance(sender, 10**20)
    env.eoa = sender

    # Deploy a contract
    src = """
counter: uint256

@external
def increment():
    self.counter += 1

@external
def get_counter() -> uint256:
    return self.counter
    """
    c = get_contract(src)

    initial_nonce = env.state.get_nonce(sender)

    # Call contract method multiple times WITH transact flag (transactions)
    c.increment(transact=True)  # transaction - nonce increments
    assert env.state.get_nonce(sender) == initial_nonce + 1

    c.increment(transact=True)  # transaction - nonce increments
    assert env.state.get_nonce(sender) == initial_nonce + 2

    c.increment(transact=True)  # transaction - nonce increments
    assert env.state.get_nonce(sender) == initial_nonce + 3

    # Verify contract state is correct
    assert c.get_counter() == 3


def test_mixed_calls_and_transactions(get_contract):
    """Test mixing message calls and transactions."""
    env = Env.get_singleton()

    # Create a test account
    sender = Address("0xc234567890123456789012345678901234567890")
    env.set_balance(sender, 10**20)
    env.eoa = sender

    # Deploy a contract
    src = """
counter: uint256
last_sender: address

@external
def increment():
    self.counter += 1
    self.last_sender = msg.sender

@external
def get_counter() -> uint256:
    return self.counter

@external
def get_last_sender() -> address:
    return self.last_sender
    """
    c = get_contract(src)

    initial_nonce = env.state.get_nonce(sender)

    # Mix message calls and transactions
    c.increment()  # message call - no nonce increment
    assert env.state.get_nonce(sender) == initial_nonce
    assert c.get_counter() == 1

    c.increment(transact=True)  # transaction - nonce increments
    assert env.state.get_nonce(sender) == initial_nonce + 1
    assert c.get_counter() == 2

    c.increment()  # message call - no nonce increment
    assert env.state.get_nonce(sender) == initial_nonce + 1
    assert c.get_counter() == 3

    # Verify last sender is always the EOA
    assert c.get_last_sender() == sender


def test_vyper_test_suite_compatibility(get_contract):
    """Test that Vyper test suite patterns work correctly."""
    env = Env.get_singleton()

    # Create multiple test accounts
    alice = Address("0xd234567890123456789012345678901234567890")
    bob = Address("0xe234567890123456789012345678901234567890")

    env.set_balance(alice, 10**20)
    env.set_balance(bob, 10**20)

    # Deploy contract as Alice
    env.eoa = alice
    src = """
balances: public(HashMap[address, uint256])

@external
@payable
def deposit():
    self.balances[msg.sender] += msg.value

@external
def transfer(to: address, amount: uint256):
    assert self.balances[msg.sender] >= amount
    self.balances[msg.sender] -= amount
    self.balances[to] += amount
    """

    c = get_contract(src)
    alice_nonce_after_deploy = env.state.get_nonce(alice)

    # Alice deposits (message call by default)
    c.deposit(value=100)
    assert env.state.get_nonce(alice) == alice_nonce_after_deploy  # No nonce increment
    assert c.balances(alice) == 100

    # Bob deposits (message call by default)
    env.eoa = bob
    c.deposit(value=50)
    assert env.state.get_nonce(bob) == 0  # Bob's nonce unchanged (never deployed)
    assert c.balances(bob) == 50

    # Alice transfers to Bob (message call by default)
    env.eoa = alice
    c.transfer(bob, 25)
    assert env.state.get_nonce(alice) == alice_nonce_after_deploy  # Still no increment
    assert c.balances(alice) == 75
    assert c.balances(bob) == 75

    # All operations happen in the same transaction context
    # This matches Vyper test suite behavior


def test_transient_storage_with_message_calls(get_contract):
    """Test that transient storage behaves correctly with message calls."""
    env = Env.get_singleton()

    sender = Address("0xf234567890123456789012345678901234567890")
    env.set_balance(sender, 10**20)
    env.eoa = sender

    # Deploy a contract that uses transient storage
    src = """
temp_value: transient(uint256)

@external
def set_temp(val: uint256):
    self.temp_value = val

@external
def get_temp() -> uint256:
    return self.temp_value

@external
def complex_operation() -> uint256:
    # Set transient value
    self.temp_value = 100
    # It should persist during this call
    return self.temp_value + 42
    """

    c = get_contract(src)

    # With message calls, transient storage persists within the same tx context
    c.set_temp(50)
    assert c.get_temp() == 50  # Should still be 50

    # Another message call in same context
    assert c.complex_operation() == 142

    # Transient storage should still have the value from complex_operation
    assert c.get_temp() == 100

    # Now with a transaction (new context), transient storage should be cleared
    c.set_temp(75, transact=True)
    assert c.get_temp(transact=True) == 0  # Cleared in new transaction

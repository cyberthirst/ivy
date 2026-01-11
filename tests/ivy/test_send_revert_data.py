"""
Tests for send() revert data handling.

Vyper's send() compiles to assert(call(...)), which discards returndata
and reverts with empty data on failure. Ivy must match this behavior and
NOT propagate the callee's revert data.
"""
import pytest
from ivy.exceptions import Assert, Revert


def test_send_does_not_propagate_callee_revert_data(get_contract, env):
    """Test that send() reverts with empty data when callee reverts with a message."""
    receiver_src = """
@external
@payable
def __default__():
    raise "nope"
    """
    sender_src = """
@external
def send_it(b: address):
    send(b, 1)
    """
    receiver = get_contract(receiver_src)
    sender = get_contract(sender_src)
    env.set_balance(sender.address, 100)

    # The send should fail (receiver reverts), but the revert data should be empty
    with pytest.raises(Assert) as excinfo:
        sender.send_it(receiver.address)

    # Vyper's send() reverts with empty data, not the callee's "nope" message
    assert excinfo.value.data == b""


def test_send_does_not_propagate_raw_revert_data(get_contract, env):
    """Test that send() reverts with empty data when callee uses raw_revert."""
    receiver_src = """
@external
@payable
def __default__():
    raw_revert(b"\\xde\\xad\\xbe\\xef")
    """
    sender_src = """
@external
def send_it(b: address):
    send(b, 1)
    """
    receiver = get_contract(receiver_src)
    sender = get_contract(sender_src)
    env.set_balance(sender.address, 100)

    with pytest.raises(Assert) as excinfo:
        sender.send_it(receiver.address)

    # Should be empty, not the callee's raw revert data
    assert excinfo.value.data == b""


def test_send_success_no_revert(get_contract, env):
    """Test that send() succeeds when the callee accepts the ether."""
    receiver_src = """
@external
@payable
def __default__():
    pass
    """
    sender_src = """
counter: public(uint256)

@external
def send_it(b: address) -> bool:
    send(b, 10)
    self.counter = self.counter + 1
    return True
    """
    receiver = get_contract(receiver_src)
    sender = get_contract(sender_src)
    env.set_balance(sender.address, 100)

    result = sender.send_it(receiver.address)
    assert result is True
    assert sender.counter() == 1

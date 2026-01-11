"""
Test delegatecall value transfer behavior.

According to EVM spec:
- DELEGATECALL should NOT transfer value
- msg.value should be inherited from the parent context
- Only CALL should transfer value
"""

import pytest


class TestDelegatecallNoValueTransfer:
    """Tests verifying that DELEGATECALL does not transfer value."""

    def test_delegatecall_does_not_transfer_value(self, get_contract, env):
        """DELEGATECALL should not transfer value, only inherit msg.value."""
        target_src = """
@external
@payable
def dummy():
    pass
        """

        caller_src = """
@external
@payable
def call_with_delegate(target: address):
    raw_call(
        target,
        method_id("dummy()"),
        is_delegate_call=True
    )
        """

        target = get_contract(target_src)
        caller = get_contract(caller_src)

        eoa = env.deployer
        env.set_balance(eoa, 1 * 10**18)
        env.set_balance(caller.address, 0)

        caller.call_with_delegate(target.address, value=1 * 10**18)

        assert env.get_balance(eoa) == 0
        assert env.get_balance(caller.address) == 1 * 10**18

    def test_delegatecall_with_zero_caller_balance(self, get_contract, env):
        """DELEGATECALL should succeed even when caller has zero balance."""
        target_src = """
@external
@payable
def get_msg_value() -> uint256:
    return msg.value
        """

        caller_src = """
@external
@payable
def delegate_get_value(target: address) -> uint256:
    response: Bytes[32] = raw_call(
        target,
        method_id("get_msg_value()"),
        max_outsize=32,
        is_delegate_call=True
    )
    return abi_decode(response, uint256)
        """

        target = get_contract(target_src)
        caller = get_contract(caller_src)

        eoa = env.deployer
        env.set_balance(eoa, 1 * 10**18)
        env.set_balance(caller.address, 0)

        # Should succeed - no value transfer happens during delegatecall
        result = caller.delegate_get_value(target.address, value=1 * 10**18)

        assert result == 1 * 10**18  # msg.value is inherited
        assert env.get_balance(caller.address) == 1 * 10**18

    def test_delegatecall_preserves_msg_value(self, get_contract, env):
        """DELEGATECALL should preserve msg.value from parent context."""
        target_src = """
@external
@payable
def check_value() -> uint256:
    return msg.value
        """

        caller_src = """
@external
@payable
def delegate_check(target: address) -> uint256:
    response: Bytes[32] = raw_call(
        target,
        method_id("check_value()"),
        max_outsize=32,
        is_delegate_call=True
    )
    return abi_decode(response, uint256)
        """

        target = get_contract(target_src)
        caller = get_contract(caller_src)

        eoa = env.deployer
        env.set_balance(eoa, 5 * 10**18)

        result = caller.delegate_check(target.address, value=3 * 10**18)

        assert result == 3 * 10**18


class TestRegularCallValueTransfer:
    """Tests verifying that regular CALL does transfer value."""

    def test_call_transfers_value(self, get_contract, env):
        """Regular CALL should transfer value to target."""
        target_src = """
received: public(uint256)

@external
@payable
def receive_eth() -> uint256:
    self.received = msg.value
    return msg.value
        """

        caller_src = """
@external
@payable
def forward_call(target: address, amount: uint256) -> uint256:
    response: Bytes[32] = raw_call(
        target,
        method_id("receive_eth()"),
        max_outsize=32,
        value=amount
    )
    return abi_decode(response, uint256)
        """

        target = get_contract(target_src)
        caller = get_contract(caller_src)

        eoa = env.deployer
        env.set_balance(eoa, 2 * 10**18)
        env.set_balance(caller.address, 0)
        env.set_balance(target.address, 0)

        caller.forward_call(target.address, 5 * 10**17, value=1 * 10**18)

        assert env.get_balance(eoa) == 1 * 10**18
        assert env.get_balance(caller.address) == 5 * 10**17
        assert env.get_balance(target.address) == 5 * 10**17
        assert target.received() == 5 * 10**17

    def test_call_with_full_value_forward(self, get_contract, env):
        """CALL can forward all received value."""
        target_src = """
@external
@payable
def sink():
    pass
        """

        caller_src = """
@external
@payable
def forward_all(target: address):
    raw_call(target, method_id("sink()"), value=msg.value)
        """

        target = get_contract(target_src)
        caller = get_contract(caller_src)

        eoa = env.deployer
        env.set_balance(eoa, 1 * 10**18)
        env.set_balance(caller.address, 0)
        env.set_balance(target.address, 0)

        caller.forward_all(target.address, value=1 * 10**18)

        assert env.get_balance(eoa) == 0
        assert env.get_balance(caller.address) == 0
        assert env.get_balance(target.address) == 1 * 10**18


class TestMixedCallPatterns:
    """Tests for mixed call/delegatecall patterns."""

    def test_delegatecall_then_call_with_value(self, get_contract, env):
        """Value transfer should work from within delegatecall context."""
        receiver_src = """
received: public(uint256)

@external
@payable
def __default__():
    self.received = msg.value
        """

        logic_src = """
@external
@payable
def send_value(to: address, amount: uint256):
    raw_call(to, b"", value=amount)
        """

        proxy_src = """
@external
@payable
def delegate_send(logic: address, receiver: address, amount: uint256):
    raw_call(
        logic,
        abi_encode(receiver, amount, method_id=method_id("send_value(address,uint256)")),
        is_delegate_call=True
    )
        """

        receiver = get_contract(receiver_src)
        logic = get_contract(logic_src)
        proxy = get_contract(proxy_src)

        eoa = env.deployer
        env.set_balance(eoa, 2 * 10**18)
        env.set_balance(proxy.address, 0)
        env.set_balance(receiver.address, 0)

        # Send 1 ETH to proxy, which delegatecalls logic to send 0.5 ETH to receiver
        proxy.delegate_send(logic.address, receiver.address, 5 * 10**17, value=1 * 10**18)

        assert env.get_balance(proxy.address) == 5 * 10**17
        assert env.get_balance(receiver.address) == 5 * 10**17

    def test_call_then_delegatecall(self, get_contract, env):
        """msg.value should be correct through call -> delegatecall chain."""
        inner_src = """
@external
@payable
def get_value() -> uint256:
    return msg.value
        """

        middle_src = """
@external
@payable
def delegate_to(target: address) -> uint256:
    response: Bytes[32] = raw_call(
        target,
        method_id("get_value()"),
        max_outsize=32,
        is_delegate_call=True
    )
    return abi_decode(response, uint256)
        """

        outer_src = """
@external
@payable
def call_middle(middle: address, inner: address, amount: uint256) -> uint256:
    response: Bytes[32] = raw_call(
        middle,
        abi_encode(inner, method_id=method_id("delegate_to(address)")),
        max_outsize=32,
        value=amount
    )
    return abi_decode(response, uint256)
        """

        inner = get_contract(inner_src)
        middle = get_contract(middle_src)
        outer = get_contract(outer_src)

        eoa = env.deployer
        env.set_balance(eoa, 2 * 10**18)

        # outer calls middle with 0.5 ETH, middle delegatecalls inner
        result = outer.call_middle(middle.address, inner.address, 5 * 10**17, value=1 * 10**18)

        # inner sees the 0.5 ETH that was sent to middle
        assert result == 5 * 10**17

    def test_nested_delegatecall_preserves_value(self, get_contract, env):
        """Nested delegatecalls should all see the same msg.value."""
        inner_src = """
@external
@payable
def get_value() -> uint256:
    return msg.value
        """

        middle_src = """
@external
@payable
def delegate_inner(target: address) -> uint256:
    response: Bytes[32] = raw_call(
        target,
        method_id("get_value()"),
        max_outsize=32,
        is_delegate_call=True
    )
    return abi_decode(response, uint256)
        """

        outer_src = """
@external
@payable
def delegate_middle(middle: address, inner: address) -> uint256:
    response: Bytes[32] = raw_call(
        middle,
        abi_encode(inner, method_id=method_id("delegate_inner(address)")),
        max_outsize=32,
        is_delegate_call=True
    )
    return abi_decode(response, uint256)
        """

        inner = get_contract(inner_src)
        middle = get_contract(middle_src)
        outer = get_contract(outer_src)

        eoa = env.deployer
        env.set_balance(eoa, 2 * 10**18)

        result = outer.delegate_middle(middle.address, inner.address, value=1 * 10**18)

        # All delegatecalls see the original msg.value
        assert result == 1 * 10**18


class TestDelegatecallBalanceRequirements:
    """Tests for balance requirements with delegatecall."""

    def test_delegatecall_no_balance_needed_for_value(self, get_contract, env):
        """Delegatecall doesn't need balance to handle msg.value."""
        target_src = """
@external
@payable
def echo_value() -> uint256:
    return msg.value
        """

        caller_src = """
@external
@payable
def delegate_echo(target: address) -> uint256:
    response: Bytes[32] = raw_call(
        target,
        method_id("echo_value()"),
        max_outsize=32,
        is_delegate_call=True
    )
    return abi_decode(response, uint256)
        """

        target = get_contract(target_src)
        caller = get_contract(caller_src)

        eoa = env.deployer
        # Give EOA exactly the amount to send - no extra
        env.set_balance(eoa, 1 * 10**18)
        env.set_balance(caller.address, 0)

        # This works because delegatecall doesn't transfer value
        result = caller.delegate_echo(target.address, value=1 * 10**18)

        assert result == 1 * 10**18
        assert env.get_balance(caller.address) == 1 * 10**18

    def test_call_after_delegatecall_uses_caller_balance(self, get_contract, env):
        """CALL from delegatecall context uses the caller's balance."""
        receiver_src = """
@external
@payable
def __default__():
    pass
        """

        logic_src = """
@external
def send_from_balance(to: address, amount: uint256):
    raw_call(to, b"", value=amount)
        """

        proxy_src = """
@external
@payable
def delegate_send(logic: address, receiver: address, amount: uint256):
    raw_call(
        logic,
        abi_encode(receiver, amount, method_id=method_id("send_from_balance(address,uint256)")),
        is_delegate_call=True
    )
        """

        receiver = get_contract(receiver_src)
        logic = get_contract(logic_src)
        proxy = get_contract(proxy_src)

        eoa = env.deployer
        # Give proxy some initial balance
        env.set_balance(proxy.address, 1 * 10**18)
        env.set_balance(receiver.address, 0)

        # No value sent, but proxy has balance that logic can spend via delegatecall
        proxy.delegate_send(logic.address, receiver.address, 5 * 10**17)

        assert env.get_balance(proxy.address) == 5 * 10**17
        assert env.get_balance(receiver.address) == 5 * 10**17

"""Tests for create_minimal_proxy_to payability - Bug #8 regression tests.

These tests verify that create_minimal_proxy_to correctly handles the value parameter,
allowing ETH to be transferred to the newly created minimal proxy contract.
"""
import pytest

from ivy.frontend.loader import loads
from ivy.types import Address


def test_create_minimal_proxy_with_zero_value(get_contract, env):
    """Baseline: create_minimal_proxy_to with value=0 should work (existing behavior)."""
    src = """
interface Self:
    def get_balance() -> uint256: view

@external
def get_balance() -> uint256:
    return self.balance

@external
def create_proxy() -> address:
    return create_minimal_proxy_to(self)
    """
    c = get_contract(src)
    proxy_addr = c.create_proxy()

    assert proxy_addr != Address(0)
    assert proxy_addr != c.address
    assert env.get_balance(proxy_addr) == 0


def test_create_minimal_proxy_with_value(get_contract, env):
    """create_minimal_proxy_to with value > 0 should transfer ETH to new proxy."""
    src = """
interface Self:
    def get_balance() -> uint256: view

@external
def get_balance() -> uint256:
    return self.balance

@external
@payable
def create_proxy_with_value() -> uint256:
    proxy: address = create_minimal_proxy_to(self, value=msg.value)
    return staticcall Self(proxy).get_balance()
    """
    c = get_contract(src)
    env.set_balance(env.deployer, 10000)

    result = c.create_proxy_with_value(value=5000)

    assert result == 5000


def test_create_minimal_proxy_value_from_contract_balance(get_contract, env):
    """create_minimal_proxy_to using contract's existing balance."""
    src = """
interface Self:
    def get_balance() -> uint256: view

@external
def get_balance() -> uint256:
    return self.balance

@external
def create_proxy_with_own_balance() -> (uint256, uint256):
    initial_balance: uint256 = self.balance
    proxy: address = create_minimal_proxy_to(self, value=1000)
    proxy_balance: uint256 = staticcall Self(proxy).get_balance()
    return proxy_balance, initial_balance - self.balance
    """
    c = get_contract(src)
    env.set_balance(c.address, 5000)

    proxy_balance, transferred = c.create_proxy_with_own_balance()

    assert proxy_balance == 1000
    assert transferred == 1000
    assert env.get_balance(c.address) == 4000


def test_create_minimal_proxy_exact_balance(get_contract, env):
    """create_minimal_proxy_to with value equal to entire contract balance."""
    src = """
interface Self:
    def get_balance() -> uint256: view

@external
def get_balance() -> uint256:
    return self.balance

@external
def create_proxy_with_all_balance() -> uint256:
    proxy: address = create_minimal_proxy_to(self, value=self.balance)
    return staticcall Self(proxy).get_balance()
    """
    c = get_contract(src)
    env.set_balance(c.address, 3000)

    result = c.create_proxy_with_all_balance()

    assert result == 3000
    assert env.get_balance(c.address) == 0


def test_create_minimal_proxy_value_with_salt(get_contract, env):
    """create_minimal_proxy_to with value and salt (CREATE2) should work."""
    src = """
interface Self:
    def get_balance() -> uint256: view

@external
def get_balance() -> uint256:
    return self.balance

@external
@payable
def create_proxy_with_salt(salt: bytes32) -> uint256:
    proxy: address = create_minimal_proxy_to(self, value=msg.value, salt=salt)
    return staticcall Self(proxy).get_balance()
    """
    c = get_contract(src)
    env.set_balance(env.deployer, 10000)

    salt = b'\x01' * 32
    result = c.create_proxy_with_salt(salt, value=2500)

    assert result == 2500


def test_create_minimal_proxy_value_with_revert_on_failure_false(get_contract, env):
    """create_minimal_proxy_to with value and revert_on_failure=False."""
    src = """
interface Self:
    def get_balance() -> uint256: view

@external
def get_balance() -> uint256:
    return self.balance

@external
def create_proxy_no_revert() -> (address, uint256):
    # Contract has enough balance
    proxy: address = create_minimal_proxy_to(self, value=1000, revert_on_failure=False)
    if proxy == empty(address):
        return proxy, 0
    proxy_balance: uint256 = staticcall Self(proxy).get_balance()
    return proxy, proxy_balance
    """
    c = get_contract(src)
    env.set_balance(c.address, 2000)

    proxy_addr, proxy_balance = c.create_proxy_no_revert()

    assert proxy_addr != Address(0)
    assert proxy_balance == 1000


def test_create_minimal_proxy_insufficient_balance_reverts(get_contract, env, tx_failed):
    """create_minimal_proxy_to with insufficient balance should revert."""
    from ivy.exceptions import EVMException

    src = """
@external
def create_proxy_with_too_much_value() -> address:
    return create_minimal_proxy_to(self, value=10000)
    """
    c = get_contract(src)
    env.set_balance(c.address, 100)  # Not enough

    with tx_failed(EVMException):
        c.create_proxy_with_too_much_value()


def test_create_minimal_proxy_insufficient_balance_no_revert(get_contract, env):
    """create_minimal_proxy_to with insufficient balance and revert_on_failure=False returns zero address."""
    src = """
@external
def create_proxy_insufficient_no_revert() -> address:
    return create_minimal_proxy_to(self, value=10000, revert_on_failure=False)
    """
    c = get_contract(src)
    env.set_balance(c.address, 100)  # Not enough

    result = c.create_proxy_insufficient_no_revert()

    assert result == Address(0)


def test_create_minimal_proxy_value_multiple_creations(get_contract, env):
    """Multiple create_minimal_proxy_to calls with different values."""
    src = """
interface Self:
    def get_balance() -> uint256: view

@external
def get_balance() -> uint256:
    return self.balance

@external
def create_multiple_proxies() -> (uint256, uint256, uint256):
    proxy1: address = create_minimal_proxy_to(self, value=100)
    proxy2: address = create_minimal_proxy_to(self, value=200)
    proxy3: address = create_minimal_proxy_to(self, value=300)

    bal1: uint256 = staticcall Self(proxy1).get_balance()
    bal2: uint256 = staticcall Self(proxy2).get_balance()
    bal3: uint256 = staticcall Self(proxy3).get_balance()

    return bal1, bal2, bal3
    """
    c = get_contract(src)
    env.set_balance(c.address, 1000)

    bal1, bal2, bal3 = c.create_multiple_proxies()

    assert bal1 == 100
    assert bal2 == 200
    assert bal3 == 300
    assert env.get_balance(c.address) == 400  # 1000 - 100 - 200 - 300


def test_create_minimal_proxy_value_proxy_can_send(get_contract, env):
    """Proxy created with value can send ETH to other addresses."""
    receiver_src = """
received: public(uint256)

@external
@payable
def receive_eth():
    self.received = msg.value
    """

    src = """
interface Receiver:
    def receive_eth(): payable

interface Self:
    def send_eth(receiver: address, amount: uint256): nonpayable

@external
def send_eth(receiver: address, amount: uint256):
    extcall Receiver(receiver).receive_eth(value=amount)

@external
def create_proxy_and_send(receiver: address) -> uint256:
    proxy: address = create_minimal_proxy_to(self, value=500)
    # Call proxy to send some ETH
    extcall Self(proxy).send_eth(receiver, 200)
    return self.balance
    """
    receiver = get_contract(receiver_src)
    c = get_contract(src)
    env.set_balance(c.address, 1000)

    remaining = c.create_proxy_and_send(receiver.address)

    assert remaining == 500  # 1000 - 500 sent to proxy
    assert receiver.received() == 200


def test_create_minimal_proxy_value_one_wei(get_contract, env):
    """Edge case: create_minimal_proxy_to with value=1 (smallest possible)."""
    src = """
interface Self:
    def get_balance() -> uint256: view

@external
def get_balance() -> uint256:
    return self.balance

@external
def create_proxy_one_wei() -> uint256:
    proxy: address = create_minimal_proxy_to(self, value=1)
    return staticcall Self(proxy).get_balance()
    """
    c = get_contract(src)
    env.set_balance(c.address, 10)

    result = c.create_proxy_one_wei()

    assert result == 1


def test_create_minimal_proxy_call_with_value(get_contract, env):
    """Call proxy's __default__ with msg.value to exercise payable fallback."""
    src = """
interface Self:
    def get_balance() -> uint256: view
    def receive_eth(): payable

@external
def get_balance() -> uint256:
    return self.balance

@external
@payable
def receive_eth():
    pass  # Just receive ETH

@external
def create_proxy_and_send_value() -> (uint256, uint256):
    proxy: address = create_minimal_proxy_to(self, value=100)
    initial_balance: uint256 = staticcall Self(proxy).get_balance()
    # Call proxy with value to test payable __default__
    extcall Self(proxy).receive_eth(value=200)
    final_balance: uint256 = staticcall Self(proxy).get_balance()
    return initial_balance, final_balance
    """
    c = get_contract(src)
    env.set_balance(c.address, 1000)

    initial, final = c.create_proxy_and_send_value()

    assert initial == 100
    assert final == 300  # 100 from create + 200 from call


def test_create_minimal_proxy_value_large_amount(get_contract, env):
    """create_minimal_proxy_to with a large value (many ether)."""
    src = """
interface Self:
    def get_balance() -> uint256: view

@external
def get_balance() -> uint256:
    return self.balance

@external
def create_proxy_large_value() -> uint256:
    proxy: address = create_minimal_proxy_to(self, value=10**18)  # 1 ether
    return staticcall Self(proxy).get_balance()
    """
    c = get_contract(src)
    env.set_balance(c.address, 10**19)  # 10 ether

    result = c.create_proxy_large_value()

    assert result == 10**18

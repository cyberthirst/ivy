"""
Tests for selfdestruct builtin (EIP-6780 semantics).

Post-Cancun: selfdestruct only deletes account if created in same transaction.
"""
import pytest
from ivy.types import Address


def test_selfdestruct_transfers_balance(get_contract, env):
    """Selfdestruct transfers all balance to beneficiary."""
    receiver_src = """
@external
@payable
def __default__():
    pass
    """
    destroyer_src = """
@external
def destroy(recipient: address):
    selfdestruct(recipient)
    """
    receiver = get_contract(receiver_src)
    destroyer = get_contract(destroyer_src)

    env.set_balance(destroyer.address, 1000)
    initial_receiver_balance = env.get_balance(receiver.address)

    destroyer.destroy(receiver.address)

    assert env.get_balance(receiver.address) == initial_receiver_balance + 1000
    assert env.get_balance(destroyer.address) == 0


def test_selfdestruct_halts_execution(get_contract, env):
    """Code after selfdestruct is not executed (verified via state not being set)."""
    receiver_src = """
@external
@payable
def __default__():
    pass
    """
    # Use a conditional to avoid Vyper's unreachable code detection
    destroyer_src = """
counter: public(uint256)

@external
def destroy_conditional(recipient: address, should_destroy: bool):
    if should_destroy:
        selfdestruct(recipient)
    self.counter = 999
    """
    receiver = get_contract(receiver_src)
    destroyer = get_contract(destroyer_src)

    # When should_destroy=True, selfdestruct halts, counter stays 0
    destroyer.destroy_conditional(receiver.address, True)
    assert destroyer.counter() == 0

    # When should_destroy=False, counter gets set
    destroyer2 = get_contract(destroyer_src)
    destroyer2.destroy_conditional(receiver.address, False)
    assert destroyer2.counter() == 999


def test_selfdestruct_existing_contract_no_delete(get_contract, env):
    """Pre-existing contract selfdestruct only transfers balance, doesn't delete (EIP-6780)."""
    receiver_src = """
@external
@payable
def __default__():
    pass
    """
    destroyer_src = """
value: public(uint256)

@deploy
def __init__():
    self.value = 42

@external
def destroy(recipient: address):
    selfdestruct(recipient)

@external
def get_value() -> uint256:
    return self.value
    """
    receiver = get_contract(receiver_src)
    destroyer = get_contract(destroyer_src)

    # Verify contract exists and has state
    assert destroyer.get_value() == 42
    env.set_balance(destroyer.address, 1000)

    # Selfdestruct in a new transaction (contract was created in previous tx)
    destroyer.destroy(receiver.address)

    # Balance should be transferred
    assert env.get_balance(destroyer.address) == 0

    # But contract should still exist (EIP-6780)
    # Storage should still be accessible
    assert destroyer.get_value() == 42


def test_selfdestruct_to_self_existing_contract(get_contract, env):
    """Selfdestruct to self on pre-existing contract: balance unchanged (EIP-6780)."""
    destroyer_src = """
@external
def destroy_to_self():
    selfdestruct(self)
    """
    destroyer = get_contract(destroyer_src)

    initial_balance = 1000
    env.set_balance(destroyer.address, initial_balance)

    destroyer.destroy_to_self()

    # Balance should remain unchanged for pre-existing contracts
    # (move_ether to self is a no-op, and account not deleted per EIP-6780)
    assert env.get_balance(destroyer.address) == 1000


def test_selfdestruct_to_nonexistent_creates_account(get_contract, env):
    """Selfdestruct to non-existent address creates it with balance."""
    destroyer_src = """
@external
def destroy(recipient: address):
    selfdestruct(recipient)
    """
    destroyer = get_contract(destroyer_src)

    # Use a fresh address that doesn't exist
    nonexistent = Address(0xDEADBEEF)
    assert env.get_balance(nonexistent) == 0

    env.set_balance(destroyer.address, 500)

    destroyer.destroy(nonexistent)

    assert env.get_balance(nonexistent) == 500


def test_selfdestruct_zero_balance(get_contract, env):
    """Selfdestruct with zero balance works correctly."""
    receiver_src = """
@external
@payable
def __default__():
    pass
    """
    destroyer_src = """
@external
def destroy(recipient: address):
    selfdestruct(recipient)
    """
    receiver = get_contract(receiver_src)
    destroyer = get_contract(destroyer_src)

    # Ensure destroyer has zero balance
    assert env.get_balance(destroyer.address) == 0
    initial_receiver = env.get_balance(receiver.address)

    destroyer.destroy(receiver.address)

    # Receiver balance unchanged
    assert env.get_balance(receiver.address) == initial_receiver


def test_selfdestruct_in_constructor(get_contract, env):
    """Selfdestruct in constructor destroys contract during deployment."""
    receiver_src = """
@external
@payable
def __default__():
    pass
    """
    receiver = get_contract(receiver_src)

    # Contract that selfdestructs in constructor
    destroyer_src = f"""
@deploy
@payable
def __init__():
    selfdestruct({receiver.address})
    """
    initial_receiver = env.get_balance(receiver.address)

    # Set balance on default deployer address first
    deployer = env.deployer
    env.set_balance(deployer, 10000)

    # Deploy with value - should selfdestruct in constructor
    destroyer = get_contract(destroyer_src, value=1000)

    # Balance should be at receiver
    assert env.get_balance(receiver.address) == initial_receiver + 1000

    # Contract was created and destroyed in same tx, so it should be deleted
    # Accessing it should return default values
    assert env.get_balance(destroyer.address) == 0


def test_selfdestruct_nested_via_raw_call(get_contract, env):
    """Contract called via raw_call can selfdestruct."""
    receiver_src = """
@external
@payable
def __default__():
    pass
    """
    destroyer_src = """
@external
def destroy(recipient: address):
    selfdestruct(recipient)
    """
    caller_src = """
interface Destroyer:
    def destroy(recipient: address): nonpayable

@external
def call_destroy(target: address, recipient: address):
    extcall Destroyer(target).destroy(recipient)
    """
    receiver = get_contract(receiver_src)
    destroyer = get_contract(destroyer_src)
    caller = get_contract(caller_src)

    env.set_balance(destroyer.address, 750)

    caller.call_destroy(destroyer.address, receiver.address)

    # Balance should be transferred
    assert env.get_balance(destroyer.address) == 0
    assert env.get_balance(receiver.address) == 750


def test_selfdestruct_reverted_restores_balance(get_contract, env):
    """Reverted selfdestruct restores original balance."""
    receiver_src = """
@external
@payable
def __default__():
    pass
    """
    destroyer_src = """
@external
def destroy(recipient: address):
    selfdestruct(recipient)
    """
    # Use a flag to force revert dynamically
    caller_src = """
interface Destroyer:
    def destroy(recipient: address): nonpayable

@external
def call_destroy_then_revert(target: address, recipient: address, should_fail: bool):
    extcall Destroyer(target).destroy(recipient)
    assert should_fail == False  # Will fail if should_fail is True
    """
    receiver = get_contract(receiver_src)
    destroyer = get_contract(destroyer_src)
    caller = get_contract(caller_src)

    env.set_balance(destroyer.address, 1000)
    initial_receiver = env.get_balance(receiver.address)

    # This should revert, rolling back the selfdestruct
    with pytest.raises(Exception):
        caller.call_destroy_then_revert(destroyer.address, receiver.address, True)

    # Balance should be restored
    assert env.get_balance(destroyer.address) == 1000
    assert env.get_balance(receiver.address) == initial_receiver


def test_selfdestruct_in_raw_call_success(get_contract, env):
    """Selfdestruct in subcall succeeds normally."""
    receiver_src = """
@external
@payable
def __default__():
    pass
    """
    destroyer_src = """
@external
def destroy(recipient: address):
    selfdestruct(recipient)
    """
    caller_src = """
@external
def try_destroy(target: address, recipient: address) -> bool:
    success: bool = raw_call(
        target,
        abi_encode(recipient, method_id=method_id("destroy(address)")),
        revert_on_failure=False
    )
    return success
    """
    receiver = get_contract(receiver_src)
    destroyer = get_contract(destroyer_src)
    caller = get_contract(caller_src)

    env.set_balance(destroyer.address, 1000)
    initial_receiver = env.get_balance(receiver.address)

    # Call should succeed (selfdestruct is a clean halt, not an error)
    success = caller.try_destroy(destroyer.address, receiver.address)
    assert success is True

    # Balance should be transferred
    assert env.get_balance(destroyer.address) == 0
    assert env.get_balance(receiver.address) == initial_receiver + 1000


def test_selfdestruct_constructor_deletes_account_eip6780(get_contract, env):
    """
    EIP-6780 regression: selfdestruct in constructor MUST delete account.

    The created_accounts set must be populated for EIP-6780 to work.
    This tests that selfdestructing in the same transaction as creation
    actually deletes the account (code and nonce are cleared).
    """
    receiver_src = """
@external
@payable
def __default__():
    pass
    """
    receiver = get_contract(receiver_src)

    # Contract that selfdestructs in constructor
    destroyer_src = f"""
value: public(uint256)

@deploy
@payable
def __init__():
    self.value = 12345
    selfdestruct({receiver.address})
    """
    deployer = env.deployer
    env.set_balance(deployer, 10000)

    # Deploy with value - selfdestructs in constructor
    destroyer = get_contract(destroyer_src, value=1000)

    # Key assertions for EIP-6780 compliance:
    # 1. Account should be deleted - no code
    assert env.state.get_code(destroyer.address) is None

    # 2. Account nonce should be 0 (deleted account)
    assert env.state.get_nonce(destroyer.address) == 0

    # 3. Balance should be 0 (transferred to receiver)
    assert env.get_balance(destroyer.address) == 0

    # 4. has_account should return False for deleted account
    assert not env.state.has_account(destroyer.address)


def test_selfdestruct_post_deployment_preserves_account_eip6780(get_contract, env):
    """
    EIP-6780 regression: selfdestruct AFTER deployment must NOT delete account.

    Post-Cancun, selfdestruct only transfers balance but preserves the account
    when called in a different transaction than creation.
    """
    receiver_src = """
@external
@payable
def __default__():
    pass
    """
    destroyer_src = """
value: public(uint256)

@deploy
def __init__():
    self.value = 99999

@external
def destroy(recipient: address):
    selfdestruct(recipient)

@external
def get_value() -> uint256:
    return self.value
    """
    receiver = get_contract(receiver_src)
    destroyer = get_contract(destroyer_src)

    # Verify contract exists with code and state
    assert destroyer.get_value() == 99999
    assert env.state.get_code(destroyer.address) is not None

    env.set_balance(destroyer.address, 1000)

    # Selfdestruct in a new transaction (different from creation)
    destroyer.destroy(receiver.address)

    # Key assertions for EIP-6780 compliance:
    # 1. Code should still exist (account NOT deleted)
    assert env.state.get_code(destroyer.address) is not None

    # 2. Nonce should be non-zero (account preserved)
    assert env.state.get_nonce(destroyer.address) > 0

    # 3. Balance should be transferred (to 0)
    assert env.get_balance(destroyer.address) == 0

    # 4. has_account should return True (account still exists)
    assert env.state.has_account(destroyer.address)

    # 5. Storage should still be accessible
    assert destroyer.get_value() == 99999


def test_selfdestruct_constructor_balance_transfer(get_contract, env):
    """
    EIP-6780: selfdestruct in constructor transfers balance to beneficiary.
    """
    receiver_src = """
@external
@payable
def __default__():
    pass
    """
    receiver = get_contract(receiver_src)
    initial_receiver_balance = env.get_balance(receiver.address)

    # Contract that selfdestructs to receiver in constructor
    destroyer_src = f"""
@deploy
@payable
def __init__():
    selfdestruct({receiver.address})
    """
    deployer = env.deployer
    env.set_balance(deployer, 5000)

    # Deploy with 2000 wei
    destroyer = get_contract(destroyer_src, value=2000)

    # Receiver should have received the 2000 wei
    assert env.get_balance(receiver.address) == initial_receiver_balance + 2000

    # Destroyed account should have 0 balance
    assert env.get_balance(destroyer.address) == 0


def test_selfdestruct_to_self_in_constructor_burns_ether(get_contract, env):
    """
    EIP-6780: selfdestruct to self in constructor burns ether.

    When beneficiary == originator and account is deleted, ether is burnt.
    """
    # Contract that selfdestructs to itself in constructor
    destroyer_src = """
@deploy
@payable
def __init__():
    selfdestruct(self)
    """
    deployer = env.deployer
    env.set_balance(deployer, 5000)

    destroyer = get_contract(destroyer_src, value=1000)

    # Account should be deleted (created and destroyed in same tx)
    assert env.state.get_code(destroyer.address) is None
    assert env.state.get_nonce(destroyer.address) == 0

    # Balance should be 0 (ether burnt because beneficiary == originator and account deleted)
    assert env.get_balance(destroyer.address) == 0


def test_factory_create_child_selfdestruct_same_tx(get_contract, env):
    """
    EIP-6780 regression: factory creates child via CREATE, child selfdestructs in same tx.

    This tests that created_accounts is properly populated for nested CREATE operations.
    When factory deploys a child and then calls the child's selfdestruct in the same
    transaction, the child SHOULD be deleted (created in same tx).
    """
    receiver_src = """
@external
@payable
def __default__():
    pass
    """
    receiver = get_contract(receiver_src)

    child_src = """
@external
def destroy(recipient: address):
    selfdestruct(recipient)
    """
    child_template = get_contract(child_src)

    # Factory that deploys child and immediately calls selfdestruct on it
    factory_src = f"""
interface Child:
    def destroy(recipient: address): nonpayable

@external
@payable
def deploy_and_destroy(recipient: address) -> address:
    child: address = create_copy_of({child_template.address}, value=msg.value)
    extcall Child(child).destroy(recipient)
    return child
    """
    factory = get_contract(factory_src)

    # Fund factory
    env.set_balance(env.deployer, 10000)
    initial_receiver_balance = env.get_balance(receiver.address)

    # Deploy child and destroy it in same transaction
    child_address = factory.deploy_and_destroy(receiver.address, value=1000)

    # Child was created and destroyed in same tx, so it should be deleted
    assert env.state.get_code(child_address) is None
    assert env.state.get_nonce(child_address) == 0
    assert env.get_balance(child_address) == 0

    # Balance should be transferred to receiver
    assert env.get_balance(receiver.address) == initial_receiver_balance + 1000


def test_factory_create2_child_selfdestruct_same_tx(get_contract, env):
    """
    EIP-6780 regression: factory creates child via CREATE2, child selfdestructs in same tx.

    Similar to CREATE test but using CREATE2 with salt. The created_accounts
    tracking should work the same for both opcodes.
    """
    receiver_src = """
@external
@payable
def __default__():
    pass
    """
    receiver = get_contract(receiver_src)

    child_src = """
@external
def destroy(recipient: address):
    selfdestruct(recipient)
    """
    child_template = get_contract(child_src)

    # Factory using CREATE2 (salt parameter)
    factory_src = f"""
interface Child:
    def destroy(recipient: address): nonpayable

@external
@payable
def deploy_and_destroy_create2(recipient: address, salt: bytes32) -> address:
    child: address = create_copy_of({child_template.address}, value=msg.value, salt=salt)
    extcall Child(child).destroy(recipient)
    return child
    """
    factory = get_contract(factory_src)

    # Fund factory
    env.set_balance(env.deployer, 10000)
    initial_receiver_balance = env.get_balance(receiver.address)

    # Deploy child with CREATE2 and destroy it in same transaction
    salt = b"\x00" * 31 + b"\x01"  # salt = 1
    child_address = factory.deploy_and_destroy_create2(receiver.address, salt, value=1000)

    # Child was created and destroyed in same tx, so it should be deleted
    assert env.state.get_code(child_address) is None
    assert env.state.get_nonce(child_address) == 0
    assert env.get_balance(child_address) == 0

    # Balance should be transferred to receiver
    assert env.get_balance(receiver.address) == initial_receiver_balance + 1000

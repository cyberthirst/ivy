"""Tests for nonce management fixes.

Tests cover:
1. EIP-161: Newly created contracts start with nonce=1
2. Insufficient balance check before nonce increment
3. Nonce overflow check (2^64 - 1)
4. Nested create address computation with correct nonces
"""

from ivy.types import Address
from ivy.utils import compute_contract_address


# =============================================================================
# EIP-161: Newly created contracts start with nonce=1
# =============================================================================


def test_created_contract_has_nonce_1(env, get_contract):
    """Test that newly created contracts start with nonce=1 per EIP-161."""
    child_src = """
@external
def get_value() -> uint256:
    return 42
    """

    factory_src = """
interface Child:
    def get_value() -> uint256: view

@external
def create_child(target: address) -> address:
    return create_copy_of(target)
    """

    child = get_contract(child_src)
    factory = get_contract(factory_src)

    # Create a new contract via factory
    new_addr = factory.create_child(child.address)

    # Per EIP-161, newly created contract should have nonce=1
    assert env.state.get_nonce(new_addr) == 1, (
        f"Newly created contract should have nonce=1, got {env.state.get_nonce(new_addr)}"
    )


def test_factory_nonce_is_1_after_deploy(env, get_contract):
    """Test that top-level deployed contracts also have nonce=1."""
    src = """
@external
def foo() -> uint256:
    return 42
    """
    c = get_contract(src)

    # Deployed contract should have nonce=1
    assert env.state.get_nonce(c.address) == 1, (
        f"Deployed contract should have nonce=1, got {env.state.get_nonce(c.address)}"
    )


def test_nested_create_address_computation(env, get_contract):
    """Test that nested creates compute addresses correctly with proper nonces.

    When a factory creates a child, the child starts with nonce=1.
    If the child then creates a grandchild, the grandchild address should
    be computed using the child's nonce=1, not nonce=0.
    """
    grandchild_src = """
@external
def id() -> uint256:
    return 999
    """

    child_src = """
interface Grandchild:
    def id() -> uint256: view

created_grandchild: public(address)

@external
def create_grandchild(target: address) -> address:
    self.created_grandchild = create_copy_of(target)
    return self.created_grandchild
    """

    factory_src = """
interface Child:
    def create_grandchild(target: address) -> address: nonpayable
    def created_grandchild() -> address: view

@external
def create_child_and_grandchild(child_template: address, grandchild_template: address) -> (address, address):
    child: address = create_copy_of(child_template)
    grandchild: address = extcall Child(child).create_grandchild(grandchild_template)
    return (child, grandchild)
    """

    grandchild_template = get_contract(grandchild_src)
    child_template = get_contract(child_src)
    factory = get_contract(factory_src)

    child_addr, grandchild_addr = factory.create_child_and_grandchild(
        child_template.address, grandchild_template.address
    )

    # Child should have nonce=1 (EIP-161) + 1 (from creating grandchild) = 2
    assert env.state.get_nonce(child_addr) == 2, (
        f"Child nonce should be 2 after creating grandchild, got {env.state.get_nonce(child_addr)}"
    )

    # Verify grandchild address was computed using child's initial nonce=1
    # The grandchild was created when child had nonce=1
    expected_grandchild = Address(
        compute_contract_address(child_addr.canonical_address, 1)
    )
    assert grandchild_addr == expected_grandchild, (
        f"Grandchild address should be computed with child nonce=1. "
        f"Expected {expected_grandchild}, got {grandchild_addr}"
    )


# =============================================================================
# Insufficient balance check before nonce increment
# =============================================================================


def test_create_insufficient_balance_nonce_not_incremented(env, get_contract):
    """Test that nonce is NOT incremented when CREATE fails due to insufficient balance.

    Per Execution Spec: if sender.balance < endowment, return early without
    incrementing nonce.
    """
    child_src = """
@external
def get_value() -> uint256:
    return 42
    """

    factory_src = """
interface Child:
    def get_value() -> uint256: view

@external
def try_create_with_value(target: address, value_to_send: uint256) -> address:
    return create_copy_of(target, value=value_to_send, revert_on_failure=False)

@external
def get_balance() -> uint256:
    return self.balance
    """

    child = get_contract(child_src)
    factory = get_contract(factory_src)
    factory_addr = factory.address

    # Give factory some balance, but not enough for the create we'll attempt
    env.set_balance(factory_addr, 100)

    # Record nonce before failed create
    nonce_before = env.state.get_nonce(factory_addr)

    # Attempt to create with more value than factory has (should fail, return address(0))
    result = factory.try_create_with_value(child.address, 200)  # Factory only has 100

    # Verify create failed (returned address(0))
    assert result == Address(0), "Create should have failed and returned address(0)"

    # Record nonce after failed create
    nonce_after = env.state.get_nonce(factory_addr)

    # Nonce should NOT be incremented when CREATE fails due to insufficient balance
    assert nonce_after == nonce_before, (
        f"Nonce should NOT be incremented when CREATE fails due to insufficient balance. "
        f"Before: {nonce_before}, After: {nonce_after}"
    )


def test_create_address_consistency_after_failed_create(env, get_contract):
    """Test that after a failed CREATE, the next successful CREATE uses correct address.

    If nonce was incorrectly incremented on failed create, the next contract
    would be deployed to the wrong address.
    """
    child_src = """
@external
def get_value() -> uint256:
    return 42
    """

    factory_src = """
interface Child:
    def get_value() -> uint256: view

@external
def try_create_with_value(target: address, value_to_send: uint256) -> address:
    return create_copy_of(target, value=value_to_send, revert_on_failure=False)

@external
def create_child(target: address) -> address:
    return create_copy_of(target)
    """

    child = get_contract(child_src)
    factory = get_contract(factory_src)
    factory_addr = factory.address

    # Give factory some balance, but not enough for the first create
    env.set_balance(factory_addr, 100)

    # Record nonce before any creates
    nonce_before = env.state.get_nonce(factory_addr)

    # Compute expected address for first successful create (using current nonce)
    expected_addr = Address(
        compute_contract_address(factory_addr.canonical_address, nonce_before)
    )

    # Attempt to create with more value than factory has (should fail)
    failed_result = factory.try_create_with_value(child.address, 200)
    assert failed_result == Address(0), "First create should have failed"

    # Now give factory enough balance and try again
    env.set_balance(factory_addr, 1000)
    success_result = factory.create_child(child.address)

    # The successful create should use the SAME nonce as before (since failed didn't increment)
    assert success_result == expected_addr, (
        f"After failed CREATE, next successful CREATE should use original nonce. "
        f"Expected {expected_addr}, got {success_result}"
    )


def test_create_sufficient_balance_increments_nonce(env, get_contract):
    """Test that successful CREATE properly increments nonce."""
    child_src = """
@external
def get_value() -> uint256:
    return 42
    """

    factory_src = """
interface Child:
    def get_value() -> uint256: view

@external
def create_child(target: address) -> address:
    return create_copy_of(target)
    """

    child = get_contract(child_src)
    factory = get_contract(factory_src)
    factory_addr = factory.address

    nonce_before = env.state.get_nonce(factory_addr)

    # Create a child (should succeed and increment nonce)
    new_addr = factory.create_child(child.address)

    assert new_addr != Address(0), "Create should have succeeded"

    nonce_after = env.state.get_nonce(factory_addr)
    assert nonce_after == nonce_before + 1, (
        f"Nonce should be incremented after successful CREATE. "
        f"Before: {nonce_before}, After: {nonce_after}"
    )


# =============================================================================
# Nonce overflow check (2^64 - 1)
# =============================================================================


def test_create_fails_when_nonce_at_max(env, get_contract):
    """CREATE should fail gracefully when sender nonce is at max (2^64 - 1).

    Per Execution Spec: if sender.nonce == 2^64 - 1, return 0 without
    incrementing nonce (nonce overflow check).
    """
    MAX_NONCE = 2**64 - 1

    child_src = """
@external
def get_value() -> uint256:
    return 42
    """

    factory_src = """
interface Child:
    def get_value() -> uint256: view

@external
def try_create(target: address) -> address:
    return create_copy_of(target, revert_on_failure=False)
    """

    child = get_contract(child_src)
    factory = get_contract(factory_src)
    factory_addr = factory.address

    # Set factory's nonce to max value (access internal state)
    env.interpreter.state._state.state[factory_addr].nonce = MAX_NONCE

    # CREATE should fail and return address(0), not overflow
    result = factory.try_create(child.address)

    # Verify create failed
    assert result == Address(0), "CREATE should fail when nonce is at max"

    # Nonce should remain at MAX_NONCE (not incremented or overflowed)
    assert env.state.get_nonce(factory_addr) == MAX_NONCE, (
        f"Nonce should remain at MAX_NONCE, got {env.state.get_nonce(factory_addr)}"
    )


def test_create_succeeds_when_nonce_below_max(env, get_contract):
    """CREATE should succeed when sender nonce is below max."""
    NEAR_MAX_NONCE = 2**64 - 2  # One below max

    child_src = """
@external
def get_value() -> uint256:
    return 42
    """

    factory_src = """
interface Child:
    def get_value() -> uint256: view

@external
def try_create(target: address) -> address:
    return create_copy_of(target, revert_on_failure=False)
    """

    child = get_contract(child_src)
    factory = get_contract(factory_src)
    factory_addr = factory.address

    # Set factory's nonce to one below max (access internal state)
    env.interpreter.state._state.state[factory_addr].nonce = NEAR_MAX_NONCE

    # CREATE should succeed
    result = factory.try_create(child.address)

    # Verify create succeeded
    assert result != Address(0), "CREATE should succeed when nonce is below max"

    # Nonce should now be at max
    assert env.state.get_nonce(factory_addr) == 2**64 - 1, (
        f"Nonce should be incremented to max, got {env.state.get_nonce(factory_addr)}"
    )


# =============================================================================
# Combined scenarios
# =============================================================================


def test_multiple_creates_correct_addresses(env, get_contract):
    """Test that multiple CREATE operations produce correct addresses."""
    child_src = """
@external
def id() -> uint256:
    return 42
    """

    factory_src = """
@external
def create_multiple(target: address) -> (address, address, address):
    a: address = create_copy_of(target)
    b: address = create_copy_of(target)
    c: address = create_copy_of(target)
    return (a, b, c)
    """

    child = get_contract(child_src)
    factory = get_contract(factory_src)
    factory_addr = factory.address

    # Record factory nonce before creates
    initial_nonce = env.state.get_nonce(factory_addr)

    # Compute expected addresses
    expected_a = Address(compute_contract_address(factory_addr.canonical_address, initial_nonce))
    expected_b = Address(compute_contract_address(factory_addr.canonical_address, initial_nonce + 1))
    expected_c = Address(compute_contract_address(factory_addr.canonical_address, initial_nonce + 2))

    addr_a, addr_b, addr_c = factory.create_multiple(child.address)

    assert addr_a == expected_a, f"First address mismatch: expected {expected_a}, got {addr_a}"
    assert addr_b == expected_b, f"Second address mismatch: expected {expected_b}, got {addr_b}"
    assert addr_c == expected_c, f"Third address mismatch: expected {expected_c}, got {addr_c}"

    # Final nonce should be initial + 3
    assert env.state.get_nonce(factory_addr) == initial_nonce + 3


def test_create_in_constructor_uses_correct_nonce(env, get_contract):
    """Test that CREATE inside __init__ uses the correct nonce (1, not 0)."""
    grandchild_src = """
@external
def id() -> uint256:
    return 999
    """

    # This contract creates a child in its constructor
    factory_src = """
created_child: public(address)

@deploy
def __init__(target: address):
    self.created_child = create_copy_of(target)

@external
def get_child() -> address:
    return self.created_child
    """

    grandchild = get_contract(grandchild_src)

    # Deploy factory which will create a child during construction
    factory = get_contract(factory_src, grandchild.address)
    factory_addr = factory.address

    child_addr = factory.created_child()

    # The factory should have nonce=2 (1 from EIP-161 + 1 from creating child)
    assert env.state.get_nonce(factory_addr) == 2, (
        f"Factory nonce should be 2, got {env.state.get_nonce(factory_addr)}"
    )

    # The child address should be computed using factory's initial nonce=1
    expected_child = Address(compute_contract_address(factory_addr.canonical_address, 1))
    assert child_addr == expected_child, (
        f"Child created in constructor should use nonce=1. "
        f"Expected {expected_child}, got {child_addr}"
    )

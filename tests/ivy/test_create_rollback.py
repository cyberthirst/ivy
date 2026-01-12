"""Tests for CREATE rollback behavior when outer calls revert."""

from ivy.types import Address
from ivy.utils import compute_contract_address


def test_create_success_outer_revert(get_contract, env, tx_failed):
    """When CREATE succeeds but outer call reverts, account should not exist.

    This is a regression test for a bug where nested CREATE operations were
    not properly rolled back when the parent call reverted. The issue was that
    process_create_message directly assigned contract_data instead of using
    the journaled set_code() method.
    """
    blueprint_src = """
value: public(uint256)

@deploy
def __init__(v: uint256):
    self.value = v
    """

    factory_src = """
interface Child:
    def value() -> uint256: view

created_child: public(address)

@external
def create_then_revert(target: address, v: uint256):
    self.created_child = create_from_blueprint(target, v)
    assert False, "outer revert"
    """

    blueprint = get_contract(blueprint_src, 0)
    factory = get_contract(factory_src)
    factory_addr = factory.address

    # Get factory nonce before the call
    nonce_before = env.state.get_nonce(factory_addr)
    expected_child_addr = Address(
        compute_contract_address(factory_addr.canonical_address, nonce_before)
    )

    # Call should revert
    with tx_failed(Exception):
        factory.create_then_revert(blueprint.address, 42)

    # After revert, the child contract should NOT exist
    child_code = env.state.get_code(expected_child_addr)
    assert child_code is None, "Child account should not exist after outer revert"

    # Nonce should also be rolled back
    nonce_after = env.state.get_nonce(factory_addr)
    assert nonce_after == nonce_before, "Factory nonce should be rolled back"


def test_create_success_no_revert(get_contract, env):
    """Sanity check: CREATE should persist when outer call succeeds."""
    blueprint_src = """
value: public(uint256)

@deploy
def __init__(v: uint256):
    self.value = v
    """

    factory_src = """
interface Child:
    def value() -> uint256: view

created_child: public(address)

@external
def create_child(target: address, v: uint256) -> address:
    self.created_child = create_from_blueprint(target, v)
    return self.created_child
    """

    blueprint = get_contract(blueprint_src, 0)
    factory = get_contract(factory_src)

    # Create child - should succeed and persist
    child_addr = factory.create_child(blueprint.address, 42)

    # Child contract should exist
    child_code = env.state.get_code(child_addr)
    assert child_code is not None, "Child should exist after successful create"


def test_nested_create_outer_revert_with_storage(get_contract, env, tx_failed):
    """Nested CREATE with storage writes should all be rolled back on outer revert."""
    blueprint_src = """
value: public(uint256)

@deploy
def __init__(v: uint256):
    self.value = v
    """

    factory_src = """
counter: public(uint256)
created_child: public(address)

@external
def create_modify_then_revert(target: address, v: uint256):
    self.counter = 100
    self.created_child = create_from_blueprint(target, v)
    self.counter = 200
    assert False, "outer revert"
    """

    blueprint = get_contract(blueprint_src, 0)
    factory = get_contract(factory_src)
    factory_addr = factory.address

    nonce_before = env.state.get_nonce(factory_addr)
    expected_child_addr = Address(
        compute_contract_address(factory_addr.canonical_address, nonce_before)
    )

    # Initial counter value
    assert factory.counter() == 0

    with tx_failed(Exception):
        factory.create_modify_then_revert(blueprint.address, 42)

    # All state should be rolled back
    child_code = env.state.get_code(expected_child_addr)
    assert child_code is None, "Child should not exist after revert"
    assert factory.counter() == 0, "Counter should be rolled back to 0"


def test_create_inner_revert_outer_success(get_contract, env):
    """When inner CREATE succeeds, outer call succeeds, child persists."""
    blueprint_src = """
value: public(uint256)

@deploy
def __init__(v: uint256):
    self.value = v
    """

    factory_src = """
success_child: public(address)
counter: public(uint256)

@external
def create_child(target: address, v: uint256) -> bool:
    self.counter = 1
    self.success_child = create_from_blueprint(target, v)
    self.counter = 2
    return True
    """

    blueprint = get_contract(blueprint_src, 0)
    factory = get_contract(factory_src)

    result = factory.create_child(blueprint.address, 42)
    assert result is True

    # Child should exist
    success_addr = factory.success_child()
    assert env.state.get_code(success_addr) is not None

    # Counter should reflect successful execution
    assert factory.counter() == 2

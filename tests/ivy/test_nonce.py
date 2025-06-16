from ivy.types import Address


def test_nonce_increments_on_deploy(env, get_contract):
    # Create a test account
    sender = Address("0x1234567890123456789012345678901234567890")
    env.set_balance(sender, 10**20)  # Give it 100 ETH
    env.eoa = sender

    # Check initial nonce
    initial_nonce = env.state.get_nonce(sender)
    assert initial_nonce == 0, "Initial nonce should be 0"

    # Deploy first contract
    src1 = """
@external
def foo() -> uint256:
    return 1
    """
    c1 = get_contract(src1)

    # Check nonce after first deployment
    nonce_after_first = env.state.get_nonce(sender)
    assert nonce_after_first == 1, "Nonce should be 1 after first deployment"

    # Deploy second contract
    src2 = """
@external
def bar() -> uint256:
    return 2
    """
    c2 = get_contract(src2)

    # Check nonce after second deployment
    nonce_after_second = env.state.get_nonce(sender)
    assert nonce_after_second == 2, "Nonce should be 2 after second deployment"

    # Contracts should have different addresses
    assert c1.address != c2.address, "Contracts should have different addresses"


def test_nonce_increments_on_deployment_transaction(env, get_contract):
    """Test that nonces increment correctly on deployment transactions."""

    # Use a predictable address for testing
    sender = Address("0x3334567890123456789012345678901234567890")
    env.set_balance(sender, 10**20)
    env.eoa = sender

    # Record initial nonce
    initial_nonce = env.state.get_nonce(sender)

    # Deploy multiple contracts and verify they have unique addresses
    addresses = []
    for i in range(5):
        src = f"""
@external
def id() -> uint256:
    return {i}
        """
        c = get_contract(src)
        addresses.append(c.address)

        # Verify nonce incremented
        assert env.state.get_nonce(sender) == initial_nonce + i + 1

    # All addresses should be unique
    assert len(set(addresses)) == len(addresses), (
        "All contract addresses should be unique"
    )


def test_nonce_with_failed_deployment(env, get_contract, tx_failed):
    """Test that nonce still increments even if deployment fails."""

    # Create a test account
    sender = Address("0x4434567890123456789012345678901234567890")
    env.set_balance(sender, 10**20)
    env.eoa = sender

    initial_nonce = env.state.get_nonce(sender)

    # Try to deploy a contract that will fail
    failing_src = """
@deploy
def __init__():
    raise "Deployment failed"
    """

    with tx_failed():
        get_contract(failing_src)

    # Nonce should still have incremented
    assert env.state.get_nonce(sender) == initial_nonce + 1

    # Deploy a successful contract
    src = """
@external
def foo() -> uint256:
    return 42
    """
    c = get_contract(src)

    # Nonce should increment again
    assert env.state.get_nonce(sender) == initial_nonce + 2
    assert c.foo() == 42


def test_nonce_persists_across_calls(env, get_contract):
    """Test that nonces persist correctly across multiple interactions."""

    # Create test accounts
    alice = Address("0x6634567890123456789012345678901234567890")
    bob = Address("0x7734567890123456789012345678901234567890")

    env.set_balance(alice, 10**20)
    env.set_balance(bob, 10**20)

    # Deploy contract as Alice
    env.eoa = alice
    src = """
owner: public(address)

@deploy
def __init__():
    self.owner = msg.sender

@external
def get_sender() -> address:
    return msg.sender
    """

    c = get_contract(src)
    alice_nonce_after_deploy = env.state.get_nonce(alice)
    assert alice_nonce_after_deploy == 1
    assert env.state.get_nonce(bob) == 0  # Bob hasn't done anything yet

    # Alice calls the contract
    assert c.get_sender() == alice
    assert env.state.get_nonce(alice) == 2

    # Bob calls the contract
    env.eoa = bob
    assert c.get_sender() == bob
    assert env.state.get_nonce(bob) == 1
    assert env.state.get_nonce(alice) == 2  # Alice's nonce unchanged

    # Verify contract owner is still Alice
    assert c.owner() == alice


def test_deterministic_contract_addresses(env, get_contract):
    """Test that contract addresses are deterministic based on sender and nonce."""
    from ivy.utils import compute_contract_address

    # Create a test account with known address
    sender = Address("0x8834567890123456789012345678901234567890")
    env.set_balance(sender, 10**20)
    env.eoa = sender

    # Compute expected addresses for nonces 0, 1, 2
    expected_addr_0 = Address(compute_contract_address(sender.canonical_address, 0))
    expected_addr_1 = Address(compute_contract_address(sender.canonical_address, 1))
    expected_addr_2 = Address(compute_contract_address(sender.canonical_address, 2))

    # Deploy first contract (nonce 0)
    src1 = """
@external
def id() -> uint256:
    return 1
    """
    c1 = get_contract(src1)
    assert c1.address == expected_addr_0, (
        f"First contract should be at {expected_addr_0}, got {c1.address}"
    )

    # Deploy second contract (nonce 1)
    src2 = """
@external
def id() -> uint256:
    return 2
    """
    c2 = get_contract(src2)
    assert c2.address == expected_addr_1, (
        f"Second contract should be at {expected_addr_1}, got {c2.address}"
    )

    # Deploy third contract (nonce 2)
    src3 = """
@external
def id() -> uint256:
    return 3
    """
    c3 = get_contract(src3)
    assert c3.address == expected_addr_2, (
        f"Third contract should be at {expected_addr_2}, got {c3.address}"
    )

    # Verify all addresses are different
    assert len({c1.address, c2.address, c3.address}) == 3, (
        "All contract addresses should be unique"
    )

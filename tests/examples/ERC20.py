import pytest

from ivy.frontend.loader import load


def test_ERC20(env):
    # deploy contract while passing data to the __init__ function,
    # and return user-facing frontend repr of the deployed contract for interaction
    contract = load("examples/ERC20.vy", "MyToken", "MTKN", 18, 100 * 10**18)
    # interact with contract
    # - the methods are exposed as attributes on the contract object
    assert contract.name() == "MyToken"
    assert contract.symbol() == "MTKN"
    assert contract.decimals() == 18
    # load 2 accounts from the environment
    alice, bob = env.accounts[:2]
    # alice doesn't have any tokens, catch error
    with pytest.raises(ValueError):
        contract.transfer(bob, 10 * 10**18, sender=alice)
    # mint tokens to alice & transfer to bob
    contract.mint(alice, 100 * 10**18)
    contract.transfer(bob, 10 * 10**18, sender=alice)
    # get logs
    logs = contract.get_logs()
    assert len(logs) == 1
    # sender and receiver are indexed, so access via topics
    assert logs[0].topics[0] == alice and logs[0].topics[1] == bob
    # value is not indexed
    assert logs[0].args[0] == 10 * 10**18
    # check that tokens were transferred
    assert contract.balanceOf(alice) == 90 * 10**18
    assert contract.balanceOf(bob) == 10 * 10**18
    # further contract interaction
    contract.approve(bob, 5 * 10**18, sender=alice)
    # bob transfers 5 tokens from alice to himself
    contract.transferFrom(alice, bob, 5 * 10**18, sender=bob)
    assert contract.balanceOf(alice) == 85 * 10**18
    assert contract.balanceOf(bob) == 15 * 10**18
    # bob transfers 5 tokens from alice to himself
    # but he doesn't have enough allowance
    with pytest.raises(ValueError):
        contract.transferFrom(alice, bob, 5 * 10**18, sender=bob)
    # alice's balance remains the same
    assert contract.balanceOf(alice) == 85 * 10**18

"""Regression test: failed deployment must not leak storage at the create address.

Before the fix, GlobalVariable.__init__ wrote default values directly to the
account's storage dict, bypassing the journal.  When __init__ reverted, the
journal rolled back nonce/code changes but the storage entries survived.
A subsequent deployment to the same address then hit "Address already taken"
because account_has_storage() returned True.
"""

from ivy.frontend.loader import loads
from ivy.types import Address
from ivy.utils import compute_contract_address


def test_failed_deploy_no_storage_leak(env, tx_failed):
    """Storage variables allocated during a failed deploy must be rolled back."""
    src = """
x: uint256
y: DynArray[uint256, 5]
z: address

@deploy
def __init__():
    assert False
    """

    deployer = env.eoa
    nonce = env.state.get_nonce(deployer)
    create_addr = Address(
        compute_contract_address(deployer.canonical_address, nonce)
    )

    with tx_failed(Exception):
        loads(src)

    assert not env.interpreter.evm.account_has_storage(create_addr)


def test_failed_deploy_retry_same_address(env, tx_failed):
    """After a failed deploy, retrying to the same address must succeed."""
    fail_src = """
x: uint256
y: DynArray[uint256, 5]

@deploy
def __init__():
    assert False
    """

    deployer = env.eoa
    nonce_before = env.state.get_nonce(deployer)

    with tx_failed(Exception):
        loads(fail_src)

    # deploy() increments the sender nonce outside the journal,
    # so restore it manually to target the same create address.
    env.state.get_account(deployer).nonce = nonce_before

    ok_src = """
x: public(uint256)

@deploy
def __init__():
    self.x = 42
    """
    c = loads(ok_src)
    assert c.x() == 42

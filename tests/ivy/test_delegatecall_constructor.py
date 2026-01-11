"""
Regression tests for delegatecall in constructor address resolution bug.

Bug: During contract creation, msg.to == b"" and the actual address is in
msg.create_address. Delegatecall was using msg.to as the storage context,
causing storage/transient operations to target the wrong address.

Fix: Use msg.create_address when msg.to == b"" (i.e., during construction).
"""


def test_delegatecall_in_constructor_writes_to_own_storage(get_contract):
    """Delegatecall in constructor should write to the new contract's storage."""
    target_src = """
t: transient(uint256)

@external
def set_transient(val: uint256):
    self.t = val
    """

    deployer_src = """
t: transient(uint256)
result: public(uint256)

@deploy
def __init__(target: address):
    x: Bytes[32] = raw_call(
        target,
        abi_encode(convert(42, uint256), method_id=method_id("set_transient(uint256)")),
        max_outsize=32,
        is_delegate_call=True
    )
    self.result = self.t
    """

    target = get_contract(target_src)
    deployer = get_contract(deployer_src, target.address)

    assert deployer.result() == 42


def test_delegatecall_in_constructor_reads_own_storage(get_contract):
    """Delegatecall in constructor should read from new contract's storage."""
    target_src = """
t: transient(uint256)

@external
def get_transient() -> uint256:
    return self.t
    """

    deployer_src = """
t: transient(uint256)
result: public(uint256)

@deploy
def __init__(target: address):
    self.t = 123
    response: Bytes[32] = raw_call(
        target,
        method_id("get_transient()"),
        max_outsize=32,
        is_delegate_call=True
    )
    self.result = abi_decode(response, uint256)
    """

    target = get_contract(target_src)
    deployer = get_contract(deployer_src, target.address)

    assert deployer.result() == 123


def test_nested_delegatecall_in_constructor(get_contract):
    """Nested delegatecalls in constructor should all use the new contract's storage."""
    inner_src = """
t: transient(uint256)

@external
def set_t(val: uint256):
    self.t = val
    """

    middle_src = """
t: transient(uint256)

@external
def delegate_set(inner: address, val: uint256):
    x: Bytes[32] = raw_call(
        inner,
        abi_encode(val, method_id=method_id("set_t(uint256)")),
        max_outsize=32,
        is_delegate_call=True
    )
    """

    outer_src = """
t: transient(uint256)
result: public(uint256)

@deploy
def __init__(middle: address, inner: address):
    x: Bytes[32] = raw_call(
        middle,
        abi_encode(inner, convert(999, uint256), method_id=method_id("delegate_set(address,uint256)")),
        max_outsize=32,
        is_delegate_call=True
    )
    self.result = self.t
    """

    inner = get_contract(inner_src)
    middle = get_contract(middle_src)
    outer = get_contract(outer_src, middle.address, inner.address)

    assert outer.result() == 999

import pytest
from ivy.abi import EncodeError

TOKEN_NAME = "Vypercoin"
TOKEN_SYMBOL = "FANG"
TOKEN_DECIMALS = 18
TOKEN_INITIAL_SUPPLY = 21 * 10**6
TOKEN_TOTAL_SUPPLY = TOKEN_INITIAL_SUPPLY * (10**TOKEN_DECIMALS)


source = """
#pragma version >0.3.10

from ethereum.ercs import IERC20
from ethereum.ercs import IERC20Detailed

implements: IERC20
implements: IERC20Detailed

name: public(String[32])
symbol: public(String[32])
decimals: public(uint8)

# NOTE: By declaring `balanceOf` as public, vyper automatically generates a 'balanceOf()' getter
#       method to allow access to account balances.
#       The _KeyType will become a required parameter for the getter and it will return _ValueType.
#       See: https://docs.vyperlang.org/en/v0.1.0-beta.8/types.html?highlight=getter#mappings
balanceOf: public(HashMap[address, uint256])
# By declaring `allowance` as public, vyper automatically generates the `allowance()` getter
allowance: public(HashMap[address, HashMap[address, uint256]])
# By declaring `totalSupply` as public, we automatically create the `totalSupply()` getter
totalSupply: public(uint256)
minter: address


@deploy
def __init__(_name: String[32], _symbol: String[32], _decimals: uint8, _supply: uint256):
    init_supply: uint256 = _supply * 10 ** convert(_decimals, uint256)
    self.name = _name
    self.symbol = _symbol
    self.decimals = _decimals
    self.balanceOf[msg.sender] = init_supply
    self.totalSupply = init_supply
    self.minter = msg.sender
    log IERC20.Transfer(sender=empty(address), receiver=msg.sender, value=init_supply)


@external
def transfer(_to : address, _value : uint256) -> bool:
    # NOTE: vyper does not allow underflows
    #       so the following subtraction would revert on insufficient balance
    self.balanceOf[msg.sender] -= _value
    self.balanceOf[_to] += _value
    log IERC20.Transfer(sender=msg.sender, receiver=_to, value=_value)
    return True


@external
def transferFrom(_from : address, _to : address, _value : uint256) -> bool:
    # NOTE: vyper does not allow underflows
    #       so the following subtraction would revert on insufficient balance
    self.balanceOf[_from] -= _value
    self.balanceOf[_to] += _value
    # NOTE: vyper does not allow underflows
    #      so the following subtraction would revert on insufficient allowance
    self.allowance[_from][msg.sender] -= _value
    log IERC20.Transfer(sender=_from, receiver=_to, value=_value)
    return True


@external
def approve(_spender : address, _value : uint256) -> bool:
    self.allowance[msg.sender][_spender] = _value
    log IERC20.Approval(owner=msg.sender, spender=_spender, value=_value)
    return True


@external
def mint(_to: address, _value: uint256):
    assert msg.sender == self.minter
    assert _to != empty(address)
    self.totalSupply += _value
    self.balanceOf[_to] += _value
    log IERC20.Transfer(sender=empty(address), receiver=_to, value=_value)


@internal
def _burn(_to: address, _value: uint256):
    assert _to != empty(address)
    self.totalSupply -= _value
    self.balanceOf[_to] -= _value
    log IERC20.Transfer(sender=_to, receiver=empty(address), value=_value)


@external
def burn(_value: uint256):
    self._burn(msg.sender, _value)


@external
def burnFrom(_to: address, _value: uint256):
    self.allowance[_to][msg.sender] -= _value
    self._burn(_to, _value)
"""


@pytest.fixture(scope="module")
def erc20(get_contract):
    contract = get_contract(
        source, *[TOKEN_NAME, TOKEN_SYMBOL, TOKEN_DECIMALS, TOKEN_INITIAL_SUPPLY]
    )
    return contract


@pytest.fixture(scope="module")
def erc20_caller(erc20, get_contract):
    erc20_caller_code = """
interface ERC20Contract:
    def name() -> String[64]: view
    def symbol() -> String[32]: view
    def decimals() -> uint256: view
    def balanceOf(_owner: address) -> uint256: view
    def totalSupply() -> uint256: view
    def transfer(_to: address, _amount: uint256) -> bool: nonpayable
    def transferFrom(_from: address, _to: address, _value: uint256) -> bool: nonpayable
    def approve(_spender: address, _amount: uint256) -> bool: nonpayable
    def allowance(_owner: address, _spender: address) -> uint256: nonpayable

token_address: ERC20Contract

@deploy
def __init__(token_addr: address):
    self.token_address = ERC20Contract(token_addr)

@external
def name() -> String[64]:
    return staticcall self.token_address.name()

@external
def symbol() -> String[32]:
    return staticcall self.token_address.symbol()

@external
def decimals() -> uint256:
    return staticcall self.token_address.decimals()

@external
def balanceOf(_owner: address) -> uint256:
    return staticcall self.token_address.balanceOf(_owner)

@external
def totalSupply() -> uint256:
    return staticcall self.token_address.totalSupply()

@external
def transfer(_to: address, _value: uint256) -> bool:
    return extcall self.token_address.transfer(_to, _value)

@external
def transferFrom(_from: address, _to: address, _value: uint256) -> bool:
    return extcall self.token_address.transferFrom(_from, _to, _value)

@external
def allowance(_owner: address, _spender: address) -> uint256:
    return extcall self.token_address.allowance(_owner, _spender)
    """
    return get_contract(erc20_caller_code, *[erc20.address])


def test_initial_state(env, erc20_caller):
    assert erc20_caller.totalSupply() == TOKEN_TOTAL_SUPPLY
    assert erc20_caller.balanceOf(env.deployer) == TOKEN_TOTAL_SUPPLY
    assert erc20_caller.balanceOf(env.accounts[1]) == 0
    assert erc20_caller.name() == TOKEN_NAME
    assert erc20_caller.symbol() == TOKEN_SYMBOL
    assert erc20_caller.decimals() == TOKEN_DECIMALS


def test_call_transfer(env, erc20, erc20_caller, tx_failed):
    # Basic transfer.
    erc20.transfer(erc20_caller.address, 10)
    assert erc20.balanceOf(erc20_caller.address) == 10
    erc20_caller.transfer(env.accounts[1], 10)
    assert erc20.balanceOf(erc20_caller.address) == 0
    assert erc20.balanceOf(env.accounts[1]) == 10

    # more than allowed
    with tx_failed():
        erc20_caller.transfer(env.accounts[1], TOKEN_TOTAL_SUPPLY)

    # Negative transfer value.
    with tx_failed(EncodeError):
        erc20_caller.transfer(env.accounts[1], -1)


def test_caller_approve_allowance(env, erc20, erc20_caller):
    assert erc20_caller.allowance(erc20.address, erc20_caller.address) == 0
    assert erc20.approve(erc20_caller.address, 10)
    assert erc20_caller.allowance(env.deployer, erc20_caller.address) == 10


def test_caller_tranfer_from(env, erc20, erc20_caller, tx_failed):
    # Cannot transfer tokens that are unavailable
    with tx_failed():
        erc20_caller.transferFrom(env.deployer, erc20_caller.address, 10)
    assert erc20.balanceOf(erc20_caller.address) == 0
    assert erc20.approve(erc20_caller.address, 10)
    erc20_caller.transferFrom(env.deployer, erc20_caller.address, 5)
    assert erc20.balanceOf(erc20_caller.address) == 5
    assert erc20_caller.allowance(env.deployer, erc20_caller.address) == 5
    erc20_caller.transferFrom(env.deployer, erc20_caller.address, 3)
    assert erc20.balanceOf(erc20_caller.address) == 8
    assert erc20_caller.allowance(env.deployer, erc20_caller.address) == 2

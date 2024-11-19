import copy
from typing import Optional

from ivy.evm.evm_core import EVMCore
from ivy.evm.evm_state import StateAccess
from ivy.evm.evm_structures import ContractData
from ivy.types import Address


def deepcopy_code(state: StateAccess, target: Address):
    code = state.get_code(target)
    return copy.deepcopy(code)


# TODO find a better name
def create_builtin_shared(
    evm: EVMCore,
    code: ContractData,
    data: bytes = b"",
    value: int = 0,
    revert_on_failure: bool = True,
    salt: Optional[bytes] = None,
    is_runtime_copy: Optional[bool] = False,
):
    res, address = evm.do_create_message_call(value, data, code, salt, is_runtime_copy)

    if not res.is_error:
        return address

    # child evm resulted in error but we shouldn't revert so return zero address
    if not revert_on_failure:
        return Address(0)

    raise res.error


class ProxyFactory:
    src = """"
implementation: address

@deploy
def __init__(implementation: address):
    self.implementation = implementation

@external
@payable
def __default__():
    return raw_call(self.implementation, msg.data, is_delegate_call=True)
    """

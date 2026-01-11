import copy
from typing import Optional

from vyper.compiler import CompilerData

from ivy.evm.evm_core import EVMCore
from ivy.evm.evm_state import StateAccess
from ivy.evm.evm_structures import ContractData
from ivy.types import Address


def deepcopy_code(state: StateAccess, target: Address, reset_global_vars: bool = False):
    from ivy.variable import GlobalVariables

    # TODO what about the case when the target is empty?
    code = state.get_code(target)
    code_copy = copy.deepcopy(code)
    if reset_global_vars:
        # For blueprint creation, we need fresh global_vars since the new contract
        # will run its constructor and allocate variables
        code_copy.global_vars = GlobalVariables()
    return code_copy


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


class MinimalProxyFactory:
    # NOTE: the contract is not semantically eq. to the minimal proxy as per EIP1167
    # however, through an escape hatch in the interpreter we maintain clean passthrough of
    # data without touching abi-encoding
    _SOURCE = """
implementation: immutable(address)

@deploy
def __init__(_implementation: address):
    implementation = _implementation

# use 2**32 which is sufficiently large not to cause runtime problems,
# and small enough # not to cause allocator exception in the frontend
@external
@payable
def __default__() -> Bytes[2**32]:
    return raw_call(implementation, msg.data, is_delegate_call=True, max_outsize=2**32)


    """
    _compiler_data: CompilerData = None

    @classmethod
    def get_proxy_contract_data(cls):
        if cls._compiler_data is None:
            cls._compiler_data = CompilerData(file_input=cls._SOURCE)
            cls._compiler_data.annotated_vyper_module._metadata["is_minimal_proxy"] = (
                True
            )

        return ContractData(cls._compiler_data)

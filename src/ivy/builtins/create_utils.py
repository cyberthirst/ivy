import copy
from typing import Optional

from vyper.compiler import CompilerData

from ivy.evm.evm_core import EVMCore
from ivy.evm.evm_state import StateAccess
from ivy.evm.evm_structures import ContractData
from ivy.types import Address


def deepcopy_code(state: StateAccess, target: Address, reset_global_vars: bool = False):
    code = state.get_code(target)
    if code is None:
        return None

    # Create a new ContractData that shares the compiler_data and module_t
    # but has fresh mutable state (global_vars, immutables, constants).
    # We can't use deepcopy because it breaks VarInfo object identity -
    # the same variable ends up with different VarInfo objects in declarations
    # vs references, causing lookups to fail.
    code_copy = ContractData(code.compiler_data)

    if not reset_global_vars:
        # Copy the existing global_vars for runtime copies.
        # We must preserve VarInfo key identity (they must match the shared AST),
        # so we can't use deepcopy on the whole GlobalVariables.
        # Instead, copy the positions dict directly (same keys) and deepcopy
        # the variables dict (new GlobalVariable instances with same values).
        code_copy.global_vars.positions = code.global_vars.positions.copy()
        code_copy.global_vars.variables = copy.deepcopy(code.global_vars.variables)
        code_copy.global_vars.reentrant_key_address = code.global_vars.reentrant_key_address
        code_copy.global_vars.adrr_to_name = code.global_vars.adrr_to_name.copy()

        # Also copy immutables and constants for runtime copies
        code_copy.immutables = code.immutables.copy()
        code_copy.constants = code.constants.copy()
    # else: reset_global_vars=True means we want fresh GlobalVariables(),
    # which ContractData.__init__ already creates

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

from vyper.utils import method_id
from vyper.semantics.types import TupleT, IntegerT

from ivy.abi import abi_encode


def test_revert_reason(env, tx_failed, get_contract):
    reverty_code = """
@external
def foo():
    data: Bytes[4] = method_id("NoFives()")
    raw_revert(data)
    """

    revert_bytes = method_id("NoFives()")

    with tx_failed(exc_text=revert_bytes.hex()):
        get_contract(reverty_code).foo()


def test_revert_reason_typed(env, tx_failed, get_contract):
    reverty_code = """
@external
def foo():
    val: uint256 = 5
    data: Bytes[100] = _abi_encode(val, method_id=method_id("NoFives(uint256)"))
    raw_revert(data)
    """

    # revert_bytes = method_id("NoFives(uint256)") + abi.encode("(uint256)", (5,))
    encode_typ = TupleT((IntegerT(False, 256),))
    revert_bytes = (
        method_id("NoFives(uint256)").hex() + abi_encode(encode_typ, (5,)).hex()
    )

    with tx_failed(exc_text=revert_bytes):
        get_contract(reverty_code).foo()


def test_revert_reason_typed_no_variable(env, tx_failed, get_contract):
    reverty_code = """
@external
def foo():
    val: uint256 = 5
    raw_revert(_abi_encode(val, method_id=method_id("NoFives(uint256)")))
    """

    encode_typ = TupleT((IntegerT(False, 256),))
    revert_bytes = (
        method_id("NoFives(uint256)").hex() + abi_encode(encode_typ, (5,)).hex()
    )

    with tx_failed(exc_text=revert_bytes):
        get_contract(reverty_code).foo()

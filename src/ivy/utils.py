from vyper.utils import method_id
from vyper.semantics.types.subscriptable import TupleT
import vyper.ast.nodes as ast


def compute_call_abi_data(func_t, num_kwargs):
    sig_kwargs = func_t.keyword_args[:num_kwargs]
    sig_args = func_t.positional_args + sig_kwargs

    calldata_args_t = TupleT(list(arg.typ for arg in sig_args))

    sig = func_t.name + calldata_args_t.abi_type.selector_name()

    selector = method_id(sig)

    return selector, calldata_args_t

from vyper.utils import method_id
from vyper.semantics.types.subscriptable import TupleT
from vyper.semantics.types.function import KeywordArg


def compute_call_abi_data(func_t, num_kwargs):
    sig_kwargs = func_t.keyword_args[:num_kwargs]
    sig_args = func_t.positional_args + sig_kwargs

    calldata_args_t = TupleT(list(arg.typ for arg in sig_args))

    sig = func_t.name + calldata_args_t.abi_type.selector_name()

    selector = method_id(sig)

    return selector, calldata_args_t


def abi_signature_for_kwargs(self, kwargs: list[KeywordArg]) -> str:
    args = self.positional_args + kwargs  # type: ignore
    return (
        self.name
        + "("
        + ",".join([arg.typ.abi_type.selector_name() for arg in args])
        + ")"
    )


def needs_external_call_wrap(typ):
    return not (isinstance(typ, tuple) and len(typ) > 1)


def calculate_type_for_external_return(typ):
    if needs_external_call_wrap(typ):
        return tuple([typ])
    return typ

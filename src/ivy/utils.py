from vyper.utils import method_id
from vyper.semantics.types.function import KeywordArg


def compute_args_abi_type(func_t, num_kwargs):
    sig_kwargs = func_t.keyword_args[:num_kwargs]
    sig_args = func_t.positional_args + sig_kwargs
    args_abi_type = (
        "(" + ",".join(arg.typ.abi_type.selector_name() for arg in sig_args) + ")"
    )
    abi_sig = func_t.name + args_abi_type

    _method_id = method_id(abi_sig)
    return (_method_id, args_abi_type)


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

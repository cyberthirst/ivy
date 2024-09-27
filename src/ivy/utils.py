from vyper.utils import method_id


def compute_args_abi_type(func_t, num_kwargs):
    sig_kwargs = func_t.keyword_args[:num_kwargs]
    sig_args = func_t.positional_args + sig_kwargs
    args_abi_type = (
        "(" + ",".join(arg.typ.abi_type.selector_name() for arg in sig_args) + ")"
    )
    abi_sig = func_t.name + args_abi_type

    _method_id = method_id(abi_sig)
    return (_method_id, args_abi_type)

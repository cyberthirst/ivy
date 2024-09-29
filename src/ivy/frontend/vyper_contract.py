from functools import cached_property
from abc import ABC, abstractmethod

from vyper.codegen.core import calculate_type_for_external_return

from vyper.compiler import CompilerData
from vyper.compiler.output import build_abi_output
from vyper.semantics.types import TupleT

from titanoboa.boa.util.abi import Address

from ivy.frontend.env import Env
from ivy.abi import abi_decode, abi_encode
from ivy.utils import compute_call_abi_data


class BaseDeployer(ABC):
    def __init__(self, compiler_data, filename=None):
        self.compiler_data = compiler_data
        self.filename = filename

    def __call__(self, *args, **kwargs):
        return self.deploy(*args, **kwargs)

    @abstractmethod
    def deploy(self, *args, **kwargs):
        pass


class VyperDeployer(BaseDeployer):
    create_compiler_data = CompilerData  # this may be a different class in plugins

    def __init__(self, compiler_data, filename=None):
        super().__init__(compiler_data, filename=filename)

    def __call__(self, *args, **kwargs):
        return self.deploy(*args, **kwargs)

    def deploy(self, *args, **kwargs):
        return VyperContract(
            self.compiler_data, *args, filename=self.filename, **kwargs
        )


class VyperContract:
    _can_line_profile = True

    def __init__(
        self,
        compiler_data: CompilerData,
        *args,
        value=0,
        env: Env = None,
        filename: str = None,
    ):
        self.compiler_data = compiler_data
        self.env = env or Env.get_singleton()
        self.filename = filename

        # TODO collect all the exposed funcs in ivy to avoid introducing
        # a potential Vyper bug
        exposed_fns = {
            fn_t.name: fn_t.decl_node
            for fn_t in compiler_data.global_ctx.exposed_functions
        }

        # set external methods as class attributes:
        self._ctor = None
        if compiler_data.global_ctx.init_function is not None:
            self._ctor = VyperFunction(
                compiler_data.global_ctx.init_function.decl_node, self
            )

        addr = self._run_init(*args, value=value)

        self._address = addr

        for fn_name, fn in exposed_fns.items():
            setattr(self, fn_name, VyperFunction(fn, self))

    @cached_property
    def abi(self):
        return build_abi_output(self.compiler_data)

    def marshal_to_python(self, computation, vyper_typ):
        if vyper_typ is None:
            return None

        return_typ = calculate_type_for_external_return(vyper_typ)
        ret = abi_decode(return_typ, computation)

        # unwrap the tuple if needed
        if not isinstance(vyper_typ, TupleT):
            (ret,) = ret

        return vyper_object(ret, vyper_typ)

    def _run_init(self, *args, value=0):
        encoded_args = b""
        if self._ctor:
            encoded_args = self._ctor.prepare_calldata(*args)

        module = self.compiler_data.annotated_vyper_module

        address = self.env.deploy(
            module=module,
            *args,
            raw_args=encoded_args,
            value=value,
        )

        return address


class VyperFunction:
    def __init__(self, fn_ast, contract):
        self.fn_ast = fn_ast
        self.contract = contract
        self.env = contract.env

        self.__doc__ = (
            fn_ast.doc_string.value if hasattr(fn_ast, "doc_string") else None
        )
        self.__module__ = self.contract.compiler_data.contract_path

    def __repr__(self):
        return f"{self.contract.compiler_data.contract_path}.{self.fn_ast.name}"

    def __str__(self):
        return repr(self.func_t)

    @property
    def func_t(self):
        return self.fn_ast._metadata["func_type"]

    def args_abi_type(self, num_kwargs):
        # hotspot, cache the signature computation
        if not hasattr(self, "_signature_cache"):
            self._signature_cache = {}

        if num_kwargs in self._signature_cache:
            return self._signature_cache[num_kwargs]

        _method_id, args_abi_type = compute_call_abi_data(self.func_t, num_kwargs)

        self._signature_cache[num_kwargs] = (_method_id, args_abi_type)

        return _method_id, args_abi_type

    def prepare_calldata(self, *args, **kwargs):
        n_total_args = self.func_t.n_total_args
        n_pos_args = self.func_t.n_positional_args

        if not n_pos_args <= len(args) <= n_total_args:
            expectation_str = f"expected between {n_pos_args} and {n_total_args}"
            if n_pos_args == n_total_args:
                expectation_str = f"expected {n_total_args}"
            raise Exception(
                f"bad args to `{repr(self.func_t)}` "
                f"({expectation_str}, got {len(args)})"
            )

        # align the kwargs with the signature
        # sig_kwargs = self.func_t.default_args[: len(kwargs)]

        total_non_base_args = len(kwargs) + len(args) - n_pos_args

        args = tuple(getattr(arg, "address", arg) for arg in args)

        method_id, args_abi_type = self.args_abi_type(total_non_base_args)
        encoded_args = abi_encode(args_abi_type, args)

        if self.func_t.is_constructor or self.func_t.is_fallback:
            return encoded_args

        return method_id + encoded_args

    def __call__(self, *args, value=0, sender=None, **kwargs):
        calldata_bytes = self.prepare_calldata(*args, **kwargs)

        res = self.env.execute_code(
            to_address=self.contract._address,
            sender=sender,
            calldata=calldata_bytes,
            value=value,
            is_modifying=self.func_t.is_mutable,
        )

        typ = self.func_t.return_type
        return self.contract.marshal_to_python(res, typ)


_typ_cache = {}


def vyper_object(val, vyper_type):
    # make a thin wrapper around whatever type val is,
    # and tag it with _vyper_type metadata

    vt = type(val)
    if vt is bool or vt is Address:
        # https://stackoverflow.com/q/2172189
        # bool is not ambiguous wrt vyper type anyways.
        return val

    if vt not in _typ_cache:
        # ex. class int_wrapper(int): pass
        _typ_cache[vt] = type(f"{vt.__name__}_wrapper", (vt,), {})

    t = _typ_cache[type(val)]

    ret = t(val)
    ret._vyper_type = vyper_type
    return ret

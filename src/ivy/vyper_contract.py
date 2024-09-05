# the main "entry point" of vyper-related functionality like
# AST handling, traceback construction and ABI (marshaling
# and unmarshaling vyper objects)

import contextlib
import copy
import warnings
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Optional

import vyper
import vyper.ast as vy_ast
import vyper.ir.compile_ir as compile_ir
import vyper.semantics.namespace as vy_ns
from eth.exceptions import VMError
from vyper.ast.nodes import VariableDecl
from vyper.ast.parse import parse_to_ast
from vyper.codegen.core import calculate_type_for_external_return
from vyper.codegen.function_definitions import (
    generate_ir_for_external_function,
    generate_ir_for_internal_function,
)
from vyper.codegen.ir_node import IRnode
from vyper.codegen.module import generate_ir_for_module
from vyper.compiler import CompilerData
from vyper.compiler import output as compiler_output
from vyper.compiler.output import build_abi_output
from vyper.compiler.settings import OptimizationLevel, anchor_settings
from vyper.exceptions import VyperException
from vyper.ir.optimizer import optimize
from vyper.semantics.types import AddressT, HashMapT, TupleT
from vyper.utils import method_id

from titanoboa.boa.util.abi import Address, abi_decode, abi_encode

from ivy.env import Env


class VyperDeployer:
    create_compiler_data = CompilerData  # this may be a different class in plugins

    def __init__(self, compiler_data, filename=None):
        self.compiler_data = compiler_data

        self.filename = filename

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


        # add all exposed functions from the interface to the contract
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

        addr = self._run_init( *args, value=value)

        self._address = addr

        for fn_name, fn in exposed_fns.items():
            setattr(self, fn_name, VyperFunction(fn, self))

    @cached_property
    def abi(self):
        return build_abi_output(self.compiler_data)


    def _run_init(self, *args, value=0):
        encoded_args = b""
        if self._ctor:
            encoded_args = self._ctor.prepare_calldata(*args)

        module = self.compiler_data.annotated_vyper_module

        address = self.env.deploy(
            module=module,
            args=encoded_args,
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

    # hotspot, cache the signature computation
    def args_abi_type(self, num_kwargs):
        if not hasattr(self, "_signature_cache"):
            self._signature_cache = {}

        if num_kwargs in self._signature_cache:
            return self._signature_cache[num_kwargs]

        # align the kwargs with the signature
        sig_kwargs = self.func_t.keyword_args[:num_kwargs]
        sig_args = self.func_t.positional_args + sig_kwargs
        args_abi_type = (
            "(" + ",".join(arg.typ.abi_type.selector_name() for arg in sig_args) + ")"
        )
        abi_sig = self.func_t.name + args_abi_type

        _method_id = method_id(abi_sig)
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

        args = [getattr(arg, "address", arg) for arg in args]

        method_id, args_abi_type = self.args_abi_type(total_non_base_args)
        encoded_args = abi_encode(args_abi_type, args)

        if self.func_t.is_constructor or self.func_t.is_fallback:
            return encoded_args

        return method_id + encoded_args

    def __call__(self, *args, value=0, gas=None, sender=None, **kwargs):
        calldata_bytes = self.prepare_calldata(*args, **kwargs)


        computation = self.env.execute_code(
            to_address=self.contract._address,
            sender=sender,
            data=calldata_bytes,
            value=value,
            gas=gas,
            is_modifying=self.func_t.is_mutable,
            contract=self.contract,
        )

        typ = self.func_t.return_type
        return self.contract.marshal_to_python(computation, typ)


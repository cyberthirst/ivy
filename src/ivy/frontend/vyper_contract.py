# this file is very closesly based on titanoboa: https://github.com/vyperlang/titanoboa/blob/f58c33cde50f6deaaeefff1136b18f92ef747c6b/boa/contracts/vyper/vyper_contract.py
from typing import Any
from dataclasses import astuple
from functools import cached_property
from abc import ABC, abstractmethod

from vyper.codegen.core import (
    calculate_type_for_external_return,
    needs_external_call_wrap,
)

from vyper.compiler import CompilerData
from vyper.compiler.output import build_abi_output
from vyper.semantics.types import TupleT
from vyper.semantics.data_locations import DataLocation

from ivy.context import ExecutionOutput
from ivy.frontend.env import Env
from ivy.abi import abi_decode, abi_encode
from ivy.utils import compute_call_abi_data
from ivy.types import Address
from ivy.frontend.event import Event, RawEvent
from ivy.frontend.decoder_utils import decode_ivy_object
from ivy.evm.evm_structures import Log


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
        encoded_constructor_args: bytes | str = None,
    ):
        self.compiler_data = compiler_data
        self.env = env or Env.get_singleton()
        self.filename = filename
        self._execution_output = None

        if args and encoded_constructor_args is not None:
            raise ValueError(
                "Cannot provide both constructor arguments and encoded constructor arguments. "
                "Use either *args or encoded_constructor_args, not both."
            )

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

        addr = self._run_init(
            *args, value=value, encoded_constructor_args=encoded_constructor_args
        )

        self._address = addr

        for fn_name, fn in exposed_fns.items():
            setattr(self, fn_name, VyperFunction(fn, self))

        self.env.register_contract(self._address, self)

    @cached_property
    def abi(self):
        return build_abi_output(self.compiler_data)

    @property
    def address(self) -> Address:
        if self._address is None:
            # avoid assert, in pytest it would call repr(self) which segfaults
            raise RuntimeError("Contract address is not set")
        return self._address

    def handle_error(self, execution_output: ExecutionOutput):
        raise execution_output.error

    @cached_property
    def event_for(self):
        module_t = self.compiler_data.global_ctx
        return {e.event_id: e for e in module_t.used_events}

    def decode_log(self, e: Log, include_id=False):
        address, topics, data = astuple(e)
        assert self._address.canonical_address == address
        event_hash = int.from_bytes(topics[0], byteorder="big")
        event_t = self.event_for[event_hash]

        topic_typs = []
        arg_typs = []
        for is_topic, typ in zip(event_t.indexed, event_t.arguments.values()):
            if not is_topic:
                arg_typs.append(typ)
            else:
                topic_typs.append(typ)

        decoded_topics = [] if not include_id else [e.topics[0]]

        for typ, t in zip(topic_typs, topics[1:]):
            if typ._is_prim_word:
                decoded_topics.append(abi_decode(typ, t, ivy_compat=False))
            else:
                decoded_topics.append(t)

        tuple_typ = TupleT(arg_typs)

        args = abi_decode(tuple_typ, data, ivy_compat=False)

        return Event(self._address, event_t, event_t.name, decoded_topics, args)

    def _get_logs(self, execution_output: ExecutionOutput) -> list[Log]:
        if execution_output is None:
            return []

        if execution_output.is_error:
            return []

        return execution_output.logs

    def get_logs(self, execution_output=None, include_id=False):
        if execution_output is None:
            execution_output = self._execution_output

        entries = self._get_logs(execution_output)

        ret = []
        for e in entries:
            logger_address = e.address
            c = self.env.lookup_contract(logger_address)
            if c is not None:
                ret.append(c.decode_log(e, include_id))
            else:
                ret.append(RawEvent(e))

        return ret

    def message_call(self, data: bytes, value: int = 0, sender=None):
        ret = self.env.message_call(
            to_address=self.address,
            data=data,
            sender=sender,
            value=value,
            get_execution_output=True,
        )
        self._execution_output = ret
        return ret.output

    def _dump_data_location(self, data_location: DataLocation) -> dict[str, Any]:
        # TODO: add a param to access self._execution_output.touched_accounts
        # and also dump their storage
        acc = self.env.get_account(self._address)
        location = (
            acc.storage if data_location == DataLocation.STORAGE else acc.transient
        )
        gvars = acc.contract_data.global_vars
        res = {}
        # Iterate over known declared variables, not raw storage items.
        # This avoids KeyError when storage contains slots from delegatecall
        # (callee writes to caller's storage using callee's slot positions).
        for (pos, loc), gvar in gvars.variables.items():
            if loc != data_location:
                continue
            if pos not in location:
                continue
            v = location[pos]
            typ = gvar.varinfo.typ
            # Always decode to convert boxed types to Python primitives
            v = decode_ivy_object(v, typ)
            name = gvars.adrr_to_name.get((pos, loc))
            if name is None:
                continue
            res[name] = v
        return res

    def storage_dump(self) -> dict[str, Any]:
        return self._dump_data_location(DataLocation.STORAGE)

    def transient_storage_dump(self) -> dict[str, Any]:
        return self._dump_data_location(DataLocation.TRANSIENT)

    def marshal_to_python(self, execution_output: ExecutionOutput, vyper_typ):
        self._execution_output = execution_output

        if execution_output.is_error:
            self.handle_error(execution_output)

        if vyper_typ is None:
            return None

        return_typ = calculate_type_for_external_return(vyper_typ)
        ret = abi_decode(return_typ, execution_output.output, ivy_compat=False)

        # unwrap the tuple if needed (single values AND single-element tuples get wrapped)
        if needs_external_call_wrap(vyper_typ):
            (ret,) = ret

        return ret

    def _run_init(self, *args, value=0, encoded_constructor_args=None):
        encoded_args = b""
        self.ctor_encoded_args = ""

        if encoded_constructor_args is not None:
            if isinstance(encoded_constructor_args, str):
                encoded_args = bytes.fromhex(encoded_constructor_args.replace("0x", ""))
            elif isinstance(encoded_constructor_args, bytes):
                encoded_args = encoded_constructor_args
            else:
                raise TypeError(
                    f"encoded_constructor_args must be bytes or str, got {type(encoded_constructor_args)}"
                )
            self.ctor_encoded_args = encoded_args.hex()
        elif self._ctor:
            encoded_args = self._ctor.prepare_calldata(*args)
            self.ctor_encoded_args = encoded_args.hex()

        # touch bytecode to ensure the contract compiles (not all semantic checks are done in the frontend)
        _ = self.compiler_data.bytecode

        address, execution_output = self.env.deploy(
            compiler_data=self.compiler_data,
            raw_args=encoded_args,
            value=value,
        )

        self._execution_output = execution_output

        if execution_output.is_error:
            self.handle_error(execution_output)

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

        # enable to pass Contract instances as arguments to external functions
        args = tuple(getattr(arg, "address", arg) for arg in args)

        method_id, args_abi_type = self.args_abi_type(total_non_base_args)
        encoded_args = abi_encode(args_abi_type, args)

        if self.func_t.is_constructor or self.func_t.is_fallback:
            return encoded_args

        return method_id + encoded_args

    def __call__(self, *args, value=0, sender=None, transact=False, **kwargs):
        calldata_bytes = self.prepare_calldata(*args, **kwargs)

        if transact:
            # Use execute_code which creates a new transaction context
            res = self.env.execute_code(
                to_address=self.contract._address,
                sender=sender,
                calldata=calldata_bytes,
                value=value,
                is_modifying=self.func_t.is_mutable,
            )
        else:
            # Use message_call for Vyper test suite compatibility
            # This keeps the same transaction context across multiple calls
            res = self.env.message_call(
                to_address=self.contract._address,
                data=calldata_bytes,
                value=value,
                get_execution_output=True,
                sender=sender,
            )

        typ = self.func_t.return_type
        return self.contract.marshal_to_python(res, typ)


_typ_cache = {}

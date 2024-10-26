from typing import Any, Optional

from attr import dataclass
from vyper.semantics.types import VyperType
from vyper.semantics.types.function import ContractFunctionT

from ivy.evaluator import VyperEvaluator
from ivy.evm_structures import Account, Message, ContractData


class FunctionContext:
    scopes: list[dict[str, Any]]
    function: ContractFunctionT

    def __init__(self, function: ContractFunctionT):
        self.scopes = []
        self.function = function

    def push(self):
        self.scopes.append({})

    def pop(self):
        self.scopes.pop()

    def new_variable(self, key, typ: VyperType):
        value = VyperEvaluator.default_value(typ)
        assert key not in self.scopes[-1]
        self.scopes[-1][key] = value

    def __contains__(self, item):
        for scope in reversed(self.scopes):
            if item in scope:
                return True
        return False

    def __setitem__(self, key, value):
        for scope in reversed(self.scopes):
            if key in scope:
                scope[key] = value
                return
        self.scopes[-1][key] = value

    def __getitem__(self, key):
        for scope in reversed(self.scopes):
            if key in scope:
                return scope[key]
        raise KeyError(key)

    def __repr__(self):
        return f"Ctx({self.scopes})"


@dataclass
class Storage:
    transient: dict[str, Any]
    storage: dict[str, Any]


@dataclass
class Code:
    immutables: dict[str, Any]
    constants: dict[str, Any]


class ExecutionContext:
    def __init__(
        self, acc: Account, msg: Message, contract_data: Optional[ContractData]
    ):
        self.acc = acc
        if acc.contract_data is None:
            assert contract_data is not None
            self.contract = contract_data
        else:
            self.contract = acc.contract_data
        self.function = None
        self.function_contexts = []
        # set storage-related attributes based on the target account
        self.storage = acc.storage
        self.transient = acc.transient
        self.globals = self.contract.global_vars
        # set code-related attributes based on message.code
        self.msg = msg
        self.immutables = msg.code.immutables
        self.constants = msg.code.constants
        self.entry_points = msg.code.entry_points
        self.returndata: bytes = b""
        self.output: Optional[bytes] = None

    def push_fun_context(self, func_t: ContractFunctionT):
        self.function_contexts.append(FunctionContext(func_t))

    def pop_fun_context(self):
        self.function_contexts.pop()

    def current_fun_context(self):
        return self.function_contexts[-1]

    def push_scope(self):
        self.current_fun_context().push()

    def pop_scope(self):
        self.current_fun_context().pop()

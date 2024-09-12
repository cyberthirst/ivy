from typing import Any
from dataclasses import dataclass

from vyper.semantics.data_locations import DataLocation
from vyper.semantics.types import VyperType
from vyper.semantics.types.function import ContractFunctionT

from ivy.evm import Account


@dataclass
class Variable:
    value: Any
    typ: VyperType
    location: DataLocation


class Context:
    scopes: list[dict[str, Any]]

    def __init__(self):
        self.scopes = [{}]

    def push(self):
        self.scopes.append({})

    def pop(self):
        self.scopes.pop()

    def __contains__(self, item):
        for scope in reversed(self.scopes):
            if item in scope:
                return True
        return False

    def __setitem__(self, key, value):
        self.scopes[-1][key] = value

    def __getitem__(self, key):
        for scope in reversed(self.scopes):
            if key in scope:
                return scope[key]
        raise KeyError(key)

    def __repr__(self):
        return f"Ctx({self.scopes})"


class ExecutionContext:
    def __init__(self, acc: Account, function: ContractFunctionT = None):
        self.contract = acc.contract_data
        self.function = function
        self.function_contexts = [Context()]
        self.storage = acc.storage
        self.transient = acc.transient
        self.immutables = self.contract.immutables
        self.returndata: bytes = b""
        # self.constants = contract.module.constants

    def push_fun_context(self):
        self.function_contexts.append(Context())

    def pop_fun_context(self):
        self.function_contexts.pop()

    def current_fun_context(self):
        return self.function_contexts[-1]

    def push_scope(self):
        self.current_fun_context().push()

    def pop_scope(self):
        self.current_fun_context().pop()

    def add_variable(self, key, value):
        self.function_contexts[-1][key] = value

from typing import Any, Optional

from vyper.semantics.data_locations import DataLocation
from vyper.semantics.types import VyperType
from vyper.semantics.types.function import ContractFunctionT
from vyper.semantics.types.module import ModuleT

from ivy.evm_structures import Account, Message, ContractData
from ivy.evaluator import VyperEvaluator


# TODO probably move this elsewhere
class Variable:
    # TODO add a reference to execution journal
    name: str
    typ: VyperType
    location: dict  # TODO is dict precise?

    def __init__(self, name: str, typ: VyperType, location: DataLocation):
        self.typ = typ
        self.location = location
        self.name = name
        self.location[self.name] = VyperEvaluator.default_value(typ)

    @property
    def value(self):
        return self.location[self.name]

    @value.setter
    def value(self, new_value):
        # TODO register old value in execution journal
        self.location[self.name] = new_value


class FunctionContext:
    scopes: list[dict[str, Any]]
    function: ContractFunctionT

    def __init__(self, function: ContractFunctionT):
        self.scopes = [{}]
        self.function = function

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
    def __init__(self, acc: Account, msg: Message, module: Optional[ModuleT]):
        self.acc = acc
        self.contract = acc.contract_data or ContractData(module)
        self.function = None
        self.function_contexts = []
        self.storage = acc.storage
        self.transient = acc.transient
        self.immutables = self.contract.immutables
        self.constants = self.contract.constants
        self.returndata: bytes = b""
        self.output: Optional[bytes] = None
        self.msg = msg

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

    def new_variable(self, key, value):
        self.function_contexts[-1][key] = value

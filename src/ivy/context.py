from typing import Any, Optional

from vyper.semantics.types import VyperType
from vyper.semantics.types.function import ContractFunctionT

from ivy.evaluator import VyperEvaluator
from ivy.evm.evm_structures import Account, Message, ContractData
from ivy.types import Address


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


class ExecutionContext:
    def __init__(self, acc: Account, msg: Message):
        self.msg = msg
        self.contract = msg.code
        self.function = None
        self.function_contexts = []
        # set storage-related attributes based on the target account
        self.storage = acc.storage
        self.transient = acc.transient
        # set code-related attributes based on message.code
        self.globals = msg.code.global_vars
        self.immutables = msg.code.immutables
        self.constants = msg.code.constants
        self.entry_points = msg.code.entry_points
        self.execution_output = ExecutionOutput()
        # return_data is not part of the ExecutionOutput, it's filled from the output
        # of child evm calls
        self.returndata: bytes = b""
        # TODO should we keep it or only use execution_output.output?
        # - is it used for internal functions?
        self.func_return: Optional[bytes] = None

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


class ExecutionOutput:
    def __init__(self):
        self._output: Optional[bytes] = None
        self.error: Optional[Exception] = None
        self.accessed_addresses: set[Address] = set()
        self.accounts_to_delete: set[Address] = set()
        self.logs: list = []
        self.refund_counter = 0
        self.touched_accounts: set[Address] = set()

    @property
    def output(self):
        if self._output is None:
            return b""
        return self._output

    @output.setter
    def output(self, value):
        self._output = value

    @property
    def is_error(self):
        return self.error is not None

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager
from enum import Enum, auto

from vyper.semantics.analysis.base import VarInfo, DataLocation, Modifiability
from vyper.semantics.types import VyperType
from vyper.semantics.types.function import StateMutability

from ..xfail import XFailExpectation


class ScopeType(Enum):
    MODULE = auto()
    FUNCTION = auto()
    IF = auto()
    FOR = auto()


class ExprMutability(Enum):
    CONST = auto()  # compile-time constants only (const exprs)
    PURE = auto()  # no state reads/writes
    VIEW = auto()  # can read state, cannot write
    STATEFUL = auto()  # full access


def state_to_expr_mutability(state_mutability: StateMutability) -> ExprMutability:
    if state_mutability == StateMutability.PURE:
        return ExprMutability.PURE
    if state_mutability == StateMutability.VIEW:
        return ExprMutability.VIEW
    return ExprMutability.STATEFUL


@dataclass
class Scope:
    scope_type: ScopeType
    vars: Dict[str, VarInfo] = field(default_factory=dict)


@dataclass
class Context:
    current_scope: Scope = field(default_factory=lambda: Scope(ScopeType.MODULE))
    scope_stack: List[Scope] = field(default_factory=list)
    all_vars: Dict[str, VarInfo] = field(default_factory=dict)
    immutables_to_init: List[tuple[str, VarInfo]] = field(default_factory=list)
    compilation_xfails: List[XFailExpectation] = field(default_factory=list)
    runtime_xfails: List[XFailExpectation] = field(default_factory=list)
    current_mutability: ExprMutability = ExprMutability.STATEFUL
    current_function_mutability: StateMutability = StateMutability.NONPAYABLE

    def _push_scope(self, scope_type: ScopeType) -> None:
        self.scope_stack.append(self.current_scope)
        self.current_scope = Scope(scope_type)

    def _pop_scope(self) -> None:
        for var_name in self.current_scope.vars:
            if var_name in self.all_vars:
                del self.all_vars[var_name]
        self.current_scope = self.scope_stack.pop()

    def add_variable(self, name: str, var_info: VarInfo) -> None:
        self.current_scope.vars[name] = var_info
        self.all_vars[name] = var_info

        # Track immutables that need initialization in __init__
        if (
            var_info.location == DataLocation.CODE
            and var_info.modifiability == Modifiability.RUNTIME_CONSTANT
        ):
            self.immutables_to_init.append((name, var_info))

    @property
    def is_module_scope(self) -> bool:
        return self.current_scope.scope_type == ScopeType.MODULE

    @contextmanager
    def new_scope(self, scope_type: ScopeType):
        self._push_scope(scope_type)
        try:
            yield
        finally:
            self._pop_scope()

    @contextmanager
    def mutability(self, new_mode: ExprMutability):
        prev = self.current_mutability
        self.current_mutability = new_mode
        try:
            yield
        finally:
            self.current_mutability = prev

    @contextmanager
    def function_mutability(self, new_mode: StateMutability):
        prev = self.current_function_mutability
        self.current_function_mutability = new_mode
        try:
            yield
        finally:
            self.current_function_mutability = prev

    def find_matching_vars(
        self, want_type: Optional[VyperType] = None
    ) -> list[tuple[str, VarInfo]]:
        """Find variables compatible with want_type and current mutability context."""
        const_only = self.current_mutability == ExprMutability.CONST
        candidates = []
        for name, var_info in self.all_vars.items():
            if want_type is not None and not want_type.compare_type(var_info.typ):
                continue
            if const_only and var_info.modifiability != Modifiability.CONSTANT:
                continue
            candidates.append((name, var_info))
        return candidates

from typing import Any, Dict, Iterable, List, Optional, Set
from dataclasses import dataclass, field
from contextlib import contextmanager
from enum import Enum, auto

from vyper.semantics.analysis.base import VarInfo, DataLocation, Modifiability
from vyper.semantics.types import VyperType, SArrayT, DArrayT
from vyper.semantics.types.function import StateMutability

from fuzzer.xfail import XFailExpectation


class ScopeType(Enum):
    MODULE = auto()
    FUNCTION = auto()
    IF = auto()
    FOR = auto()


class AccessMode(Enum):
    READ = auto()
    WRITE = auto()


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


def effective_call_mutability(context: "GenerationContext") -> StateMutability:
    caller_mutability = context.current_function_mutability
    if context.in_iterable_expr and caller_mutability not in (
        StateMutability.PURE,
        StateMutability.VIEW,
    ):
        return StateMutability.VIEW
    return caller_mutability


@dataclass
class Scope:
    scope_type: ScopeType
    vars: Dict[str, VarInfo] = field(default_factory=dict)


@dataclass
class GenerationContext:
    current_scope: Scope = field(default_factory=lambda: Scope(ScopeType.MODULE))
    scope_stack: List[Scope] = field(default_factory=list)
    all_vars: Dict[str, VarInfo] = field(default_factory=dict)
    constants: Dict[str, Any] = field(default_factory=dict)
    immutables_to_init: Dict[str, VarInfo] = field(default_factory=dict)
    compilation_xfails: List[XFailExpectation] = field(default_factory=list)
    runtime_xfails: List[XFailExpectation] = field(default_factory=list)
    current_mutability: ExprMutability = ExprMutability.STATEFUL
    current_function_mutability: StateMutability = StateMutability.NONPAYABLE
    current_access_mode: AccessMode = AccessMode.READ
    in_init: bool = False
    remaining_immutables: Set[str] = field(default_factory=set)
    iterable_expr_depth: int = 0

    def _push_scope(self, scope_type: ScopeType) -> None:
        self.scope_stack.append(self.current_scope)
        self.current_scope = Scope(scope_type)

    def _pop_scope(self) -> None:
        for var_name in self.current_scope.vars:
            if var_name in self.all_vars:
                del self.all_vars[var_name]
                # Restore shadowed variable from outer scope if it exists
                for outer_scope in reversed(self.scope_stack):
                    if var_name in outer_scope.vars:
                        self.all_vars[var_name] = outer_scope.vars[var_name]
                        break
        self.current_scope = self.scope_stack.pop()

    def add_variable(self, name: str, var_info: VarInfo) -> None:
        self.current_scope.vars[name] = var_info
        self.all_vars[name] = var_info

    def mark_immutable_assigned(self, name: str) -> None:
        self.remaining_immutables.discard(name)

    def remaining_immutable_items(self) -> list[tuple[str, VarInfo]]:
        return [
            (name, self.immutables_to_init[name])
            for name in self.remaining_immutables
            if name in self.immutables_to_init
        ]

    @contextmanager
    def init_assignments(self, immutables: Optional[Iterable[str]] = None):
        prev_in_init = self.in_init
        prev_remaining = self.remaining_immutables
        self.in_init = True
        if immutables is None:
            self.remaining_immutables = set(self.immutables_to_init.keys())
        else:
            self.remaining_immutables = set(immutables)
        try:
            yield
        finally:
            self.in_init = prev_in_init
            self.remaining_immutables = prev_remaining

    def add_constant(self, name: str, value: Any) -> None:
        self.constants[name] = value

    @property
    def is_module_scope(self) -> bool:
        return self.current_scope.scope_type == ScopeType.MODULE

    def is_inside_for_scope(self) -> bool:
        if self.current_scope.scope_type == ScopeType.FOR:
            return True
        return any(scope.scope_type == ScopeType.FOR for scope in self.scope_stack)

    @property
    def in_iterable_expr(self) -> bool:
        return self.iterable_expr_depth > 0

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

    @contextmanager
    def access_mode(self, mode: AccessMode):
        prev = self.current_access_mode
        self.current_access_mode = mode
        try:
            yield
        finally:
            self.current_access_mode = prev

    @contextmanager
    def iterable_expr(self):
        self.iterable_expr_depth += 1
        try:
            yield
        finally:
            self.iterable_expr_depth -= 1

    def _is_var_accessible(self, name: str, var_info: VarInfo) -> bool:
        if self.current_mutability == ExprMutability.CONST:
            if var_info.modifiability != Modifiability.CONSTANT:
                return False
        elif self.current_mutability == ExprMutability.PURE:
            if var_info.is_state_variable():
                return False
        elif self.current_mutability == ExprMutability.VIEW:
            if self.current_access_mode == AccessMode.WRITE and var_info.location in (
                DataLocation.STORAGE,
                DataLocation.TRANSIENT,
            ):
                return False

        if self.current_access_mode == AccessMode.WRITE:
            if var_info.modifiability == Modifiability.MODIFIABLE:
                return True
            if (
                self.in_init
                and var_info.location == DataLocation.CODE
                and var_info.modifiability == Modifiability.RUNTIME_CONSTANT
                and name in self.remaining_immutables
            ):
                return True
            return False

        return True

    def find_matching_vars(
        self, want_type: Optional[VyperType] = None
    ) -> list[tuple[str, VarInfo]]:
        """Find variables compatible with want_type and current mutability/access context."""
        candidates = []
        for name, var_info in self.all_vars.items():
            if want_type is not None and not want_type.compare_type(var_info.typ):
                continue

            if not self._is_var_accessible(name, var_info):
                continue

            candidates.append((name, var_info))
        return candidates

    def find_iterable_arrays(self) -> list[tuple[str, VarInfo]]:
        """Find arrays (SArrayT or DArrayT) that can be iterated over."""
        return [
            (name, var_info)
            for name, var_info in self.all_vars.items()
            if isinstance(var_info.typ, (SArrayT, DArrayT))
            # Constant DynArrays fold to literal lists and fail in for-loops.
            # https://github.com/vyperlang/vyper/issues/4823
            and not (
                var_info.modifiability == Modifiability.CONSTANT
                and isinstance(var_info.typ, DArrayT)
            )
            and self._is_var_accessible(name, var_info)
        ]

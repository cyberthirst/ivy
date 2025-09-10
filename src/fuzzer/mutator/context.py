from typing import Dict, List, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager
from enum import Enum, auto

from vyper.semantics.analysis.base import VarInfo, DataLocation, Modifiability
from vyper.semantics.types import VyperType


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
    compilation_xfail: Optional[bool] = (
        None  # True = must fail, None = don't check, False = must not fail
    )
    runtime_xfail: Optional[bool] = (
        None  # True = must fail, None = don't know, False = must not fail (future)
    )
    current_mutability: ExprMutability = ExprMutability.STATEFUL

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

    # TODO probably remove this
    def add_local(self, name: str, typ: VyperType) -> None:
        var_info = VarInfo(
            typ=typ,
            location=DataLocation.MEMORY,
            modifiability=Modifiability.MODIFIABLE,
            is_public=False,
            decl_node=None,
        )
        self.add_variable(name, var_info)

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

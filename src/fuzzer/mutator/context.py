from typing import Dict, List
from dataclasses import dataclass, field

from vyper.semantics.analysis.base import VarInfo, DataLocation, Modifiability
from vyper.semantics.types import VyperType


@dataclass
class Scope:
    vars: Dict[str, VarInfo] = field(default_factory=dict)


@dataclass
class Context:
    current_scope: Scope = field(default_factory=Scope)
    scope_stack: List[Scope] = field(default_factory=list)
    all_vars: Dict[str, VarInfo] = field(default_factory=dict)
    immutables_to_init: List[tuple[str, VarInfo]] = field(default_factory=list)

    def push_scope(self) -> None:
        self.scope_stack.append(self.current_scope)
        self.current_scope = Scope()

    def pop_scope(self) -> None:
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
        return len(self.scope_stack) == 1

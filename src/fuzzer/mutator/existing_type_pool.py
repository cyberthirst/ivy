from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from vyper.semantics.types import VyperType

from fuzzer.mutator.context import GenerationContext
from fuzzer.mutator.type_utils import collect_dereference_types

if TYPE_CHECKING:
    from fuzzer.mutator.function_registry import FunctionRegistry


def collect_existing_reachable_types(
    *,
    context: GenerationContext,
    deref_max_steps: int,
    function_registry: Optional[FunctionRegistry] = None,
    skip: Optional[set[type]] = None,
) -> list[VyperType]:
    skip_types = skip or set()
    seen: set[str] = set()
    all_types: list[VyperType] = []

    def add_type(typ: VyperType) -> None:
        if type(typ) in skip_types:
            return
        key = str(typ)
        if key in seen:
            return
        seen.add(key)
        all_types.append(typ)

    def add_root_and_children(root_type: VyperType) -> None:
        add_type(root_type)
        for child_t, _depth in collect_dereference_types(
            root_type, max_steps=deref_max_steps
        ):
            add_type(child_t)

    for _name, var_info in context.find_matching_vars():
        add_root_and_children(var_info.typ)

    if function_registry is None or function_registry.current_function is None:
        return all_types

    callable_funcs = function_registry.get_callable_functions(
        from_function=function_registry.current_function,
        caller_mutability=context.current_function_mutability,
    )
    for func in callable_funcs:
        if func.return_type is None:
            continue
        add_root_and_children(func.return_type)

    return all_types

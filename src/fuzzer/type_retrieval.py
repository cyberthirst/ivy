"""
Type retrieval utilities for fuzzer.

This module provides utilities to retrieve Vyper types from
annotated AST and ABI for use in value mutation.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from vyper.semantics.types import (
    VyperType,
    IntegerT,
    AddressT,
    BoolT,
    BytesM_T,
    BytesT,
    StringT,
    DArrayT,
    SArrayT,
    StructT,
    TupleT,
    ContractFunctionT,
    MemberFunctionT,
    ModuleT,
)
from vyper.ast import nodes as ast
from vyper.utils import method_id


def get_function_types_from_annotated_ast(
    annotated_ast: ast.Module,
) -> Dict[str, Union[ContractFunctionT, MemberFunctionT]]:
    """Extract function types from annotated AST."""
    module_type = annotated_ast._metadata.get("type")
    if not isinstance(module_type, ModuleT):
        return {}

    function_types = {}

    # Get exposed functions
    for func in module_type.exposed_functions:
        if func.name != "__default__":
            function_types[func.name] = func

    return function_types


def get_entry_point_types(
    module_t: ModuleT,
) -> Dict[bytes, Tuple[Union[ContractFunctionT, MemberFunctionT], List[VyperType]]]:
    """Get entry point types with their selectors."""
    from vyper.codegen.core import calculate_arg_totals
    from vyper.codegen.function_definitions.common import get_function_sig

    entry_points = {}

    for func_t in module_t.exposed_functions:
        if func_t.name == "__default__":
            continue

        # Get function selector
        sig = get_function_sig(func_t)
        selector = method_id(sig)

        # Get argument types
        arg_types = []
        for arg in func_t.arguments:
            arg_types.append(arg.typ)

        entry_points[selector] = (func_t, arg_types)

    return entry_points

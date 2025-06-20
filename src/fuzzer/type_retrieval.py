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


def abi_type_to_vyper_type(abi_type: str) -> Optional[VyperType]:
    """Convert ABI type string to VyperType instance."""
    if abi_type == "address":
        return AddressT()
    elif abi_type == "bool":
        return BoolT()
    elif abi_type == "string":
        # Default max string length
        return StringT(1024)
    elif abi_type == "bytes":
        # Default max bytes length
        return BytesT(1024)
    elif abi_type.startswith("bytes"):
        # Fixed bytes like bytes32
        try:
            size = int(abi_type[5:])
            return BytesM_T(size)
        except:
            return None
    elif abi_type.startswith("uint"):
        # Unsigned integer
        bits = 256
        if abi_type != "uint":
            try:
                bits = int(abi_type[4:])
            except:
                pass
        return IntegerT(False, bits)
    elif abi_type.startswith("int"):
        # Signed integer
        bits = 256
        if abi_type != "int":
            try:
                bits = int(abi_type[3:])
            except:
                pass
        return IntegerT(True, bits)
    # TODO: Handle arrays and tuples
    return None


def get_function_types_from_abi(abi: List[Dict[str, Any]]) -> Dict[str, List[VyperType]]:
    """Extract function argument types from ABI."""
    function_types = {}
    
    for item in abi:
        if item.get("type") == "function":
            fn_name = item.get("name", "")
            inputs = item.get("inputs", [])
            
            arg_types = []
            for input_spec in inputs:
                abi_type_str = input_spec.get("type", "")
                vyper_type = abi_type_to_vyper_type(abi_type_str)
                if vyper_type:
                    arg_types.append(vyper_type)
            
            function_types[fn_name] = arg_types
    
    return function_types


def get_constructor_types_from_abi(abi: List[Dict[str, Any]]) -> Tuple[List[VyperType], bool]:
    """Extract constructor argument types and payability from ABI."""
    for item in abi:
        if item.get("type") == "constructor":
            inputs = item.get("inputs", [])
            is_payable = item.get("stateMutability") == "payable"
            
            arg_types = []
            for input_spec in inputs:
                abi_type_str = input_spec.get("type", "")
                vyper_type = abi_type_to_vyper_type(abi_type_str)
                if vyper_type:
                    arg_types.append(vyper_type)
            
            return arg_types, is_payable
    
    return [], False


def get_function_types_from_annotated_ast(annotated_ast: ast.Module) -> Dict[str, Union[ContractFunctionT, MemberFunctionT]]:
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


def get_entry_point_types(module_t: ModuleT) -> Dict[bytes, Tuple[Union[ContractFunctionT, MemberFunctionT], List[VyperType]]]:
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
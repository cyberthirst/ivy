"""Type introspection utilities for Vyper types.

Pure functions for analyzing and traversing Vyper type structures,
particularly for subscriptable types (HashMaps, arrays, tuples).
"""

from typing import Generator

from vyper.semantics.types import (
    VyperType,
    HashMapT,
    SArrayT,
    DArrayT,
    TupleT,
)
from vyper.semantics.analysis.base import VarInfo


def is_subscriptable(t: VyperType) -> bool:
    """Check if type supports subscript access."""
    return isinstance(t, (HashMapT, SArrayT, DArrayT, TupleT))


def child_types(t: VyperType) -> Generator[VyperType, None, None]:
    """Yield element/value types accessible via subscript."""
    if isinstance(t, HashMapT):
        yield t.value_type
    elif isinstance(t, (SArrayT, DArrayT)):
        yield t.value_type
    elif isinstance(t, TupleT):
        yield from getattr(t, "member_types", [])


def can_reach_type(base_t: VyperType, target_t: VyperType, max_depth: int) -> bool:
    """Check if target_t is reachable from base_t via subscripting.

    Recursively explores nested subscriptable types up to max_depth levels.
    """
    if max_depth <= 0:
        return False

    for ct in child_types(base_t):
        if target_t.compare_type(ct):
            return True
        if is_subscriptable(ct) and can_reach_type(ct, target_t, max_depth - 1):
            return True
    return False


def find_subscript_bases(
    target_type: VyperType,
    vars_dict: dict[str, VarInfo],
) -> list[tuple[str, VyperType]]:
    """Find variables that can be subscripted to yield target_type.

    Supports HashMapT[key_type, value_type] and sequences (SArrayT, DArrayT, TupleT).
    For TupleT, any element matching target_type is acceptable.

    Args:
        target_type: The type we want to produce via subscripting
        vars_dict: Mapping of variable names to VarInfo

    Returns:
        List of (var_name, var_type) tuples for variables whose subscript yields target_type
    """
    candidates: list[tuple[str, VyperType]] = []

    for name, var_info in vars_dict.items():
        var_t = var_info.typ

        # HashMap[key->value]
        if isinstance(var_t, HashMapT):
            if target_type.compare_type(var_t.value_type):
                candidates.append((name, var_t))
            continue

        # Static/Dynamic arrays
        if isinstance(var_t, (SArrayT, DArrayT)):
            if target_type.compare_type(var_t.value_type):
                candidates.append((name, var_t))
            continue

        # Tuples: allow if any member matches target_type
        if isinstance(var_t, TupleT):
            for mt in getattr(var_t, "member_types", []):
                if target_type.compare_type(mt):
                    candidates.append((name, var_t))
                    break

    return candidates


def find_nested_subscript_bases(
    target_type: VyperType,
    vars_dict: dict[str, VarInfo],
    max_steps: int,
) -> list[tuple[str, VyperType]]:
    """Find variables that can reach target_type via nested subscripting.

    Unlike find_subscript_bases, this searches through multiple levels
    of subscripting (e.g., hashmap[key1][key2]).

    Args:
        target_type: The type we want to produce via subscripting
        vars_dict: Mapping of variable names to VarInfo
        max_steps: Maximum depth of subscript chain to explore

    Returns:
        List of (var_name, var_type) tuples for variables that can reach target_type
    """
    result: list[tuple[str, VyperType]] = []

    for name, var_info in vars_dict.items():
        t = var_info.typ
        if not is_subscriptable(t):
            continue
        if can_reach_type(t, target_type, max_steps):
            result.append((name, t))

    return result

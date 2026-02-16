"""Type introspection utilities for Vyper types.

Pure functions for analyzing and traversing Vyper type structures,
particularly for subscriptable types (HashMaps, arrays, tuples).
"""

from typing import Generator, Iterable, TypeVar

from vyper.semantics.types import (
    VyperType,
    HashMapT,
    SArrayT,
    DArrayT,
    TupleT,
    StructT,
)
from vyper.semantics.analysis.base import VarInfo

TSource = TypeVar("TSource")


def is_subscriptable(t: VyperType) -> bool:
    """Check if type supports subscript access."""
    return isinstance(t, (HashMapT, SArrayT, DArrayT, TupleT))


def is_dereferenceable(
    t: VyperType,
    *,
    allow_attribute: bool = True,
    allow_subscript: bool = True,
) -> bool:
    """Check if type supports attribute or subscript dereferencing."""
    if allow_subscript and isinstance(t, (HashMapT, SArrayT, DArrayT, TupleT)):
        return True
    if allow_attribute and isinstance(t, StructT):
        return True
    return False


def child_types(t: VyperType) -> Generator[VyperType, None, None]:
    """Yield element/value types accessible via subscript."""
    if isinstance(t, HashMapT):
        yield t.value_type
    elif isinstance(t, (SArrayT, DArrayT)):
        yield t.value_type
    elif isinstance(t, TupleT):
        yield from getattr(t, "member_types", [])


def dereference_child_types(
    t: VyperType,
    *,
    allow_attribute: bool = True,
    allow_subscript: bool = True,
) -> Generator[VyperType, None, None]:
    """Yield child types accessible via attribute or subscript dereference."""
    if allow_subscript:
        yield from child_types(t)
    if allow_attribute and isinstance(t, StructT):
        yield from t.members.values()


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


def can_reach_type_via_deref(
    base_t: VyperType,
    target_t: VyperType,
    max_depth: int,
    *,
    allow_attribute: bool = True,
    allow_subscript: bool = True,
) -> bool:
    """Check if target_t is reachable via attribute/subscript dereferencing."""
    if max_depth <= 0:
        return False

    for ct in dereference_child_types(
        base_t,
        allow_attribute=allow_attribute,
        allow_subscript=allow_subscript,
    ):
        if target_t.compare_type(ct):
            return True
        if is_dereferenceable(
            ct, allow_attribute=allow_attribute, allow_subscript=allow_subscript
        ) and can_reach_type_via_deref(
            ct,
            target_t,
            max_depth - 1,
            allow_attribute=allow_attribute,
            allow_subscript=allow_subscript,
        ):
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


def find_dereference_bases(
    target_type: VyperType,
    vars_dict: dict[str, VarInfo],
    max_steps: int,
    *,
    allow_attribute: bool = True,
    allow_subscript: bool = True,
) -> list[tuple[str, VyperType]]:
    """Find variables that can reach target_type via dereferencing."""
    return find_dereference_sources(
        target_type,
        ((name, var_info.typ) for name, var_info in vars_dict.items()),
        max_steps,
        allow_attribute=allow_attribute,
        allow_subscript=allow_subscript,
    )


def find_dereference_sources(
    target_type: VyperType,
    sources: Iterable[tuple[TSource, VyperType]],
    max_steps: int,
    *,
    allow_attribute: bool = True,
    allow_subscript: bool = True,
) -> list[tuple[TSource, VyperType]]:
    """Find typed sources that can reach target_type via dereferencing."""
    result: list[tuple[TSource, VyperType]] = []

    for source, source_t in sources:
        if not is_dereferenceable(
            source_t,
            allow_attribute=allow_attribute,
            allow_subscript=allow_subscript,
        ):
            continue
        if can_reach_type_via_deref(
            source_t,
            target_type,
            max_steps,
            allow_attribute=allow_attribute,
            allow_subscript=allow_subscript,
        ):
            result.append((source, source_t))

    return result

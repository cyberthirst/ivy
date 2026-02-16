"""Type introspection utilities for Vyper types.

Pure functions for analyzing and traversing Vyper type structures,
particularly for subscriptable types (HashMaps, arrays, tuples).
"""

import random
from dataclasses import dataclass
from typing import Callable, Generator, Iterable, Optional, TypeVar

from vyper.semantics.analysis.base import VarInfo
from vyper.semantics.types import (
    DArrayT,
    HashMapT,
    SArrayT,
    StructT,
    TupleT,
    VyperType,
)

TSource = TypeVar("TSource")


@dataclass(frozen=True)
class DerefCandidate:
    kind: str
    child_type: VyperType
    attr_name: Optional[str] = None
    tuple_index: Optional[int] = None


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


def find_dereferenceable_vars(
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


def allow_deref_candidate(
    child_t: VyperType,
    *,
    target_type: Optional[VyperType],
    max_steps_remaining: int,
    allow_attribute: bool,
    allow_subscript: bool,
) -> bool:
    if target_type is None:
        return True
    if target_type.compare_type(child_t):
        return True
    if max_steps_remaining <= 0:
        return False
    if not is_dereferenceable(
        child_t,
        allow_attribute=allow_attribute,
        allow_subscript=allow_subscript,
    ):
        return False
    return can_reach_type_via_deref(
        child_t,
        target_type,
        max_steps_remaining,
        allow_attribute=allow_attribute,
        allow_subscript=allow_subscript,
    )


def dereference_candidates(
    cur_t: VyperType,
    *,
    target_type: Optional[VyperType],
    max_steps_remaining: int,
    allow_attribute: bool,
    allow_subscript: bool,
) -> list[DerefCandidate]:
    candidates: list[DerefCandidate] = []

    if allow_attribute and isinstance(cur_t, StructT):
        for field_name, field_type in cur_t.members.items():
            if not allow_deref_candidate(
                field_type,
                target_type=target_type,
                max_steps_remaining=max_steps_remaining,
                allow_attribute=allow_attribute,
                allow_subscript=allow_subscript,
            ):
                continue
            candidates.append(
                DerefCandidate(
                    kind="attribute",
                    child_type=field_type,
                    attr_name=field_name,
                )
            )

    if allow_subscript and is_subscriptable(cur_t):
        if isinstance(cur_t, HashMapT):
            child_t = cur_t.value_type
            if allow_deref_candidate(
                child_t,
                target_type=target_type,
                max_steps_remaining=max_steps_remaining,
                allow_attribute=allow_attribute,
                allow_subscript=allow_subscript,
            ):
                candidates.append(DerefCandidate(kind="subscript", child_type=child_t))

        elif isinstance(cur_t, (SArrayT, DArrayT)):
            child_t = cur_t.value_type
            if allow_deref_candidate(
                child_t,
                target_type=target_type,
                max_steps_remaining=max_steps_remaining,
                allow_attribute=allow_attribute,
                allow_subscript=allow_subscript,
            ):
                candidates.append(DerefCandidate(kind="subscript", child_type=child_t))

        elif isinstance(cur_t, TupleT):
            mtypes = list(getattr(cur_t, "member_types", []))
            for i, mt in enumerate(mtypes):
                if not allow_deref_candidate(
                    mt,
                    target_type=target_type,
                    max_steps_remaining=max_steps_remaining,
                    allow_attribute=allow_attribute,
                    allow_subscript=allow_subscript,
                ):
                    continue
                candidates.append(
                    DerefCandidate(
                        kind="subscript",
                        child_type=mt,
                        tuple_index=i,
                    )
                )

    return candidates


def collect_dereference_types(
    base_type: VyperType,
    max_steps: int,
    *,
    allow_attribute: bool = True,
    allow_subscript: bool = True,
) -> list[tuple[VyperType, int]]:
    types: list[tuple[VyperType, int]] = []

    def walk(cur_t: VyperType, steps_left: int, depth: int) -> None:
        if steps_left <= 0:
            return
        for child_t in dereference_child_types(
            cur_t,
            allow_attribute=allow_attribute,
            allow_subscript=allow_subscript,
        ):
            types.append((child_t, depth))
            if is_dereferenceable(
                child_t,
                allow_attribute=allow_attribute,
                allow_subscript=allow_subscript,
            ):
                walk(child_t, steps_left - 1, depth + 1)

    walk(base_type, max_steps, 1)
    return types


def pick_dereference_target_type(
    base_type: VyperType,
    *,
    max_steps: int,
    predicate: Callable[[VyperType], bool],
    rng: random.Random,
    continue_prob: float,
    allow_attribute: bool = True,
    allow_subscript: bool = True,
) -> Optional[VyperType]:
    all_candidates = [
        (t, depth)
        for t, depth in collect_dereference_types(
            base_type,
            max_steps,
            allow_attribute=allow_attribute,
            allow_subscript=allow_subscript,
        )
        if predicate(t)
    ]
    if not all_candidates:
        return None

    desired_depth = 1
    while desired_depth < max_steps and rng.random() < continue_prob:
        desired_depth += 1

    depth_candidates = [t for t, depth in all_candidates if depth == desired_depth]
    if depth_candidates:
        return rng.choice(depth_candidates)

    return rng.choice([t for t, _ in all_candidates])

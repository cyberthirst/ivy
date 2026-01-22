from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Optional

from vyper.semantics.types import (
    VyperType,
    StructT,
    HashMapT,
    SArrayT,
    DArrayT,
    TupleT,
)

from fuzzer.mutator.type_utils import (
    is_subscriptable,
    is_dereferenceable,
    can_reach_type_via_deref,
    dereference_child_types,
)


@dataclass(frozen=True)
class DerefCandidate:
    kind: str
    child_type: VyperType
    attr_name: Optional[str] = None
    tuple_index: Optional[int] = None


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
                candidates.append(
                    DerefCandidate(kind="subscript", child_type=child_t)
                )

        elif isinstance(cur_t, (SArrayT, DArrayT)):
            child_t = cur_t.value_type
            if allow_deref_candidate(
                child_t,
                target_type=target_type,
                max_steps_remaining=max_steps_remaining,
                allow_attribute=allow_attribute,
                allow_subscript=allow_subscript,
            ):
                candidates.append(
                    DerefCandidate(kind="subscript", child_type=child_t)
                )

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

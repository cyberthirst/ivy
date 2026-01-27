from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Type

from vyper.semantics.types import (
    VyperType,
    IntegerT,
    BoolT,
    AddressT,
    BytesM_T,
    BytesT,
    StringT,
)


@dataclass(frozen=True)
class ConvertRule:
    target: Type[VyperType]
    sources: tuple[Type[VyperType], ...]
    constraints: str


# Phase 1: bool/int/address/bytesM/bytes/string only.
CONVERT_MATRIX: tuple[ConvertRule, ...] = (
    ConvertRule(
        target=BoolT,
        sources=(IntegerT, BytesM_T, AddressT, BoolT, BytesT, StringT),
        constraints="Bytes/String require N<=32.",
    ),
    ConvertRule(
        target=IntegerT,
        sources=(IntegerT, BytesM_T, AddressT, BoolT, BytesT),
        constraints="Bytes[N] require N<=32. Address only to unsigned.",
    ),
    ConvertRule(
        target=BytesM_T,
        sources=(IntegerT, BytesM_T, AddressT, BoolT, BytesT),
        constraints="Bytes[N] require N<=M. If M*8 < input_bits -> error.",
    ),
    ConvertRule(
        target=AddressT,
        sources=(BytesM_T, IntegerT, BytesT),
        constraints="Signed integers rejected. Bytes[N] require N<=32.",
    ),
    ConvertRule(
        target=BytesT,
        sources=(BytesT, StringT),
        constraints="Pointer cast only. Same class cannot widen/same-size.",
    ),
    ConvertRule(
        target=StringT,
        sources=(StringT, BytesT),
        constraints="Pointer cast only. Same class cannot widen/same-size.",
    ),
)

_CONVERT_RULES = {rule.target: rule for rule in CONVERT_MATRIX}


def _convert_same_type_allowed(src: VyperType, dst: VyperType) -> bool:
    if not isinstance(src, IntegerT) or not isinstance(dst, IntegerT):
        return False
    return src.bits == 256 and dst.bits == 256 and src.is_signed == dst.is_signed


def convert_target_supported(dst: VyperType) -> bool:
    return type(dst) in _CONVERT_RULES


def convert_source_kinds(dst: VyperType) -> tuple[Type[VyperType], ...]:
    rule = _CONVERT_RULES.get(type(dst))
    if rule is None:
        return ()
    return rule.sources


def convert_is_valid(
    src: VyperType,
    dst: VyperType,
    *,
    allow_same_type: bool = True,
) -> bool:
    src_kind = type(src)
    dst_kind = type(dst)
    rule = _CONVERT_RULES.get(dst_kind)
    if rule is None:
        return False
    if src_kind not in rule.sources:
        return False

    same_type = src.compare_type(dst)
    if same_type:
        if not allow_same_type:
            return False
        if not _convert_same_type_allowed(src, dst):
            return False

    if dst_kind is BoolT:
        if src_kind in {BytesT, StringT} and src.length > 32:
            return False
        return True

    if dst_kind is IntegerT:
        if src_kind is AddressT and dst.is_signed:
            return False
        if src_kind is BytesT and src.length > 32:
            return False
        return True

    if dst_kind is BytesM_T:
        if src_kind is BytesT:
            if src.length > dst.length:
                return False
            return True
        if src_kind is BytesM_T:
            if src.length <= dst.length:
                return False
            return True
        if src_kind is IntegerT:
            if dst.length * 8 < src.bits:
                return False
            return True
        if src_kind is AddressT:
            if dst.length * 8 < 160:
                return False
            return True
        if src_kind is BoolT:
            return True
        return False

    if dst_kind is AddressT:
        if src_kind is IntegerT and src.is_signed:
            return False
        if src_kind is BytesT and src.length > 32:
            return False
        return True

    if dst_kind in {BytesT, StringT}:
        if src_kind not in {BytesT, StringT}:
            return False
        if type(src) is type(dst) and src.length <= dst.length:
            return False
        return True

    return False


def random_convert_type(
    rng: random.Random, kind: Type[VyperType], target_type: VyperType
) -> Optional[VyperType]:
    if kind is IntegerT:
        bit_options = [8, 16, 32, 64, 128, 256]
        max_bits = None
        if isinstance(target_type, BytesM_T):
            max_bits = target_type.length * 8
        if max_bits is not None:
            bit_options = [b for b in bit_options if b <= max_bits]
        if not bit_options:
            return None
        bits = rng.choice(bit_options)
        signed = rng.choice([True, False])
        if isinstance(target_type, AddressT):
            signed = False
        return IntegerT(signed, bits)

    if kind is AddressT:
        return AddressT()

    if kind is BoolT:
        return BoolT()

    if kind is BytesM_T:
        min_len = 1
        max_len = 32
        if isinstance(target_type, BytesM_T):
            min_len = target_type.length + 1
        if max_len < min_len:
            return None
        return BytesM_T(rng.randint(min_len, max_len))

    if kind is BytesT:
        max_dyn_len = 128
        min_len = 1
        max_len = max_dyn_len
        if isinstance(target_type, BytesM_T):
            max_len = min(max_dyn_len, target_type.length)
        elif isinstance(target_type, (BoolT, IntegerT, AddressT)):
            max_len = min(max_dyn_len, 32)
        elif isinstance(target_type, BytesT):
            min_len = target_type.length + 1
        elif isinstance(target_type, StringT):
            max_len = max_dyn_len
        else:
            return None
        if max_len < min_len:
            return None
        return BytesT(rng.randint(min_len, max_len))

    if kind is StringT:
        max_dyn_len = 128
        min_len = 1
        max_len = max_dyn_len
        if isinstance(target_type, StringT):
            min_len = target_type.length + 1
        elif isinstance(target_type, BytesT):
            max_len = max_dyn_len
        else:
            return None
        if max_len < min_len:
            return None
        return StringT(rng.randint(min_len, max_len))

    return None


def pick_convert_source_type(
    rng: random.Random, target_type: VyperType
) -> Optional[VyperType]:
    kinds = list(convert_source_kinds(target_type))
    if not kinds:
        return None

    for _ in range(6):
        kind = rng.choice(kinds)
        src_t = random_convert_type(rng, kind, target_type)
        if src_t is None:
            continue
        if convert_is_valid(src_t, target_type, allow_same_type=False):
            return src_t
    return None

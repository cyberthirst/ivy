from __future__ import annotations

from vyper.ast import nodes as ast
from vyper.semantics.types import (
    VyperType,
    IntegerT,
    DecimalT,
    BytesM_T,
)
from vyper.semantics.types.user import FlagT

from fuzzer.mutator.type_utils import is_dereferenceable, dereference_child_types


def augassign_ops_for_type(typ: VyperType) -> list[type[ast.VyperNode]]:
    if not getattr(typ, "_is_prim_word", False):
        return []

    if isinstance(typ, IntegerT):
        ops = [
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.FloorDiv,
            ast.Mod,
            ast.Pow,
            ast.BitAnd,
            ast.BitOr,
            ast.BitXor,
        ]
        if typ.bits == 256:
            ops.extend([ast.LShift, ast.RShift])
        return ops

    if isinstance(typ, DecimalT):
        return [ast.Add, ast.Sub, ast.Mult, ast.Div]

    if isinstance(typ, BytesM_T):
        ops = [ast.BitAnd, ast.BitOr, ast.BitXor]
        if typ.length == 32:
            ops.extend([ast.LShift, ast.RShift])
        return ops

    if isinstance(typ, FlagT):
        return [ast.BitAnd, ast.BitOr, ast.BitXor]

    return []


def is_augassignable_type(typ: VyperType) -> bool:
    return bool(augassign_ops_for_type(typ))


def augassign_rhs_type(
    op_cls: type[ast.VyperNode], target_type: VyperType
) -> VyperType:
    if op_cls in (ast.LShift, ast.RShift):
        return IntegerT(False, 256)
    return target_type


def can_reach_augassignable(typ: VyperType, max_steps: int) -> bool:
    if is_augassignable_type(typ):
        return True
    if max_steps <= 0:
        return False
    if not is_dereferenceable(typ):
        return False
    for child_t in dereference_child_types(typ):
        if can_reach_augassignable(child_t, max_steps - 1):
            return True
    return False

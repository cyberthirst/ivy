from __future__ import annotations

from typing import Optional

from vyper.abi_types import ABI_Tuple
from vyper.ast import nodes as ast
from vyper.semantics.types import BoolT, BytesT, StringT, TupleT, VyperType


def raw_call_target_spec(target_type: Optional[VyperType]) -> Optional[tuple[str, int]]:
    if target_type is None:
        return "void", 0
    if isinstance(target_type, BoolT):
        return "bool", 0
    if isinstance(target_type, BytesT):
        if target_type.length < 1:
            return None
        return "bytes", target_type.length
    if isinstance(target_type, TupleT) and len(target_type.member_types) == 2:
        first_t, second_t = target_type.member_types
        if (
            first_t.compare_type(BoolT())
            and isinstance(second_t, BytesT)
            and second_t.length > 0
        ):
            return "tuple", second_t.length
    return None


def literal_len(node: ast.VyperNode) -> Optional[int]:
    if isinstance(node, ast.Bytes):
        return len(node.value)
    if isinstance(node, ast.Str):
        return len(node.value)
    return None


def convert_literal_length_ok(src_expr: ast.VyperNode, dst_type: VyperType) -> bool:
    if not isinstance(dst_type, (BytesT, StringT)):
        return True
    lit_len = literal_len(src_expr)
    if lit_len is None:
        return True
    return lit_len <= dst_type.length


def abi_encode_maxlen(
    arg_types: list[VyperType], *, ensure_tuple: bool, has_method_id: bool
) -> Optional[int]:
    if not arg_types:
        return None

    try:
        abi_types = [arg_t.abi_type for arg_t in arg_types]
        encoded_shape = (
            abi_types[0]
            if len(abi_types) == 1 and not ensure_tuple
            else ABI_Tuple(abi_types)
        )
        maxlen = encoded_shape.size_bound()
    except Exception:
        return None

    if has_method_id:
        maxlen += 4
    return maxlen

"""Index generation helpers for sequence subscripting.

Extracted from ExprGenerator._generate_index_for_sequence to enable
unit testing and improve readability.
"""

from __future__ import annotations

import random

from vyper.ast import nodes as ast
from vyper.semantics.types import IntegerT, BoolT

from fuzzer.mutator import ast_builder


# Standard index type for array access
INDEX_TYPE = IntegerT(False, 256)


def small_literal_index(rng: random.Random, seq_length: int) -> ast.Int:
    """Generate a small literal index within bounds for the given length."""
    if seq_length <= 1:
        val = 0
    else:
        val = rng.randint(0, min(2, seq_length - 1))
    return ast_builder.literal(val, INDEX_TYPE)


def build_len_call(base_node: ast.VyperNode) -> ast.Call:
    """Build len(base_node) call with proper type metadata."""
    len_call = ast.Call(func=ast.Name(id="len"), args=[base_node], keywords=[])
    len_call._metadata = getattr(len_call, "_metadata", {})
    len_call._metadata["type"] = INDEX_TYPE
    return len_call


def build_guarded_index(
    i_expr: ast.VyperNode,
    len_call: ast.VyperNode,
) -> ast.IfExp:
    """Build a bounds-guarded index expression.

    Returns: i if i < len else (len-1 if len > 0 else 0)

    This ensures the index is always valid at runtime, even if i_expr
    could be out of bounds.
    """
    zero = ast_builder.uint256_literal(0)
    one = ast_builder.uint256_literal(1)

    # len > 0
    len_gt_zero = ast.Compare(
        left=len_call,
        ops=[ast.Gt()],
        comparators=[zero],
    )
    len_gt_zero._metadata = {"type": BoolT()}

    # len - 1
    len_minus_one = ast.BinOp(left=len_call, op=ast.Sub(), right=one)
    len_minus_one._metadata = {"type": INDEX_TYPE}

    # (len-1 if len > 0 else 0)
    safe_fallback = ast.IfExp(
        test=len_gt_zero,
        body=len_minus_one,
        orelse=zero,
    )
    safe_fallback._metadata = {"type": INDEX_TYPE}

    # i < len
    cond = ast.Compare(left=i_expr, ops=[ast.Lt()], comparators=[len_call])
    cond._metadata = {"type": BoolT()}

    # i if i < len else safe_fallback
    guarded = ast.IfExp(test=cond, body=i_expr, orelse=safe_fallback)
    guarded._metadata = {"type": INDEX_TYPE}
    return guarded


def pick_oob_value(seq_length: int, rng: random.Random, cap_prob: float) -> int:
    """Pick an out-of-bounds index value.

    Args:
        seq_length: The declared length/capacity of the sequence
        rng: Random number generator
        cap_prob: Probability of choosing exactly seq_length vs seq_length+1

    Returns:
        An index value >= seq_length (guaranteed out of bounds)
    """
    if rng.random() < cap_prob:
        return seq_length if seq_length > 0 else 1
    else:
        return seq_length + 1 if seq_length > 0 else 1

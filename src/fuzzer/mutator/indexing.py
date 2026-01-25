"""Index generation helpers for sequence subscripting.

Extracted from ExprGenerator._generate_index_for_sequence to enable
unit testing and improve readability.
"""

from __future__ import annotations

import random

from vyper.ast import nodes as ast
from vyper.semantics.types import IntegerT

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
) -> ast.BinOp:
    """Build a bounds-guarded index expression.

    Returns: i % max(len, 1)

    This keeps the index in-bounds when len > 0, while avoiding a zero divisor
    for empty arrays.
    """
    one = ast_builder.uint256_literal(1)
    max_len = ast.Call(func=ast.Name(id="max"), args=[len_call, one], keywords=[])
    max_len._metadata = {"type": INDEX_TYPE}
    return ast_builder.uint256_binop(i_expr, ast.Mod(), max_len)


def build_dyn_last_index(len_call: ast.VyperNode) -> ast.BinOp:
    """Build a last-element index for dynarrays.

    Returns: max(len, 1) - 1
    """
    one = ast_builder.uint256_literal(1)
    max_len = ast.Call(func=ast.Name(id="max"), args=[len_call, one], keywords=[])
    max_len._metadata = {"type": INDEX_TYPE}
    return ast_builder.uint256_binop(max_len, ast.Sub(), one)


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

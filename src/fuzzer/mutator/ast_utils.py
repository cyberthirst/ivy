from __future__ import annotations

from typing import Any

from vyper.ast import nodes as ast

_IGNORED_FIELDS = {
    "ast_type",
    "col_offset",
    "end_col_offset",
    "end_lineno",
    "full_source_code",
    "lineno",
    "node_id",
    "node_source_code",
    "src",
}


def ast_equivalent(left: Any, right: Any) -> bool:
    if left is right:
        return True
    if left is None or right is None:
        return False
    if type(left) is not type(right):
        return False

    if isinstance(left, ast.VyperNode):
        for field in sorted(left.get_fields()):
            if field in _IGNORED_FIELDS:
                continue
            if not ast_equivalent(getattr(left, field), getattr(right, field)):
                return False
        return True

    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(ast_equivalent(l, r) for l, r in zip(left, right))

    if isinstance(left, dict):
        if left.keys() != right.keys():
            return False
        return all(ast_equivalent(left[k], right[k]) for k in left)

    return left == right


def body_is_terminated(body: list[ast.VyperNode]) -> bool:
    if not body:
        return False
    last_stmt = body[-1]
    return isinstance(last_stmt, (ast.Continue, ast.Break, ast.Return, ast.Raise))

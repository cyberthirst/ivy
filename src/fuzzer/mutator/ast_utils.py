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
        return all(ast_equivalent(lhs, rhs) for lhs, rhs in zip(left, right))

    if isinstance(left, dict):
        if left.keys() != right.keys():
            return False
        return all(ast_equivalent(left[k], right[k]) for k in left)

    return left == right


_BODY_FIELDS: dict[type, set[str]] = {
    ast.Module: {"body"},
    ast.FunctionDef: {"body"},
    ast.If: {"body", "orelse"},
    ast.For: {"body"},
    ast.InterfaceDef: {"body"},
}


def _collect_hoisted(node: ast.VyperNode) -> list[tuple[int, ast.VyperNode]]:
    found: list[tuple[int, ast.VyperNode]] = []
    meta = getattr(node, "_metadata", None)
    if meta and "hoisted_prelude" in meta:
        found.append((meta["hoist_seq"], meta["hoisted_prelude"]))

    # Skip body/orelse fields — those are handled by _process_body's own
    # recursion.  Descending into them here would double-collect any hoisted
    # decls from nested statements.
    skip_fields = _BODY_FIELDS.get(type(node), set())
    for field_name in node.get_fields():
        if field_name in skip_fields:
            continue
        value = getattr(node, field_name, None)
        if isinstance(value, ast.VyperNode):
            found.extend(_collect_hoisted(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, ast.VyperNode):
                    found.extend(_collect_hoisted(item))
    return found


def _process_body(body: list[ast.VyperNode]) -> None:
    i = 0
    while i < len(body):
        stmt = body[i]
        hoisted = _collect_hoisted(stmt)
        if hoisted:
            hoisted.sort(key=lambda pair: pair[0])
            decls = [decl for _, decl in hoisted]
            body[i:i] = decls
            i += len(decls)

        for field_name in _BODY_FIELDS.get(type(stmt), set()):
            nested = getattr(stmt, field_name, None)
            if isinstance(nested, list):
                _process_body(nested)

        i += 1


def hoist_prelude_decls(root: ast.Module) -> None:
    _process_body(root.body)


def contains_call(node: ast.VyperNode) -> bool:
    if isinstance(node, (ast.Call, ast.StaticCall, ast.ExtCall)):
        return True
    for field in node.get_fields():
        val = getattr(node, field, None)
        if isinstance(val, ast.VyperNode) and contains_call(val):
            return True
        if isinstance(val, list):
            if any(isinstance(v, ast.VyperNode) and contains_call(v) for v in val):
                return True
    return False


def body_is_terminated(body: list[ast.VyperNode]) -> bool:
    if not body:
        return False
    last_stmt = body[-1]
    return isinstance(last_stmt, (ast.Continue, ast.Break, ast.Return, ast.Raise))


def expr_type(node: ast.VyperNode):
    return getattr(node, "_metadata", {}).get("type")

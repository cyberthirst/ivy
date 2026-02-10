from __future__ import annotations

from typing import Any, Iterator, Set

from vyper.ast import nodes as ast

from fuzzer.coverage_types import RuntimeBranchOutcome, RuntimeStmtSite
from ivy.source_utils import source_id_for_node


_STMT_NODE_TYPE = getattr(ast, "Stmt")


def iter_runtime_modules(root_module_t: Any) -> Iterator[Any]:
    seen: Set[Any] = set()
    stack = [root_module_t]
    while stack:
        module_t = stack.pop()
        if module_t in seen:
            continue
        seen.add(module_t)

        decl_node = getattr(module_t, "decl_node", None)
        if getattr(decl_node, "is_interface", False):
            continue

        yield module_t

        imported_modules = getattr(module_t, "imported_modules", {})
        if not isinstance(imported_modules, dict):
            continue
        for import_info in imported_modules.values():
            imported_module_t = getattr(import_info, "module_t", None)
            if imported_module_t is not None:
                stack.append(imported_module_t)


def iter_runtime_functions(root_module_t: Any) -> set[Any]:
    runtime_functions: set[Any] = set()
    for module_t in iter_runtime_modules(root_module_t):
        exposed_functions = getattr(module_t, "exposed_functions", [])
        for fn_t in exposed_functions:
            if getattr(fn_t, "is_constructor", False):
                continue
            runtime_functions.add(fn_t)
            runtime_functions.update(
                internal_fn
                for internal_fn in getattr(fn_t, "reachable_internal_functions", [])
                if not getattr(internal_fn, "is_constructor", False)
            )
    return runtime_functions


def collect_static_coverage_sites_for_contract(
    contract: Any,
) -> tuple[Set[RuntimeStmtSite], Set[RuntimeBranchOutcome]]:
    compiler_data = getattr(contract, "compiler_data", None)
    address = getattr(contract, "address", None)
    if compiler_data is None or address is None:
        return set(), set()

    root_module_t = getattr(compiler_data, "global_ctx", None)
    if root_module_t is None:
        return set(), set()

    address_str = str(address)
    stmt_sites: Set[RuntimeStmtSite] = set()
    branch_outcomes: Set[RuntimeBranchOutcome] = set()

    runtime_functions = iter_runtime_functions(root_module_t)
    for fn_t in runtime_functions:
        decl_node = getattr(fn_t, "decl_node", None)
        if not isinstance(decl_node, ast.FunctionDef):
            continue

        for node in decl_node.get_descendants(_STMT_NODE_TYPE):
            node_id = getattr(node, "node_id", None)
            if node_id is None:
                continue
            stmt_sites.add((address_str, source_id_for_node(node), node_id))

        for node in decl_node.get_descendants((ast.If, ast.Assert, ast.IfExp)):
            node_id = getattr(node, "node_id", None)
            if node_id is None:
                continue
            source_id = source_id_for_node(node)
            branch_outcomes.add((address_str, source_id, node_id, True))
            branch_outcomes.add((address_str, source_id, node_id, False))

    return stmt_sites, branch_outcomes

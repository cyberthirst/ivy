from __future__ import annotations

from vyper.ast import nodes as ast
from vyper.semantics.types.function import ContractFunctionT, StateMutability

from fuzzer.mutator.strategy import strategy
from fuzzer.mutator.mutations.base import MutationCtx


def _has_body(*, ctx: MutationCtx, **_) -> bool:
    return len(ctx.node.body) > 0


def _decorator_name(node: ast.VyperNode) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _has_decorator(func_def: ast.FunctionDef, name: str) -> bool:
    return any(_decorator_name(dec) == name for dec in func_def.decorator_list)


def _remove_decorator(func_def: ast.FunctionDef, name: str) -> bool:
    before = len(func_def.decorator_list)
    func_def.decorator_list = [
        dec for dec in func_def.decorator_list if _decorator_name(dec) != name
    ]
    return len(func_def.decorator_list) != before


def _module_nonreentrancy_by_default(func_def: ast.FunctionDef) -> bool:
    module = func_def.get_ancestor(ast.Module)
    settings = getattr(module, "settings", None)
    return bool(getattr(settings, "nonreentrancy_by_default", False))


def _is_interface_function(func_def: ast.FunctionDef) -> bool:
    return func_def.get_ancestor(ast.InterfaceDef) is not None


def _effective_nonreentrant(
    func_t: ContractFunctionT, func_def: ast.FunctionDef
) -> bool:
    if func_t.mutability == StateMutability.PURE:
        return False
    if _has_decorator(func_def, "reentrant"):
        return False
    if _has_decorator(func_def, "nonreentrant"):
        return True
    if _module_nonreentrancy_by_default(func_def) and func_t.is_external:
        return True
    return False


def _can_add_nonreentrant(*, ctx: MutationCtx, **_) -> bool:
    func_t = ctx.node._metadata.get("func_type")
    if not isinstance(func_t, ContractFunctionT):
        return False
    if _is_interface_function(ctx.node):
        return False
    if func_t.is_constructor:
        return False
    if func_t.mutability == StateMutability.PURE:
        return False
    if func_t.nonreentrant or _has_decorator(ctx.node, "nonreentrant"):
        return False
    if _has_decorator(ctx.node, "reentrant"):
        return False
    if func_t.is_external and _module_nonreentrancy_by_default(ctx.node):
        return False
    if ctx.function_registry.reachable_from_nonreentrant(func_t.name):
        return False
    if ctx.function_registry.reaches_nonreentrant(func_t.name):
        return False
    prob = ctx.function_registry.nonreentrant_probability(
        visibility=func_t.visibility,
        mutability=func_t.mutability,
    )
    if prob <= 0:
        return False
    return ctx.rng.random() < prob


def _can_remove_nonreentrant(*, ctx: MutationCtx, **_) -> bool:
    if not isinstance(ctx.node._metadata.get("func_type"), ContractFunctionT):
        return False
    if _is_interface_function(ctx.node):
        return False
    return _has_decorator(ctx.node, "nonreentrant")


def _can_add_reentrant(*, ctx: MutationCtx, **_) -> bool:
    func_t = ctx.node._metadata.get("func_type")
    if not isinstance(func_t, ContractFunctionT):
        return False
    if _is_interface_function(ctx.node):
        return False
    if not _module_nonreentrancy_by_default(ctx.node):
        return False
    if _has_decorator(ctx.node, "reentrant"):
        return False
    if _has_decorator(ctx.node, "nonreentrant"):
        return False
    return True


def _can_remove_reentrant(*, ctx: MutationCtx, **_) -> bool:
    if not isinstance(ctx.node._metadata.get("func_type"), ContractFunctionT):
        return False
    if _is_interface_function(ctx.node):
        return False
    return _has_decorator(ctx.node, "reentrant")


@strategy(
    name="function.inject_statement",
    type_classes=(ast.FunctionDef,),
    tags=frozenset({"mutation", "function"}),
    is_applicable="_has_body",
)
def _inject_statement(*, ctx: MutationCtx, **_) -> ast.FunctionDef:
    ctx.stmt_gen.inject_statements(
        ctx.node.body,
        ctx.context,
        ctx.node,
        depth=ctx.stmt_gen.root_depth(),
        min_stmts=1,
        max_stmts=1,
    )
    return ctx.node


@strategy(
    name="function.add_nonreentrant",
    type_classes=(ast.FunctionDef,),
    tags=frozenset({"mutation", "function"}),
    is_applicable="_can_add_nonreentrant",
)
def _add_nonreentrant(*, ctx: MutationCtx, **_) -> ast.FunctionDef:
    ctx.node.decorator_list.append(ast.Name(id="nonreentrant"))
    func_t = ctx.node._metadata.get("func_type")
    if isinstance(func_t, ContractFunctionT):
        func_t.nonreentrant = _effective_nonreentrant(func_t, ctx.node)
        ctx.function_registry.refresh_reachable_from_nonreentrant()
    return ctx.node


@strategy(
    name="function.remove_nonreentrant",
    type_classes=(ast.FunctionDef,),
    tags=frozenset({"mutation", "function"}),
    is_applicable="_can_remove_nonreentrant",
)
def _remove_nonreentrant(*, ctx: MutationCtx, **_) -> ast.FunctionDef:
    removed = _remove_decorator(ctx.node, "nonreentrant")
    if removed:
        func_t = ctx.node._metadata.get("func_type")
        if isinstance(func_t, ContractFunctionT):
            func_t.nonreentrant = _effective_nonreentrant(func_t, ctx.node)
            ctx.function_registry.refresh_reachable_from_nonreentrant()
    return ctx.node


@strategy(
    name="function.add_reentrant",
    type_classes=(ast.FunctionDef,),
    tags=frozenset({"mutation", "function"}),
    is_applicable="_can_add_reentrant",
)
def _add_reentrant(*, ctx: MutationCtx, **_) -> ast.FunctionDef:
    ctx.node.decorator_list.append(ast.Name(id="reentrant"))
    func_t = ctx.node._metadata.get("func_type")
    if isinstance(func_t, ContractFunctionT):
        func_t.nonreentrant = _effective_nonreentrant(func_t, ctx.node)
        ctx.function_registry.refresh_reachable_from_nonreentrant()
    return ctx.node


@strategy(
    name="function.remove_reentrant",
    type_classes=(ast.FunctionDef,),
    tags=frozenset({"mutation", "function"}),
    is_applicable="_can_remove_reentrant",
)
def _remove_reentrant(*, ctx: MutationCtx, **_) -> ast.FunctionDef:
    removed = _remove_decorator(ctx.node, "reentrant")
    if removed:
        func_t = ctx.node._metadata.get("func_type")
        if isinstance(func_t, ContractFunctionT):
            func_t.nonreentrant = _effective_nonreentrant(func_t, ctx.node)
            ctx.function_registry.refresh_reachable_from_nonreentrant()
    return ctx.node

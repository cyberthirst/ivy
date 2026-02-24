from __future__ import annotations

from vyper.ast import nodes as ast
from vyper.semantics.types.function import ContractFunctionT

from fuzzer.mutator.strategy import strategy
from fuzzer.mutator.mutations.base import MutationCtx


def _get_func_type(node: ast.Call) -> ContractFunctionT | None:
    func_type = getattr(node.func, "_metadata", {}).get("type")
    if isinstance(func_type, ContractFunctionT):
        return func_type
    return None


def _arg_type(func_t: ContractFunctionT, idx: int):
    n_pos = len(func_t.positional_args)
    if idx < n_pos:
        return func_t.positional_args[idx].typ
    return func_t.keyword_args[idx - n_pos].typ


def _call(ctx: MutationCtx) -> ast.Call:
    assert isinstance(ctx.node, ast.Call)
    return ctx.node


def _has_args(*, ctx: MutationCtx, **_) -> bool:
    node = _call(ctx)
    func_t = _get_func_type(node)
    if func_t is None:
        return False
    return len(node.args) > 0


def _has_unprovided_kwargs(*, ctx: MutationCtx, **_) -> bool:
    node = _call(ctx)
    func_t = _get_func_type(node)
    if func_t is None:
        return False
    max_args = len(func_t.positional_args) + len(func_t.keyword_args)
    return len(node.args) < max_args


@strategy(
    name="call.mutate_argument",
    type_classes=(ast.Call,),
    tags=frozenset({"mutation", "call"}),
    is_applicable="_has_args",
)
def _mutate_argument(*, ctx: MutationCtx, **_) -> ast.Call:
    node = _call(ctx)
    func_t = _get_func_type(node)
    assert func_t is not None

    idx = ctx.rng.randrange(len(node.args))
    typ = _arg_type(func_t, idx)

    ctx.expr_gen._active_context = ctx.context
    new_expr = ctx.expr_gen.generate(
        typ, ctx.context, depth=ctx.expr_gen.root_depth()
    )
    if new_expr is not None:
        node.args[idx] = new_expr
    return node


@strategy(
    name="call.provide_default_arg",
    type_classes=(ast.Call,),
    tags=frozenset({"mutation", "call"}),
    is_applicable="_has_unprovided_kwargs",
)
def _provide_default_arg(*, ctx: MutationCtx, **_) -> ast.Call:
    node = _call(ctx)
    func_t = _get_func_type(node)
    assert func_t is not None

    kw_idx = len(node.args) - len(func_t.positional_args)
    kw_arg = func_t.keyword_args[kw_idx]

    ctx.expr_gen._active_context = ctx.context
    new_expr = ctx.expr_gen.generate(
        kw_arg.typ, ctx.context, depth=ctx.expr_gen.root_depth()
    )
    if new_expr is not None:
        node.args.append(new_expr)
    return node

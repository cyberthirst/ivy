from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Optional, Union

from vyper.abi_types import ABI_Tuple
from vyper.ast import nodes as ast
from vyper.semantics.types import (
    VyperType,
    IntegerT,
    BoolT,
    AddressT,
    StructT,
    BytesM_T,
    BytesT,
    StringT,
    SArrayT,
    DArrayT,
    TupleT,
    HashMapT,
    DecimalT,
    InterfaceT,
)
from vyper.semantics.types.subscriptable import _SequenceT
from vyper.semantics.analysis.base import DataLocation, VarInfo, Modifiability
from vyper.semantics.types.function import StateMutability, ContractFunctionT

from fuzzer.mutator import ast_builder
from fuzzer.mutator.ast_utils import contains_call
from fuzzer.mutator.base_generator import BaseGenerator
from fuzzer.mutator.convert_utils import (
    convert_is_valid,
    convert_target_supported,
    pick_convert_source_type,
)
from fuzzer.mutator.config import ExprGeneratorConfig, DepthConfig
from fuzzer.mutator.constant_folding import (
    ConstEvalError,
    ConstEvalNonConstant,
    constant_folds_to_zero,
    evaluate_constant_expression,
    fold_constant_expression_status,
)
from fuzzer.mutator.context import GenerationContext, ExprMutability
from fuzzer.mutator.existing_type_pool import collect_existing_reachable_types
from fuzzer.mutator.function_registry import FunctionRegistry
from fuzzer.mutator.indexing import (
    small_literal_index,
    random_literal_index,
    random_index_type,
    build_len_call,
    build_guarded_index,
    build_dyn_last_index,
    INDEX_TYPE,
)
from fuzzer.mutator.interface_registry import InterfaceRegistry
from fuzzer.mutator.literal_generator import LiteralGenerator
from fuzzer.mutator.name_generator import FreshNameGenerator
from fuzzer.mutator.strategy import strategy
from fuzzer.mutator.type_utils import (
    DerefCandidate,
    dereference_candidates,
    find_dereference_sources,
    find_dereferenceable_vars,
    is_dereferenceable,
)
from fuzzer.type_generator import TypeGenerator
from ivy.builtins.builtins import builtin_convert


@dataclass
class ExprGenCtx:
    target_type: VyperType
    context: GenerationContext
    depth: int
    gen: ExprGenerator
    allow_tuple_literal: bool
    allow_empty_list: bool = True


class ExprGenerator(BaseGenerator):
    def __init__(
        self,
        literal_generator: LiteralGenerator,
        rng: random.Random,
        interface_registry: InterfaceRegistry,
        function_registry: FunctionRegistry,
        type_generator: TypeGenerator,
        cfg: Optional[ExprGeneratorConfig] = None,
        depth_cfg: Optional[DepthConfig] = None,
        name_generator: Optional[FreshNameGenerator] = None,
    ):
        self.literal_generator = literal_generator
        self.function_registry: FunctionRegistry = function_registry
        self.interface_registry: InterfaceRegistry = interface_registry
        self.type_generator = type_generator
        self.cfg = cfg or ExprGeneratorConfig()
        self.name_generator = name_generator or FreshNameGenerator()

        super().__init__(rng, depth_cfg)
        self._hoist_seq: int = 0
        self._active_context: Optional[GenerationContext] = None

        self._builtin_handlers = {
            "min": self._builtin_min_max,
            "max": self._builtin_min_max,
            "abs": self._builtin_abs,
            "floor": self._builtin_floor_ceil,
            "ceil": self._builtin_floor_ceil,
            "len": self._builtin_len,
            "keccak256": self._builtin_keccak256,
            "concat": self._builtin_concat,
            "convert": self._builtin_convert,
            "slice": self._builtin_slice,
            "abi_encode": self._builtin_abi_encode,
        }

    def reset_state(self) -> None:
        self._hoist_seq = 0
        self._active_context = None

    def hoist_to_tmp_var(
        self, expr: ast.VyperNode, *, prefix: Optional[str] = None
    ) -> ast.VyperNode:
        context = self._active_context
        if context is None:
            raise ValueError("hoist_to_tmp_var requires an active generation context")

        expr_type = self._expr_type(expr)
        name = self.name_generator.generate(prefix=prefix)

        if context.is_module_scope:
            var_info = VarInfo(
                typ=expr_type,
                location=DataLocation.UNSET,
                modifiability=Modifiability.CONSTANT,
            )
            annotation = ast.Call(
                func=ast.Name(id="constant"),
                args=[ast.Name(id=str(expr_type))],
            )
            decl: ast.VyperNode = ast.VariableDecl(
                target=ast.Name(id=name),
                annotation=annotation,
                value=expr,
            )
            try:
                value = evaluate_constant_expression(expr, context.constants)
            except Exception:
                pass
            else:
                context.add_constant(name, value)
        else:
            var_info = VarInfo(
                typ=expr_type,
                location=DataLocation.MEMORY,
                modifiability=Modifiability.MODIFIABLE,
            )
            decl = ast.AnnAssign(
                target=ast.Name(id=name),
                annotation=ast.Name(id=str(expr_type)),
                value=expr,
            )

        # Don't add to context: hoisted temps are internal to the expression
        # being built. Adding them makes them visible to find_matching_vars(),
        # which lets later statement generation emit assignments to the name
        # before hoist_prelude_decls inserts the declaration.
        ref = ast_builder.var_ref(name, var_info)
        ref._metadata = {"type": expr_type, "varinfo": var_info}
        ref._metadata["hoisted_prelude"] = decl
        ref._metadata["hoist_seq"] = self._hoist_seq
        self._hoist_seq += 1
        return ref

    def generate(
        self,
        target_type: VyperType,
        context: GenerationContext,
        depth: int = 0,
        *,
        allow_tuple_literal: bool = False,
        allow_recursion: bool = False,
        allow_empty_list: bool = True,
    ) -> ast.VyperNode:
        self._active_context = context
        ctx = ExprGenCtx(
            target_type=target_type,
            context=context,
            depth=depth,
            gen=self,
            allow_tuple_literal=allow_tuple_literal,
            allow_empty_list=allow_empty_list,
        )

        # Use tag-based filtering for terminal vs recursive strategies
        if allow_recursion or self.should_continue(depth):
            include_tags = ("expr",)
        else:
            include_tags = ("expr", "terminal")

        # Collect strategies via registry
        strategies = self._strategy_registry.collect(
            type_class=type(target_type),
            include_tags=include_tags,
            context={"ctx": ctx},
        )

        if isinstance(target_type, TupleT) and not allow_tuple_literal:
            # Tuple literals are rejected in assignment-like contexts; use empty(T).
            fallback = lambda: self._build_empty(target_type)
        else:
            fallback = lambda: self._generate_literal(ctx=ctx)

        return self._strategy_executor.execute_with_retry(
            strategies,
            policy="weighted_random",  # TODO nested hash maps
            context={"ctx": ctx},
            fallback=fallback,
        )

    def generate_nonzero_expr(
        self,
        target_type: VyperType,
        context: GenerationContext,
        depth: int,
    ) -> ast.VyperNode:
        while True:
            expr = self.generate(target_type, context, depth)
            if not constant_folds_to_zero(expr, context.constants):
                return expr

    def generate_nonconstant_expr(
        self,
        target_type: VyperType,
        context: GenerationContext,
        depth: int,
        *,
        allow_tuple_literal: bool = False,
        allow_recursion: bool = False,
        retries: Optional[int] = None,
    ) -> ast.VyperNode:
        def reject_if(expr: ast.VyperNode) -> bool:
            status, _ = fold_constant_expression_status(expr, context.constants)
            return status != "non_constant"

        attempts = self.cfg.nonconst_expr_retries if retries is None else retries

        return self._retry(
            make_candidate=lambda: self.generate(
                target_type,
                context,
                depth,
                allow_tuple_literal=allow_tuple_literal,
                allow_recursion=allow_recursion,
            ),
            reject_if=reject_if,
            retries=attempts,
        )

    # Applicability/weight helpers

    def _is_var_ref_applicable(self, *, ctx: ExprGenCtx, **_) -> bool:
        return bool(ctx.context.find_matching_vars(ctx.target_type))

    def _is_literal_applicable(self, *, ctx: ExprGenCtx, **_) -> bool:
        if isinstance(ctx.target_type, TupleT):
            return ctx.allow_tuple_literal
        return True

    def _env_var_candidates(
        self, *, ctx: ExprGenCtx
    ) -> list[tuple[Optional[str], str, VyperType]]:
        if ctx.context.is_module_scope:
            return []
        if ctx.context.current_mutability in (
            ExprMutability.CONST,
            ExprMutability.PURE,
        ):
            return []

        candidates: list[tuple[Optional[str], str, VyperType]] = []
        if ctx.target_type.compare_type(AddressT()):
            candidates.extend(
                [
                    ("msg", "sender", AddressT()),
                    ("tx", "origin", AddressT()),
                    (None, "self", AddressT()),
                ]
            )
        else:
            uint256_t = IntegerT(False, 256)
            if not ctx.target_type.compare_type(uint256_t):
                return candidates

            candidates.append(("block", "timestamp", uint256_t))
            if ctx.context.current_function_mutability == StateMutability.PAYABLE:
                candidates.append(("msg", "value", uint256_t))
            # TODO: support msg.data

        return candidates

    def _is_env_var_applicable(self, *, ctx: ExprGenCtx, **_) -> bool:
        return bool(self._env_var_candidates(ctx=ctx))

    def _weight_var_ref(self, *, ctx: ExprGenCtx, **_) -> float:
        # Slightly bias towards var refs when there are many
        n = len(ctx.context.find_matching_vars(ctx.target_type))
        cfg = self.cfg
        if n == 0:
            return 1.0
        return min(
            cfg.var_ref_weight_max,
            cfg.var_ref_weight_base + cfg.var_ref_weight_scale * n,
        )

    def _is_unary_minus_applicable(self, *, ctx: ExprGenCtx, **_) -> bool:
        return isinstance(ctx.target_type, IntegerT) and ctx.target_type.is_signed

    def _is_func_call_applicable(self, *, ctx: ExprGenCtx, **_) -> bool:
        # TODO do we allow constant folding of some builtins?
        # if yes, we'd want to drop this restriction
        if ctx.context.current_mutability == ExprMutability.CONST:
            return False
        return not ctx.context.is_module_scope

    def _raw_call_target_spec(
        self, target_type: Optional[VyperType]
    ) -> Optional[tuple[str, int]]:
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

    def can_generate_raw_call(
        self,
        context: GenerationContext,
        target_type: Optional[VyperType],
    ) -> bool:
        if context.is_module_scope:
            return False
        if context.current_mutability in (ExprMutability.CONST, ExprMutability.PURE):
            return False
        if context.current_function_mutability == StateMutability.PURE:
            return False
        return self._raw_call_target_spec(target_type) is not None

    def _is_raw_call_expr_applicable(self, *, ctx: ExprGenCtx, **_) -> bool:
        return self.can_generate_raw_call(ctx.context, ctx.target_type)

    def _existing_reachable_types(
        self,
        context: GenerationContext,
        *,
        skip: Optional[set[type]] = None,
    ) -> list[VyperType]:
        return collect_existing_reachable_types(
            context=context,
            deref_max_steps=self.cfg.subscript_chain_max_steps,
            function_registry=self.function_registry,
            skip=skip,
        )

    def _find_deref_bases(
        self,
        *,
        ctx: ExprGenCtx,
        allow_attribute: bool,
        allow_subscript: bool,
    ) -> list[tuple[str, VyperType]]:
        vars_dict = dict(ctx.context.find_matching_vars(None))
        if allow_subscript:
            vars_dict = {
                name: var_info
                for name, var_info in vars_dict.items()
                if not self._is_empty_constant_array(name, var_info, ctx.context)
            }
        return find_dereferenceable_vars(
            ctx.target_type,
            vars_dict,
            max_steps=self.cfg.subscript_chain_max_steps,
            allow_attribute=allow_attribute,
            allow_subscript=allow_subscript,
        )

    def _find_deref_func_bases(
        self,
        *,
        ctx: ExprGenCtx,
        allow_attribute: bool,
        allow_subscript: bool,
    ) -> list[tuple[ContractFunctionT, VyperType]]:
        funcs = self._get_callable_functions(ctx.context)
        return find_dereference_sources(
            ctx.target_type,
            (
                (func, func.return_type)
                for func in funcs
                if func.return_type is not None
            ),
            max_steps=self.cfg.subscript_chain_max_steps,
            allow_attribute=allow_attribute,
            allow_subscript=allow_subscript,
        )

    def _is_empty_constant_array(
        self,
        name: str,
        var_info: VarInfo,
        context: GenerationContext,
    ) -> bool:
        if var_info.modifiability != Modifiability.CONSTANT:
            return False
        if not isinstance(var_info.typ, (SArrayT, DArrayT)):
            return False
        if name not in context.constants:
            return False
        value = context.constants[name]
        try:
            return len(value) == 0
        except Exception:
            return getattr(value, "length", None) == 0

    def _folded_constant_sequence_length(
        self,
        node: ast.VyperNode,
        constants: dict[str, object],
    ) -> Optional[int]:
        try:
            value = evaluate_constant_expression(node, constants)
        except Exception:
            return None
        return len(value)

    def _weight_deref_like(self, n: int) -> float:
        cfg = self.cfg
        if n == 0:
            return 1.0
        return min(
            cfg.subscript_weight_max,
            cfg.subscript_weight_base + cfg.subscript_weight_scale * n,
        )

    def _is_attribute_applicable(self, *, ctx: ExprGenCtx, **_) -> bool:
        return bool(
            self._find_deref_bases(ctx=ctx, allow_attribute=True, allow_subscript=False)
        )

    def _is_subscript_applicable(self, *, ctx: ExprGenCtx, **_) -> bool:
        return bool(
            self._find_deref_bases(ctx=ctx, allow_attribute=False, allow_subscript=True)
        )

    def _is_dereference_var_applicable(self, *, ctx: ExprGenCtx, **_) -> bool:
        return bool(
            self._find_deref_bases(ctx=ctx, allow_attribute=True, allow_subscript=True)
        )

    def _is_dereference_func_call_applicable(self, *, ctx: ExprGenCtx, **_) -> bool:
        if not self._is_func_call_applicable(ctx=ctx):
            return False
        return bool(
            self._find_deref_func_bases(
                ctx=ctx, allow_attribute=True, allow_subscript=True
            )
        )

    def _is_ifexp_applicable(self, *, ctx: ExprGenCtx, **_) -> bool:
        # Disallow if-expressions in constant contexts due to compiler limitation
        if ctx.context.current_mutability == ExprMutability.CONST:
            return False
        # TODO: enable once https://github.com/vyperlang/vyper/issues/3480
        # or https://github.com/vyperlang/vyper/issues/4825 is fixed
        if isinstance(ctx.target_type, TupleT):
            return False
        return True

    def _is_convert_applicable(self, *, ctx: ExprGenCtx, **_) -> bool:
        if ctx.context.current_mutability == ExprMutability.CONST:
            return False
        return convert_target_supported(ctx.target_type)

    def _weight_literal(self, **_) -> float:
        return self.cfg.literal_weight

    def _weight_ifexp(self, **_) -> float:
        return self.cfg.ifexp_weight

    def _weight_env_var(self, **_) -> float:
        return self.cfg.env_var_weight

    def _weight_convert(self, **_) -> float:
        return self.cfg.convert_weight

    def _weight_raw_call(self, **_) -> float:
        return self.cfg.raw_call_weight

    def _weight_attribute(self, *, ctx: ExprGenCtx, **_) -> float:
        n = len(
            self._find_deref_bases(ctx=ctx, allow_attribute=True, allow_subscript=False)
        )
        return self._weight_deref_like(n)

    def _weight_subscript(self, *, ctx: ExprGenCtx, **_) -> float:
        n = len(
            self._find_deref_bases(ctx=ctx, allow_attribute=False, allow_subscript=True)
        )
        return self._weight_deref_like(n)

    def _weight_dereference_var(self, *, ctx: ExprGenCtx, **_) -> float:
        n = len(
            self._find_deref_bases(ctx=ctx, allow_attribute=True, allow_subscript=True)
        )
        return self._weight_deref_like(n)

    def _weight_dereference_func_call(self, *, ctx: ExprGenCtx, **_) -> float:
        return self.cfg.dereference_func_call_weight

    def _expr_type(self, node: ast.VyperNode) -> Optional[VyperType]:
        return getattr(node, "_metadata", {}).get("type")

    def _literal_len(self, node: ast.VyperNode) -> Optional[int]:
        if isinstance(node, ast.Bytes):
            return len(node.value)
        if isinstance(node, ast.Str):
            return len(node.value)
        return None

    def _convert_literal_length_ok(
        self, src_expr: ast.VyperNode, dst_type: VyperType
    ) -> bool:
        if not isinstance(dst_type, (BytesT, StringT)):
            return True
        lit_len = self._literal_len(src_expr)
        if lit_len is None:
            return True
        return lit_len <= dst_type.length

    def _convert_literal_ok(
        self,
        src_expr: ast.VyperNode,
        dst_type: VyperType,
        context: GenerationContext,
    ) -> bool:
        status, folded = fold_constant_expression_status(src_expr, context.constants)
        if status == "invalid_constant":
            return False

        lit_node = folded if status == "value" and folded is not None else src_expr
        if not self._convert_literal_length_ok(lit_node, dst_type):
            return False

        if status != "value":
            return True
        if not isinstance(dst_type, (IntegerT, AddressT)):
            return True

        try:
            value = evaluate_constant_expression(src_expr, context.constants)
        except ConstEvalNonConstant:
            return True
        except ConstEvalError:
            return False
        try:
            builtin_convert(value, dst_type)
            return True
        except Exception:
            return False

    def _convert_type_literal(self, target_type: VyperType) -> ast.Name:
        return ast.Name(id=str(target_type))

    def maybe_convert_expr(
        self,
        src_expr: ast.VyperNode,
        target_type: VyperType,
        context: GenerationContext,
    ) -> Optional[ast.Call]:
        src_type = self._expr_type(src_expr)
        if src_type is None:
            return None
        if not convert_is_valid(src_type, target_type):
            return None
        if not self._convert_literal_ok(src_expr, target_type, context):
            return None

        func_node = ast.Name(id="convert")
        func_node._metadata = {}
        if "convert" in self.function_registry.builtins:
            func_node._metadata["type"] = self.function_registry.builtins["convert"]
        type_node = self._convert_type_literal(target_type)

        call_node = ast.Call(func=func_node, args=[src_expr, type_node], keywords=[])
        call_node._metadata = {"type": target_type}
        return call_node

    def _pick_convert_source_expr(
        self, target_type: VyperType, context: GenerationContext, depth: int
    ) -> Optional[ast.VyperNode]:
        var_candidates = [
            (name, info)
            for name, info in context.find_matching_vars(None)
            if convert_is_valid(info.typ, target_type)
        ]
        if var_candidates:
            return self._generate_variable_ref(self.rng.choice(var_candidates), context)

        for _ in range(self.cfg.convert_max_attempts):
            src_t = pick_convert_source_type(self.rng, target_type)
            if src_t is None:
                return None
            src_expr = self.generate(src_t, context, self.child_depth(depth))
            src_type = self._expr_type(src_expr)
            if src_type is None:
                continue
            if not convert_is_valid(src_type, target_type):
                continue
            if not self._convert_literal_ok(src_expr, target_type, context):
                continue
            return src_expr

        return None

    def _get_external_functions(self) -> list[ContractFunctionT]:
        return [
            f
            for f in self.function_registry.functions.values()
            if f.is_external and f.name not in ("__init__", "__default__")
        ]

    def _build_abi_encoded_calldata(
        self,
        func: ContractFunctionT,
        context: GenerationContext,
        depth: int,
    ) -> ast.VyperNode:
        selector_int = next(iter(func.method_ids.values()))
        selector_bytes = selector_int.to_bytes(4, "big")
        method_id_node = ast_builder.literal(selector_bytes, BytesM_T(4))

        arg_types = [arg.typ for arg in func.positional_args]
        if not arg_types:
            method_id_node._metadata = {"type": BytesM_T(4)}
            return method_id_node

        args = [
            self._maybe_hoist_abi_encode_arg(
                self._generate_abi_encode_arg_expr(arg_t, context, depth),
                context,
            )
            for arg_t in arg_types
        ]

        keywords = [
            ast.keyword(arg="method_id", value=method_id_node),
            ast.keyword(
                arg="ensure_tuple",
                value=ast_builder.literal(True, BoolT()),
            ),
        ]

        func_node = ast.Name(id="abi_encode")
        func_node._metadata = {}
        if "abi_encode" in self.function_registry.builtins:
            func_node._metadata["type"] = self.function_registry.builtins["abi_encode"]

        call_node = ast.Call(func=func_node, args=args, keywords=keywords)
        maxlen = self._abi_encode_maxlen(
            arg_types, ensure_tuple=True, has_method_id=True
        )
        if maxlen is None:
            maxlen = 4 + 32 * len(arg_types)
        call_node._metadata = {"type": BytesT(maxlen)}
        return call_node

    def generate_raw_call_call(
        self,
        context: GenerationContext,
        depth: int,
        *,
        target_type: Optional[VyperType],
    ) -> Optional[ast.Call]:
        if not self.can_generate_raw_call(context, target_type):
            return None

        target_spec = self._raw_call_target_spec(target_type)
        if target_spec is None:
            return None

        target_kind, target_typ_len = target_spec
        uint256_t = IntegerT(False, 256)
        bool_t = BoolT()
        arg_depth = self.child_depth(depth)

        force_static_call = context.current_function_mutability == StateMutability.VIEW
        if force_static_call:
            is_static_call = True
        else:
            is_static_call = (
                self.rng.random() < self.cfg.raw_call_set_is_static_call_prob
            )

        revert_on_failure = target_kind in {"void", "bytes"}

        max_outsize = 0
        max_outsize_optional = target_kind in {"void", "bool"}
        set_max_outsize = (
            not max_outsize_optional
            or self.rng.random() < self.cfg.raw_call_set_max_outsize_prob
        )
        if set_max_outsize:
            max_outsize = self.rng.randint(0, target_typ_len)
            if target_kind in {"bytes", "tuple"} and max_outsize == 0:
                max_outsize = self.rng.randint(1, target_typ_len)

        external_funcs = self._get_external_functions()
        use_self = (
            external_funcs and self.rng.random() < self.cfg.raw_call_target_self_prob
        )

        if use_self:
            func = self.rng.choice(external_funcs)
            to_expr = ast.Name(id="self")
            to_expr._metadata = {"type": AddressT()}
            data_expr = self._build_abi_encoded_calldata(func, context, arg_depth)
        else:
            to_expr = self.generate(AddressT(), context, arg_depth)
            data_expr = self.generate(
                BytesT(self.rng.randint(0, 256)), context, arg_depth
            )

        func_node = ast.Name(id="raw_call")
        func_node._metadata = {}
        if (
            self.function_registry is not None
            and "raw_call" in self.function_registry.builtins
        ):
            func_node._metadata["type"] = self.function_registry.builtins["raw_call"]

        keywords: list[ast.keyword] = []
        if set_max_outsize:
            keywords.append(
                ast.keyword(
                    arg="max_outsize",
                    value=ast_builder.uint256_literal(max_outsize),
                )
            )

        if not is_static_call and self.rng.random() < self.cfg.raw_call_set_value_prob:
            randexpr = self.generate(uint256_t, context, arg_depth)
            wei_scale = ast_builder.uint256_literal(10**18)
            value_expr = ast.BinOp(left=randexpr, op=ast.Mod(), right=wei_scale)
            value_expr._metadata = {"type": uint256_t}
            keywords.append(ast.keyword(arg="value", value=value_expr))

        if is_static_call:
            keywords.append(
                ast.keyword(
                    arg="is_static_call", value=ast_builder.literal(True, bool_t)
                )
            )

        keywords.append(
            ast.keyword(
                arg="revert_on_failure",
                value=ast_builder.literal(revert_on_failure, bool_t),
            )
        )

        call_node = ast.Call(
            func=func_node, args=[to_expr, data_expr], keywords=keywords
        )

        if target_kind == "void":
            ret_type = None
        elif target_kind == "bool":
            ret_type = BoolT()
        elif target_kind == "bytes":
            ret_type = BytesT(max_outsize)
        elif target_kind == "tuple":
            ret_type = TupleT((BoolT(), BytesT(max_outsize)))
        else:
            return None

        call_node._metadata = {"type": ret_type}
        return call_node

    # Strategy methods

    @strategy(
        name="expr.var_ref",
        tags=frozenset({"expr", "terminal"}),
        is_applicable="_is_var_ref_applicable",
        weight="_weight_var_ref",
    )
    def _generate_var_ref(self, *, ctx: ExprGenCtx, **_):
        matches = ctx.context.find_matching_vars(ctx.target_type)
        if not matches:
            return None
        return self._generate_variable_ref(ctx.gen.rng.choice(matches), ctx.context)

    @strategy(
        name="expr.literal",
        tags=frozenset({"expr", "terminal"}),
        is_applicable="_is_literal_applicable",
        weight="_weight_literal",
    )
    def _generate_literal(self, *, ctx: ExprGenCtx, **_) -> ast.VyperNode:
        if isinstance(ctx.target_type, (SArrayT, DArrayT)):
            return self._generate_array_literal(ctx)
        if isinstance(ctx.target_type, TupleT):
            return self._generate_tuple_literal(ctx)
        if isinstance(ctx.target_type, StructT):
            return self._generate_struct_literal(ctx)
        if isinstance(ctx.target_type, InterfaceT):
            return self._generate_interface_literal(ctx)
        value = self.literal_generator.generate(ctx.target_type)
        literal_type = ctx.target_type
        if isinstance(ctx.target_type, BytesT):
            literal_type = BytesT(len(value))
        elif isinstance(ctx.target_type, StringT):
            literal_type = StringT(len(value))
        return ast_builder.literal(value, literal_type)

    def _generate_array_literal(self, ctx: ExprGenCtx) -> ast.List:
        target_type = ctx.target_type
        assert isinstance(target_type, (SArrayT, DArrayT))

        if isinstance(target_type, DArrayT):
            max_len = target_type.length
            min_len = 0 if ctx.allow_empty_list else 1
            if self.rng.random() < 0.01:
                length = self.rng.randint(min_len, max_len)
            else:
                length = self.rng.randint(min_len, min(10, max_len))
        else:
            length = target_type.length

        next_depth = self.child_depth(ctx.depth)
        elements = [
            self.generate(
                target_type.value_type,
                ctx.context,
                next_depth,
                allow_tuple_literal=ctx.allow_tuple_literal,
                allow_empty_list=False,
            )
            for _ in range(length)
        ]
        node = ast.List(elements=elements)
        node._metadata = {"type": target_type}
        return node

    def _generate_tuple_literal(self, ctx: ExprGenCtx) -> ast.Tuple:
        target_type = ctx.target_type
        assert isinstance(target_type, TupleT)

        next_depth = self.child_depth(ctx.depth)
        elements = [
            self.generate(
                member_type,
                ctx.context,
                next_depth,
                allow_tuple_literal=ctx.allow_tuple_literal,
            )
            for member_type in target_type.member_types
        ]
        node = ast.Tuple(elements=elements)
        node._metadata = {"type": target_type}
        return node

    def _generate_struct_literal(self, ctx: ExprGenCtx) -> ast.Call:
        target_type = ctx.target_type
        assert isinstance(target_type, StructT)

        call_node = ast.Call(func=ast.Name(id=target_type._id), args=[], keywords=[])
        next_depth = self.child_depth(ctx.depth)

        for field_name, field_type in target_type.members.items():
            field_expr = self.generate(
                field_type,
                ctx.context,
                next_depth,
                allow_tuple_literal=ctx.allow_tuple_literal,
            )
            keyword = ast.keyword(arg=field_name, value=field_expr)
            call_node.keywords.append(keyword)

        call_node._metadata = {"type": target_type}
        return call_node

    def _generate_interface_literal(self, ctx: ExprGenCtx) -> ast.Call:
        target_type = ctx.target_type
        assert isinstance(target_type, InterfaceT)

        allow_self = (
            not ctx.context.is_module_scope
            and ctx.context.current_mutability
            not in (ExprMutability.CONST, ExprMutability.PURE)
        )
        if allow_self and self.rng.random() < 0.25:
            address_node = ast.Name(id="self")
            address_node._metadata = {"type": AddressT()}
        else:
            address_value = self.literal_generator.generate(AddressT())
            address_node = ast_builder.literal(address_value, AddressT())

        return ast_builder.interface_cast(target_type, address_node)

    @strategy(
        name="expr.env_var",
        tags=frozenset({"expr", "terminal"}),
        is_applicable="_is_env_var_applicable",
        weight="_weight_env_var",
    )
    def _generate_env_var(self, *, ctx: ExprGenCtx, **_) -> Optional[ast.VyperNode]:
        candidates = self._env_var_candidates(ctx=ctx)
        if not candidates:
            return None

        base, name, typ = ctx.gen.rng.choice(candidates)
        if base is None:
            node = ast.Name(id=name)
        else:
            node = ast.Attribute(value=ast.Name(id=base), attr=name)
        node._metadata = {"type": typ}
        return node

    def _build_empty(self, target_type: VyperType) -> ast.Call:
        type_node = ast.Name(id=str(target_type))
        call_node = ast.Call(func=ast.Name(id="empty"), args=[type_node], keywords=[])
        call_node._metadata["type"] = target_type
        return call_node

    def random_var_ref(
        self, target_type: VyperType, context: GenerationContext
    ) -> Optional[Union[ast.Attribute, ast.Name]]:
        """Pick a random variable matching target_type, returns proper AST ref."""
        matches = context.find_matching_vars(target_type)
        if not matches:
            return None
        return self._generate_variable_ref(self.rng.choice(matches), context)

    def _generate_variable_ref(
        self, target: Union[str, tuple[str, VarInfo]], context: GenerationContext
    ) -> Union[ast.Attribute, ast.Name]:
        if isinstance(target, str):
            name = target
            var_info = context.all_vars[name]
        else:
            name, var_info = target

        node = ast_builder.var_ref(name, var_info)
        node._metadata["type"] = var_info.typ
        node._metadata["varinfo"] = var_info
        return node

    @strategy(
        name="expr.attribute",
        tags=frozenset({"expr", "recursive"}),
        is_applicable="_is_attribute_applicable",
        weight="_weight_attribute",
    )
    def _generate_attribute(self, *, ctx: ExprGenCtx, **_) -> Optional[ast.Attribute]:
        bases = self._find_deref_bases(
            ctx=ctx, allow_attribute=True, allow_subscript=False
        )
        if not bases:
            return None

        name, base_t = ctx.gen.rng.choice(bases)
        base_node: ast.VyperNode = self._generate_variable_ref(name, ctx.context)
        built = self.build_dereference_chain_to_target(
            base_node,
            base_t,
            ctx.target_type,
            ctx.context,
            self.child_depth(ctx.depth),
            max_steps=self.cfg.subscript_chain_max_steps,
            allow_attribute=True,
            allow_subscript=False,
        )
        if not built:
            return None
        node, node_t = built
        assert isinstance(node, ast.Attribute)
        node._metadata["type"] = node_t
        return node

    @strategy(
        name="expr.subscript",
        tags=frozenset({"expr", "recursive"}),
        is_applicable="_is_subscript_applicable",
        weight="_weight_subscript",
    )
    def _generate_subscript(self, *, ctx: ExprGenCtx, **_) -> Optional[ast.Subscript]:
        bases = self._find_deref_bases(
            ctx=ctx, allow_attribute=False, allow_subscript=True
        )
        if not bases:
            return None

        name, base_t = ctx.gen.rng.choice(bases)
        base_node: ast.VyperNode = self._generate_variable_ref(name, ctx.context)
        built = self.build_dereference_chain_to_target(
            base_node,
            base_t,
            ctx.target_type,
            ctx.context,
            self.child_depth(ctx.depth),
            max_steps=self.cfg.subscript_chain_max_steps,
            allow_attribute=False,
            allow_subscript=True,
        )
        if not built:
            return None
        node, node_t = built
        assert isinstance(node, ast.Subscript)
        node._metadata["type"] = node_t
        return node

    @strategy(
        name="expr.dereference_var",
        tags=frozenset({"expr", "recursive"}),
        is_applicable="_is_dereference_var_applicable",
        weight="_weight_dereference_var",
    )
    def _generate_dereference_var(
        self, *, ctx: ExprGenCtx, **_
    ) -> Optional[ast.VyperNode]:
        bases = self._find_deref_bases(
            ctx=ctx, allow_attribute=True, allow_subscript=True
        )
        if not bases:
            return None

        name, base_t = ctx.gen.rng.choice(bases)
        base_node: ast.VyperNode = self._generate_variable_ref(name, ctx.context)
        built = self.build_dereference_chain_to_target(
            base_node,
            base_t,
            ctx.target_type,
            ctx.context,
            self.child_depth(ctx.depth),
            max_steps=self.cfg.subscript_chain_max_steps,
            allow_attribute=True,
            allow_subscript=True,
        )
        if not built:
            return None
        node, node_t = built
        node._metadata["type"] = node_t
        return node

    @strategy(
        name="expr.dereference_func_call",
        tags=frozenset({"expr", "recursive"}),
        is_applicable="_is_dereference_func_call_applicable",
        weight="_weight_dereference_func_call",
    )
    def _generate_dereference_func_call(
        self, *, ctx: ExprGenCtx, **_
    ) -> Optional[ast.VyperNode]:
        bases = self._find_deref_func_bases(
            ctx=ctx, allow_attribute=True, allow_subscript=True
        )
        if not bases:
            return None

        func_t, base_t = ctx.gen.rng.choice(bases)
        base_node = self._generate_call_to_function(func_t, ctx.context, ctx.depth)
        if base_node is None:
            return None

        built = self.build_dereference_chain_to_target(
            base_node,
            base_t,
            ctx.target_type,
            ctx.context,
            self.child_depth(ctx.depth),
            max_steps=self.cfg.subscript_chain_max_steps,
            allow_attribute=True,
            allow_subscript=True,
        )
        if not built:
            return None

        node, node_t = built
        node._metadata["type"] = node_t
        return node

    @strategy(
        name="expr.ifexp",
        tags=frozenset({"expr", "recursive"}),
        is_applicable="_is_ifexp_applicable",
        weight="_weight_ifexp",
    )
    def _generate_ifexp(self, *, ctx: ExprGenCtx, **_) -> ast.IfExp:
        # Condition must be bool; branches must yield the same type
        next_depth = self.child_depth(ctx.depth)
        test = self.generate(BoolT(), ctx.context, next_depth)
        # Avoid ambiguous tuple literal typing inside if-expressions.
        allow_tuple_literal = False
        body = self.generate(
            ctx.target_type,
            ctx.context,
            next_depth,
            allow_tuple_literal=allow_tuple_literal,
            allow_empty_list=False,
        )
        orelse = self.generate(
            ctx.target_type,
            ctx.context,
            next_depth,
            allow_tuple_literal=allow_tuple_literal,
            allow_empty_list=False,
        )

        node = ast.IfExp(test=test, body=body, orelse=orelse)
        node._metadata["type"] = ctx.target_type
        return node

    @strategy(
        name="expr.convert",
        tags=frozenset({"expr", "recursive"}),
        type_classes=(BoolT, IntegerT, AddressT, BytesM_T, BytesT, StringT),
        is_applicable="_is_convert_applicable",
        weight="_weight_convert",
    )
    def _generate_convert(self, *, ctx: ExprGenCtx, **_) -> Optional[ast.Call]:
        src_expr = self._pick_convert_source_expr(
            ctx.target_type, ctx.context, ctx.depth
        )
        if src_expr is None:
            return None
        return self.maybe_convert_expr(src_expr, ctx.target_type, ctx.context)

    # -------------------------
    # Shared dereference utils
    # -------------------------

    def _random_integer_type(self) -> IntegerT:
        bits = self.rng.choice([8, 16, 32, 64, 128, 256])
        signed = self.rng.choice([True, False])
        return IntegerT(signed, bits)

    def _generate_index_for_sequence(
        self,
        base_node: ast.VyperNode,
        seq_t: _SequenceT,
        context: GenerationContext,
        depth: int,
    ) -> ast.VyperNode:
        """Generate an index expression for arrays with three modes:
        - Guarded by len(base) (preferred)
        - Random expression (unconstrained)
        - OOB literal that forces compilation xfail (rare)
        """
        assert isinstance(seq_t, (SArrayT, DArrayT))

        cfg = self.cfg
        if self.rng.random() < cfg.index_guard_prob:
            return self._generate_guarded_index(base_node, seq_t, context, depth)
        else:
            return self._generate_random_index(seq_t, context, depth)

    def _build_len_call_maybe_hoisted(self, base_node: ast.VyperNode) -> ast.VyperNode:
        # Hoist len() to avoid duplicating calls in the same
        # expression, which triggers a Vyper ICE (overlapping memory).
        len_call = build_len_call(base_node)
        if contains_call(base_node):
            return self.hoist_to_tmp_var(len_call, prefix="tmp_len")
        return len_call

    def _generate_guarded_index(
        self,
        base_node: ast.VyperNode,
        seq_t: _SequenceT,
        context: GenerationContext,
        depth: int,
    ) -> ast.VyperNode:
        """Generate a bounds-guarded index expression."""
        cfg = self.cfg
        if isinstance(seq_t, SArrayT):
            roll = self.rng.random()
            if roll < 1 / 3:
                return random_literal_index(self.rng, seq_t.length)
            if roll < 2 / 3:
                return small_literal_index(
                    self.rng,
                    seq_t.length,
                    max_value=3,
                )
            idx_expr = self._retry_index_expr(
                seq_length=seq_t.length,
                make_candidate=lambda: self.generate(INDEX_TYPE, context, depth),
                fallback=lambda: random_literal_index(self.rng, seq_t.length),
                reject_if=self._static_index_is_invalid_constant,
            )
            len_call = ast_builder.uint256_literal(seq_t.length)
            return ast_builder.uint256_binop(idx_expr, ast.Mod(), len_call)
        dyn_const_len: Optional[int] = None
        if isinstance(seq_t, DArrayT):
            dyn_const_len = self._folded_constant_sequence_length(
                base_node, context.constants
            )
            if self.rng.random() < cfg.dynarray_last_index_in_guard_prob:
                if dyn_const_len is not None and dyn_const_len > 0:
                    return ast_builder.uint256_literal(dyn_const_len - 1)
                return build_dyn_last_index(
                    self._build_len_call_maybe_hoisted(base_node)
                )

        # Choose the raw index expression (biased towards small literals for DynArray)
        use_small = (
            isinstance(seq_t, DArrayT)
            and self.rng.random() < cfg.dynarray_small_literal_in_guard_prob
        )
        if use_small:
            i_expr = small_literal_index(self.rng, seq_t.length)
        else:
            i_expr = self._retry_index_expr(
                seq_length=seq_t.length,
                make_candidate=lambda: self.generate(INDEX_TYPE, context, depth),
                fallback=lambda: small_literal_index(self.rng, seq_t.length),
                reject_if=self._static_index_is_invalid_constant,
            )

        if dyn_const_len is not None and dyn_const_len > 0:
            len_lit = ast_builder.uint256_literal(dyn_const_len)
            return ast_builder.uint256_binop(i_expr, ast.Mod(), len_lit)

        return build_guarded_index(
            i_expr, self._build_len_call_maybe_hoisted(base_node)
        )

    def _generate_random_index(
        self,
        seq_t: _SequenceT,
        context: GenerationContext,
        depth: int,
    ) -> ast.VyperNode:
        """Generate a random (unconstrained) index expression."""

        def reject_oob(idx_expr: ast.VyperNode, length: int) -> bool:
            return self._static_index_is_constant_oob(
                idx_expr, length, context.constants
            )

        if isinstance(seq_t, SArrayT):
            return self._retry_index_expr(
                seq_length=seq_t.length,
                make_candidate=lambda: self.generate(
                    random_index_type(self.rng), context, depth
                ),
                fallback=lambda: random_literal_index(self.rng, seq_t.length),
                reject_if=reject_oob,
            )

        return self._retry_index_expr(
            seq_length=seq_t.length,
            make_candidate=lambda: self.generate(
                random_index_type(self.rng), context, depth
            ),
            fallback=lambda: small_literal_index(self.rng, seq_t.length),
            reject_if=reject_oob,
        )

    def _retry_index_expr(
        self,
        *,
        seq_length: int,
        make_candidate: Callable[[], ast.VyperNode],
        fallback: Callable[[], ast.VyperNode],
        reject_if: Callable[[ast.VyperNode, int], bool],
        retries: int = 3,
    ) -> ast.VyperNode:
        return self._retry(
            make_candidate=make_candidate,
            reject_if=lambda candidate: reject_if(candidate, seq_length),
            retries=retries,
            fallback=fallback,
        )

    def _static_index_is_invalid_constant(
        self, idx_expr: ast.VyperNode, _seq_length: int
    ) -> bool:
        status, _ = fold_constant_expression_status(idx_expr, {})
        return status == "invalid_constant"

    def _static_index_is_constant_oob(
        self, idx_expr: ast.VyperNode, seq_length: int, constants: dict[str, object]
    ) -> bool:
        status, folded = fold_constant_expression_status(idx_expr, constants)
        if status == "invalid_constant":
            return True
        if status != "value" or not isinstance(folded, ast.Int):
            return False
        return folded.value < 0 or folded.value >= seq_length

    def _apply_dereference_step(
        self,
        node: ast.VyperNode,
        cur_t: VyperType,
        candidate: DerefCandidate,
        context: GenerationContext,
        depth: int,
    ) -> tuple[ast.VyperNode, VyperType, str]:
        if candidate.kind == "attribute":
            assert candidate.attr_name is not None
            attr_node = ast.Attribute(value=node, attr=candidate.attr_name)
            attr_node._metadata = {"type": candidate.child_type}
            return attr_node, candidate.child_type, "attribute"

        if isinstance(cur_t, HashMapT):
            idx_expr = self.generate(cur_t.key_type, context, depth)
        elif isinstance(cur_t, (SArrayT, DArrayT)):
            idx_expr = self._generate_index_for_sequence(node, cur_t, context, depth)
        elif isinstance(cur_t, TupleT):
            assert candidate.tuple_index is not None
            idx_expr = ast_builder.literal(candidate.tuple_index, IntegerT(False, 256))
        else:
            raise ValueError(f"unsupported deref type: {type(cur_t).__name__}")

        sub_node = ast.Subscript(value=node, slice=idx_expr)
        sub_node._metadata = {"type": candidate.child_type}
        return sub_node, candidate.child_type, "subscript"

    def dereference_once(
        self,
        node: ast.VyperNode,
        cur_t: VyperType,
        context: GenerationContext,
        depth: int,
        *,
        target_type: Optional[VyperType] = None,
        max_steps_remaining: int = 0,
        allow_attribute: bool = True,
        allow_subscript: bool = True,
    ) -> Optional[tuple[ast.VyperNode, VyperType, str]]:
        candidates = dereference_candidates(
            cur_t,
            target_type=target_type,
            max_steps_remaining=max_steps_remaining,
            allow_attribute=allow_attribute,
            allow_subscript=allow_subscript,
        )
        if not candidates:
            return None
        candidate = self.rng.choice(candidates)
        return self._apply_dereference_step(node, cur_t, candidate, context, depth)

    def build_dereference_chain_to_target(
        self,
        base_node: ast.VyperNode,
        base_type: VyperType,
        target_type: VyperType,
        context: GenerationContext,
        depth: int,
        *,
        max_steps: int = 3,
        allow_attribute: bool = True,
        allow_subscript: bool = True,
    ) -> Optional[tuple[ast.VyperNode, VyperType]]:
        cur_t = base_type
        node = base_node
        steps_remaining = max(1, max_steps)

        while steps_remaining > 0:
            candidates = dereference_candidates(
                cur_t,
                target_type=target_type,
                max_steps_remaining=steps_remaining - 1,
                allow_attribute=allow_attribute,
                allow_subscript=allow_subscript,
            )
            if not candidates:
                break

            direct = [c for c in candidates if target_type.compare_type(c.child_type)]
            candidate = self.rng.choice(direct or candidates)
            node, cur_t, _ = self._apply_dereference_step(
                node, cur_t, candidate, context, depth
            )
            steps_remaining -= 1

            if target_type.compare_type(cur_t):
                return node, cur_t

            if not is_dereferenceable(
                cur_t,
                allow_attribute=allow_attribute,
                allow_subscript=allow_subscript,
            ):
                break

        return None

    def build_random_dereference_chain(
        self,
        base_node: ast.VyperNode,
        base_type: VyperType,
        context: GenerationContext,
        depth: int,
        *,
        max_steps: int = 2,
        allow_attribute: bool = True,
        allow_subscript: bool = True,
    ) -> Optional[tuple[ast.VyperNode, VyperType]]:
        if not is_dereferenceable(
            base_type,
            allow_attribute=allow_attribute,
            allow_subscript=allow_subscript,
        ):
            return None

        cur_t = base_type
        node = base_node
        steps = self.rng.randint(1, max(1, max_steps))

        for i in range(steps):
            step = self.dereference_once(
                node,
                cur_t,
                context,
                depth,
                allow_attribute=allow_attribute,
                allow_subscript=allow_subscript,
            )
            if step is None:
                break
            node, cur_t, _ = step

            if not is_dereferenceable(
                cur_t,
                allow_attribute=allow_attribute,
                allow_subscript=allow_subscript,
            ):
                break

            if i == steps - 1:
                break

        return node, cur_t

    def build_dereference_chain(
        self,
        base_node: ast.VyperNode,
        base_type: VyperType,
        context: GenerationContext,
        depth: int,
        *,
        target_type: Optional[VyperType] = None,
        max_steps: int = 3,
        allow_attribute: bool = True,
        allow_subscript: bool = True,
    ) -> Optional[tuple[ast.VyperNode, VyperType]]:
        if target_type is None:
            return self.build_random_dereference_chain(
                base_node,
                base_type,
                context,
                depth,
                max_steps=max_steps,
                allow_attribute=allow_attribute,
                allow_subscript=allow_subscript,
            )
        return self.build_dereference_chain_to_target(
            base_node,
            base_type,
            target_type,
            context,
            depth,
            max_steps=max_steps,
            allow_attribute=allow_attribute,
            allow_subscript=allow_subscript,
        )

    def build_chain_to_target(
        self,
        base_node: ast.VyperNode,
        base_type: VyperType,
        target_type: VyperType,
        context: GenerationContext,
        depth: int,
        max_steps: int = 3,
    ) -> Optional[tuple[ast.VyperNode, VyperType]]:
        return self.build_dereference_chain_to_target(
            base_node,
            base_type,
            target_type,
            context,
            depth,
            max_steps=max_steps,
            allow_attribute=False,
            allow_subscript=True,
        )

    def build_random_chain(
        self,
        base_node: ast.VyperNode,
        base_type: VyperType,
        context: GenerationContext,
        depth: int,
        max_steps: int = 2,
    ) -> tuple[ast.VyperNode, VyperType]:
        built = self.build_random_dereference_chain(
            base_node,
            base_type,
            context,
            depth,
            max_steps=max_steps,
            allow_attribute=False,
            allow_subscript=True,
        )
        if built is None:
            return base_node, base_type
        return built

    @strategy(
        name="expr.arithmetic",
        tags=frozenset({"expr", "recursive"}),
        type_classes=(IntegerT,),
    )
    def _generate_arithmetic(self, *, ctx: ExprGenCtx, **_) -> Optional[ast.BinOp]:
        op_classes = [ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod]
        next_depth = self.child_depth(ctx.depth)
        for _ in range(self.cfg.arithmetic_max_attempts):
            op_class = ctx.gen.rng.choice(op_classes)
            left = self.generate(ctx.target_type, ctx.context, next_depth)
            if op_class in (ast.FloorDiv, ast.Mod, ast.Div):
                right = self.generate_nonzero_expr(
                    ctx.target_type, ctx.context, next_depth
                )
            else:
                right = self.generate(ctx.target_type, ctx.context, next_depth)

            node = ast.BinOp(left=left, op=op_class(), right=right)
            node._metadata["type"] = ctx.target_type

            status, _ = fold_constant_expression_status(node, ctx.context.constants)
            if status != "invalid_constant":
                return node

        return None

    @strategy(
        name="expr.comparison",
        tags=frozenset({"expr", "recursive"}),
        type_classes=(BoolT,),
    )
    def _generate_comparison(self, *, ctx: ExprGenCtx, **_) -> ast.Compare:
        op_classes = [ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq]
        op_class = ctx.gen.rng.choice(op_classes)

        # For equality/inequality, we can use more types
        # TODO parametrize typeclasses so we don't always get the same size
        if isinstance(op_class(), (ast.Eq, ast.NotEq)):
            comparable_types = [
                IntegerT(True, 256),
                IntegerT(True, 128),
                AddressT(),
                BytesT(32),
                BytesM_T(32),
                StringT(100),
                BoolT(),
            ]
        else:
            # For ordering comparisons, only use numeric
            comparable_types = [IntegerT(True, 256), IntegerT(True, 128)]

        comparable_type = ctx.gen.rng.choice(comparable_types)

        next_depth = self.child_depth(ctx.depth)
        left = self.generate(comparable_type, ctx.context, next_depth)
        right = self.generate(comparable_type, ctx.context, next_depth)

        node = ast.Compare(left=left, ops=[op_class()], comparators=[right])
        node._metadata["type"] = BoolT()
        return node

    @strategy(
        name="expr.boolean_op",
        tags=frozenset({"expr", "recursive"}),
        type_classes=(BoolT,),
    )
    def _generate_boolean_op(self, *, ctx: ExprGenCtx, **_) -> ast.BoolOp:
        op_classes = [ast.And, ast.Or]
        op_class = ctx.gen.rng.choice(op_classes)

        next_depth = self.child_depth(ctx.depth)
        values = [self.generate(BoolT(), ctx.context, next_depth) for _ in range(2)]

        node = ast.BoolOp(op=op_class(), values=values)
        node._metadata["type"] = BoolT()
        return node

    @strategy(
        name="expr.not",
        tags=frozenset({"expr", "recursive"}),
        type_classes=(BoolT,),
    )
    def _generate_not(self, *, ctx: ExprGenCtx, **_) -> ast.UnaryOp:
        operand = self.generate(BoolT(), ctx.context, self.child_depth(ctx.depth))
        node = ast.UnaryOp(op=ast.Not(), operand=operand)
        node._metadata["type"] = BoolT()
        return node

    @strategy(
        name="expr.unary_minus",
        tags=frozenset({"expr", "recursive"}),
        type_classes=(IntegerT,),
        is_applicable="_is_unary_minus_applicable",
    )
    def _generate_unary_minus(self, *, ctx: ExprGenCtx, **_) -> ast.UnaryOp:
        operand = self.generate(
            ctx.target_type, ctx.context, self.child_depth(ctx.depth)
        )
        node = ast.UnaryOp(op=ast.USub(), operand=operand)
        node._metadata["type"] = ctx.target_type
        return node

    @strategy(
        name="expr.struct",
        tags=frozenset({"expr", "terminal"}),
        type_classes=(StructT,),
    )
    def _generate_struct(self, *, ctx: ExprGenCtx, **_) -> ast.Call:
        return self._generate_struct_literal(ctx)

    def _effective_call_mutability(self, context: GenerationContext) -> StateMutability:
        caller_mutability = context.current_function_mutability
        if context.in_iterable_expr and caller_mutability not in (
            StateMutability.PURE,
            StateMutability.VIEW,
        ):
            return StateMutability.VIEW
        return caller_mutability

    def _get_callable_functions(
        self,
        context: GenerationContext,
        *,
        return_type: Optional[VyperType] = None,
    ) -> list[ContractFunctionT]:
        current_func = self.function_registry.current_function
        if current_func is None:
            return []

        caller_mutability = self._effective_call_mutability(context)
        funcs = self.function_registry.get_callable_functions(
            return_type=return_type,
            from_function=current_func,
            caller_mutability=caller_mutability,
        )
        return funcs

    def _get_compatible_builtins(
        self, target_type: VyperType, context: GenerationContext
    ) -> list[tuple[str, object]]:
        caller_mutability = self._effective_call_mutability(context)
        candidates = self.function_registry.get_compatible_builtins(
            target_type,
            caller_mutability=caller_mutability,
        )
        return [(name, builtin) for name, builtin in candidates if name != "convert"]

    def _create_callable_function(
        self,
        *,
        return_type: VyperType,
        context: GenerationContext,
    ) -> Optional[ContractFunctionT]:
        current_func = self.function_registry.current_function
        if current_func is None:
            return None

        caller_nonreentrant_ctx = self.function_registry.reachable_from_nonreentrant(
            current_func
        )
        func_def = self.function_registry.create_new_function(
            return_type=return_type,
            type_generator=self.type_generator,
            max_args=2,
            caller_mutability=self._effective_call_mutability(context),
            allow_nonreentrant=not caller_nonreentrant_ctx,
        )
        if func_def is None:
            return None

        func_t = func_def._metadata["func_type"]
        return func_t

    def _generate_call_to_function(
        self,
        func_t: ContractFunctionT,
        context: GenerationContext,
        depth: int,
    ) -> Optional[Union[ast.Call, ast.StaticCall, ast.ExtCall]]:
        if func_t.return_type is None:
            return None

        arg_depth = self.child_depth(depth)
        args = []
        for pos_arg in func_t.positional_args:
            if pos_arg.typ:
                args.append(self.generate(pos_arg.typ, context, arg_depth))

        if func_t.is_external:
            result_node = self._generate_external_call(func_t, args)
        else:
            assert func_t.is_internal, (
                f"Expected internal or external function, got {func_t}"
            )
            func_node = ast.Attribute(value=ast.Name(id="self"), attr=func_t.name)
            func_node._metadata = {"type": func_t}
            result_node = self._finalize_call(func_node, args, func_t.return_type)

        if self.function_registry.current_function:
            self.function_registry.add_call(
                self.function_registry.current_function,
                func_t.name,
                internal=func_t.is_internal,
            )

        return result_node

    @strategy(
        name="expr.raw_call",
        tags=frozenset({"expr", "recursive"}),
        type_classes=(BoolT, BytesT, TupleT),
        is_applicable="_is_raw_call_expr_applicable",
        weight="_weight_raw_call",
    )
    def _generate_raw_call_expr(self, *, ctx: ExprGenCtx, **_) -> Optional[ast.Call]:
        return self.generate_raw_call_call(
            ctx.context,
            ctx.depth,
            target_type=ctx.target_type,
        )

    @strategy(
        name="expr.func_call",
        tags=frozenset({"expr", "recursive"}),
        is_applicable="_is_func_call_applicable",
    )
    def _generate_func_call(
        self, *, ctx: ExprGenCtx, **_
    ) -> Optional[Union[ast.Call, ast.StaticCall, ast.ExtCall]]:
        compatible_funcs = self._get_callable_functions(
            ctx.context, return_type=ctx.target_type
        )
        compatible_builtins = self._get_compatible_builtins(
            ctx.target_type, ctx.context
        )

        # Decide path: prefer user function, but sometimes pick builtin
        use_builtin = False
        if compatible_builtins:
            if not compatible_funcs:
                use_builtin = True
            else:
                use_builtin = (
                    ctx.gen.rng.random() < self.cfg.use_builtin_when_both_available_prob
                )

        if use_builtin:
            name, builtin = ctx.gen.rng.choice(compatible_builtins)
            return self._generate_builtin_call(
                name, builtin, ctx.target_type, ctx.context, ctx.depth
            )

        func_t: Optional[ContractFunctionT] = None
        if compatible_funcs and (
            ctx.gen.rng.random() >= self.cfg.create_new_function_prob
        ):
            func_t = ctx.gen.rng.choice(compatible_funcs)
        else:
            func_t = self._create_callable_function(
                return_type=ctx.target_type,
                context=ctx.context,
            )
            if func_t is None and compatible_funcs:
                func_t = ctx.gen.rng.choice(compatible_funcs)

        if func_t is None:
            if compatible_builtins:
                name, builtin = ctx.gen.rng.choice(compatible_builtins)
                return self._generate_builtin_call(
                    name, builtin, ctx.target_type, ctx.context, ctx.depth
                )
            return None

        return self._generate_call_to_function(func_t, ctx.context, ctx.depth)

    def _generate_external_call(
        self,
        func: ContractFunctionT,
        args: list,
    ) -> Union[ast.StaticCall, ast.ExtCall]:
        """Build AST for an external call through an interface."""
        _, iface_type = self.interface_registry.create_interface(func)

        # Address is always `self` for now.
        address_node = ast.Name(id="self")
        address_node._metadata = {"type": AddressT()}

        # Build: InterfaceName(address)
        iface_cast = ast_builder.interface_cast(iface_type, address_node)

        # Build: Interface(address).func
        attr_node = ast.Attribute(value=iface_cast, attr=func.name)
        attr_node._metadata = {"type": func}

        ret = self._finalize_call(attr_node, args, func.return_type, func)
        assert isinstance(ret, (ast.StaticCall, ast.ExtCall))
        return ret

    def _finalize_call(
        self, func_node, args, return_type, func_t=None
    ) -> Union[ast.Call, ast.StaticCall, ast.ExtCall]:
        """Create ast.Call and annotate; wrap external calls if func_t provided."""
        call_node = ast.Call(func=func_node, args=args, keywords=[])
        call_node._metadata = getattr(call_node, "_metadata", {})
        call_node._metadata["type"] = return_type

        if func_t is None:
            return call_node

        if func_t.is_external:
            if func_t.mutability in (StateMutability.PURE, StateMutability.VIEW):
                wrapped = ast.StaticCall(value=call_node)
            else:
                wrapped = ast.ExtCall(value=call_node)
            wrapped._metadata = getattr(wrapped, "_metadata", {})
            wrapped._metadata["type"] = return_type
            return wrapped
        return call_node

    def _generate_builtin_call(
        self,
        name: str,
        builtin,
        target_type: VyperType,
        context: GenerationContext,
        depth: int,
    ) -> Optional[Union[ast.Call, ast.StaticCall, ast.ExtCall]]:
        # func node
        func_node = ast.Name(id=name)
        func_node._metadata = getattr(func_node, "_metadata", {})
        func_node._metadata["type"] = builtin

        handler = self._builtin_handlers.get(name)
        if handler is None:
            return None

        return handler(
            func_node=func_node,
            name=name,
            builtin=builtin,
            target_type=target_type,
            context=context,
            depth=depth,
        )

    def _builtin_convert(
        self,
        *,
        target_type: VyperType,
        context: GenerationContext,
        depth: int,
        **_,
    ) -> Optional[ast.Call]:
        src_expr = self._pick_convert_source_expr(target_type, context, depth)
        if src_expr is None:
            return None
        return self.maybe_convert_expr(src_expr, target_type, context)

    def _random_abi_encode_arg_type(self, payload_budget: int) -> VyperType:
        static_choices = [
            BoolT(),
            DecimalT(),
            BytesM_T(4),
            BytesM_T(32),
            IntegerT(False, 256),
            IntegerT(True, 128),
            AddressT(),
        ]
        can_use_dynamic = payload_budget >= 64 and self.rng.random() < 0.35
        if not can_use_dynamic:
            return self.rng.choice(static_choices)

        max_dynamic_len = min(64, max(1, payload_budget - 32))
        if self.rng.random() < 0.5:
            return BytesT(self.rng.randint(1, max_dynamic_len))
        return StringT(self.rng.randint(1, max_dynamic_len))

    def _abi_encode_maxlen(
        self, arg_types: list[VyperType], *, ensure_tuple: bool, has_method_id: bool
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

    def _generate_abi_encode_arg_expr(
        self,
        arg_t: VyperType,
        context: GenerationContext,
        depth: int,
    ) -> ast.VyperNode:
        scope_matches = context.find_matching_vars(arg_t)
        if scope_matches and self.rng.random() < 0.65:
            return self._generate_variable_ref(self.rng.choice(scope_matches), context)
        return self.generate(arg_t, context, depth)

    def _generate_abi_encode_method_id(self) -> ast.VyperNode:
        value = bytes(self.rng.getrandbits(8) for _ in range(4))
        if self.rng.random() < 0.5:
            return ast_builder.literal(value, BytesM_T(4))
        return ast_builder.literal(value, BytesT(4))

    def _pick_abi_encode_arg_types(
        self,
        context: GenerationContext,
        arity: int,
        ensure_tuple: bool,
        has_method_id: bool,
        target_length: int,
    ) -> Optional[list[VyperType]]:
        existing = self._existing_reachable_types(context, skip={HashMapT})
        self.rng.shuffle(existing)

        arg_types: list[VyperType] = []
        for t in existing:
            if len(arg_types) >= arity:
                break
            candidate = arg_types + [t]
            maxlen = self._abi_encode_maxlen(
                candidate, ensure_tuple=ensure_tuple, has_method_id=has_method_id
            )
            if maxlen is not None and maxlen <= target_length:
                arg_types.append(t)

        # Greedily fill remaining slots; arity is a soft max
        while len(arg_types) < arity:
            if arg_types:
                used = (
                    self._abi_encode_maxlen(
                        arg_types,
                        ensure_tuple=ensure_tuple,
                        has_method_id=has_method_id,
                    )
                    or 0
                )
            else:
                used = 4 if has_method_id else 0

            budget = target_length - used

            candidates: list[VyperType] = []
            if budget >= 32:
                candidates.extend(
                    [
                        BoolT(),
                        DecimalT(),
                        BytesM_T(4),
                        BytesM_T(32),
                        IntegerT(False, 256),
                        IntegerT(True, 128),
                        AddressT(),
                    ]
                )
            # Dynamic types: offset(32) + length(32) + ceil(n/32)*32 data
            if budget >= 96:
                max_dyn_len = max(1, (budget - 64) // 32 * 32)
                candidates.append(BytesT(self.rng.randint(1, max_dyn_len)))
                candidates.append(StringT(self.rng.randint(1, max_dyn_len)))

            if not candidates:
                break

            self.rng.shuffle(candidates)
            for t in candidates:
                trial = arg_types + [t]
                maxlen = self._abi_encode_maxlen(
                    trial, ensure_tuple=ensure_tuple, has_method_id=has_method_id
                )
                if maxlen is not None and maxlen <= target_length:
                    arg_types.append(t)
                    break
            else:
                break

        return arg_types if arg_types else None

    def _maybe_hoist_abi_encode_arg(
        self, expr: ast.VyperNode, context: GenerationContext
    ) -> ast.VyperNode:
        # Literal lists must be hoisted to get a type annotation
        if isinstance(expr, ast.List):
            return self.hoist_to_tmp_var(expr)

        # Tuple literals: hoist any list-literal elements within
        if isinstance(expr, ast.Tuple):
            for i, elt in enumerate(expr.elements):
                if isinstance(elt, ast.List):
                    expr.elements[i] = self.hoist_to_tmp_var(elt)
            return expr

        # Expressions that constant-fold to an integer are ambiguous
        # (could be uint8, int256, uint256, etc.) — but variable references
        # (Name, Attribute) have declared types so are not ambiguous.
        if not isinstance(expr, (ast.Name, ast.Attribute)):
            status, folded = fold_constant_expression_status(expr, context.constants)
            if status == "value" and isinstance(folded, ast.Int):
                return self.hoist_to_tmp_var(expr)

        # 0x 20-byte hex literals are ambiguous between address and bytes20
        if isinstance(expr, ast.Hex):
            hex_val = expr.value
            if hex_val.startswith(("0x", "0X")) and len(hex_val) == 42:
                return self.hoist_to_tmp_var(expr)

        return expr

    def _builtin_abi_encode(
        self,
        *,
        func_node: ast.Name,
        target_type: VyperType,
        context: GenerationContext,
        depth: int,
        **_,
    ) -> Optional[Union[ast.Call, ast.StaticCall, ast.ExtCall]]:
        if not isinstance(target_type, BytesT):
            return None
        if target_type.length < 32:
            return None

        has_method_id = target_type.length >= 36 and self.rng.random() < 0.35
        max_arity = max(1, min(5, target_type.length // 32))
        weights = [0.5**i for i in range(max_arity)]
        arity = self.rng.choices(range(1, max_arity + 1), weights=weights, k=1)[0]

        ensure_tuple = self.rng.choice([True, False]) if arity == 1 else True

        arg_types = self._pick_abi_encode_arg_types(
            context=context,
            arity=arity,
            ensure_tuple=ensure_tuple,
            has_method_id=has_method_id,
            target_length=target_type.length,
        )
        if arg_types is None:
            return None

        maxlen = self._abi_encode_maxlen(
            arg_types, ensure_tuple=ensure_tuple, has_method_id=has_method_id
        )
        if maxlen is None or maxlen > target_type.length:
            return None

        arg_depth = self.child_depth(depth)
        args = [
            self._maybe_hoist_abi_encode_arg(
                self._generate_abi_encode_arg_expr(arg_t, context, arg_depth),
                context,
            )
            for arg_t in arg_types
        ]

        keywords = []
        if not ensure_tuple:
            keywords.append(
                ast.keyword(
                    arg="ensure_tuple",
                    value=ast_builder.literal(False, BoolT()),
                )
            )
        elif self.rng.random() < 0.5:
            keywords.append(
                ast.keyword(
                    arg="ensure_tuple",
                    value=ast_builder.literal(True, BoolT()),
                )
            )
        if has_method_id:
            keywords.append(
                ast.keyword(
                    arg="method_id", value=self._generate_abi_encode_method_id()
                )
            )

        call_node = ast.Call(func=func_node, args=args, keywords=keywords)
        call_node._metadata = {"type": BytesT(maxlen)}
        return call_node

    def _builtin_min_max(
        self,
        *,
        func_node: ast.Name,
        target_type: VyperType,
        context: GenerationContext,
        depth: int,
        **_,
    ) -> Optional[Union[ast.Call, ast.StaticCall, ast.ExtCall]]:
        if not isinstance(target_type, (IntegerT, DecimalT)):
            return None
        arg_t = target_type
        arg_depth = self.child_depth(depth)
        a0 = self.generate(arg_t, context, arg_depth)
        a1 = self.generate(arg_t, context, arg_depth)
        call_node = ast.Call(func=func_node, args=[a0, a1], keywords=[])
        call_node._metadata = {"type": arg_t}
        return call_node

    def _builtin_abs(
        self,
        *,
        func_node: ast.Name,
        target_type: VyperType,
        context: GenerationContext,
        depth: int,
        **_,
    ) -> Optional[Union[ast.Call, ast.StaticCall, ast.ExtCall]]:
        if isinstance(target_type, IntegerT) and not target_type.is_signed:
            return None
        if not isinstance(target_type, (IntegerT, DecimalT)):
            return None
        arg_t = target_type
        a0 = self.generate(arg_t, context, self.child_depth(depth))
        return self._finalize_call(func_node, [a0], arg_t)

    def _builtin_floor_ceil(
        self,
        *,
        func_node: ast.Name,
        builtin,
        context: GenerationContext,
        depth: int,
        **_,
    ) -> Optional[Union[ast.Call, ast.StaticCall, ast.ExtCall]]:
        # Expect decimal arg, return concrete integer type from builtin
        a0 = self.generate(DecimalT(), context, self.child_depth(depth))
        ret_t = getattr(builtin, "_return_type", None) or IntegerT(True, 256)
        return self._finalize_call(func_node, [a0], ret_t)

    def _builtin_len(
        self,
        *,
        func_node: ast.Name,
        builtin,
        context: GenerationContext,
        depth: int,
        **_,
    ) -> Optional[Union[ast.Call, ast.StaticCall, ast.ExtCall]]:
        # Only support BytesT/StringT to start
        max_len = self.rng.randint(1, 128)

        arg_t = self.rng.choice([BytesT(max_len), StringT(max_len)])
        a0 = self.generate(arg_t, context, self.child_depth(depth))
        ret_t = getattr(builtin, "_return_type", IntegerT(False, 256))
        return self._finalize_call(func_node, [a0], ret_t)

    def _builtin_keccak256(
        self,
        *,
        func_node: ast.Name,
        builtin,
        target_type: VyperType,
        context: GenerationContext,
        depth: int,
        **_,
    ) -> Optional[Union[ast.Call, ast.StaticCall, ast.ExtCall]]:
        if not (isinstance(target_type, BytesM_T) and target_type.length == 32):
            return None

        max_len = self.rng.randint(1, 128)
        arg_t = self.rng.choice([BytesT(max_len), StringT(max_len), BytesM_T(32)])
        a0 = self.generate(arg_t, context, self.child_depth(depth))
        ret_t = getattr(builtin, "_return_type", None) or target_type
        return self._finalize_call(func_node, [a0], ret_t)

    def _builtin_concat(
        self,
        *,
        func_node: ast.Name,
        target_type: VyperType,
        context: GenerationContext,
        depth: int,
        **_,
    ) -> Optional[Union[ast.Call, ast.StaticCall, ast.ExtCall]]:
        if not isinstance(target_type, (StringT, BytesT)):
            return None

        tgt_len = target_type.length
        is_string_target = isinstance(target_type, StringT)
        # Decay arg count so concat is usually low arity:
        # weights ~ [1, 1/4, 1/16, ...] for k=[2,3,4,...].
        k_values = list(range(2, 9))
        decay = 0.25
        k_weights = [decay**i for i in range(len(k_values))]
        k = self.rng.choices(k_values, weights=k_weights, k=1)[0]
        arg_depth = self.child_depth(depth)

        def make_dynamic_typ(n: int) -> VyperType:
            return StringT(n) if is_string_target else BytesT(n)

        def arg_capacity(typ: VyperType) -> Optional[int]:
            if is_string_target:
                return typ.length if isinstance(typ, StringT) else None
            if isinstance(typ, (BytesT, BytesM_T)):
                return typ.length
            return None

        def folded_arg_length(node: ast.VyperNode) -> Optional[int]:
            status, folded = fold_constant_expression_status(node, context.constants)
            if status != "value":
                return None
            assert folded is not None

            if isinstance(folded, ast.Bytes):
                return len(folded.value)
            if isinstance(folded, ast.Str):
                return len(folded.value.encode("utf-8", errors="surrogateescape"))

            folded_t = self._expr_type(folded)
            if isinstance(folded_t, BytesM_T):
                return folded_t.length
            return None

        def arg_return_length(node: ast.VyperNode, fallback_capacity: int) -> int:
            folded_len = folded_arg_length(node)
            if folded_len is not None:
                return folded_len

            expr_t = self._expr_type(node)
            expr_cap = arg_capacity(expr_t) if expr_t is not None else None
            if expr_cap is not None:
                return expr_cap
            return fallback_capacity

        # Candidate-driven pass: consume compatible in-scope expressions first.
        scope_candidates = []
        for name, var_info in context.find_matching_vars(None):
            cap = arg_capacity(var_info.typ)
            if cap is None or cap > tgt_len:
                continue
            scope_candidates.append((name, var_info, cap))

        remaining = tgt_len
        args: list[ast.VyperNode] = []
        arg_caps: list[int] = []

        if scope_candidates and self.rng.random() < 0.65:
            pool = list(scope_candidates)
            max_take = min(k, len(pool), 3)
            take = self.rng.randint(1, max_take)
            for _ in range(take):
                fitting_idxs = [
                    i for i, entry in enumerate(pool) if entry[2] <= remaining
                ]
                if not fitting_idxs:
                    break
                idx = self.rng.choice(fitting_idxs)
                name, var_info, cap = pool.pop(idx)
                expr = self._generate_variable_ref((name, var_info), context)
                args.append(expr)
                arg_caps.append(arg_return_length(expr, cap))
                remaining -= cap

        while len(args) < k:
            slots_left = k - len(args)
            if remaining == 0:
                n = 0
            elif slots_left == 1:
                n = remaining
            else:
                # Prefer non-empty chunks while keeping some zero-length tails.
                n = self.rng.randint(0, remaining)
                if n == 0 and self.rng.random() < 0.75:
                    n = self.rng.randint(1, remaining)

            if is_string_target:
                t = StringT(n)
                if n == 0:
                    expr = ast_builder.literal("", t)
                else:
                    expr = self.generate(t, context, arg_depth)
            else:
                use_fixed_bytes = n > 0 and self.rng.random() < 0.35
                if use_fixed_bytes:
                    fixed_sizes = [m for m in (1, 2, 3, 4, 8, 16, 24, 32) if m <= n]
                    if fixed_sizes:
                        m = self.rng.choice(fixed_sizes)
                        t = BytesM_T(m)
                        expr = self.generate(t, context, arg_depth)
                        args.append(expr)
                        arg_caps.append(arg_return_length(expr, m))
                        remaining -= m
                        continue

                t = BytesT(n)
                if n == 0:
                    expr = ast_builder.literal(b"", t)
                else:
                    expr = self.generate(t, context, arg_depth)

            args.append(expr)
            arg_caps.append(arg_return_length(expr, n))
            remaining -= n

        self.rng.shuffle(args)
        return self._finalize_call(func_node, args, make_dynamic_typ(sum(arg_caps)))

    def _builtin_slice(
        self,
        *,
        func_node: ast.Name,
        target_type: VyperType,
        context: GenerationContext,
        depth: int,
        **_,
    ) -> Optional[Union[ast.Call, ast.StaticCall, ast.ExtCall]]:
        if not isinstance(target_type, (BytesT, StringT)):
            return None
        if target_type.length < 1:
            return None

        target_bound = target_type.length
        target_is_string = isinstance(target_type, StringT)
        uint256_t = IntegerT(False, 256)
        arg_depth = self.child_depth(depth)

        def _folded_uint256(node: ast.VyperNode) -> tuple[str, Optional[int]]:
            status, folded = fold_constant_expression_status(node, context.constants)
            if status == "value" and isinstance(folded, ast.Int) and folded.value >= 0:
                return status, folded.value
            return status, None

        def _slice_ret_type(length: int) -> VyperType:
            return StringT(length) if target_is_string else BytesT(length)

        def _max_constexpr_len(start_const: Optional[int]) -> int:
            max_len = min(target_bound, src_bound)
            if start_const is not None:
                max_len = min(max_len, src_bound - start_const)
            return max_len

        def _constexpr_len_predicate(
            length_expr: ast.VyperNode, start_const: Optional[int]
        ) -> bool:
            status, value = _folded_uint256(length_expr)
            if status != "value" or value is None:
                return False
            max_len = _max_constexpr_len(start_const)
            return 1 <= value <= max_len

        def _generate_constexpr_len(
            *, start_const: Optional[int], retries: int = 5
        ) -> Optional[tuple[ast.VyperNode, int]]:
            max_len = _max_constexpr_len(start_const)
            if max_len < 1:
                return None

            expr = self._retry(
                make_candidate=lambda: self.generate(uint256_t, context, arg_depth),
                reject_if=lambda candidate: not _constexpr_len_predicate(
                    candidate, start_const
                ),
                retries=retries,
                fallback=lambda: ast_builder.uint256_literal(
                    self.rng.randint(1, max_len)
                ),
            )
            status, value = _folded_uint256(expr)
            assert status == "value" and value is not None and 1 <= value <= max_len
            return expr, value

        if target_is_string:
            src_req_t: VyperType = StringT(max(target_bound, 1000))
        elif self.rng.random() < 0.5:
            src_req_t = BytesM_T(32)
        else:
            src_req_t = BytesT(max(target_bound, 1000))

        arg0 = self.generate(src_req_t, context, arg_depth)
        src_t = self._expr_type(arg0)
        assert src_t is not None

        src_bound = src_t.length
        if src_bound < 1:
            return None
        force_constexpr_len = src_bound > target_bound

        start_expr = self.generate(uint256_t, context, arg_depth)
        start_status, start_value = _folded_uint256(start_expr)
        if start_status == "value" and start_value >= src_bound:
            start_expr = ast_builder.uint256_literal(self.rng.randint(0, src_bound - 1))
            start_value = start_expr.value

        use_constexpr_len = force_constexpr_len or self.rng.random() < 0.1
        if use_constexpr_len:
            constexpr = _generate_constexpr_len(start_const=start_value, retries=5)
            length_expr, length_value = constexpr
            ret_t = _slice_ret_type(length_value)
            return self._finalize_call(
                func_node, [arg0, start_expr, length_expr], ret_t
            )

        length_expr = self.generate(uint256_t, context, arg_depth)
        len_status, len_value = _folded_uint256(length_expr)
        if len_status == "value":
            max_const_len = _max_constexpr_len(start_value)
            if not (1 <= len_value <= max_const_len):
                constexpr = _generate_constexpr_len(start_const=start_value, retries=5)
                length_expr, len_value = constexpr
            ret_t = _slice_ret_type(len_value)
        else:
            ret_t = _slice_ret_type(src_bound)

        return self._finalize_call(func_node, [arg0, start_expr, length_expr], ret_t)

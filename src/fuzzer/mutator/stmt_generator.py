from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Union

from vyper.ast import nodes as ast
from vyper.semantics.types import (
    VyperType,
    BoolT,
    HashMapT,
    IntegerT,
    DArrayT,
)
from vyper.semantics.analysis.base import DataLocation, Modifiability, VarInfo

from fuzzer.mutator import ast_builder
from fuzzer.mutator.ast_utils import ast_equivalent, body_is_terminated
from fuzzer.mutator.augassign_utils import (
    augassign_ops_for_type,
    is_augassignable_type,
    augassign_rhs_type,
    can_reach_augassignable,
)
from fuzzer.mutator.base_generator import BaseGenerator
from fuzzer.mutator.config import StmtGeneratorConfig, DepthConfig
from fuzzer.mutator.constant_folding import evaluate_constant_expression
from fuzzer.mutator.context import GenerationContext, ScopeType, ExprMutability, AccessMode
from fuzzer.mutator.strategy import strategy
from fuzzer.mutator.type_utils import (
    is_dereferenceable,
    pick_dereference_target_type,
    collect_dereference_types,
)


@dataclass
class StmtGenCtx:
    context: GenerationContext
    parent: Optional[ast.VyperNode]
    depth: int
    return_type: Optional[VyperType]
    gen: StatementGenerator


class FreshNameGenerator:
    def __init__(self, prefix: str = "gen_var"):
        self.prefix = prefix
        self.counter = 0

    def generate(self) -> str:
        name = f"{self.prefix}{self.counter}"
        self.counter += 1
        return name


class StatementGenerator(BaseGenerator):
    def __init__(
        self,
        expr_generator,
        type_generator,
        rng: random.Random,
        cfg: Optional[StmtGeneratorConfig] = None,
        depth_cfg: Optional[DepthConfig] = None,
    ):
        self.expr_generator = expr_generator
        self.type_generator = type_generator
        self.cfg = cfg or StmtGeneratorConfig()
        self.name_generator = FreshNameGenerator()

        super().__init__(rng, depth_cfg)

    def _is_vardecl_applicable(self, *, ctx: StmtGenCtx, **_) -> bool:
        return True

    def _is_assign_applicable(self, *, ctx: StmtGenCtx, **_) -> bool:
        if ctx.context.is_module_scope:
            return False
        return bool(self.get_writable_variables(ctx.context))

    def _is_augassign_applicable(self, *, ctx: StmtGenCtx, **_) -> bool:
        if ctx.context.is_module_scope:
            return False
        return bool(self._get_augassign_candidates(ctx.context))

    def _is_append_applicable(self, *, ctx: StmtGenCtx, **_) -> bool:
        if ctx.context.is_module_scope:
            return False
        return bool(self._get_append_candidates(ctx.context))

    def _is_if_applicable(self, *, ctx: StmtGenCtx, **_) -> bool:
        return not ctx.context.is_module_scope

    def _is_for_applicable(self, *, ctx: StmtGenCtx, **_) -> bool:
        if ctx.context.is_module_scope:
            return False
        return not self.at_max_depth(ctx.depth)

    def _weight_vardecl(self, **_) -> float:
        return self.cfg.vardecl_weight

    def _weight_assign(self, **_) -> float:
        return self.cfg.assign_weight

    def _weight_augassign(self, **_) -> float:
        return self.cfg.augassign_weight

    def _weight_append(self, **_) -> float:
        return self.cfg.append_weight

    def _weight_if(self, **_) -> float:
        return self.cfg.if_weight

    def _weight_for(self, **_) -> float:
        return self.cfg.for_weight

    def _create_var_info(
        self,
        typ: VyperType,
        location: DataLocation = DataLocation.MEMORY,
        modifiability=None,  # Will use VarInfo's default
        is_public: bool = False,
        decl_node: Optional[ast.VyperNode] = None,
    ) -> VarInfo:
        kwargs = {
            "typ": typ,
            "location": location,
            "is_public": is_public,
            "decl_node": decl_node,
        }
        if modifiability is not None:
            kwargs["modifiability"] = modifiability

        return VarInfo(**kwargs)

    def add_variable(self, context, name: str, var_info: VarInfo):
        context.add_variable(name, var_info)

    def generate_type(
        self,
        context,
        nesting: int = 3,
        skip: Optional[set] = None,
        size_budget: Optional[int] = None,
    ) -> VyperType:
        """Generate a type, biasing towards existing types and preferred fuzzing types."""
        skip = skip or set()

        # Bias towards accessible variable types and callable return types
        accessible_vars = context.find_matching_vars()
        if self.rng.random() < self.cfg.existing_type_bias_prob and accessible_vars:
            seen: set[str] = set()
            all_types: list[VyperType] = []

            def add_type(typ: VyperType) -> None:
                if type(typ) in skip:
                    return
                key = str(typ)
                if key in seen:
                    return
                seen.add(key)
                all_types.append(typ)

            for _name, var_info in accessible_vars:
                var_type = var_info.typ
                add_type(var_type)
                for child_t, _depth in collect_dereference_types(
                    var_type, max_steps=self.cfg.deref_chain_max_steps
                ):
                    add_type(child_t)

            # Include return types of callable functions
            func_registry = self.expr_generator.function_registry
            if func_registry and func_registry.current_function:
                callable_funcs = func_registry.get_callable_functions(
                    from_function=func_registry.current_function,
                    caller_mutability=context.current_function_mutability,
                )
                for func in callable_funcs:
                    if func.return_type is not None:
                        add_type(func.return_type)
                        for child_t, _depth in collect_dereference_types(
                            func.return_type,
                            max_steps=self.cfg.deref_chain_max_steps,
                        ):
                            add_type(child_t)

            if all_types:
                return self.rng.choice(all_types)

        # Struct fragments are stored in type_generator.source_fragments
        return self.type_generator.generate_biased_type(
            nesting=nesting, skip=skip, size_budget=size_budget
        )

    def _generate_varinfo(self, context: GenerationContext) -> tuple[str, VarInfo]:
        """Generate a random variable with VarInfo.

        Returns:
            Tuple of (variable_name, VarInfo)
        """
        # Generate a unique name
        var_name = self.name_generator.generate()
        cfg = self.cfg

        # Determine location and modifiability based on scope
        if context.is_module_scope:
            # Module-level variables can be storage, transient, immutable, or constant
            location_choices = [
                (DataLocation.STORAGE, Modifiability.MODIFIABLE, cfg.storage_location_weight),
                (DataLocation.TRANSIENT, Modifiability.MODIFIABLE, cfg.transient_location_weight),
                (DataLocation.CODE, Modifiability.RUNTIME_CONSTANT, cfg.immutable_location_weight),
                (DataLocation.UNSET, Modifiability.CONSTANT, cfg.constant_location_weight),
            ]

            # Choose location based on weights
            rand = self.rng.random()
            cumulative = 0
            for location, modifiability, weight in location_choices:
                cumulative += weight
                if rand < cumulative:
                    selected_location = location
                    selected_modifiability = modifiability
                    break

            # Module-level variables can be public
            is_public = self.rng.choice([True, False])

            # Skip HashMapT for constant and immutable variables
            skip_types = set()
            if selected_modifiability in (
                Modifiability.CONSTANT,
                Modifiability.RUNTIME_CONSTANT,
            ):
                skip_types = {HashMapT}

        else:
            # Function and block level variables are always in memory and modifiable
            selected_location = DataLocation.MEMORY
            selected_modifiability = Modifiability.MODIFIABLE
            is_public = False

            # HashMapT can't be in memory
            skip_types = {HashMapT}

        # Apply a size budget for immutables, constants, and memory variables
        budget = None
        if selected_location == DataLocation.MEMORY or selected_modifiability in (
            Modifiability.CONSTANT,
            Modifiability.RUNTIME_CONSTANT,
        ):
            budget = 10000

        var_type = self.generate_type(
            context, nesting=2, skip=skip_types, size_budget=budget
        )

        # Create VarInfo
        var_info = self._create_var_info(
            typ=var_type,
            location=selected_location,
            modifiability=selected_modifiability,
            is_public=is_public,
            decl_node=None,  # We won't assign a declaration node as requested
        )

        return var_name, var_info

    def inject_variable_decls(
        self,
        body: list,
        context: GenerationContext,
        parent: Optional[ast.VyperNode] = None,
        depth: int = 0,
        min_stmts: int = 0,
        max_stmts: int = 2,
    ) -> int:
        """Inject variable declarations into body."""
        if self.at_max_depth(depth):
            return 0

        if max_stmts < min_stmts:
            max_stmts = min_stmts

        num_vars = self.rng.randint(min_stmts, max_stmts)
        for i in range(num_vars):
            ctx = StmtGenCtx(
                context=context,
                parent=parent,
                depth=depth,
                return_type=None,
                gen=self,
            )
            var_decl = self.create_vardecl_and_register(ctx=ctx)
            body.insert(i, var_decl)

        return num_vars

    def inject_random_statements(
        self,
        body: list,
        context: GenerationContext,
        parent: Optional[ast.VyperNode] = None,
        depth: int = 0,
        min_stmts: Optional[int] = None,
        max_stmts: Optional[int] = None,
        min_vardecls: int = 0,
        max_vardecls: int = 2,
        *,
        include_vardecls: bool = True,
        leading_vars: int = 0,
        allow_loop_terminator: bool = True,
    ) -> None:
        """
        Inject random statements into body.

        min_stmts/max_stmts apply to non-variable statements.
        include_vardecls controls whether variable declarations are injected.
        """
        cfg = self.cfg
        if self.at_max_depth(depth):
            return

        min_count = cfg.min_stmts if min_stmts is None else min_stmts
        max_count = cfg.max_stmts if max_stmts is None else max_stmts
        if max_count < min_count:
            max_count = min_count

        num_vars = max(0, min(leading_vars, len(body)))
        if include_vardecls:
            num_vars = self.inject_variable_decls(
                body,
                context,
                parent=parent,
                depth=depth,
                min_stmts=min_vardecls,
                max_stmts=max_vardecls,
            )

        if context.is_module_scope:
            return

        num_other_stmts = self.rng.randint(min_count, max_count)
        for _ in range(num_other_stmts):
            stmt = self.generate(context, parent, depth)
            if isinstance(stmt, ast.AnnAssign):
                body.insert(num_vars, stmt)
                num_vars += 1
                continue
            # Insert before the last statement to avoid inserting after return
            # If body is empty or only has vars, append at the end
            max_pos = max(num_vars, len(body) - 1)
            if num_vars <= max_pos:
                insert_pos = self.rng.randint(num_vars, max_pos)
            else:
                insert_pos = num_vars
            body.insert(insert_pos, stmt)

        # Ensure function scope terminates with a return when required.
        if isinstance(parent, ast.FunctionDef):
            func_type = getattr(parent, "_metadata", {}).get("func_type")
            return_type = getattr(func_type, "return_type", None)

            if return_type is not None and not body_is_terminated(body):
                ret_expr = self.expr_generator.generate(
                    return_type,
                    context,
                    depth=self.expr_generator.root_depth(),
                    allow_tuple_literal=True,
                )
                body.append(ast.Return(value=ret_expr))

        if allow_loop_terminator:
            self._maybe_append_loop_terminator(body, context, parent)

    def inject_statements(
        self,
        body: list,
        context: GenerationContext,
        parent: Optional[ast.VyperNode] = None,
        depth: int = 0,
        min_stmts: Optional[int] = None,
        max_stmts: Optional[int] = None,
        min_vardecls: int = 0,
        max_vardecls: int = 2,
        *,
        allow_loop_terminator: bool = True,
    ) -> None:
        """Inject variable declarations, then random statements (legacy behavior)."""
        if self.at_max_depth(depth):
            return

        num_vars = self.inject_variable_decls(
            body,
            context,
            parent=parent,
            depth=depth,
            min_stmts=min_vardecls,
            max_stmts=max_vardecls,
        )
        self.inject_random_statements(
            body,
            context,
            parent=parent,
            depth=depth,
            min_stmts=min_stmts,
            max_stmts=max_stmts,
            include_vardecls=False,
            leading_vars=num_vars,
            allow_loop_terminator=allow_loop_terminator,
        )

    def generate(
        self,
        context,
        parent: Optional[ast.VyperNode] = None,
        depth: int = 0,
        return_type: Optional[VyperType] = None,
    ) -> ast.VyperNode:
        ctx = StmtGenCtx(
            context=context,
            parent=parent,
            depth=depth,
            return_type=return_type,
            gen=self,
        )

        # Use tag-based filtering for terminal vs recursive strategies
        if self.should_continue(depth):
            include_tags = ("stmt",)
        else:
            include_tags = ("stmt", "terminal")

        # Collect available statement strategies
        strategies = self._strategy_registry.collect(
            include_tags=include_tags,
            context={"ctx": ctx},
        )

        return self._strategy_executor.execute_with_retry(
            strategies,
            policy="weighted_random",
            fallback=lambda: ast.Pass(),
            context={"ctx": ctx},
        )

    @strategy(
        name="stmt.if",
        tags=frozenset({"stmt", "recursive"}),
        is_applicable="_is_if_applicable",
        weight="_weight_if",
    )
    def generate_if(self, *, ctx: StmtGenCtx, **_) -> ast.If:
        test_expr = self.expr_generator.generate_nonconstant_expr(
            BoolT(),
            ctx.context,
            depth=self.expr_generator.root_depth(),
            allow_recursion=True,
        )

        if_node = ast.If(test=test_expr, body=[], orelse=[])
        inside_for = ctx.context.is_inside_for_scope()
        want_terminator = inside_for and (
            ctx.gen.rng.random() < self.cfg.loop_terminator_in_if_prob
        )
        generate_else = ctx.gen.rng.random() < self.cfg.generate_else_branch_prob
        force_else = False
        if want_terminator and not generate_else:
            if ctx.gen.rng.random() < self.cfg.loop_terminator_force_else_prob:
                generate_else = True
                force_else = True

        with ctx.context.new_scope(ScopeType.IF):
            self.inject_statements(
                if_node.body,
                ctx.context,
                if_node,
                self.child_depth(ctx.depth),
                min_stmts=self.cfg.min_stmts,
                max_stmts=self.cfg.max_stmts,
                allow_loop_terminator=False,
            )

            if not if_node.body:
                if_node.body.append(ast.Pass())

        if generate_else:
            with ctx.context.new_scope(ScopeType.IF):
                self.inject_statements(
                    if_node.orelse,
                    ctx.context,
                    if_node,
                    self.child_depth(ctx.depth),
                    min_stmts=self.cfg.min_stmts,
                    max_stmts=self.cfg.max_stmts,
                    allow_loop_terminator=False,
                )

        if want_terminator:
            target_body = if_node.body
            if force_else:
                target_body = if_node.orelse
            elif generate_else and ctx.gen.rng.random() < 0.5:
                target_body = if_node.orelse
            self._append_loop_terminator(target_body, rng=ctx.gen.rng)

        return if_node

    # TODO add current scope - what if it's module scope
    def _append_loop_terminator(
        self,
        body: list,
        *,
        rng: Optional[random.Random] = None,
    ) -> None:
        if body_is_terminated(body):
            return

        if body and isinstance(body[-1], ast.Pass):
            body.pop()

        rng = rng or self.rng
        if rng.random() < 0.5:
            body.append(ast.Break())
        else:
            body.append(ast.Continue())

    def _maybe_append_loop_terminator(
        self,
        body: list,
        context: GenerationContext,
        parent: Optional[ast.VyperNode],
        *,
        rng: Optional[random.Random] = None,
    ) -> None:
        if parent is None or not context.is_inside_for_scope():
            return

        if isinstance(parent, ast.For):
            prob = self.cfg.loop_terminator_direct_prob
        elif isinstance(parent, ast.If):
            prob = self.cfg.loop_terminator_in_if_prob
        else:
            return

        rng = rng or self.rng
        if rng.random() >= prob:
            return

        self._append_loop_terminator(body, rng=rng)

    def get_writable_variables(self, context: GenerationContext) -> list[tuple[str, VarInfo]]:
        with context.access_mode(AccessMode.WRITE):
            return context.find_matching_vars()

    def _get_augassign_candidates(
        self, context: GenerationContext
    ) -> list[tuple[str, VarInfo]]:
        writable_vars = self.get_writable_variables(context)
        if not writable_vars:
            return []
        max_steps = self.cfg.deref_chain_max_steps
        return [
            (name, var_info)
            for name, var_info in writable_vars
            if var_info.modifiability == Modifiability.MODIFIABLE
            if can_reach_augassignable(var_info.typ, max_steps)
        ]

    def _can_reach_dynarray(self, base_type: VyperType) -> bool:
        if isinstance(base_type, DArrayT):
            return True
        if not is_dereferenceable(base_type):
            return False
        for child_t, _depth in collect_dereference_types(
            base_type, self.cfg.deref_chain_max_steps
        ):
            if isinstance(child_t, DArrayT):
                return True
        return False

    def _get_append_candidates(
        self, context: GenerationContext
    ) -> list[tuple[str, VarInfo]]:
        writable_vars = self.get_writable_variables(context)
        if not writable_vars:
            return []
        return [
            (name, var_info)
            for name, var_info in writable_vars
            if var_info.modifiability == Modifiability.MODIFIABLE
            if self._can_reach_dynarray(var_info.typ)
        ]

    def _build_dereference_target(
        self,
        ctx: StmtGenCtx,
        base_node: ast.VyperNode,
        base_type: VyperType,
        *,
        target_type: Optional[VyperType],
    ) -> Optional[tuple[ast.VyperNode, VyperType]]:
        return self.expr_generator.build_dereference_chain(
            base_node,
            base_type,
            ctx.context,
            self.expr_generator.root_depth(),
            target_type=target_type,
            max_steps=self.cfg.deref_chain_max_steps,
            allow_attribute=True,
            allow_subscript=True,
        )

    def _resolve_target_with_deref(
        self,
        ctx: StmtGenCtx,
        base_node: ast.VyperNode,
        base_type: VyperType,
        *,
        predicate,
    ) -> Optional[tuple[ast.VyperNode, VyperType]]:
        base_ok = predicate(base_type)
        must_deref = isinstance(base_type, HashMapT)
        should_deref = must_deref or (
            is_dereferenceable(base_type)
            and (
                not base_ok or ctx.gen.rng.random() < self.cfg.deref_assignment_prob
            )
        )

        if should_deref:
            # We could build the deref target directly; picking a reachable type
            # first reuses the shared deref chain logic.
            target_pick = pick_dereference_target_type(
                base_type,
                max_steps=self.cfg.deref_chain_max_steps,
                predicate=predicate,
                rng=ctx.gen.rng,
                continue_prob=self.cfg.deref_continue_prob,
            )
            if target_pick is None:
                if not base_ok:
                    return None
                return base_node, base_type

            return self._build_dereference_target(
                ctx,
                base_node,
                base_type,
                target_type=target_pick,
            )

        if base_ok:
            return base_node, base_type
        return None

    def _generate_assign_value(
        self,
        context: GenerationContext,
        target_node: ast.VyperNode,
        target_type: VyperType,
        *,
        rng: Optional[random.Random] = None,
    ) -> ast.VyperNode:
        rng = rng or self.rng
        retries = max(0, self.cfg.self_assign_max_retries)
        # TODO: This retry loop might be a performance hotspot; consider prechecking
        # matching vars or biasing expr generation to avoid self-assigns upfront.
        for _ in range(retries + 1):
            value = self.expr_generator.generate(
                target_type, context, depth=self.expr_generator.root_depth()
            )
            if not ast_equivalent(target_node, value):
                return value
            if rng.random() < self.cfg.self_assign_prob:
                return value

        return value

    @strategy(
        name="stmt.assign",
        tags=frozenset({"stmt", "terminal"}),
        is_applicable="_is_assign_applicable",
        weight="_weight_assign",
    )
    def generate_assign(self, *, ctx: StmtGenCtx, **_) -> Optional[ast.Assign]:
        writable_vars = self.get_writable_variables(ctx.context)
        if not writable_vars:
            return None

        var_name, var_info = ctx.gen.rng.choice(writable_vars)

        # Build base reference (self.x or local)
        base = ast_builder.var_ref(var_name, var_info)
        base._metadata = {"type": var_info.typ, "varinfo": var_info}

        if var_info.modifiability == Modifiability.RUNTIME_CONSTANT:
            target_node = base
            target_type = var_info.typ
        else:
            resolved = self._resolve_target_with_deref(
                ctx,
                base,
                var_info.typ,
                predicate=lambda t: not isinstance(t, HashMapT),
            )
            if resolved is None:
                return None
            target_node, target_type = resolved

        value = self._generate_assign_value(
            ctx.context, target_node, target_type, rng=ctx.gen.rng
        )

        if var_info.modifiability == Modifiability.RUNTIME_CONSTANT:
            ctx.context.mark_immutable_assigned(var_name)

        return ast.Assign(targets=[target_node], value=value)

    def generate_immutable_assignments(
        self,
        context: GenerationContext,
        immutables: list[tuple[str, VarInfo]],
    ) -> list[ast.Assign]:
        assignments = []
        for name, var_info in immutables:
            target = ast_builder.var_ref(name, var_info)
            target._metadata = {"type": var_info.typ, "varinfo": var_info}
            value_expr = self.expr_generator.generate(
                var_info.typ,
                context,
                depth=self.expr_generator.root_depth(),
                allow_tuple_literal=False,
            )
            assignments.append(ast.Assign(targets=[target], value=value_expr))
            context.mark_immutable_assigned(name)
        return assignments

    @strategy(
        name="stmt.vardecl",
        tags=frozenset({"stmt", "terminal"}),
        is_applicable="_is_vardecl_applicable",
        weight="_weight_vardecl",
    )
    def create_vardecl_and_register(
        self, *, ctx: StmtGenCtx, **_
    ) -> Union[ast.VariableDecl, ast.AnnAssign]:
        var_decl, var_name, var_info = self.generate_vardecl(ctx.context, ctx.parent)
        self.add_variable(ctx.context, var_name, var_info)
        if (
            ctx.context.is_module_scope
            and var_info.location == DataLocation.CODE
            and var_info.modifiability == Modifiability.RUNTIME_CONSTANT
        ):
            ctx.context.immutables_to_init[var_name] = var_info
        self._register_constant_value(ctx.context, var_name, var_info, var_decl)
        return var_decl

    def _register_constant_value(
        self,
        context: GenerationContext,
        name: str,
        var_info: VarInfo,
        var_decl: Union[ast.VariableDecl, ast.AnnAssign],
    ) -> None:
        if (
            var_info.modifiability != Modifiability.CONSTANT
            or not context.is_module_scope
        ):
            return
        value_node = getattr(var_decl, "value", None)
        if value_node is None:
            return
        try:
            const_value = evaluate_constant_expression(value_node, context.constants)
        except Exception:
            return
        context.add_constant(name, const_value)

    def generate_vardecl(
        self, context, parent: Optional[ast.VyperNode]
    ) -> tuple[Union[ast.VariableDecl, ast.AnnAssign], str, VarInfo]:
        """
        Build a variable declaration valid for the current scope.

        Module scope: returns ast.VariableDecl (supports flags like constant, public, etc.)
        Function/block scope: returns ast.AnnAssign (always has init value, no flags)
        """
        var_name, var_info = self._generate_varinfo(context)

        anno: ast.VyperNode = ast.Name(id=str(var_info.typ))

        # These flags only apply to module scope (var_info already reflects this)
        if var_info.modifiability == Modifiability.CONSTANT:
            anno = ast.Call(func=ast.Name(id="constant"), args=[anno])
        if var_info.modifiability == Modifiability.RUNTIME_CONSTANT:
            anno = ast.Call(func=ast.Name(id="immutable"), args=[anno])

        if var_info.location == DataLocation.TRANSIENT:
            anno = ast.Call(func=ast.Name(id="transient"), args=[anno])

        if var_info.is_public:
            anno = ast.Call(func=ast.Name(id="public"), args=[anno])

        needs_init = (
            var_info.modifiability == Modifiability.CONSTANT
            or not context.is_module_scope  # inside a function / block
        )
        if needs_init:
            allow_tuple_literal = (
                not context.is_module_scope
                or var_info.modifiability == Modifiability.CONSTANT
            )
            if (
                context.is_module_scope
                and var_info.modifiability == Modifiability.CONSTANT
            ):
                with context.mutability(ExprMutability.CONST):
                    init_val = self.expr_generator.generate(
                        var_info.typ,
                        context,
                        depth=self.expr_generator.root_depth(),
                        allow_tuple_literal=allow_tuple_literal,
                        allow_empty_list=not isinstance(var_info.typ, DArrayT),
                    )
            else:
                init_val = self.expr_generator.generate(
                    var_info.typ,
                    context,
                    depth=self.expr_generator.root_depth(),
                    allow_tuple_literal=allow_tuple_literal,
                )
        else:
            init_val = None

        if context.is_module_scope:
            var_decl = ast.VariableDecl(
                parent=parent,
                target=ast.Name(id=var_name),
                annotation=anno,
                value=init_val,
            )
        else:
            var_decl = ast.AnnAssign(
                target=ast.Name(id=var_name),
                annotation=anno,
                value=init_val,
            )

        return var_decl, var_name, var_info

    def _generate_range_iter(
        self, ctx: StmtGenCtx
    ) -> tuple[ast.Call, IntegerT]:
        """Generate a range(N) iteration with literal N.

        Returns (iter_node, element_type).
        """
        # For now, always use uint256 for the loop variable
        target_type = IntegerT(False, 256)  # uint256

        # Generate a random stop value between 1 and max_range_stop
        stop_val = ctx.gen.rng.randint(1, self.cfg.for_max_range_stop)

        iter_node = ast.Call(
            func=ast.Name(id="range"),
            args=[ast.Int(value=stop_val)],
            keywords=[],
        )

        return iter_node, target_type

    # TODO we should just construct target type and ctx and route through
    # expr_generator.generate()
    # many issues though - can't use func. calls, const dynarrays, some ifexprs
    # as the iterables
    def _generate_array_iter(
        self, ctx: StmtGenCtx
    ) -> tuple[ast.VyperNode, VyperType, Optional[str], Optional[VarInfo]]:
        """Generate array iteration (over existing variable or literal).

        Returns (iter_node, element_type, iterated_var_name, iterated_var_info).
        iterated_var_name/info are None if iterating over a literal array.
        """
        iterable_arrays = ctx.context.find_iterable_arrays()

        # Prefer existing array if available and probability check passes
        if (
            iterable_arrays
            and ctx.gen.rng.random() < self.cfg.for_prefer_existing_array_prob
        ):
            var_name, var_info = ctx.gen.rng.choice(iterable_arrays)
            element_type = var_info.typ.value_type
            iter_node = ast_builder.var_ref(var_name, var_info)
            return iter_node, element_type, var_name, var_info

        # Fall back to generating a literal array
        # Generate a base type for elements (leaf type, nesting=0)
        element_type = self.type_generator.generate_simple_type()

        # Generate 1-5 elements
        num_elements = ctx.gen.rng.randint(1, 5)
        elements = []
        for _ in range(num_elements):
            elem = self.expr_generator.generate(
                element_type, ctx.context, depth=self.expr_generator.root_depth()
            )
            elements.append(elem)

        iter_node = ast.List(elements=elements)
        return iter_node, element_type, None, None

    @strategy(
        name="stmt.for",
        tags=frozenset({"stmt", "recursive"}),
        is_applicable="_is_for_applicable",
        weight="_weight_for",
    )
    def generate_for(self, *, ctx: StmtGenCtx, **_) -> ast.For:
        """Generate a for loop statement.

        Supports:
        - range(N) iteration with literal N
        - Array iteration over existing SArrayT/DArrayT variables
        - Literal array iteration as fallback
        """
        # Decide: range vs array iteration
        use_range = ctx.gen.rng.random() < self.cfg.for_use_range_prob

        iterated_var_name: Optional[str] = None
        iterated_var_info: Optional[VarInfo] = None

        if use_range:
            iter_node, target_type = self._generate_range_iter(ctx)
        else:
            iter_node, target_type, iterated_var_name, iterated_var_info = (
                self._generate_array_iter(ctx)
            )

        # Generate loop variable name
        loop_var_name = self.name_generator.generate()

        # Build target (AnnAssign structure as expected by Vyper)
        target = ast.AnnAssign(
            target=ast.Name(id=loop_var_name),
            annotation=ast.Name(id=str(target_type)),
            value=None,
        )

        # Create For node
        for_node = ast.For(target=target, iter=iter_node, body=[])

        # Enter FOR scope and generate body
        with ctx.context.new_scope(ScopeType.FOR):
            # Register loop variable as RUNTIME_CONSTANT (cannot be reassigned)
            loop_var_info = self._create_var_info(
                typ=target_type,
                location=DataLocation.MEMORY,
                modifiability=Modifiability.RUNTIME_CONSTANT,
            )
            self.add_variable(ctx.context, loop_var_name, loop_var_info)

            # Shadow the iterated array variable as read-only to prevent direct writes.
            # Note: This doesn't prevent indirect modifications via function calls;
            # the Vyper compiler will catch those cases.
            if iterated_var_name is not None and iterated_var_info is not None:
                readonly_var_info = self._create_var_info(
                    typ=iterated_var_info.typ,
                    location=iterated_var_info.location,
                    modifiability=Modifiability.RUNTIME_CONSTANT,
                )
                self.add_variable(ctx.context, iterated_var_name, readonly_var_info)

            # Generate body statements
            self.inject_statements(
                for_node.body,
                ctx.context,
                for_node,
                self.child_depth(ctx.depth),
                min_stmts=self.cfg.min_stmts,
                max_stmts=self.cfg.max_stmts,
            )

            # Ensure body is not empty
            if not for_node.body:
                for_node.body.append(ast.Pass())

        return for_node

    @strategy(
        name="stmt.augassign",
        tags=frozenset({"stmt", "terminal"}),
        is_applicable="_is_augassign_applicable",
        weight="_weight_augassign",
    )
    def generate_augassign(self, *, ctx: StmtGenCtx, **_) -> Optional[ast.AugAssign]:
        candidates = self._get_augassign_candidates(ctx.context)
        if not candidates:
            return None

        var_name, var_info = ctx.gen.rng.choice(candidates)
        base = ast_builder.var_ref(var_name, var_info)
        base._metadata = {"type": var_info.typ, "varinfo": var_info}

        base_type = var_info.typ
        resolved = self._resolve_target_with_deref(
            ctx,
            base,
            base_type,
            predicate=is_augassignable_type,
        )
        if resolved is None:
            return None
        target_node, target_type = resolved

        ops = augassign_ops_for_type(target_type)
        if not ops:
            return None

        op_class = ctx.gen.rng.choice(ops)
        rhs_type = augassign_rhs_type(op_class, target_type)

        if op_class is ast.Pow:
            rhs_value = ctx.gen.rng.randint(0, 8)
            rhs = ast_builder.literal(rhs_value, rhs_type)
        elif op_class in (ast.FloorDiv, ast.Mod, ast.Div):
            rhs = self.expr_generator.generate_nonzero_expr(
                rhs_type, ctx.context, depth=self.expr_generator.root_depth()
            )
        else:
            rhs = self.expr_generator.generate(
                rhs_type, ctx.context, depth=self.expr_generator.root_depth()
            )

        return ast.AugAssign(target=target_node, op=op_class(), value=rhs)

    @strategy(
        name="stmt.append",
        tags=frozenset({"stmt", "terminal"}),
        is_applicable="_is_append_applicable",
        weight="_weight_append",
    )
    def generate_append(self, *, ctx: StmtGenCtx, **_) -> Optional[ast.Expr]:
        candidates = self._get_append_candidates(ctx.context)
        if not candidates:
            return None

        var_name, var_info = ctx.gen.rng.choice(candidates)
        base = ast_builder.var_ref(var_name, var_info)
        base._metadata = {"type": var_info.typ, "varinfo": var_info}

        resolved = self._resolve_target_with_deref(
            ctx,
            base,
            var_info.typ,
            predicate=lambda t: isinstance(t, DArrayT),
        )
        if resolved is None:
            return None
        target_node, target_type = resolved

        value = self.expr_generator.generate(
            target_type.value_type, ctx.context, depth=self.expr_generator.root_depth()
        )
        call = ast.Call(
            func=ast.Attribute(value=target_node, attr="append"),
            args=[value],
            keywords=[],
        )
        return ast.Expr(value=call)

    def generate_assert(self, context, parent: Optional[ast.VyperNode]) -> ast.Assert:
        pass

    def generate_raise(self, context, parent: Optional[ast.VyperNode]) -> ast.Raise:
        pass

    def generate_log(self, context, parent: Optional[ast.VyperNode]) -> ast.Log:
        pass

    def generate_break(self, context, parent: Optional[ast.VyperNode]) -> ast.Break:
        return ast.Break()

    def generate_continue(
        self, context, parent: Optional[ast.VyperNode]
    ) -> ast.Continue:
        return ast.Continue()

    def generate_return(
        self, context, parent: Optional[ast.VyperNode], return_type: VyperType
    ) -> ast.Return:
        pass

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
)
from vyper.semantics.analysis.base import DataLocation, Modifiability, VarInfo

from fuzzer.mutator.context import GenerationContext, ScopeType, ExprMutability, AccessMode
from fuzzer.mutator.config import StmtGeneratorConfig, DepthConfig
from fuzzer.mutator.base_generator import BaseGenerator
from fuzzer.mutator import ast_builder
from fuzzer.mutator.strategy import strategy


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
        return bool(ctx.context.is_module_scope)

    def _is_assign_applicable(self, *, ctx: StmtGenCtx, **_) -> bool:
        if ctx.context.is_module_scope:
            return False
        return bool(self.get_writable_variables(ctx.context))

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
        """Generate a type, biasing towards existing types in context."""
        skip = skip or set()

        # Bias towards existing variable types to enable compatible expressions
        if self.rng.random() < self.cfg.existing_type_bias_prob and context.all_vars:
            valid_vars = []
            for var_info in context.all_vars.values():
                if type(var_info.typ) not in skip:
                    valid_vars.append(var_info)

            if valid_vars:
                var_info = self.rng.choice(valid_vars)
                return var_info.typ

        # Struct fragments are stored in type_generator.source_fragments
        return self.type_generator.generate_type(
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
        inject_prob: Optional[float] = None,
        include_vardecls: bool = True,
        leading_vars: int = 0,
    ) -> None:
        """
        Inject random statements into body.

        min_stmts/max_stmts apply to non-variable statements.
        include_vardecls controls whether variable declarations are injected.
        """
        cfg = self.cfg
        if self.at_max_depth(depth):
            return

        if inject_prob is not None and self.rng.random() > inject_prob:
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

            if return_type is not None and not self.scope_is_terminated(body):
                ret_expr = self.expr_generator.generate(
                    return_type, context, depth=self.expr_generator.root_depth()
                )
                body.append(ast.Return(value=ret_expr))

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
        inject_prob: Optional[float] = None,
    ) -> None:
        """Inject variable declarations, then random statements (legacy behavior)."""
        if self.at_max_depth(depth):
            return

        if inject_prob is not None and self.rng.random() > inject_prob:
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
            inject_prob=None,
            include_vardecls=False,
            leading_vars=num_vars,
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
        test_expr = self.expr_generator.generate(
            BoolT(), ctx.context, depth=self.expr_generator.root_depth()
        )

        if_node = ast.If(test=test_expr, body=[], orelse=[])

        with ctx.context.new_scope(ScopeType.IF):
            self.inject_statements(
                if_node.body,
                ctx.context,
                if_node,
                self.child_depth(ctx.depth),
                min_stmts=self.cfg.min_stmts,
                max_stmts=self.cfg.max_stmts,
                inject_prob=self.cfg.inject_prob,
            )

            if not if_node.body:
                if_node.body.append(ast.Pass())

        if ctx.gen.rng.random() < self.cfg.generate_else_branch_prob:
            with ctx.context.new_scope(ScopeType.IF):
                self.inject_statements(
                    if_node.orelse,
                    ctx.context,
                    if_node,
                    self.child_depth(ctx.depth),
                    min_stmts=self.cfg.min_stmts,
                    max_stmts=self.cfg.max_stmts,
                    inject_prob=self.cfg.inject_prob,
                )

        return if_node

    # TODO add current scope - what if it's module scope
    def scope_is_terminated(self, body: list) -> bool:
        if not body:
            return False

        last_stmt = body[-1]
        return isinstance(last_stmt, (ast.Continue, ast.Break, ast.Return))

    def get_writable_variables(self, context: GenerationContext) -> list[tuple[str, VarInfo]]:
        with context.access_mode(AccessMode.WRITE):
            return context.find_matching_vars()

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

        # Decide if we target a subscript (for arrays/hashmaps/tuples) and allow nesting
        target_type = var_info.typ
        target_node: ast.VyperNode = base

        # HashMapT must always be subscripted - can't assign directly
        # For other subscriptable types, bias towards subscripting
        is_hashmap = isinstance(var_info.typ, HashMapT)
        if is_hashmap or (
            self.expr_generator.is_subscriptable_type(var_info.typ)
            and ctx.gen.rng.random() < self.cfg.subscript_assignment_prob
        ):
            cur_node, cur_t = self.expr_generator.build_random_chain(
                base,
                var_info.typ,
                ctx.context,
                depth=self.expr_generator.root_depth(),
                max_steps=self.expr_generator.cfg.subscript_random_chain_max_steps,
            )
            target_node = cur_node
            target_type = cur_t

            # Keep subscripting while we still have a HashMapT
            while isinstance(target_type, HashMapT):
                cur_node, cur_t = self.expr_generator.build_random_chain(
                    target_node,
                    target_type,
                    ctx.context,
                    depth=self.expr_generator.root_depth(),
                    max_steps=self.expr_generator.cfg.subscript_hashmap_chain_max_steps,
                )
                target_node = cur_node
                target_type = cur_t

        value = self.expr_generator.generate(
            target_type, ctx.context, depth=self.expr_generator.root_depth()
        )

        return ast.Assign(targets=[target_node], value=value)

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
        return var_decl

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
            if (
                context.is_module_scope
                and var_info.modifiability == Modifiability.CONSTANT
            ):
                with context.mutability(ExprMutability.CONST):
                    init_val = self.expr_generator.generate(
                        var_info.typ, context, depth=self.expr_generator.root_depth()
                    )
            else:
                init_val = self.expr_generator.generate(
                    var_info.typ, context, depth=self.expr_generator.root_depth()
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
                inject_prob=self.cfg.inject_prob,
            )

            # Ensure body is not empty
            if not for_node.body:
                for_node.body.append(ast.Pass())

        return for_node

    def generate_augassign(
        self, context, parent: Optional[ast.VyperNode]
    ) -> ast.AugAssign:
        pass

    def generate_assert(self, context, parent: Optional[ast.VyperNode]) -> ast.Assert:
        pass

    def generate_raise(self, context, parent: Optional[ast.VyperNode]) -> ast.Raise:
        pass

    def generate_log(self, context, parent: Optional[ast.VyperNode]) -> ast.Log:
        pass

    def generate_break(self, context, parent: Optional[ast.VyperNode]) -> ast.Break:
        pass

    def generate_continue(
        self, context, parent: Optional[ast.VyperNode]
    ) -> ast.Continue:
        pass

    def generate_return(
        self, context, parent: Optional[ast.VyperNode], return_type: VyperType
    ) -> ast.Return:
        pass

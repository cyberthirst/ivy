import random
from typing import Optional, Union
from vyper.ast import nodes as ast
from vyper.semantics.types import (
    VyperType,
    BoolT,
    HashMapT,
    TupleT,
)
from vyper.semantics.analysis.base import DataLocation, Modifiability, VarInfo

from fuzzer.mutator.context import Context, ScopeType, ExprMutability, AccessMode
from fuzzer.mutator.strategy import (
    Strategy,
    StrategyRegistry,
    StrategySelector,
    StrategyExecutor,
)


class FreshNameGenerator:
    def __init__(self, prefix: str = "gen_var"):
        self.prefix = prefix
        self.counter = 0

    def generate(self) -> str:
        name = f"{self.prefix}{self.counter}"
        self.counter += 1
        return name


class StatementGenerator:
    # Probability of allowing recursive statements (if, for, etc.)
    # Terminal statements (assign, vardecl) are always allowed.
    # With 0.2: ~80% terminal only, ~20% may nest
    CONTINUATION_PROB = 0.2

    def __init__(self, expr_generator, type_generator, rng: random.Random):
        self.expr_generator = expr_generator
        self.type_generator = type_generator
        self.rng = rng
        self.name_generator = FreshNameGenerator()

        self.max_depth = 5
        self.nest_decay = 0.7

        self.inject_prob = 0.3
        self.min_stmts = 1
        self.max_stmts = 3

        self.statement_weights = {
            "vardecl": 0.4,
            "assign": 0.3,
            "if": 0.3,
        }

        self._strategy_registry = StrategyRegistry()
        self._strategy_selector = StrategySelector(self.rng)
        self._strategy_executor = StrategyExecutor(self._strategy_selector)
        self._register_strategies()

    def _register_strategies(self) -> None:
        # Variable declaration: only applicable in module scope
        self._strategy_registry.register(
            Strategy(
                name="stmt.vardecl",
                tags=frozenset({"stmt", "terminal"}),
                is_applicable=self._is_vardecl_applicable,
                weight=self._weight_vardecl,
                run=self._run_stmt_vardecl,
            )
        )

        # Assignment: only inside functions/blocks and if there are modifiable vars.
        self._strategy_registry.register(
            Strategy(
                name="stmt.assign",
                tags=frozenset({"stmt", "terminal"}),
                is_applicable=self._is_assign_applicable,
                weight=self._weight_assign,
                run=self._run_stmt_assign,
            )
        )

        # If-statement: recursive; only inside functions/blocks.
        self._strategy_registry.register(
            Strategy(
                name="stmt.if",
                tags=frozenset({"stmt", "recursive"}),
                is_applicable=self._is_if_applicable,
                weight=self._weight_if,
                run=self._run_stmt_if,
            )
        )

    def _is_vardecl_applicable(self, **ctx) -> bool:
        context: Context = ctx["context"]
        return bool(context.is_module_scope)

    def _is_assign_applicable(self, **ctx) -> bool:
        context: Context = ctx["context"]
        if context.is_module_scope:
            return False
        return bool(self.get_writable_variables(context))

    def _is_if_applicable(self, **ctx) -> bool:
        context: Context = ctx["context"]
        return not context.is_module_scope

    def _weight_vardecl(self, **ctx) -> float:
        return float(self.statement_weights.get("vardecl", 1.0))

    def _weight_assign(self, **ctx) -> float:
        return float(self.statement_weights.get("assign", 1.0))

    def _weight_if(self, **ctx) -> float:
        return float(self.statement_weights.get("if", 1.0))

    def _run_stmt_vardecl(self, **ctx):
        return self.create_vardecl_and_register(ctx["context"], ctx.get("parent"))

    def _run_stmt_assign(self, **ctx):
        return self.generate_assign(ctx["context"], ctx.get("parent"))

    def _run_stmt_if(self, **ctx):
        depth = ctx.get("depth", 0)
        return self.generate_if(ctx["context"], ctx.get("parent"), depth)

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
        if self.rng.random() < 0.4 and context.all_vars:
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

    def _generate_varinfo(self, context: Context) -> tuple[str, VarInfo]:
        """Generate a random variable with VarInfo.

        Returns:
            Tuple of (variable_name, VarInfo)
        """
        # Generate a unique name
        var_name = self.name_generator.generate()

        # Determine location and modifiability based on scope
        if context.is_module_scope:
            # Module-level variables can be storage, transient, immutable, or constant
            location_choices = [
                (DataLocation.STORAGE, Modifiability.MODIFIABLE, 0.4),
                (DataLocation.TRANSIENT, Modifiability.MODIFIABLE, 0.2),
                (DataLocation.CODE, Modifiability.RUNTIME_CONSTANT, 0.2),  # immutable
                (DataLocation.UNSET, Modifiability.CONSTANT, 0.2),  # constant
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

    def inject_statements(
        self,
        body: list,
        context: Context,
        parent: Optional[ast.VyperNode] = None,
        depth: int = 0,
        n_stmts: Optional[int] = None,
    ) -> None:
        """
        Inject statements into body. Doesn't inject functions, those are generated lazily
        based on call_expr demand.
        """
        if depth > self.max_depth:
            return

        if n_stmts is None:
            if self.rng.random() > self.inject_prob:
                return

        # Generate variables first and insert at beginning
        num_vars = self.rng.randint(0, 2)
        if context.is_module_scope and n_stmts is not None and num_vars == 0:
            num_vars = 1
        for i in range(num_vars):
            var_decl = self.create_vardecl_and_register(context, parent)
            body.insert(i, var_decl)

        if context.is_module_scope:
            return

        # Generate other statements
        if n_stmts is not None:
            num_other_stmts = n_stmts
        else:
            num_other_stmts = self.rng.randint(self.min_stmts, self.max_stmts)

        for _ in range(num_other_stmts):
            stmt = self.generate_statement(context, parent, depth)
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
                ret_expr = self.expr_generator.generate(return_type, context, depth=2)
                body.append(ast.Return(value=ret_expr))

    def generate_statement(
        self,
        context,
        parent: Optional[ast.VyperNode] = None,
        depth: int = 0,
        return_type: Optional[VyperType] = None,
    ) -> ast.VyperNode:
        if depth >= self.max_depth:
            return self._generate_simple_statement(context, parent)

        # Decide whether to allow recursive (nesting) statements
        # ~80% of the time, only terminal statements are considered
        allow_recursive = self.rng.random() < self.CONTINUATION_PROB
        if allow_recursive:
            include_tags = ("stmt",)  # All statement strategies
        else:
            include_tags = ("terminal",)  # Only terminal (assign, vardecl)

        # Collect available statement strategies and execute with retry
        strategies = self._strategy_registry.collect(
            include_tags=include_tags,
            context={
                "context": context,
                "parent": parent,
                "depth": depth,
                "return_type": return_type,
                "rng": self.rng,
            },
        )

        return self._strategy_executor.execute_with_retry(
            strategies,
            policy="weighted_random",
            fallback=lambda: ast.Pass(),
            context={
                "context": context,
                "parent": parent,
                "depth": depth,
                "return_type": return_type,
                "rng": self.rng,
                "nest_decay": self.nest_decay,
            },
        )

    def _generate_simple_statement(
        self, context, parent: Optional[ast.VyperNode]
    ) -> ast.VyperNode:
        if context.all_vars and self.rng.random() < 0.6:
            assign = self.generate_assign(context, parent)
            if assign is not None:
                return assign
        return ast.Pass()

    def generate_if(
        self, context: Context, parent: Optional[ast.VyperNode], depth: int
    ) -> ast.If:
        test_expr = self.expr_generator.generate(BoolT(), context, depth=2)

        if_node = ast.If(test=test_expr, body=[], orelse=[])

        with context.new_scope(ScopeType.IF):
            self.inject_statements(if_node.body, context, if_node, depth + 1)

            if not if_node.body:
                if_node.body.append(ast.Pass())

        if self.rng.random() < 0.4:
            with context.new_scope(ScopeType.IF):
                self.inject_statements(if_node.orelse, context, if_node, depth + 1)

        return if_node

    # TODO add current scope - what if it's module scope
    def scope_is_terminated(self, body: list) -> bool:
        if not body:
            return False

        last_stmt = body[-1]
        if isinstance(last_stmt, (ast.Continue, ast.Break, ast.Return)):
            return True

        if isinstance(last_stmt, ast.If):
            return self.scope_is_terminated(last_stmt.body) and self.scope_is_terminated(
                last_stmt.orelse
            )

        return False

    def get_writable_variables(self, context: Context) -> list[tuple[str, VarInfo]]:
        with context.access_mode(AccessMode.WRITE):
            return context.find_matching_vars()

    def generate_assign(
        self, context, parent: Optional[ast.VyperNode]
    ) -> Optional[ast.Assign]:
        writable_vars = self.get_writable_variables(context)
        if not writable_vars:
            return None

        var_name, var_info = self.rng.choice(writable_vars)

        # Build base reference (self.x or local)
        if var_info.location in (DataLocation.STORAGE, DataLocation.TRANSIENT):
            base = ast.Attribute(value=ast.Name(id="self"), attr=var_name)
        else:
            base = ast.Name(id=var_name)
        base._metadata = {"type": var_info.typ, "varinfo": var_info}

        # Decide if we target a subscript (for arrays/hashmaps/tuples) and allow nesting
        target_type = var_info.typ
        target_node: ast.VyperNode = base

        # HashMapT must always be subscripted - can't assign directly
        # For other subscriptable types, bias towards subscripting
        is_hashmap = isinstance(var_info.typ, HashMapT)
        if is_hashmap or (
            self.expr_generator.is_subscriptable_type(var_info.typ)
            and self.rng.random() < 0.7
        ):
            cur_node, cur_t = self.expr_generator.build_random_chain(
                base,
                var_info.typ,
                context,
                depth=2,
                max_steps=2,
            )
            target_node = cur_node
            target_type = cur_t

            # Keep subscripting while we still have a HashMapT
            while isinstance(target_type, HashMapT):
                cur_node, cur_t = self.expr_generator.build_random_chain(
                    target_node,
                    target_type,
                    context,
                    depth=2,
                    max_steps=1,
                )
                target_node = cur_node
                target_type = cur_t

        if isinstance(target_type, TupleT):
            value = self.expr_generator.random_var_ref(target_type, context)
            if value is None:
                value = self.expr_generator._generate_func_call(
                    target_type, context, depth=2
                )
            if value is None:
                return None
        else:
            value = self.expr_generator.generate(target_type, context, depth=3)

        return ast.Assign(targets=[target_node], value=value)

    def create_vardecl_and_register(
        self, context, parent: Optional[ast.VyperNode]
    ) -> Union[ast.VariableDecl, ast.AnnAssign]:
        var_decl, var_name, var_info = self.generate_vardecl(context, parent)
        self.add_variable(context, var_name, var_info)
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
                        var_info.typ, context, depth=3
                    )
            else:
                init_val = self.expr_generator.generate(var_info.typ, context, depth=3)
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

    def generate_for(
        self, context, parent: Optional[ast.VyperNode], depth: int
    ) -> ast.For:
        pass

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

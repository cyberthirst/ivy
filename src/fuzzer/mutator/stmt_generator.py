import random
from typing import Optional
from vyper.ast import nodes as ast
from vyper.semantics.types import VyperType, BoolT, HashMapT
from vyper.semantics.analysis.base import DataLocation, Modifiability, VarInfo


class FreshNameGenerator:
    def __init__(self, prefix: str = "gen_var"):
        self.prefix = prefix
        self.counter = 0

    def generate(self) -> str:
        name = f"{self.prefix}{self.counter}"
        self.counter += 1
        return name


class StatementGenerator:
    def __init__(self, expr_generator, type_generator, rng: random.Random):
        self.expr_generator = expr_generator
        self.type_generator = type_generator
        self.rng = rng
        self.name_generator = FreshNameGenerator()
        # Collect source fragments from type generation
        self.source_fragments = []

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

    def add_local(self, context, name: str, typ: VyperType):
        """Convenience method to add a local variable."""
        context.add_local(name, typ)

    def generate_type(
        self, context, nesting: int = 3, skip: Optional[set] = None
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

        typ, fragments = self.type_generator.generate_type(nesting=nesting, skip=skip)

        if fragments:
            for fragment in fragments:
                if fragment not in self.source_fragments:
                    self.source_fragments.append(fragment)

        return typ

    def _generate_varinfo(self, context) -> tuple[str, VarInfo]:
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

        var_type = self.generate_type(context, nesting=2, skip=skip_types)

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
        context,
        parent: Optional[ast.VyperNode] = None,
        depth: int = 0,
    ) -> None:
        if depth > self.max_depth:
            return

        if self.rng.random() > self.inject_prob:
            return

        # Generate variables first and insert at beginning
        num_vars = self.rng.randint(0, 2)
        for i in range(num_vars):
            var_decl = self.create_vardecl_and_register(context, parent)
            body.insert(i, var_decl)

        # Generate other statements
        num_other_stmts = self.rng.randint(self.min_stmts, self.max_stmts)
        for _ in range(num_other_stmts):
            stmt = self.generate_statement(context, parent, depth)
            insert_pos = self.rng.randint(num_vars, len(body) - 1)
            body.insert(insert_pos, stmt)

    def generate_statement(
        self,
        context,
        parent: Optional[ast.VyperNode] = None,
        depth: int = 0,
        return_type: Optional[VyperType] = None,
    ) -> Optional[ast.VyperNode]:
        if depth >= self.max_depth:
            return self._generate_simple_statement(context, parent)

        nest_prob = self.nest_decay**depth

        weights = []
        choices = []

        # In module scope, we can only add variable declarations
        if context.is_module_scope:
            choices.append("vardecl")
            weights.append(self.statement_weights["vardecl"])
        else:
            # Inside functions/blocks we can add various statements
            if context.all_vars:
                choices.append("assign")
                weights.append(self.statement_weights["assign"])

            choices.append("vardecl")
            weights.append(self.statement_weights["vardecl"])

            choices.append("if")
            weights.append(self.statement_weights["if"] * nest_prob)

        if not choices:
            return None

        total = sum(weights)
        weights = [w / total for w in weights]

        choice = self.rng.choices(choices, weights=weights)[0]

        if choice == "if":
            return self.generate_if(context, parent, depth)
        elif choice == "assign":
            return self.generate_assign(context, parent)
        elif choice == "vardecl":
            return self.create_vardecl_and_register(context, parent)

        return None

    def _generate_simple_statement(
        self, context, parent: Optional[ast.VyperNode]
    ) -> Optional[ast.VyperNode]:
        if context.all_vars and self.rng.random() < 0.6:
            return self.generate_assign(context, parent)
        else:
            return self.create_vardecl_and_register(context, parent)

    def generate_if(
        self, context, parent: Optional[ast.VyperNode], depth: int
    ) -> ast.If:
        test_expr = self.expr_generator.generate(BoolT(), context, depth=2)

        if_node = ast.If(test=test_expr, body=[], orelse=[])

        context.push_scope()

        num_body_stmts = self.rng.randint(1, 3)
        for _ in range(num_body_stmts):
            stmt = self.generate_statement(context, if_node, depth + 1)
            if stmt:
                if_node.body.append(stmt)

        if not if_node.body:
            if_node.body.append(ast.Pass())

        context.pop_scope()

        if self.rng.random() < 0.4:
            context.push_scope()

            num_else_stmts = self.rng.randint(1, 2)
            for _ in range(num_else_stmts):
                stmt = self.generate_statement(context, if_node, depth + 1)
                if stmt:
                    if_node.orelse.append(stmt)

            context.pop_scope()

        return if_node

    def get_modifiable_variables(self, context) -> list[tuple[str, VarInfo]]:
        modifiable_vars = []
        for name, var_info in context.all_vars.items():
            if var_info.modifiability == Modifiability.MODIFIABLE:
                modifiable_vars.append((name, var_info))
        return modifiable_vars

    def generate_assign(
        self, context, parent: Optional[ast.VyperNode]
    ) -> Optional[ast.Assign]:
        if not context.all_vars:
            return None

        modifiable_vars = self.get_modifiable_variables(context)
        if not modifiable_vars:
            return None

        var_name, var_info = self.rng.choice(modifiable_vars)

        target = ast.Name(id=var_name)
        target._metadata = {"type": var_info.typ, "varinfo": var_info}
        value = self.expr_generator.generate(var_info.typ, context, depth=3)

        return ast.Assign(targets=[target], value=value)

    def create_vardecl_and_register(
        self, context, parent: Optional[ast.VyperNode]
    ) -> ast.VariableDecl:
        var_decl, var_name, var_info = self.generate_vardecl(context, parent)
        self.add_variable(context, var_name, var_info)
        return var_decl

    def generate_vardecl(
        self, context, parent: Optional[ast.VyperNode]
    ) -> tuple[ast.VariableDecl, str, VarInfo]:
        """
        Build a random VariableDecl that is *valid* for the Vyper AST.

        Returns
        -------
        (var_decl, var_name, var_info)
        """
        var_name, var_info = self._generate_varinfo(context)

        anno: ast.VyperNode = ast.Name(id=str(var_info.typ))

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
        init_val = (
            self.expr_generator.generate(var_info.typ, context, depth=3)
            if needs_init
            else None
        )

        var_decl = ast.VariableDecl(
            parent=parent,
            target=ast.Name(id=var_name),
            annotation=anno,
            value=init_val,  # may be None
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

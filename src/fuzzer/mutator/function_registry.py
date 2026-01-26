import random
from typing import Optional, Dict, List, Set, Tuple
from vyper.semantics.types import (
    VyperType,
    HashMapT,
    IntegerT,
    DecimalT,
    TYPE_T,
    StringT,
    BytesT,
)
from vyper.semantics.types.function import (
    ContractFunctionT,
    FunctionVisibility,
    StateMutability,
    PositionalArg,
)
from vyper.builtins.functions import DISPATCH_TABLE
from vyper.builtins._signatures import BuiltinFunctionT
from vyper.ast import nodes as ast


class FreshFunctionNameGenerator:
    def __init__(self, prefix: str = "gen_func"):
        self.prefix = prefix
        self.counter = 0

    def generate(self) -> str:
        name = f"{self.prefix}_{self.counter}"
        self.counter += 1
        return name


class FunctionRegistry:
    def __init__(
        self,
        rng: random.Random,
        max_initial_functions: int = 5,
        max_dynamic_functions: int = 5,
    ):
        self.rng = rng
        self.functions: Dict[str, ContractFunctionT] = {}
        # Map from return type to set of function names
        self.functions_by_return_type: Dict[type, Set[str]] = {}
        # Keep builtin function objects by name (e.g., "min")
        self.builtins: Dict[str, BuiltinFunctionT] = {}
        # Track call graph to prevent cycles
        self.call_graph: Dict[str, Set[str]] = {}  # caller -> callees
        self.current_function: Optional[str] = (
            None  # Track which function we're currently generating
        )
        self.name_generator = FreshFunctionNameGenerator()
        # Separate budgets for initial (generate mode) and dynamic (during generation)
        self.max_initial_functions = max_initial_functions
        self.max_dynamic_functions = max_dynamic_functions
        self.initial_count = 0
        self.dynamic_count = 0
        self._initialize_builtins()

    @staticmethod
    def _is_mutability_compatible(
        caller_mutability: Optional[StateMutability],
        callee_mutability: Optional[StateMutability],
    ) -> bool:
        if caller_mutability is None or callee_mutability is None:
            return True
        return (
            callee_mutability <= caller_mutability
            or caller_mutability >= StateMutability.NONPAYABLE
        )

    def _initialize_builtins(self):
        """Add builtin functions from Vyper's DISPATCH_TABLE."""
        # Add selected builtins that are useful for fuzzing
        useful_builtins = [
            "min",
            "max",
            "len",
            "empty",  # currently skipped by selector (type-literal); kept for future
            "abs",
            "floor",
            "ceil",
            "concat",
            "slice",
        ]

        for name, builtin in DISPATCH_TABLE.items():
            if name in useful_builtins:
                # Builtins are already properly typed, just register them
                self.register_builtin(name, builtin)

    def register_builtin(self, name: str, builtin: BuiltinFunctionT):
        """Register a builtin function type."""
        # Store in our registry for lookup
        # Note: builtins don't go in functions dict since they're not ContractFunctionT
        # but we track them for return type lookups
        self.builtins[name] = builtin
        if hasattr(builtin, "_return_type") and builtin._return_type:
            type_key = type(builtin._return_type)
            if type_key not in self.functions_by_return_type:
                self.functions_by_return_type[type_key] = set()
            self.functions_by_return_type[type_key].add(f"__builtin__{name}")

    def get_compatible_builtins(
        self,
        return_type: VyperType,
        caller_mutability: Optional[StateMutability] = None,
    ) -> List[Tuple[str, BuiltinFunctionT]]:
        """Return a list of (name, builtin) where the builtin can produce return_type.

        - If builtin has a concrete _return_type: require compare_type.
        - For polymorphic returns (e.g., min/max/abs): allow when return_type is numeric.
        - Skip builtins which require type-literal args (TYPE_T) for now.
        """
        compat: List[Tuple[str, BuiltinFunctionT]] = []
        for name, b in self.builtins.items():
            # Skip builtins which require TYPE_T (e.g., empty/convert) for now.
            try:
                inputs = getattr(b, "_inputs", []) or []
                kwargs = getattr(b, "_kwargs", {}) or {}
            except Exception:
                inputs = []
                kwargs = {}

            # If any positional or kwarg expects a TYPE_T (type literal), skip for now
            def _is_type_literal(expected: VyperType) -> bool:
                try:
                    return TYPE_T.any().compare_type(expected)
                except Exception:
                    return False

            has_type_literal = any(_is_type_literal(exp) for _, exp in inputs) or any(
                _is_type_literal(getattr(kw, "typ", None)) for kw in kwargs.values()
            )
            if has_type_literal:
                continue

            if not self._is_mutability_compatible(caller_mutability, b.mutability):
                continue

            ret_t = getattr(b, "_return_type", None)
            if ret_t is not None:
                if return_type.compare_type(ret_t):
                    compat.append((name, b))
                continue

            # Polymorphic returns: support common numeric-family builtins
            if name in {"min", "max", "abs"}:
                if isinstance(return_type, (IntegerT, DecimalT)):
                    compat.append((name, b))
                continue

            if name in {"concat", "slice"}:
                # returns same family as inputs; allow for dynamic bytes/string
                if isinstance(return_type, (StringT, BytesT)):
                    compat.append((name, b))
                continue

            # Otherwise, skip for now (not supported)

        return compat

    def register_function(self, func: ContractFunctionT):
        """Register a ContractFunctionT."""
        self.functions[func.name] = func

        # Update return type index
        if func.return_type:
            type_key = type(func.return_type)
            if type_key not in self.functions_by_return_type:
                self.functions_by_return_type[type_key] = set()
            self.functions_by_return_type[type_key].add(func.name)

        # Initialize call graph entry
        if func.name not in self.call_graph:
            self.call_graph[func.name] = set()

    def add_call(self, caller: str, callee: str):
        """Record a function call in the call graph."""
        if caller not in self.call_graph:
            self.call_graph[caller] = set()
        self.call_graph[caller].add(callee)

    def would_create_cycle(self, caller: str, callee: str) -> bool:
        """Check if adding call from caller to callee would create a cycle."""
        if caller == callee:
            return True

        # Check if callee can transitively reach caller
        visited = set()
        stack = [callee]

        while stack:
            current = stack.pop()
            if current == caller:
                return True
            if current in visited:
                continue
            visited.add(current)
            if current in self.call_graph:
                stack.extend(self.call_graph[current])

        return False

    def get_callable_functions(
        self,
        return_type: VyperType,
        from_function: Optional[str] = None,
        caller_mutability: Optional[StateMutability] = None,
    ) -> List[ContractFunctionT]:
        """Get functions that can be called without creating cycles."""
        type_key = type(return_type)
        if type_key not in self.functions_by_return_type:
            return []

        matching_names = self.functions_by_return_type[type_key]
        callable_funcs = []

        for name in matching_names:
            # Skip builtins in this method (handled separately)
            if name.startswith("__builtin__"):
                continue

            if name not in self.functions:
                continue

            func = self.functions[name]
            # Check type compatibility
            if not return_type.compare_type(func.return_type):
                continue

            # Check for cycles if we have a caller context
            if from_function and self.would_create_cycle(from_function, name):
                continue

            if not self._is_mutability_compatible(caller_mutability, func.mutability):
                continue

            callable_funcs.append(func)

        return callable_funcs

    def get_compatible_function(
        self,
        return_type: VyperType,
        from_function: Optional[str] = None,
        caller_mutability: Optional[StateMutability] = None,
    ) -> Optional[ContractFunctionT]:
        """Get a random compatible function that won't create a cycle."""
        compatible = self.get_callable_functions(
            return_type, from_function, caller_mutability
        )
        if compatible:
            return self.rng.choice(compatible)
        return None

    def create_new_function(
        self,
        return_type: VyperType,
        type_generator,
        max_args: int = 3,
        caller_mutability: Optional[StateMutability] = None,
        *,
        initial: bool = False,
        visibility: Optional[FunctionVisibility] = None,
    ) -> Optional[ast.FunctionDef]:
        """Create a new function with empty body and ContractFunctionT in metadata.
        The body is created later, once we have more information. That allows
        e.g. calls to other functions which are yet to be registered

        Args:
            initial: If True, counts against initial_functions budget (generate mode).
                     If False, counts against dynamic_functions budget.
            visibility: Force a specific visibility when provided.
        """
        # Check the appropriate budget
        if initial:
            if self.initial_count >= self.max_initial_functions:
                return None
            self.initial_count += 1
        else:
            if self.dynamic_count >= self.max_dynamic_functions:
                return None
            self.dynamic_count += 1

        name = self.name_generator.generate()

        # Generate arguments (struct fragments are stored in type_generator)
        num_args = self.rng.randint(0, max_args)
        positional_args = []
        for i in range(num_args):
            arg_type = type_generator.generate_biased_type(skip={HashMapT}, nesting=1)
            arg_name = f"arg{i}"
            positional_args.append(PositionalArg(arg_name, arg_type))

        # Choose function properties
        if visibility is None:
            visibility = self.rng.choice(
                [
                    FunctionVisibility.INTERNAL,
                    FunctionVisibility.EXTERNAL,
                ]
            )

        # Choose state mutability
        mutability_options = [
            m
            for m in (
                StateMutability.PURE,
                StateMutability.VIEW,
                StateMutability.NONPAYABLE,
                StateMutability.PAYABLE,
            )
            if self._is_mutability_compatible(caller_mutability, m)
        ]
        assert mutability_options
        state_mutability = self.rng.choice(mutability_options)

        # Create decorators
        decorator_list = []
        if visibility.name.lower() in ["external"]:
            decorator_list.append(ast.Name(id=visibility.name.lower()))

        if state_mutability.name.lower() in ["view", "pure", "payable"]:
            decorator_list.append(ast.Name(id=state_mutability.name.lower()))

        # Create arguments AST
        args = ast.arguments(args=[], defaults=[], default=None)

        for pos_arg in positional_args:
            arg = ast.arg(
                arg=pos_arg.name,
                annotation=ast.Name(id=str(pos_arg.typ)) if pos_arg.typ else None,
            )
            args.args.append(arg)

        # Create the function definition with empty body
        func_def = ast.FunctionDef(
            name=name,
            args=args,
            body=[],  # Empty body - will be filled by visitor
            decorator_list=decorator_list,
            returns=ast.Name(id=str(return_type)) if return_type else None,
        )

        # Create ContractFunctionT and attach to AST metadata
        func_t = ContractFunctionT(
            name=name,
            positional_args=positional_args,
            keyword_args=[],
            return_type=return_type,
            function_visibility=visibility,
            state_mutability=state_mutability,
            nonreentrant=False,
            ast_def=func_def,
        )

        func_def._metadata["func_type"] = func_t

        self.register_function(func_t)
        return func_def

    def get_pending_implementations(self) -> List[ContractFunctionT]:
        """Get all functions that need implementation (have empty body)."""
        return [
            func
            for func in self.functions.values()
            # check if body was already filled or not
            if func.ast_def and not func.ast_def.body
        ]

    def set_current_function(self, name: Optional[str]):
        """Set the context of which function we're currently generating."""
        self.current_function = name

    def can_generate_more_functions(self, *, initial: bool = False) -> bool:
        if initial:
            return self.initial_count < self.max_initial_functions
        return self.dynamic_count < self.max_dynamic_functions

    def reset(self):
        """Reset the registry state for a new mutation."""
        self.functions.clear()
        self.functions_by_return_type.clear()
        self.builtins.clear()
        self.call_graph.clear()
        self.current_function = None
        self.name_generator.counter = 0
        self.initial_count = 0
        self.dynamic_count = 0
        # Re-initialize builtins
        self._initialize_builtins()

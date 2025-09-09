import random
from typing import Optional, Dict, List, Set
from vyper.semantics.types import VyperType, HashMapT
from vyper.semantics.types.function import (
    ContractFunctionT,
    FunctionVisibility,
    StateMutability,
    PositionalArg,
)
from vyper.builtins.functions import DISPATCH_TABLE
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
    def __init__(self, rng: random.Random, max_generated_functions: int = 5):
        self.rng = rng
        self.functions: Dict[str, ContractFunctionT] = {}
        # Map from return type to set of function names
        self.functions_by_return_type: Dict[type, Set[str]] = {}
        # Track call graph to prevent cycles
        self.call_graph: Dict[str, Set[str]] = {}  # caller -> callees
        self.current_function: Optional[str] = (
            None  # Track which function we're currently generating
        )
        self.name_generator = FreshFunctionNameGenerator()
        self.max_generated_functions = max_generated_functions
        self.generated_count = 0
        self._initialize_builtins()

    def _initialize_builtins(self):
        """Add builtin functions from Vyper's DISPATCH_TABLE."""
        # Add selected builtins that are useful for fuzzing
        useful_builtins = ["min", "max", "len", "empty", "abs", "floor", "ceil"]

        for name, builtin in DISPATCH_TABLE.items():
            if name in useful_builtins:
                # Builtins are already properly typed, just register them
                self.register_builtin(name, builtin)

    def register_builtin(self, name: str, builtin):
        """Register a builtin function type."""
        # Store in our registry for lookup
        # Note: builtins don't go in functions dict since they're not ContractFunctionT
        # but we track them for return type lookups
        if hasattr(builtin, "_return_type") and builtin._return_type:
            type_key = type(builtin._return_type)
            if type_key not in self.functions_by_return_type:
                self.functions_by_return_type[type_key] = set()
            self.functions_by_return_type[type_key].add(f"__builtin__{name}")

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
        self, return_type: VyperType, from_function: Optional[str] = None
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

            callable_funcs.append(func)

        return callable_funcs

    def get_compatible_function(
        self, return_type: VyperType, from_function: Optional[str] = None
    ) -> Optional[ContractFunctionT]:
        """Get a random compatible function that won't create a cycle."""
        compatible = self.get_callable_functions(return_type, from_function)
        if compatible:
            return self.rng.choice(compatible)
        return None

    def create_new_function(
        self, return_type: VyperType, type_generator, max_args: int = 3
    ) -> Optional[ast.FunctionDef]:
        """Create a new function with empty body and ContractFunctionT in metadata.
        The body is created later, once we have more information. That allows
        e.g. calls to other functions which are yet to be registered
        """
        # Check if we've reached the limit
        if self.generated_count >= self.max_generated_functions:
            return None

        name = self.name_generator.generate()
        self.generated_count += 1

        # Generate arguments
        num_args = self.rng.randint(0, max_args)
        positional_args = []
        for i in range(num_args):
            arg_type, _ = type_generator.generate_type(skip={HashMapT}, nesting=1)
            arg_name = f"arg{i}"
            positional_args.append(PositionalArg(arg_name, arg_type))

        # Choose function properties
        visibility = self.rng.choice(
            [
                FunctionVisibility.INTERNAL,
                FunctionVisibility.EXTERNAL,
            ]
        )

        # Choose state mutability
        state_mutability = self.rng.choice(
            [
                StateMutability.PURE,
                StateMutability.VIEW,
                StateMutability.NONPAYABLE,
                StateMutability.PAYABLE,
            ]
        )

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

    def can_generate_more_functions(self) -> bool:
        """Check if more functions can be generated."""
        return self.generated_count < self.max_generated_functions
    
    def reset(self):
        """Reset the registry state for a new mutation."""
        self.functions.clear()
        self.functions_by_return_type.clear()
        self.call_graph.clear()
        self.current_function = None
        self.name_generator.counter = 0
        self.generated_count = 0
        # Re-initialize builtins
        self._initialize_builtins()

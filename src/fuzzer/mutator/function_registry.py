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
    InterfaceT,
)
from vyper.semantics.types.function import (
    ContractFunctionT,
    FunctionVisibility,
    StateMutability,
    PositionalArg,
    KeywordArg,
)
from vyper.builtins.functions import DISPATCH_TABLE
from vyper.builtins._signatures import BuiltinFunctionT
from vyper.ast import nodes as ast

from fuzzer.mutator.convert_utils import convert_target_supported

KWARG_PLACEHOLDER_TAG = "kwarg_placeholder"


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
        nonreentrant_external_prob: float = 0.04,
        nonreentrant_internal_prob: float = 0.01,
        nonreentrancy_by_default: bool = False,
    ):
        self.rng = rng
        self.functions: Dict[str, ContractFunctionT] = {}
        # Map from return type to set of function names
        self.functions_by_return_type: Dict[type, Set[str]] = {}
        # Keep builtin function objects by name (e.g., "min")
        self.builtins: Dict[str, BuiltinFunctionT] = {}
        # Track call graphs to prevent cycles
        self.call_graph: Dict[str, Set[str]] = {}  # caller -> callees (all edges)
        self.internal_call_graph: Dict[str, Set[str]] = {}  # internal-only edges
        self.current_function: Optional[str] = (
            None  # Track which function we're currently generating
        )
        self.name_generator = FreshFunctionNameGenerator()
        # Separate budgets for initial (generate mode) and dynamic (during generation)
        self.max_initial_functions = max_initial_functions
        self.max_dynamic_functions = max_dynamic_functions
        self.nonreentrant_external_prob = nonreentrant_external_prob
        self.nonreentrant_internal_prob = nonreentrant_internal_prob
        self.nonreentrancy_by_default = nonreentrancy_by_default
        self.initial_count = 0
        self.dynamic_count = 0
        self._reachable_from_nonreentrant: Set[str] = set()
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
            "keccak256",
            "convert",
            "abi_encode",
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
            if has_type_literal and name != "convert":
                continue

            if not self._is_mutability_compatible(caller_mutability, b.mutability):
                continue

            if name == "convert":
                if convert_target_supported(return_type):
                    compat.append((name, b))
                continue

            if name == "abi_encode":
                # `abi_encode` returns Bytes[maxlen] where maxlen depends on args.
                # We can only prefilter by target family and minimum feasible size.
                if isinstance(return_type, BytesT) and return_type.length >= 32:
                    compat.append((name, b))
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
        if func.name not in self.internal_call_graph:
            self.internal_call_graph[func.name] = set()
        self._recompute_reachable_from_nonreentrant()

    def set_nonreentrancy_by_default(self, value: bool) -> None:
        self.nonreentrancy_by_default = bool(value)

    def add_call(self, caller: str, callee: str, *, internal: bool):
        """Record a function call in the call graph."""
        if caller not in self.call_graph:
            self.call_graph[caller] = set()
        self.call_graph[caller].add(callee)
        if internal:
            if caller not in self.internal_call_graph:
                self.internal_call_graph[caller] = set()
            self.internal_call_graph[caller].add(callee)
        self._recompute_reachable_from_nonreentrant()

    def _recompute_reachable_from_nonreentrant(self) -> None:
        """Track functions reachable from any @nonreentrant function."""
        self._reachable_from_nonreentrant.clear()
        stack = [name for name, f in self.functions.items() if f.nonreentrant]
        while stack:
            name = stack.pop()
            if name in self._reachable_from_nonreentrant:
                continue
            self._reachable_from_nonreentrant.add(name)
            for callee in self.internal_call_graph.get(name, ()):
                stack.append(callee)

    def reachable_from_nonreentrant(self, name: Optional[str]) -> bool:
        if name is None:
            return False
        return name in self._reachable_from_nonreentrant

    def refresh_reachable_from_nonreentrant(self) -> None:
        self._recompute_reachable_from_nonreentrant()

    def _reaches_nonreentrant(self, name: str) -> bool:
        """Return True if name can reach any nonreentrant function via call graph."""
        visited = set()
        stack = [name]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            func = self.functions.get(current)
            if func is not None and func.nonreentrant:
                return True
            stack.extend(self.internal_call_graph.get(current, ()))
        return False

    def reaches_nonreentrant(self, name: str) -> bool:
        return self._reaches_nonreentrant(name)

    def nonreentrant_probability(
        self,
        *,
        visibility: FunctionVisibility,
        mutability: StateMutability,
    ) -> float:
        if self.nonreentrancy_by_default and visibility == FunctionVisibility.EXTERNAL:
            return 0.0
        if visibility == FunctionVisibility.DEPLOY:
            return 0.0
        if mutability == StateMutability.PURE:
            return 0.0
        if visibility == FunctionVisibility.INTERNAL:
            return self.nonreentrant_internal_prob
        if visibility == FunctionVisibility.EXTERNAL:
            return self.nonreentrant_external_prob
        return 0.0

    def would_create_internal_cycle(self, caller: str, callee: str) -> bool:
        """Check if adding an internal call would create a cycle."""
        return self._would_create_cycle(self.internal_call_graph, caller, callee)

    def would_create_any_cycle(self, caller: str, callee: str) -> bool:
        """Check if adding any call would create a cycle."""
        return self._would_create_cycle(self.call_graph, caller, callee)

    def _would_create_cycle(
        self, graph: Dict[str, Set[str]], caller: str, callee: str
    ) -> bool:
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
            if current in graph:
                stack.extend(graph[current])

        return False

    def _allow_external_cycle(self, *, caller: str, callee: ContractFunctionT) -> bool:
        caller_func = self.functions.get(caller)
        if caller_func is not None and caller_func.nonreentrant and callee.nonreentrant:
            return self.rng.random() < 0.1
        return False

    def get_callable_functions(
        self,
        return_type: Optional[VyperType] = None,
        from_function: Optional[str] = None,
        caller_mutability: Optional[StateMutability] = None,
    ) -> List[ContractFunctionT]:
        """Get functions that can be called without creating cycles.

        If return_type is None, all functions passing the other filters are returned.
        """
        if return_type is not None:
            type_key = type(return_type)
            if type_key not in self.functions_by_return_type:
                return []
            candidate_names = self.functions_by_return_type[type_key]
        else:
            candidate_names = self.functions.keys()

        callable_funcs = []

        for name in candidate_names:
            # Skip builtins in this method (handled separately)
            if name.startswith("__builtin__"):
                continue

            # __init__ and __default__ cannot be called directly
            if name in ("__init__", "__default__"):
                continue

            if name not in self.functions:
                continue

            func = self.functions[name]

            if caller_mutability is StateMutability.PURE and func.is_external:
                # TODO: we can't generate the `self` address in pure functions.
                continue

            # Check type compatibility when a target type is specified
            if return_type is not None:
                if func.return_type is None or not return_type.compare_type(
                    func.return_type
                ):
                    continue

            # Check for cycles if we have a caller context
            if from_function:
                if func.is_internal:
                    if self.would_create_internal_cycle(from_function, name):
                        continue
                    if self.would_create_any_cycle(from_function, name):
                        if not self._allow_external_cycle(
                            caller=from_function, callee=func
                        ):
                            continue
                else:
                    if self.would_create_any_cycle(from_function, name):
                        if not self._allow_external_cycle(
                            caller=from_function, callee=func
                        ):
                            continue

            if not self._is_mutability_compatible(caller_mutability, func.mutability):
                continue

            if from_function and func.is_internal:
                if self.reachable_from_nonreentrant(from_function):
                    if self._reaches_nonreentrant(name):
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
        allow_nonreentrant: bool = True,
        max_kwargs: int = 0,
    ) -> Optional[ast.FunctionDef]:
        """Create a new function with empty body and ContractFunctionT in metadata.
        The body is created later, once we have more information. That allows
        e.g. calls to other functions which are yet to be registered

        Args:
            initial: If True, counts against initial_functions budget (generate mode).
                     If False, counts against dynamic_functions budget.
            visibility: Force a specific visibility when provided.
            allow_nonreentrant: If False, never mark the new function @nonreentrant.
            max_kwargs: Maximum number of keyword arguments to generate.
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
        positional_args = self._generate_positional_args(type_generator, max_args)
        keyword_args: List[KeywordArg] = []
        if max_kwargs > 0:
            keyword_args = self._generate_keyword_args(
                type_generator, max_kwargs, start_index=len(positional_args)
            )

        # Choose function properties
        if visibility is None:
            visibility_choices = [
                FunctionVisibility.INTERNAL,
                FunctionVisibility.EXTERNAL,
            ]
            if caller_mutability is StateMutability.PURE:
                visibility_choices = [FunctionVisibility.INTERNAL]
            visibility = self.rng.choice(visibility_choices)
        elif (
            caller_mutability is StateMutability.PURE
            and visibility is FunctionVisibility.EXTERNAL
        ):
            return None

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

        default_nonreentrant = (
            self.nonreentrancy_by_default
            and visibility == FunctionVisibility.EXTERNAL
            and state_mutability != StateMutability.PURE
        )
        nonreentrant = default_nonreentrant
        reentrant = False
        emit_nonreentrant = False

        if default_nonreentrant and not allow_nonreentrant:
            reentrant = True
            nonreentrant = False
        elif allow_nonreentrant and not default_nonreentrant:
            prob = self.nonreentrant_probability(
                visibility=visibility,
                mutability=state_mutability,
            )
            if prob > 0 and self.rng.random() < prob:
                nonreentrant = True
                emit_nonreentrant = True

        return self._create_function_def(
            name=name,
            positional_args=positional_args,
            return_type=return_type,
            visibility=visibility,
            state_mutability=state_mutability,
            nonreentrant=nonreentrant,
            emit_nonreentrant=emit_nonreentrant,
            reentrant=reentrant,
            keyword_args=keyword_args,
        )

    def create_init(
        self,
        type_generator,
        max_args: int = 3,
        *,
        payable: Optional[bool] = None,
    ) -> Optional[ast.FunctionDef]:
        if "__init__" in self.functions:
            existing = self.functions["__init__"].ast_def
            if existing is not None:
                return existing
            return None

        if payable is None:
            state_mutability = self.rng.choice(
                [StateMutability.NONPAYABLE, StateMutability.PAYABLE]
            )
        else:
            state_mutability = (
                StateMutability.PAYABLE if payable else StateMutability.NONPAYABLE
            )

        positional_args = self._generate_positional_args(type_generator, max_args)

        return self._create_function_def(
            name="__init__",
            positional_args=positional_args,
            return_type=None,
            visibility=FunctionVisibility.DEPLOY,
            state_mutability=state_mutability,
        )

    def create_default(self) -> Optional[ast.FunctionDef]:
        assert "__default__" not in self.functions

        if self.rng.random() < 0.8:
            state_mutability = StateMutability.PAYABLE
        else:
            state_mutability = self.rng.choice(
                [
                    StateMutability.NONPAYABLE,
                    StateMutability.VIEW,
                    StateMutability.PURE,
                ]
            )

        # Always make __default__ nonreentrant (when not @pure) to prevent
        # exponential call trees: raw_call(self, ...) with unmatched calldata
        # re-enters __default__, and multiple such calls create 2^n branching.
        nonreentrant = state_mutability != StateMutability.PURE
        emit_nonreentrant = not self.nonreentrancy_by_default and nonreentrant
        reentrant = False

        return self._create_function_def(
            name="__default__",
            positional_args=[],
            return_type=None,
            visibility=FunctionVisibility.EXTERNAL,
            state_mutability=state_mutability,
            nonreentrant=nonreentrant,
            emit_nonreentrant=emit_nonreentrant,
            reentrant=reentrant,
        )

    def _generate_positional_args(
        self, type_generator, max_args: int
    ) -> List[PositionalArg]:
        num_args = self.rng.randint(0, max_args)
        positional_args = []
        for i in range(num_args):
            arg_type = type_generator.generate_biased_type(skip={HashMapT}, nesting=2)
            arg_name = f"arg{i}"
            positional_args.append(PositionalArg(arg_name, arg_type))
        return positional_args

    def _generate_keyword_args(
        self, type_generator, max_kwargs: int, start_index: int
    ) -> List[KeywordArg]:
        num_kwargs = self.rng.randint(0, max_kwargs)
        keyword_args: List[KeywordArg] = []
        for i in range(num_kwargs):
            arg_type = type_generator.generate_biased_type(
                skip={HashMapT, InterfaceT}, nesting=1
            )
            placeholder = ast.Ellipsis(value=...)
            placeholder._metadata[KWARG_PLACEHOLDER_TAG] = True
            keyword_args.append(
                KeywordArg(
                    f"arg{start_index + i}",
                    arg_type,
                    default_value=placeholder,
                )
            )
        return keyword_args

    def _create_function_def(
        self,
        *,
        name: str,
        positional_args: List[PositionalArg],
        return_type: Optional[VyperType],
        visibility: FunctionVisibility,
        state_mutability: StateMutability,
        nonreentrant: bool = False,
        emit_nonreentrant: bool = True,
        reentrant: bool = False,
        keyword_args: Optional[List[KeywordArg]] = None,
    ) -> ast.FunctionDef:
        if keyword_args is None:
            keyword_args = []
        assert not (nonreentrant and reentrant)
        decorator_list = []
        if visibility == FunctionVisibility.EXTERNAL:
            decorator_list.append(ast.Name(id="external"))
        elif visibility == FunctionVisibility.DEPLOY:
            decorator_list.append(ast.Name(id="deploy"))

        if state_mutability in (
            StateMutability.VIEW,
            StateMutability.PURE,
            StateMutability.PAYABLE,
        ):
            decorator_list.append(ast.Name(id=state_mutability.name.lower()))
        if nonreentrant and emit_nonreentrant:
            decorator_list.append(ast.Name(id="nonreentrant"))
        if reentrant:
            decorator_list.append(ast.Name(id="reentrant"))

        defaults = [kw.default_value for kw in keyword_args]
        args = ast.arguments(args=[], defaults=defaults, default=None)
        for pos_arg in positional_args:
            arg = ast.arg(
                arg=pos_arg.name,
                annotation=ast.Name(id=str(pos_arg.typ)) if pos_arg.typ else None,
            )
            args.args.append(arg)
        for kw_arg in keyword_args:
            arg = ast.arg(
                arg=kw_arg.name,
                annotation=ast.Name(id=str(kw_arg.typ)) if kw_arg.typ else None,
            )
            args.args.append(arg)

        func_def = ast.FunctionDef(
            name=name,
            args=args,
            body=[],
            decorator_list=decorator_list,
            returns=ast.Name(id=str(return_type)) if return_type else None,
        )

        func_t = ContractFunctionT(
            name=name,
            positional_args=positional_args,
            keyword_args=keyword_args,
            return_type=return_type,
            function_visibility=visibility,
            state_mutability=state_mutability,
            nonreentrant=nonreentrant,
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
        self.internal_call_graph.clear()
        self.current_function = None
        self.name_generator.counter = 0
        self.initial_count = 0
        self.dynamic_count = 0
        self._reachable_from_nonreentrant.clear()
        self.nonreentrancy_by_default = False
        # Re-initialize builtins
        self._initialize_builtins()

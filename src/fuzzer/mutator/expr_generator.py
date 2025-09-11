import random

from typing import Optional

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
)
from vyper.semantics.analysis.base import Modifiability

from .value_mutator import ValueMutator
from .context import Context, ExprMutability
from .function_registry import FunctionRegistry
from .strategy import Strategy, StrategyRegistry, StrategySelector, StrategyExecutor
from vyper.semantics.analysis.base import DataLocation
from vyper.semantics.types.function import StateMutability


class ExprGenerator:
    def __init__(
        self,
        value_mutator: ValueMutator,
        rng: random.Random,
        function_registry: FunctionRegistry = None,
        type_generator=None,
    ):
        self.value_mutator = value_mutator
        self.rng = rng
        self.function_registry = function_registry
        self.type_generator = type_generator

        # Build dispatch table for efficient type-to-AST conversion
        self._ast_builders = {
            IntegerT: self._int_to_ast,
            BoolT: self._bool_to_ast,
            AddressT: self._address_to_ast,
            BytesM_T: self._bytesm_to_ast,
            BytesT: self._bytes_to_ast,
            StringT: self._string_to_ast,
            SArrayT: self._array_to_ast,
            DArrayT: self._array_to_ast,
            TupleT: self._tuple_to_ast,
            StructT: self._struct_to_ast,
            DecimalT: self._decimal_to_ast,
        }

        self._strategy_registry = StrategyRegistry()
        self._strategy_selector = StrategySelector(self.rng)
        self._strategy_executor = StrategyExecutor(self._strategy_selector)

        self._register_strategies()

    def generate(
        self, target_type: VyperType, context: Context, depth: int = 3
    ) -> ast.VyperNode:
        # TODO we probably need a special case also for tuples
        if isinstance(target_type, StructT):
            return self._generate_struct(target_type, context, depth)

        if depth == 0:
            return self._generate_terminal(target_type, context)

        # Collect strategies via registry and execute with retry
        strategies = self._strategy_registry.collect(
            type_class=type(target_type),
            include_tags=("expr",),
            context={
                "target_type": target_type,
                "context": context,
                "depth": depth,
                "rng": self.rng,
                "gen": self,
                "function_registry": self.function_registry,
                "mutability": context.current_mutability,
            },
        )

        # Always include literal as a safe fallback if not already in the set
        # (Registry likely contains it already, but make sure.)
        # Execution context is passed through executor.
        def _fallback_literal():
            return self._generate_literal(target_type, context)

        return self._strategy_executor.execute_with_retry(
            strategies,
            policy="weighted_random",  # TODO nested hash maps
            fallback=_fallback_literal,
            context={
                "target_type": target_type,
                "context": context,
                "depth": depth - 1,  # recursive strategies consume depth
                "rng": self.rng,
                "gen": self,
                "function_registry": self.function_registry,
                "mutability": context.current_mutability,
            },
        )

    def _register_strategies(self) -> None:
        # Base literal (terminal; always applicable)
        self._strategy_registry.register(
            Strategy(
                name="expr.literal",
                tags=frozenset({"expr", "terminal"}),
                is_applicable=lambda **_: True,
                # prefer non-literals; keep as low-weight but reliable fallback
                weight=lambda **_: 0.15,
                run=self._run_literal,
            )
        )

        # Variable reference (terminal)
        self._strategy_registry.register(
            Strategy(
                name="expr.var_ref",
                tags=frozenset({"expr", "terminal"}),
                is_applicable=self._is_var_ref_applicable,
                weight=self._weight_var_ref,
                run=self._run_var_ref,
            )
        )

        # Integer arithmetic (recursive)
        self._strategy_registry.register(
            Strategy(
                name="expr.arithmetic",
                tags=frozenset({"expr", "recursive"}),
                type_classes=(IntegerT,),
                is_applicable=lambda **_: True,
                weight=lambda **_: 1.0,
                run=self._run_arithmetic,
            )
        )

        # Unary minus for signed integers (recursive)
        self._strategy_registry.register(
            Strategy(
                name="expr.unary_minus",
                tags=frozenset({"expr", "recursive"}),
                type_classes=(IntegerT,),
                is_applicable=self._is_unary_minus_applicable,
                weight=lambda **_: 1.0,
                run=self._run_unary_minus,
            )
        )

        # Boolean ops and comparisons (recursive targeting BoolT)
        self._strategy_registry.register(
            Strategy(
                name="expr.comparison",
                tags=frozenset({"expr", "recursive"}),
                type_classes=(BoolT,),
                is_applicable=lambda **_: True,
                weight=lambda **_: 1.0,
                run=self._run_comparison,
            )
        )
        self._strategy_registry.register(
            Strategy(
                name="expr.boolean_op",
                tags=frozenset({"expr", "recursive"}),
                type_classes=(BoolT,),
                is_applicable=lambda **_: True,
                weight=lambda **_: 1.0,
                run=self._run_boolean_op,
            )
        )
        self._strategy_registry.register(
            Strategy(
                name="expr.not",
                tags=frozenset({"expr", "recursive"}),
                type_classes=(BoolT,),
                is_applicable=lambda **_: True,
                weight=lambda **_: 1.0,
                run=self._run_not,
            )
        )

        # If-expression (ternary) supporting any target type (recursive)
        self._strategy_registry.register(
            Strategy(
                name="expr.ifexp",
                tags=frozenset({"expr", "recursive"}),
                is_applicable=lambda **_: True,
                weight=lambda **_: 0.3,
                run=self._run_ifexp,
            )
        )

        # Function calls (recursive)
        self._strategy_registry.register(
            Strategy(
                name="expr.func_call",
                tags=frozenset({"expr", "recursive"}),
                is_applicable=self._is_func_call_applicable,
                weight=lambda **_: 1.0,
                run=self._run_func_call,
            )
        )

        # Subscript expressions (recursive)
        self._strategy_registry.register(
            Strategy(
                name="expr.subscript",
                tags=frozenset({"expr", "recursive"}),
                is_applicable=self._is_subscript_applicable,
                weight=self._weight_subscript,
                run=self._run_subscript,
            )
        )

    # Applicability/weight helpers

    def _is_var_ref_applicable(self, **ctx) -> bool:
        target_type: VyperType = ctx["target_type"]
        context: Context = ctx["context"]
        return bool(self._find_matching_variables(target_type, context))

    def _weight_var_ref(self, **ctx) -> float:
        # Slightly bias towards var refs when there are many
        target_type: VyperType = ctx["target_type"]
        context: Context = ctx["context"]
        n = len(self._find_matching_variables(target_type, context))
        return 1.0 if n == 0 else min(2.0, 0.5 + 0.1 * n)

    def _is_unary_minus_applicable(self, **ctx) -> bool:
        target_type: VyperType = ctx["target_type"]
        return isinstance(target_type, IntegerT) and target_type.is_signed

    def _is_func_call_applicable(self, **ctx) -> bool:
        # TODO do we allow constant folding of some builtins?
        # if yes, we'd want to drop this restriction
        mutability: ExprMutability = ctx.get("mutability", ExprMutability.STATEFUL)
        if mutability in (ExprMutability.CONST):
            return False
        return not ctx["context"].is_module_scope

    def _is_subscript_applicable(self, **ctx) -> bool:
        target_type: VyperType = ctx["target_type"]
        context: Context = ctx["context"]
        return bool(
            self._find_nested_subscript_bases(target_type, context, max_steps=3)
        )

    def _weight_subscript(self, **ctx) -> float:
        target_type: VyperType = ctx["target_type"]
        context: Context = ctx["context"]
        # Slight bias based on number of available bases
        n = len(self._find_subscript_bases(target_type, context))
        return 1.0 if n == 0 else min(2.5, 0.5 + 0.2 * n)

    # Runner helpers (consume context kwargs)

    def _run_literal(self, **ctx):
        return self._generate_literal(ctx["target_type"], ctx["context"])

    def _run_var_ref(self, **ctx):
        target_type: VyperType = ctx["target_type"]
        context: Context = ctx["context"]
        matches = self._find_matching_variables(target_type, context)
        if not matches:
            return None
        return self._generate_variable_ref(self.rng.choice(matches), context)

    def _run_arithmetic(self, **ctx):
        return self._generate_arithmetic(
            ctx["target_type"], ctx["context"], ctx["depth"]
        )

    def _run_unary_minus(self, **ctx):
        return self._generate_unary_minus(
            ctx["target_type"], ctx["context"], ctx["depth"]
        )

    def _run_comparison(self, **ctx):
        return self._generate_comparison(ctx["context"], ctx["depth"])

    def _run_boolean_op(self, **ctx):
        return self._generate_boolean_op(ctx["context"], ctx["depth"])

    def _run_not(self, **ctx):
        return self._generate_not(ctx["context"], ctx["depth"])

    def _run_ifexp(self, **ctx):
        return self._generate_ifexp(ctx["target_type"], ctx["context"], ctx["depth"])

    def _run_func_call(self, **ctx):
        return self._generate_func_call(
            ctx["target_type"], ctx["context"], ctx["depth"]
        )

    def _run_subscript(self, **ctx):
        return self._generate_subscript(
            ctx["target_type"], ctx["context"], ctx["depth"]
        )

    def _generate_terminal(
        self, target_type: VyperType, context: Context
    ) -> ast.VyperNode:
        if isinstance(target_type, StructT):
            return self._generate_struct(target_type, context, depth=0)

        matching_vars = self._find_matching_variables(target_type, context)

        if matching_vars and self.rng.random() < 0.5:
            return self._generate_variable_ref(self.rng.choice(matching_vars), context)
        else:
            return self._generate_literal(target_type, context)

    def _generate_literal(
        self, target_type: VyperType, context: Context
    ) -> ast.VyperNode:
        """Generate AST node for a literal value of the given type."""
        value = self.value_mutator.generate_value_for_type(target_type)
        return self._value_to_ast(value, target_type)

    def _value_to_ast(self, value, typ: VyperType) -> ast.VyperNode:
        """Convert a Python value to an AST node"""
        builder = self._ast_builders.get(type(typ))
        if builder:
            return builder(value, typ)
        raise NotImplementedError(
            f"Value to AST conversion not implemented for {type(typ).__name__}"
        )

    def _int_to_ast(self, value: int, typ: IntegerT) -> ast.Int:
        node = ast.Int(value=value)
        node._metadata["type"] = typ
        return node

    def _bool_to_ast(self, value: bool, typ: BoolT) -> ast.NameConstant:
        node = ast.NameConstant(value=value)
        node._metadata["type"] = typ
        return node

    def _address_to_ast(self, value: str, typ: AddressT) -> ast.Hex:
        # Hex node expects string value
        if not value.startswith("0x"):
            value = f"0x{value}"
        node = ast.Hex(value=value)
        node._metadata["type"] = typ
        return node

    def _bytesm_to_ast(self, value: bytes, typ: BytesM_T) -> ast.HexBytes:
        # HexBytes expects bytes value
        node = ast.HexBytes(value=value)
        node._metadata["type"] = typ
        return node

    def _bytes_to_ast(self, value: bytes, typ: BytesT) -> ast.Bytes:
        # Bytes expects bytes value
        node = ast.Bytes(value=value)
        node._metadata["type"] = typ
        return node

    def _string_to_ast(self, value: str, typ: StringT) -> ast.Str:
        node = ast.Str(value=value)
        node._metadata["type"] = typ
        return node

    def _decimal_to_ast(self, value, typ: DecimalT) -> ast.Decimal:
        node = ast.Decimal(value=value)
        node._metadata["type"] = typ
        return node

    def _array_to_ast(self, value: list, typ) -> ast.List:
        """Handle both SArrayT and DArrayT - elements must be AST nodes."""
        elements = [self._value_to_ast(v, typ.value_type) for v in value]
        node = ast.List(elements=elements)
        node._metadata["type"] = typ
        return node

    def _tuple_to_ast(self, value: tuple, typ: TupleT) -> ast.Tuple:
        """Handle TupleT - elements must be AST nodes."""
        elements = [self._value_to_ast(v, t) for v, t in zip(value, typ.member_types)]
        node = ast.Tuple(elements=elements)
        node._metadata["type"] = typ
        return node

    def _struct_to_ast(self, value: dict, typ: StructT) -> ast.Call:
        """Generate struct constructor call from dict value."""
        call_node = ast.Call(func=ast.Name(id=typ._id), args=[], keywords=[])

        for field_name, field_value in value.items():
            field_type = typ.members[field_name]
            field_expr = self._value_to_ast(field_value, field_type)
            keyword = ast.keyword(arg=field_name, value=field_expr)
            call_node.keywords.append(keyword)

        call_node._metadata["type"] = typ
        return call_node

    def _find_matching_variables(
        self, target_type: VyperType, context: Context
    ) -> list[str]:
        matches = []
        const_only = context.current_mutability == ExprMutability.CONST
        for name, var_info in context.all_vars.items():
            # Directional assignability: a value of var_info.typ must be assignable
            # to a expr of target_type
            if target_type.compare_type(var_info.typ):
                if const_only and var_info.modifiability != Modifiability.CONSTANT:
                    continue
            # Directional: can var_type be used where target_type is expected?
            if target_type.compare_type(var_info.typ):
                matches.append(name)
        return matches

    def _find_subscript_bases(
        self, target_type: VyperType, context: Context
    ) -> list[tuple[str, VyperType]]:
        """Return list of (var_name, var_type) that can be subscripted to yield target_type.

        Supports HashMapT[key_type, value_type] and sequences (SArrayT, DArrayT, TupleT).
        For TupleT, any element matching target_type is acceptable.
        """
        candidates: list[tuple[str, VyperType]] = []
        for name, var_info in context.all_vars.items():
            var_t = var_info.typ
            # HashMap[key->value]
            # TODO nested hash maps
            if isinstance(var_t, HashMapT):
                if target_type.compare_type(var_t.value_type):
                    candidates.append((name, var_t))
                continue

            # Static/Dynamic arrays
            if isinstance(var_t, (SArrayT, DArrayT)):
                if target_type.compare_type(var_t.value_type):
                    candidates.append((name, var_t))
                continue

            # Tuples: allow if any member matches target_type
            if isinstance(var_t, TupleT):
                for mt in getattr(var_t, "member_types", []):
                    if target_type.compare_type(mt):
                        candidates.append((name, var_t))
                        break

        return candidates

    def _find_nested_subscript_bases(
        self, target_type: VyperType, context: Context, max_steps: int
    ) -> list[tuple[str, VyperType]]:
        result: list[tuple[str, VyperType]] = []
        for name, var_info in context.all_vars.items():
            t = var_info.typ
            if not self.is_subscriptable_type(t):
                continue
            if self.can_reach_via_subscript(t, target_type, max_steps):
                result.append((name, t))
        return result

    def _generate_variable_ref(self, name: str, context: Context) -> ast.VyperNode:
        var_info = context.all_vars[name]

        if var_info.location in (DataLocation.STORAGE, DataLocation.TRANSIENT):
            node = ast.Attribute(value=ast.Name(id="self"), attr=name)
        else:
            node = ast.Name(id=name)

        node._metadata["type"] = var_info.typ
        node._metadata["varinfo"] = var_info
        return node

    def _generate_subscript(
        self, target_type: VyperType, context: Context, depth: int
    ) -> Optional[ast.Subscript]:
        bases = self._find_nested_subscript_bases(target_type, context, max_steps=3)
        if not bases:
            return None

        name, base_t = self.rng.choice(bases)
        base_node: ast.VyperNode = self._generate_variable_ref(name, context)
        built = self.build_chain_to_target(
            base_node,
            base_t,
            target_type,
            context,
            depth,
            max_steps=3,
        )
        if not built:
            return None
        node, _ = built
        node._metadata["type"] = target_type
        return node

    def _generate_ifexp(
        self, target_type: VyperType, context: Context, depth: int
    ) -> ast.IfExp:
        # Condition must be bool; branches must yield the same type
        test = self.generate(BoolT(), context, depth)
        body = self.generate(target_type, context, depth)
        orelse = self.generate(target_type, context, depth)

        node = ast.IfExp(test=test, body=body, orelse=orelse)
        node._metadata["type"] = target_type
        return node

    # ----------------------
    # Shared subscript utils
    # ----------------------

    def is_subscriptable_type(self, t: VyperType) -> bool:
        return isinstance(t, (HashMapT, SArrayT, DArrayT, TupleT))

    def _random_integer_type(self) -> IntegerT:
        bits = self.rng.choice([8, 16, 32, 64, 128, 256])
        signed = self.rng.choice([True, False])
        return IntegerT(signed, bits)

    def _generate_index_for_sequence(
        self, seq_t: VyperType, context: Context, depth: int
    ) -> ast.VyperNode:
        assert isinstance(seq_t, (SArrayT, DArrayT))
        idx_t = self._random_integer_type()

        capacity = seq_t.length
        small_window = max(0, min(capacity - 1, 3))
        p_small = 0.8 if isinstance(seq_t, DArrayT) else 0.6

        if capacity > 0 and small_window >= 0 and self.rng.random() < p_small:
            val = self.rng.randint(0, small_window)
            return self._int_to_ast(val, idx_t)

        return self.generate(idx_t, context, max(0, depth))

    def can_reach_via_subscript(
        self, base_t: VyperType, target_t: VyperType, steps_left: int
    ) -> bool:
        if steps_left <= 0:
            return False

        def child_types(t: VyperType):
            if isinstance(t, HashMapT):
                yield t.value_type
            elif isinstance(t, (SArrayT, DArrayT)):
                yield t.value_type
            elif isinstance(t, TupleT):
                for mt in getattr(t, "member_types", []):
                    yield mt

        for ct in child_types(base_t):
            if target_t.compare_type(ct):
                return True
            if self.is_subscriptable_type(ct) and self.can_reach_via_subscript(
                ct, target_t, steps_left - 1
            ):
                return True
        return False

    def build_chain_to_target(
        self,
        base_node: ast.VyperNode,
        base_type: VyperType,
        target_type: VyperType,
        context: Context,
        depth: int,
        max_steps: int = 3,
    ) -> Optional[tuple[ast.VyperNode, VyperType]]:
        cur_t = base_type
        node = base_node
        steps_remaining = max(1, min(max_steps, depth + 1))

        while steps_remaining > 0 and self.is_subscriptable_type(cur_t):
            next_options: list[tuple[VyperType, ast.VyperNode]] = []

            if isinstance(cur_t, HashMapT):
                key_expr = self.generate(cur_t.key_type, context, max(0, depth))
                next_options.append((cur_t.value_type, key_expr))

            elif isinstance(cur_t, (SArrayT, DArrayT)):
                idx_expr = self._generate_index_for_sequence(cur_t, context, depth)
                next_options.append((cur_t.value_type, idx_expr))

            elif isinstance(cur_t, TupleT):
                mtypes = list(getattr(cur_t, "member_types", []))
                choices = []
                for i, mt in enumerate(mtypes):
                    if target_type.compare_type(mt) or (
                        self.is_subscriptable_type(mt)
                        and self.can_reach_via_subscript(
                            mt, target_type, steps_left=steps_remaining - 1
                        )
                    ):
                        choices.append((i, mt))
                if not choices:
                    break
                idx, child_t = self.rng.choice(choices)
                idx_expr = self._int_to_ast(idx, IntegerT(False, 256))
                next_options.append((child_t, idx_expr))

            direct = [
                (ct, idx) for (ct, idx) in next_options if target_type.compare_type(ct)
            ]
            chosen_ct, idx_expr = self.rng.choice(direct or next_options)

            node = ast.Subscript(value=node, slice=idx_expr)
            node._metadata = {"type": chosen_ct}
            cur_t = chosen_ct
            steps_remaining -= 1

            if target_type.compare_type(cur_t):
                return node, cur_t

        if target_type.compare_type(cur_t):
            return node, cur_t
        return None

    def build_random_chain(
        self,
        base_node: ast.VyperNode,
        base_type: VyperType,
        context: Context,
        depth: int,
        max_steps: int = 2,
    ) -> tuple[ast.VyperNode, VyperType]:
        cur_t = base_type
        node = base_node
        steps = self.rng.randint(1, max_steps)

        for _ in range(steps):
            if isinstance(cur_t, HashMapT):
                idx_expr = self.generate(cur_t.key_type, context, max(0, depth))
                cur_t = cur_t.value_type
            elif isinstance(cur_t, (SArrayT, DArrayT)):
                idx_expr = self._generate_index_for_sequence(cur_t, context, depth)
                cur_t = cur_t.value_type
            elif isinstance(cur_t, TupleT):
                mtypes = list(getattr(cur_t, "member_types", []))
                if not mtypes:
                    break
                idx = self.rng.randrange(len(mtypes))
                idx_expr = self._int_to_ast(idx, IntegerT(False, 256))
                cur_t = mtypes[idx]
            else:
                break

            node = ast.Subscript(value=node, slice=idx_expr)
            node._metadata = {"type": cur_t}

            if not self.is_subscriptable_type(cur_t):
                break

        return node, cur_t

    def _generate_arithmetic(
        self, target_type: IntegerT, context: Context, depth: int
    ) -> ast.BinOp:
        op_classes = [ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod]
        op_class = self.rng.choice(op_classes)

        left = self.generate(target_type, context, depth)
        right = self.generate(target_type, context, depth)

        node = ast.BinOp(left=left, op=op_class(), right=right)

        if isinstance(node.op, (ast.FloorDiv, ast.Mod, ast.Div)):
            if isinstance(right, ast.Int) and getattr(right, "value", None) == 0:
                context.compilation_xfail = True

        node._metadata["type"] = target_type
        return node

    def _generate_comparison(self, context: Context, depth: int) -> ast.Compare:
        op_classes = [ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq]
        op_class = self.rng.choice(op_classes)

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

        comparable_type = self.rng.choice(comparable_types)

        left = self.generate(comparable_type, context, depth)
        right = self.generate(comparable_type, context, depth)

        node = ast.Compare(left=left, ops=[op_class()], comparators=[right])
        node._metadata["type"] = BoolT()
        return node

    def _generate_boolean_op(self, context: Context, depth: int) -> ast.BoolOp:
        op_classes = [ast.And, ast.Or]
        op_class = self.rng.choice(op_classes)

        values = [self.generate(BoolT(), context, depth) for _ in range(2)]

        node = ast.BoolOp(op=op_class(), values=values)
        node._metadata["type"] = BoolT()
        return node

    def _generate_not(self, context: Context, depth: int) -> ast.UnaryOp:
        operand = self.generate(BoolT(), context, depth)
        node = ast.UnaryOp(op=ast.Not(), operand=operand)
        node._metadata["type"] = BoolT()
        return node

    def _generate_unary_minus(
        self, target_type: IntegerT, context: Context, depth: int
    ) -> ast.UnaryOp:
        operand = self.generate(target_type, context, depth)
        node = ast.UnaryOp(op=ast.USub(), operand=operand)
        node._metadata["type"] = target_type
        return node

    def _generate_struct(self, target_type, context: Context, depth: int) -> ast.Call:
        assert isinstance(target_type, StructT)

        # Create the struct constructor call
        call_node = ast.Call(func=ast.Name(id=target_type._id), args=[], keywords=[])

        for field_name, field_type in target_type.members.items():
            field_expr = self.generate(field_type, context, max(0, depth - 1))

            keyword = ast.keyword(arg=field_name, value=field_expr)
            call_node.keywords.append(keyword)

        call_node._metadata["type"] = target_type
        return call_node

    def _generate_func_call(
        self, target_type: VyperType, context: Context, depth: int
    ) -> Optional[ast.Call]:
        """Generate a function call, selecting between user functions and builtins."""
        if not self.function_registry:
            return None

        current_func = self.function_registry.current_function
        assert current_func is not None

        # Gather candidates
        compatible_func = self.function_registry.get_compatible_function(
            target_type, current_func
        )
        compatible_builtins = self.function_registry.get_compatible_builtins(
            target_type
        )

        # Decide path: prefer user function, but sometimes pick builtin
        use_builtin = False
        if compatible_builtins:
            if not compatible_func:
                use_builtin = True
            else:
                use_builtin = self.rng.random() < 0.4

        if use_builtin:
            name, builtin = self.rng.choice(compatible_builtins)
            return self._generate_builtin_call(
                name, builtin, target_type, context, depth
            )

        # Fall back to user function (existing or create new)
        if not compatible_func or self.rng.random() >= 0.9:
            # Create a new function (returns ast.FunctionDef or None)
            func_def = self.function_registry.create_new_function(
                return_type=target_type, type_generator=self.type_generator, max_args=2
            )
            if func_def is None:
                # Can't create more functions; try builtin if available
                if compatible_builtins:
                    name, builtin = self.rng.choice(compatible_builtins)
                    return self._generate_builtin_call(
                        name, builtin, target_type, context, depth
                    )
                return None
            func_t = func_def._metadata["func_type"]
        else:
            func_t = compatible_func

        func_name = func_t.name
        if func_t.is_external:
            func_node = ast.Name(id=func_name)
        else:
            assert func_t.is_internal, (
                f"Expected internal or external function, got {func_t}"
            )
            func_node = ast.Attribute(value=ast.Name(id="self"), attr=func_name)

        func_node._metadata = getattr(func_node, "_metadata", {})
        func_node._metadata["type"] = func_t

        # Generate arguments
        args = []
        for pos_arg in func_t.positional_args:
            if pos_arg.typ:
                arg_expr = self.generate(pos_arg.typ, context, max(0, depth))
                args.append(arg_expr)

        # Build and maybe wrap
        result_node = self._finalize_call(func_node, args, func_t.return_type, func_t)

        # Record the call in the call graph
        if self.function_registry.current_function:
            self.function_registry.add_call(
                self.function_registry.current_function, func_name
            )

        return result_node

    def _finalize_call(self, func_node, args, return_type, func_t=None):
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

    def _generate_uint256_literal(self, value: int) -> ast.Int:
        node = ast.Int(value=value)
        node._metadata["type"] = IntegerT(False, 256)
        return node

    def _generate_builtin_call(
        self,
        name: str,
        builtin,
        target_type: VyperType,
        context: Context,
        depth: int,
    ) -> Optional[ast.Call]:
        # func node
        func_node = ast.Name(id=name)
        func_node._metadata = getattr(func_node, "_metadata", {})
        func_node._metadata["type"] = builtin

        # Dispatch per builtin for arg synthesis + return typing
        if name in {"min", "max"}:
            if not isinstance(target_type, (IntegerT, DecimalT)):
                return None
            arg_t = target_type
            a0 = self.generate(arg_t, context, max(0, depth))
            a1 = self.generate(arg_t, context, max(0, depth))
            return self._finalize_call(func_node, [a0, a1], arg_t)

        if name == "abs":
            if isinstance(target_type, IntegerT) and not target_type.is_signed:
                return None
            if not isinstance(target_type, (IntegerT, DecimalT)):
                return None
            arg_t = target_type
            a0 = self.generate(arg_t, context, max(0, depth))
            return self._finalize_call(func_node, [a0], arg_t)

        if name in {"floor", "ceil"}:
            # Expect decimal arg, return concrete integer type from builtin

            a0 = self.generate(DecimalT(), context, max(0, depth))
            ret_t = getattr(builtin, "_return_type", None) or IntegerT(True, 256)
            return self._finalize_call(func_node, [a0], ret_t)

        if name == "len":
            # Only support BytesT/StringT to start
            max_len = self.rng.randint(1, 128)

            arg_t = self.rng.choice([BytesT(max_len), StringT(max_len)])
            a0 = self.generate(arg_t, context, max(0, depth))
            ret_t = getattr(builtin, "_return_type", IntegerT(False, 256))
            return self._finalize_call(func_node, [a0], ret_t)

        if name == "concat":
            # Choose k in [2,8] and a budget B in [0, target.length].
            # Partition B into k nonnegative parts and generate that many args.
            if not isinstance(target_type, (StringT, BytesT)):
                return None

            tgt_len = target_type.length
            k = self.rng.randint(2, 8)
            budget = self.rng.randint(0, tgt_len)

            # composition of `budget` into `k` nonnegative parts
            parts = []
            remaining = budget
            for i in range(k - 1):
                li = self.rng.randint(0, remaining)
                parts.append(li)
                remaining -= li
            parts.append(remaining)

            # shuffle to avoid monotone order bias
            self.rng.shuffle(parts)

            def make_typ(n):
                return StringT(n) if isinstance(target_type, StringT) else BytesT(n)

            arg_types = [make_typ(n) for n in parts]
            args = [self.generate(t, context, max(0, depth - 1)) for t in arg_types]

            # The concat return length is sum(parts) which is <= target length by design
            return self._finalize_call(func_node, args, make_typ(sum(parts)))

        if name == "slice":
            if isinstance(target_type, (BytesT, StringT)):
                ret_len = target_type.length
                arg_len = ret_len + self.rng.randint(0, 32)
                arg_t = (
                    StringT(arg_len)
                    if isinstance(target_type, StringT)
                    else BytesT(arg_len)
                )
                arg0 = self.generate(arg_t, context, max(0, depth))
                # pick literal start/length within bounds for static safety
                start = self.rng.randint(0, max(0, arg_len - ret_len))
                length = (
                    ret_len
                    if ret_len > 0
                    else self.rng.randint(1, max(1, arg_len - start))
                )
                a1 = self._generate_uint256_literal(start)
                a2 = self._generate_uint256_literal(length)
                return self._finalize_call(func_node, [arg0, a1, a2], target_type)
            return None

        # Unknown builtin (not yet supported)
        return None

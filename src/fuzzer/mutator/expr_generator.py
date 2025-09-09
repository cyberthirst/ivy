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
    DecimalT,
)

from .value_mutator import ValueMutator
from .context import Context
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
            },
        )

        # Always include literal as a safe fallback if not already in the set
        # (Registry likely contains it already, but make sure.)
        # Execution context is passed through executor.
        def _fallback_literal():
            return self._generate_literal(target_type, context)

        return self._strategy_executor.execute_with_retry(
            strategies,
            policy="weighted_random",
            fallback=_fallback_literal,
            context={
                "target_type": target_type,
                "context": context,
                "depth": depth - 1,  # recursive strategies consume depth
                "rng": self.rng,
                "gen": self,
                "function_registry": self.function_registry,
            },
        )

    def _register_strategies(self) -> None:
        # Base literal (terminal; always applicable)
        self._strategy_registry.register(
            Strategy(
                name="expr.literal",
                tags=frozenset({"expr", "terminal"}),
                is_applicable=lambda **_: True,
                weight=lambda **_: 1.0,
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
        return not ctx["context"].is_module_scope

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

    def _run_func_call(self, **ctx):
        return self._generate_func_call(
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
        for name, var_info in context.all_vars.items():
            # Directional assignability: a value of var_info.typ must be assignable
            # to a expr of target_type
            if target_type.compare_type(var_info.typ):
                matches.append(name)
        return matches

    def _generate_variable_ref(self, name: str, context: Context) -> ast.VyperNode:
        var_info = context.all_vars[name]

        if var_info.location in (DataLocation.STORAGE, DataLocation.TRANSIENT):
            node = ast.Attribute(value=ast.Name(id="self"), attr=name)
        else:
            node = ast.Name(id=name)

        node._metadata["type"] = var_info.typ
        node._metadata["varinfo"] = var_info
        return node

    def _generate_arithmetic(
        self, target_type: IntegerT, context: Context, depth: int
    ) -> ast.BinOp:
        op_classes = [ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod]
        op_class = self.rng.choice(op_classes)

        left = self.generate(target_type, context, depth)
        right = self.generate(target_type, context, depth)

        node = ast.BinOp(left=left, op=op_class(), right=right)
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
        """Generate a function call, either to existing or new function."""
        if not self.function_registry:
            return None

        current_func = self.function_registry.current_function
        assert current_func is not None

        compatible_func = self.function_registry.get_compatible_function(
            target_type, current_func
        )

        # Try to use existing function with 90% probability
        if compatible_func and self.rng.random() < 0.9:
            func_t = compatible_func
            func_name = func_t.name
        else:
            # Create a new function (returns ast.FunctionDef or None)
            func_def = self.function_registry.create_new_function(
                return_type=target_type, type_generator=self.type_generator, max_args=2
            )
            if func_def is None:
                # Can't create more functions
                return None
            # Get the ContractFunctionT from metadata
            func_t = func_def._metadata["func_type"]
            func_name = func_def.name

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

        # Create the call node
        call_node = ast.Call(func=func_node, args=args, keywords=[])
        call_node._metadata["type"] = func_t.return_type

        # Wrap external calls using ExtCall/StaticCall
        if func_t.is_external:
            if not hasattr(func_t, "state_mutability"):
                pass
            if func_t.mutability in (StateMutability.PURE, StateMutability.VIEW):
                wrapped = ast.StaticCall(value=call_node)
            else:
                wrapped = ast.ExtCall(value=call_node)
            wrapped._metadata = getattr(wrapped, "_metadata", {})
            wrapped._metadata["type"] = func_t.return_type
            result_node = wrapped
        else:
            result_node = call_node

        # Record the call in the call graph
        if self.function_registry.current_function:
            self.function_registry.add_call(
                self.function_registry.current_function, func_name
            )

        return result_node

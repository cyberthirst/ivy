import random

from typing import Optional, Union

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
from vyper.semantics.types.subscriptable import _SequenceT

from .literal_generator import LiteralGenerator
from .context import Context, ExprMutability
from .function_registry import FunctionRegistry
from .strategy import Strategy, StrategyRegistry, StrategySelector, StrategyExecutor
from .mode import MutationMode
from vyper.semantics.analysis.base import DataLocation
from vyper.semantics.types.function import StateMutability

from src.fuzzer.xfail import XFailExpectation


class ExprGenerator:
    def __init__(
        self,
        literal_generator: LiteralGenerator,
        rng: random.Random,
        function_registry: Optional[FunctionRegistry] = None,
        type_generator=None,
        *,
        mode: MutationMode,
    ):
        self.literal_generator = literal_generator
        self.rng = rng
        self.function_registry = function_registry
        self.type_generator = type_generator
        self.mode = mode

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

    # Probability of continuing to build recursive expressions.
    # At each level, only this fraction will attempt complex structures.
    # P(reaching depth k) = continuation_prob^k
    # With 0.2: depth1=20%, depth2=4%, depth3=0.8%
    CONTINUATION_PROB = 0.2

    def generate(
        self, target_type: VyperType, context: Context, depth: int = 3
    ) -> ast.VyperNode:
        # TODO we probably need a special case also for tuples
        if isinstance(target_type, StructT):
            return self._generate_struct(target_type, context, depth)

        # Early termination: probabilistic + hard depth limit
        if depth <= 0 or self.rng.random() > self.CONTINUATION_PROB:
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
                is_applicable=self._is_ifexp_applicable,
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
        if self.mode == MutationMode.MUTATE:
            return False
        mutability: ExprMutability = ctx.get("mutability", ExprMutability.STATEFUL)
        if mutability == ExprMutability.CONST:
            return False
        return not ctx["context"].is_module_scope

    def _is_subscript_applicable(self, **ctx) -> bool:
        target_type: VyperType = ctx["target_type"]
        context: Context = ctx["context"]
        return bool(
            self._find_nested_subscript_bases(target_type, context, max_steps=3)
        )

    def _is_ifexp_applicable(self, **ctx) -> bool:
        # Disallow if-expressions in constant contexts due to compiler limitation
        mutability: ExprMutability = ctx.get("mutability", ExprMutability.STATEFUL)
        return mutability != ExprMutability.CONST

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

        if matching_vars and self.rng.random() < 0.95:
            return self._generate_variable_ref(self.rng.choice(matching_vars), context)
        else:
            return self._generate_literal(target_type, context)

    def _generate_literal(
        self, target_type: VyperType, context: Context
    ) -> ast.VyperNode:
        """Generate AST node for a literal value of the given type."""
        value = self.literal_generator.generate(target_type)
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
        assert isinstance(node, ast.Subscript)
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
        self,
        base_node: ast.VyperNode,
        seq_t: _SequenceT,
        context: Context,
        depth: int,
    ) -> ast.VyperNode:
        """Generate an index expression for arrays with three modes:
        - Guarded by len(base) (preferred)
        - Random expression (unconstrained)
        - OOB literal that forces compilation xfail (rare)

        Notes:
        - Use uint256-only indices (no signedness for now).
        - Bias DynArray towards small literals because runtime len tends to be small.
        - For SArray, guarding is still fine and semantically interesting.
        """
        assert isinstance(seq_t, (SArrayT, DArrayT))

        # Fixed unsigned 256-bit type for indices
        idx_t = IntegerT(False, 256)

        # Strategy weights
        p_guard = 0.60
        p_rand = 0.35
        # remaining 0.05 -> OOB literal

        roll = self.rng.random()

        # Helper: literal small index for DArray bias
        def _small_literal_for_dynarray():
            # very small indices [0..2]
            val = self.rng.randint(0, 2)
            return self._int_to_ast(val, idx_t)

        # Helper: generate random uint256 index expression
        def _random_uint_index():
            return self.generate(idx_t, context, max(0, depth))

        # Helper: guarded index using len(base)
        def _guarded_index():
            # i if i < len(base) else (len(base)-1 if len(base) > 0 else 0)
            i_expr = (
                _small_literal_for_dynarray()
                if isinstance(seq_t, DArrayT) and self.rng.random() < 0.75
                else _random_uint_index()
            )

            len_call = ast.Call(func=ast.Name(id="len"), args=[base_node], keywords=[])
            len_call._metadata = getattr(len_call, "_metadata", {})
            len_call._metadata["type"] = IntegerT(False, 256)

            zero = self._generate_uint256_literal(0)
            one = self._generate_uint256_literal(1)

            len_gt_zero = ast.Compare(
                left=len_call,
                ops=[ast.Gt()],
                comparators=[zero],
            )
            len_gt_zero._metadata = {"type": BoolT()}

            len_minus_one = ast.BinOp(left=len_call, op=ast.Sub(), right=one)
            len_minus_one._metadata = {"type": idx_t}

            safe_fallback = ast.IfExp(
                test=len_gt_zero,
                body=len_minus_one,
                orelse=zero,
            )
            safe_fallback._metadata = {"type": idx_t}

            cond = ast.Compare(left=i_expr, ops=[ast.Lt()], comparators=[len_call])
            cond._metadata = {"type": BoolT()}

            guarded = ast.IfExp(test=cond, body=i_expr, orelse=safe_fallback)
            guarded._metadata = {"type": idx_t}
            return guarded

        def _oob_literal():
            cap = seq_t.length  # SArray: fixed size, DArray: max capacity
            # choose either cap (== length/capacity) or cap+1 as out-of-bounds
            if self.rng.random() < 0.5:
                val = cap if cap > 0 else 1
            else:
                val = cap + 1 if cap > 0 else 1
            # For both SArray and DArray, literal index >= declared length/capacity
            # is detectable at compile time by Vyper
            context.compilation_xfails.append(
                XFailExpectation(
                    kind="compilation",
                    reason="generated out-of-bounds array index",
                )
            )
            return self._int_to_ast(val, idx_t)

        if roll < p_guard:
            return _guarded_index()
        elif roll < p_guard + p_rand:
            # Bias for DynArray to small literal sometimes
            if isinstance(seq_t, DArrayT) and self.rng.random() < 0.6:
                return _small_literal_for_dynarray()
            return _random_uint_index()
        else:
            # Only makes sense for SArray where cap is static; still set xfail
            return _oob_literal()

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
                idx_expr = self._generate_index_for_sequence(
                    node, cur_t, context, depth
                )
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
                idx_expr = self._generate_index_for_sequence(
                    node, cur_t, context, depth
                )
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
                context.compilation_xfails.append(
                    XFailExpectation(
                        kind="compilation",
                        reason="division or modulo by zero should fail compilation",
                    )
                )

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
    ) -> Optional[Union[ast.Call, ast.StaticCall, ast.ExtCall]]:
        """Generate a function call, selecting between user functions and builtins."""
        if not self.function_registry:
            return None

        current_func = self.function_registry.current_function
        assert current_func is not None

        caller_mutability = context.current_function_mutability

        # Gather candidates
        compatible_func = self.function_registry.get_compatible_function(
            target_type,
            current_func,
            caller_mutability=caller_mutability,
        )
        compatible_builtins = self.function_registry.get_compatible_builtins(
            target_type,
            caller_mutability=caller_mutability,
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
                return_type=target_type,
                type_generator=self.type_generator,
                max_args=2,
                caller_mutability=caller_mutability,
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

    def _generate_uint256_literal(self, value: int) -> ast.Int:
        node = ast.Int(value=value)
        node._metadata["type"] = IntegerT(False, 256)
        return node

    # ---------
    # AST utils
    # ---------
    def _builtin_call_expr(self, name: str, args: list, ret_t: VyperType) -> ast.Call:
        b = None
        if self.function_registry and name in self.function_registry.builtins:
            b = self.function_registry.builtins[name]
        fn = ast.Name(id=name)
        fn._metadata = getattr(fn, "_metadata", {})
        if b is not None:
            fn._metadata["type"] = b
        call = ast.Call(func=fn, args=args, keywords=[])
        call._metadata = getattr(call, "_metadata", {})
        call._metadata["type"] = ret_t
        return call

    def _uint_binop_expr(
        self, left: ast.VyperNode, op, right: ast.VyperNode
    ) -> ast.BinOp:
        node = ast.BinOp(left=left, op=op, right=right)
        node._metadata = getattr(node, "_metadata", {})
        node._metadata["type"] = IntegerT(False, 256)
        return node

    def _uint_cmp_expr(
        self, left: ast.VyperNode, op, right: ast.VyperNode
    ) -> ast.Compare:
        node = ast.Compare(left=left, ops=[op], comparators=[right])
        node._metadata = getattr(node, "_metadata", {})
        node._metadata["type"] = BoolT()
        return node

    def _ifexp_typed(
        self,
        test: ast.VyperNode,
        body: ast.VyperNode,
        orelse: ast.VyperNode,
        ret_t: VyperType,
    ) -> ast.IfExp:
        node = ast.IfExp(test=test, body=body, orelse=orelse)
        node._metadata = getattr(node, "_metadata", {})
        node._metadata["type"] = ret_t
        return node

    def _generate_builtin_call(
        self,
        name: str,
        builtin,
        target_type: VyperType,
        context: Context,
        depth: int,
    ) -> Optional[Union[ast.Call, ast.StaticCall, ast.ExtCall]]:
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
            # Generate slice with dynamic expressions for start/length.
            # Bias towards valid slices (~70%), but still produce some invalid
            # ones to ensure compiler/runtime checks are exercised.
            if not isinstance(target_type, (BytesT, StringT)):
                return None

            # Helper to build builtin calls directly (min/max/len)
            def _builtin_call(bname: str, args, ret_t):
                b = None
                if self.function_registry and bname in self.function_registry.builtins:
                    b = self.function_registry.builtins[bname]
                fn = ast.Name(id=bname)
                fn._metadata = getattr(fn, "_metadata", {})
                if b is not None:
                    fn._metadata["type"] = b
                call = ast.Call(func=fn, args=args, keywords=[])
                call._metadata = getattr(call, "_metadata", {})
                call._metadata["type"] = ret_t
                return call

            def _uint_binop(a, op, b):
                node = ast.BinOp(left=a, op=op, right=b)
                node._metadata = getattr(node, "_metadata", {})
                node._metadata["type"] = IntegerT(False, 256)
                return node

            def _uint_cmp(a, op, b):
                node = ast.Compare(left=a, ops=[op], comparators=[b])
                node._metadata = getattr(node, "_metadata", {})
                node._metadata["type"] = BoolT()
                return node

            def _ifexp(test, body, orelse, ret_t):
                node = ast.IfExp(test=test, body=body, orelse=orelse)
                node._metadata = getattr(node, "_metadata", {})
                node._metadata["type"] = ret_t
                return node

            ret_len = target_type.length

            # Choose source type:
            # - For String target, must slice a String
            # - For Bytes target, pick Bytes[..] or sometimes bytes32
            if isinstance(target_type, StringT):
                arg_len = ret_len + self.rng.randint(0, 32)
                src_t = StringT(arg_len)
            else:
                # 25% chance to slice bytes32
                if self.rng.random() < 0.25:
                    src_t = BytesM_T(32)
                    arg_len = 32
                else:
                    arg_len = ret_len + self.rng.randint(0, 32)
                    src_t = BytesT(arg_len)

            # Generate the source expression
            arg0 = self.generate(src_t, context, max(0, depth))

            # Build len(arg0) where applicable (Bytes/String only)
            len_ret_t = IntegerT(False, 256)
            if isinstance(src_t, (BytesT, StringT)):
                len_arg = arg0
                len_call = _builtin_call("len", [len_arg], len_ret_t)
            else:
                # bytes32 has fixed length 32
                len_call = self._generate_uint256_literal(32)

            # Random uint expressions to feed into min/max
            rand_u = self.generate(IntegerT(False, 256), context, max(0, depth))
            rand_v = self.generate(IntegerT(False, 256), context, max(0, depth))

            # Valid vs invalid selection
            make_valid = self.rng.random() < 0.7

            if make_valid:
                # Start: if len == 0 -> 0 else rand % len
                one = self._generate_uint256_literal(1)
                zero = self._generate_uint256_literal(0)

                len_is_zero = _uint_cmp(len_call, ast.Eq(), zero)
                start_else = _uint_binop(rand_u, ast.Mod(), len_call)
                a1 = _ifexp(len_is_zero, zero, start_else, IntegerT(False, 256))

                # remaining = len - start
                remaining = _uint_binop(len_call, ast.Sub(), a1)

                # length: if remaining == 0 -> 1 (will be invalid but rare)
                # else min((rand_v % bound)+1, remaining) where bound = ret_len (if >0) else remaining
                rem_is_zero = _uint_cmp(remaining, ast.Eq(), zero)
                if ret_len > 0:
                    bound = self._generate_uint256_literal(ret_len)
                else:
                    bound = remaining
                # Avoid modulo by 0 by ensuring bound >= 1 when it's a literal
                rand_mod = _uint_binop(rand_v, ast.Mod(), bound)
                plus_one = _uint_binop(rand_mod, ast.Add(), one)
                len_else = _builtin_call("min", [plus_one, remaining], len_ret_t)
                a2 = _ifexp(rem_is_zero, one, len_else, IntegerT(False, 256))
            else:
                # Intentionally produce out-of-bounds in a dynamic way
                one = self._generate_uint256_literal(1)
                two = self._generate_uint256_literal(2)
                ten = self._generate_uint256_literal(10)

                choice = self.rng.random()
                if choice < 0.34:
                    # start = len(arg0) (or more); length = 1
                    a1 = len_call
                    a2 = one
                elif choice < 0.67:
                    # start = len(arg0) + (rand % 10); length = (rand_v % (ret_len+1)) + 1
                    a1 = _uint_binop(
                        len_call, ast.Add(), _uint_binop(rand_u, ast.Mod(), ten)
                    )
                    a2 = _uint_binop(
                        _uint_binop(
                            rand_v,
                            ast.Mod(),
                            self._generate_uint256_literal(max(1, ret_len + 1)),
                        ),
                        ast.Add(),
                        one,
                    )
                else:
                    # start = (len(arg0) * 2); length = 1
                    a1 = _uint_binop(len_call, ast.Mult(), two)
                    a2 = one

            return self._finalize_call(func_node, [arg0, a1, a2], target_type)

        # Unknown builtin (not yet supported)
        return None

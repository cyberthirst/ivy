from __future__ import annotations

import random
from dataclasses import dataclass
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
    TYPE_T,
)
from vyper.semantics.types.subscriptable import _SequenceT
from vyper.semantics.analysis.base import DataLocation, VarInfo
from vyper.semantics.types.function import StateMutability, ContractFunctionT

from fuzzer.mutator.literal_generator import LiteralGenerator
from fuzzer.mutator.context import GenerationContext, ExprMutability
from fuzzer.mutator.function_registry import FunctionRegistry
from fuzzer.mutator.interface_registry import InterfaceRegistry
from fuzzer.mutator import ast_builder
from fuzzer.mutator.config import ExprGeneratorConfig
from fuzzer.mutator.strategy import (
    StrategyRegistry,
    StrategySelector,
    StrategyExecutor,
    register_decorated,
    strategy,
)
from fuzzer.mutator.type_utils import (
    is_subscriptable,
    can_reach_type,
    find_subscript_bases,
    find_nested_subscript_bases,
)
from fuzzer.mutator.indexing import (
    small_literal_index,
    build_len_call,
    build_guarded_index,
    pick_oob_value,
    INDEX_TYPE,
)
from fuzzer.xfail import XFailExpectation
from fuzzer.type_generator import TypeGenerator


@dataclass
class ExprGenCtx:
    target_type: VyperType
    context: GenerationContext
    depth: int
    rng: random.Random
    function_registry: FunctionRegistry
    mutability: ExprMutability
    gen: ExprGenerator


class ExprGenerator:
    def __init__(
        self,
        literal_generator: LiteralGenerator,
        rng: random.Random,
        interface_registry: InterfaceRegistry,
        function_registry: FunctionRegistry,
        type_generator: TypeGenerator,
        cfg: Optional[ExprGeneratorConfig] = None,
    ):
        self.literal_generator = literal_generator
        self.rng = rng
        self.function_registry = function_registry
        self.interface_registry = interface_registry
        self.type_generator = type_generator
        self.cfg = cfg or ExprGeneratorConfig()

        self._strategy_registry = StrategyRegistry()
        self._strategy_selector = StrategySelector(self.rng)
        self._strategy_executor = StrategyExecutor(self._strategy_selector)

        self._register_strategies()
        self._builtin_handlers = {
            "min": self._builtin_min_max,
            "max": self._builtin_min_max,
            "abs": self._builtin_abs,
            "floor": self._builtin_floor_ceil,
            "ceil": self._builtin_floor_ceil,
            "len": self._builtin_len,
            "concat": self._builtin_concat,
            "slice": self._builtin_slice,
        }

    def generate(
        self, target_type: VyperType, context: GenerationContext, depth: int = 3
    ) -> ast.VyperNode:
        # TODO we probably need a special case also for tuples
        if isinstance(target_type, StructT):
            return self._generate_struct(target_type, context, depth)

        # Early termination: probabilistic + hard depth limit
        if depth <= 0 or self.rng.random() > self.cfg.continuation_prob:
            return self._generate_terminal(target_type, context)

        ctx = ExprGenCtx(
            target_type=target_type,
            context=context,
            depth=depth,
            rng=self.rng,
            function_registry=self.function_registry,
            mutability=context.current_mutability,
            gen=self,
        )

        # Collect strategies via registry
        strategies = self._strategy_registry.collect(
            type_class=type(target_type),
            include_tags=("expr",),
            context={"ctx": ctx},
        )

        # Recursive strategies consume depth
        ctx.depth -= 1

        return self._strategy_executor.execute_with_retry(
            strategies,
            policy="weighted_random",  # TODO nested hash maps
            context={"ctx": ctx},
            fallback=lambda: self._generate_terminal(target_type, context),
        )

    def _register_strategies(self) -> None:
        register_decorated(self._strategy_registry, self)

    # Applicability/weight helpers

    def _is_var_ref_applicable(self, *, ctx: ExprGenCtx, **_) -> bool:
        return bool(ctx.context.find_matching_vars(ctx.target_type))

    def _weight_var_ref(self, *, ctx: ExprGenCtx, **_) -> float:
        # Slightly bias towards var refs when there are many
        n = len(ctx.context.find_matching_vars(ctx.target_type))
        cfg = self.cfg
        if n == 0:
            return 1.0
        return min(cfg.var_ref_weight_max, cfg.var_ref_weight_base + cfg.var_ref_weight_scale * n)

    def _is_unary_minus_applicable(self, *, ctx: ExprGenCtx, **_) -> bool:
        return isinstance(ctx.target_type, IntegerT) and ctx.target_type.is_signed

    def _is_func_call_applicable(self, *, ctx: ExprGenCtx, **_) -> bool:
        # TODO do we allow constant folding of some builtins?
        # if yes, we'd want to drop this restriction
        if ctx.mutability == ExprMutability.CONST:
            return False
        return not ctx.context.is_module_scope

    def _is_subscript_applicable(self, *, ctx: ExprGenCtx, **_) -> bool:
        vars_dict = dict(ctx.context.find_matching_vars(None))
        return bool(find_nested_subscript_bases(ctx.target_type, vars_dict, max_steps=3))

    def _is_ifexp_applicable(self, *, ctx: ExprGenCtx, **_) -> bool:
        # Disallow if-expressions in constant contexts due to compiler limitation
        return ctx.mutability != ExprMutability.CONST

    def _weight_literal(self, **_) -> float:
        return self.cfg.literal_weight

    def _weight_ifexp(self, **_) -> float:
        return self.cfg.ifexp_weight

    def _weight_subscript(self, *, ctx: ExprGenCtx, **_) -> float:
        # Slight bias based on number of available bases
        vars_dict = dict(ctx.context.find_matching_vars(None))
        n = len(find_subscript_bases(ctx.target_type, vars_dict))
        cfg = self.cfg
        if n == 0:
            return 1.0
        return min(cfg.subscript_weight_max, cfg.subscript_weight_base + cfg.subscript_weight_scale * n)

    # Strategy methods

    @strategy(
        name="expr.var_ref",
        tags=frozenset({"expr", "terminal"}),
        is_applicable="_is_var_ref_applicable",
        weight="_weight_var_ref",
    )
    def _generate_var_ref(self, *, ctx: ExprGenCtx, **_):
        matches = ctx.context.find_matching_vars(ctx.target_type)
        if not matches:
            return None
        return self._generate_variable_ref(ctx.rng.choice(matches), ctx.context)

    def _generate_terminal(
        self, target_type: VyperType, context: GenerationContext
    ) -> ast.VyperNode:
        if isinstance(target_type, StructT):
            return self._generate_struct(target_type, context, depth=0)

        matching_vars = context.find_matching_vars(target_type)

        if matching_vars and self.rng.random() < self.cfg.terminal_var_ref_prob:
            return self._generate_variable_ref(self.rng.choice(matching_vars), context)
        else:
            ctx = ExprGenCtx(
                target_type=target_type,
                context=context,
                depth=0,
                rng=self.rng,
                function_registry=self.function_registry,
                mutability=context.current_mutability,
                gen=self,
            )
            return self._generate_literal(ctx=ctx)

    @strategy(
        name="expr.literal",
        tags=frozenset({"expr", "terminal"}),
        weight="_weight_literal",
    )
    def _generate_literal(self, *, ctx: ExprGenCtx, **_) -> ast.VyperNode:
        value = self.literal_generator.generate(ctx.target_type)
        return ast_builder.literal(value, ctx.target_type)

    def random_var_ref(
        self, target_type: VyperType, context: GenerationContext
    ) -> Optional[Union[ast.Attribute, ast.Name]]:
        """Pick a random variable matching target_type, returns proper AST ref."""
        matches = context.find_matching_vars(target_type)
        if not matches:
            return None
        return self._generate_variable_ref(self.rng.choice(matches), context)

    def _generate_variable_ref(
        self, target: Union[str, tuple[str, VarInfo]], context: GenerationContext
    ) -> Union[ast.Attribute, ast.Name]:
        if isinstance(target, str):
            name = target
            var_info = context.all_vars[name]
        else:
            name, var_info = target

        node = ast_builder.var_ref(name, var_info)
        node._metadata["type"] = var_info.typ
        node._metadata["varinfo"] = var_info
        return node

    @strategy(
        name="expr.subscript",
        tags=frozenset({"expr", "recursive"}),
        is_applicable="_is_subscript_applicable",
        weight="_weight_subscript",
    )
    def _generate_subscript(self, *, ctx: ExprGenCtx, **_) -> Optional[ast.Subscript]:
        vars_dict = dict(ctx.context.find_matching_vars(None))
        bases = find_nested_subscript_bases(ctx.target_type, vars_dict, max_steps=3)
        if not bases:
            return None

        name, base_t = ctx.rng.choice(bases)
        base_node: ast.VyperNode = self._generate_variable_ref(name, ctx.context)
        built = self.build_chain_to_target(
            base_node,
            base_t,
            ctx.target_type,
            ctx.context,
            ctx.depth,
            max_steps=3,
        )
        if not built:
            return None
        node, _ = built
        assert isinstance(node, ast.Subscript)
        node._metadata["type"] = ctx.target_type
        return node

    @strategy(
        name="expr.ifexp",
        tags=frozenset({"expr", "recursive"}),
        is_applicable="_is_ifexp_applicable",
        weight="_weight_ifexp",
    )
    def _generate_ifexp(self, *, ctx: ExprGenCtx, **_) -> ast.IfExp:
        # Condition must be bool; branches must yield the same type
        test = self.generate(BoolT(), ctx.context, ctx.depth)
        body = self.generate(ctx.target_type, ctx.context, ctx.depth)
        orelse = self.generate(ctx.target_type, ctx.context, ctx.depth)

        node = ast.IfExp(test=test, body=body, orelse=orelse)
        node._metadata["type"] = ctx.target_type
        return node

    # ----------------------
    # Shared subscript utils
    # ----------------------

    def _random_integer_type(self) -> IntegerT:
        bits = self.rng.choice([8, 16, 32, 64, 128, 256])
        signed = self.rng.choice([True, False])
        return IntegerT(signed, bits)

    def _generate_index_for_sequence(
        self,
        base_node: ast.VyperNode,
        seq_t: _SequenceT,
        context: GenerationContext,
        depth: int,
    ) -> ast.VyperNode:
        """Generate an index expression for arrays with three modes:
        - Guarded by len(base) (preferred)
        - Random expression (unconstrained)
        - OOB literal that forces compilation xfail (rare)
        """
        assert isinstance(seq_t, (SArrayT, DArrayT))
        cfg = self.cfg
        roll = self.rng.random()

        if roll < cfg.index_guard_prob:
            return self._generate_guarded_index(base_node, seq_t, context, depth)
        elif roll < cfg.index_guard_prob + cfg.index_random_prob:
            return self._generate_random_index(seq_t, context, depth)
        else:
            return self._generate_oob_index(seq_t, context)

    def _generate_guarded_index(
        self,
        base_node: ast.VyperNode,
        seq_t: _SequenceT,
        context: GenerationContext,
        depth: int,
    ) -> ast.IfExp:
        """Generate a bounds-guarded index expression."""
        cfg = self.cfg

        # Choose the raw index expression (biased towards small literals for DynArray)
        use_small = (
            isinstance(seq_t, DArrayT)
            and self.rng.random() < cfg.dynarray_small_literal_in_guard_prob
        )
        if use_small:
            i_expr = small_literal_index(self.rng)
        else:
            i_expr = self.generate(INDEX_TYPE, context, max(0, depth))

        len_call = build_len_call(base_node)
        return build_guarded_index(i_expr, len_call)

    def _generate_random_index(
        self,
        seq_t: _SequenceT,
        context: GenerationContext,
        depth: int,
    ) -> ast.VyperNode:
        """Generate a random (unconstrained) index expression."""
        cfg = self.cfg

        # Bias DynArray towards small literals
        use_small = (
            isinstance(seq_t, DArrayT)
            and self.rng.random() < cfg.dynarray_small_literal_in_random_prob
        )
        if use_small:
            return small_literal_index(self.rng)
        return self.generate(INDEX_TYPE, context, max(0, depth))

    def _generate_oob_index(
        self,
        seq_t: _SequenceT,
        context: GenerationContext,
    ) -> ast.Int:
        """Generate an out-of-bounds literal index (triggers compilation xfail)."""
        cfg = self.cfg
        val = pick_oob_value(seq_t.length, self.rng, cfg.oob_cap_vs_cap_plus_one_prob)

        context.compilation_xfails.append(
            XFailExpectation(
                kind="compilation",
                reason="generated out-of-bounds array index",
            )
        )
        return ast_builder.literal(val, INDEX_TYPE)

    def build_chain_to_target(
        self,
        base_node: ast.VyperNode,
        base_type: VyperType,
        target_type: VyperType,
        context: GenerationContext,
        depth: int,
        max_steps: int = 3,
    ) -> Optional[tuple[ast.VyperNode, VyperType]]:
        cur_t = base_type
        node = base_node
        steps_remaining = max(1, min(max_steps, depth + 1))

        while steps_remaining > 0 and is_subscriptable(cur_t):
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
                        is_subscriptable(mt)
                        and can_reach_type(mt, target_type, steps_remaining - 1)
                    ):
                        choices.append((i, mt))
                if not choices:
                    break
                idx, child_t = self.rng.choice(choices)
                idx_expr = ast_builder.literal(idx, IntegerT(False, 256))
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
        context: GenerationContext,
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
                idx_expr = ast_builder.literal(idx, IntegerT(False, 256))
                cur_t = mtypes[idx]
            else:
                break

            node = ast.Subscript(value=node, slice=idx_expr)
            node._metadata = {"type": cur_t}

            if not is_subscriptable(cur_t):
                break

        return node, cur_t

    @strategy(
        name="expr.arithmetic",
        tags=frozenset({"expr", "recursive"}),
        type_classes=(IntegerT,),
    )
    def _generate_arithmetic(self, *, ctx: ExprGenCtx, **_) -> ast.BinOp:
        op_classes = [ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod]
        op_class = ctx.rng.choice(op_classes)

        left = self.generate(ctx.target_type, ctx.context, ctx.depth)
        right = self.generate(ctx.target_type, ctx.context, ctx.depth)

        node = ast.BinOp(left=left, op=op_class(), right=right)

        if isinstance(node.op, (ast.FloorDiv, ast.Mod, ast.Div)):
            if isinstance(right, ast.Int) and getattr(right, "value", None) == 0:
                ctx.context.compilation_xfails.append(
                    XFailExpectation(
                        kind="compilation",
                        reason="division or modulo by zero should fail compilation",
                    )
                )

        node._metadata["type"] = ctx.target_type
        return node

    @strategy(
        name="expr.comparison",
        tags=frozenset({"expr", "recursive"}),
        type_classes=(BoolT,),
    )
    def _generate_comparison(self, *, ctx: ExprGenCtx, **_) -> ast.Compare:
        op_classes = [ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq]
        op_class = ctx.rng.choice(op_classes)

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

        comparable_type = ctx.rng.choice(comparable_types)

        left = self.generate(comparable_type, ctx.context, ctx.depth)
        right = self.generate(comparable_type, ctx.context, ctx.depth)

        node = ast.Compare(left=left, ops=[op_class()], comparators=[right])
        node._metadata["type"] = BoolT()
        return node

    @strategy(
        name="expr.boolean_op",
        tags=frozenset({"expr", "recursive"}),
        type_classes=(BoolT,),
    )
    def _generate_boolean_op(self, *, ctx: ExprGenCtx, **_) -> ast.BoolOp:
        op_classes = [ast.And, ast.Or]
        op_class = ctx.rng.choice(op_classes)

        values = [self.generate(BoolT(), ctx.context, ctx.depth) for _ in range(2)]

        node = ast.BoolOp(op=op_class(), values=values)
        node._metadata["type"] = BoolT()
        return node

    @strategy(
        name="expr.not",
        tags=frozenset({"expr", "recursive"}),
        type_classes=(BoolT,),
    )
    def _generate_not(self, *, ctx: ExprGenCtx, **_) -> ast.UnaryOp:
        operand = self.generate(BoolT(), ctx.context, ctx.depth)
        node = ast.UnaryOp(op=ast.Not(), operand=operand)
        node._metadata["type"] = BoolT()
        return node

    @strategy(
        name="expr.unary_minus",
        tags=frozenset({"expr", "recursive"}),
        type_classes=(IntegerT,),
        is_applicable="_is_unary_minus_applicable",
    )
    def _generate_unary_minus(self, *, ctx: ExprGenCtx, **_) -> ast.UnaryOp:
        operand = self.generate(ctx.target_type, ctx.context, ctx.depth)
        node = ast.UnaryOp(op=ast.USub(), operand=operand)
        node._metadata["type"] = ctx.target_type
        return node

    def _generate_struct(self, target_type, context: GenerationContext, depth: int) -> ast.Call:
        assert isinstance(target_type, StructT)

        # Create the struct constructor call
        call_node = ast.Call(func=ast.Name(id=target_type._id), args=[], keywords=[])

        for field_name, field_type in target_type.members.items():
            field_expr = self.generate(field_type, context, max(0, depth - 1))

            keyword = ast.keyword(arg=field_name, value=field_expr)
            call_node.keywords.append(keyword)

        call_node._metadata["type"] = target_type
        return call_node

    @strategy(
        name="expr.func_call",
        tags=frozenset({"expr", "recursive"}),
        is_applicable="_is_func_call_applicable",
    )
    def _generate_func_call(
        self, *, ctx: ExprGenCtx, **_
    ) -> Optional[Union[ast.Call, ast.StaticCall, ast.ExtCall]]:
        if not ctx.function_registry:
            return None

        current_func = ctx.function_registry.current_function
        assert current_func is not None

        caller_mutability = ctx.context.current_function_mutability

        # Gather candidates
        compatible_func = ctx.function_registry.get_compatible_function(
            ctx.target_type,
            current_func,
            caller_mutability=caller_mutability,
        )
        compatible_builtins = ctx.function_registry.get_compatible_builtins(
            ctx.target_type,
            caller_mutability=caller_mutability,
        )

        # Decide path: prefer user function, but sometimes pick builtin
        use_builtin = False
        if compatible_builtins:
            if not compatible_func:
                use_builtin = True
            else:
                use_builtin = ctx.rng.random() < self.cfg.use_builtin_when_both_available_prob

        if use_builtin:
            name, builtin = ctx.rng.choice(compatible_builtins)
            return self._generate_builtin_call(
                name, builtin, ctx.target_type, ctx.context, ctx.depth
            )

        # Fall back to user function (existing or create new)
        if not compatible_func or ctx.rng.random() < self.cfg.create_new_function_prob:
            # Create a new function (returns ast.FunctionDef or None)
            func_def = ctx.function_registry.create_new_function(
                return_type=ctx.target_type,
                type_generator=self.type_generator,
                max_args=2,
                caller_mutability=caller_mutability,
            )
            if func_def is None:
                # Can't create more functions; try builtin if available
                if compatible_builtins:
                    name, builtin = ctx.rng.choice(compatible_builtins)
                    return self._generate_builtin_call(
                        name, builtin, ctx.target_type, ctx.context, ctx.depth
                    )
                return None
            func_t = func_def._metadata["func_type"]
        else:
            func_t = compatible_func

        func_name = func_t.name

        # Generate arguments
        args = []
        for pos_arg in func_t.positional_args:
            if pos_arg.typ:
                arg_expr = self.generate(pos_arg.typ, ctx.context, max(0, ctx.depth))
                args.append(arg_expr)

        if func_t.is_external:
            # External functions must be called through an interface
            if not self.interface_registry:
                return None
            # External calls use `self` as address, which is not allowed in @pure
            # TODO: support literal addresses for pure function calls
            if ctx.context.current_function_mutability == StateMutability.PURE:
                return None
            result_node = self._generate_external_call(func_t, args)
        else:
            assert func_t.is_internal, (
                f"Expected internal or external function, got {func_t}"
            )
            func_node = ast.Attribute(value=ast.Name(id="self"), attr=func_name)
            func_node._metadata = {"type": func_t}
            result_node = self._finalize_call(func_node, args, func_t.return_type)

        # Record the call in the call graph
        if ctx.function_registry.current_function:
            ctx.function_registry.add_call(
                ctx.function_registry.current_function, func_name
            )

        return result_node

    def _generate_external_call(
        self, func: ContractFunctionT, args: list
    ) -> Union[ast.StaticCall, ast.ExtCall]:
        """Build AST for an external call through an interface."""
        iface_name, iface_type = self.interface_registry.create_interface(func)

        # Address is always `self` for now
        # TODO support random addresses
        address_node = ast.Name(id="self")
        address_node._metadata = {"type": AddressT()}

        # Build: InterfaceName(address)
        iface_name_node = ast.Name(id=iface_name)
        iface_name_node._metadata = {"type": TYPE_T(iface_type)}

        iface_cast = ast.Call(func=iface_name_node, args=[address_node], keywords=[])
        iface_cast._metadata = {"type": iface_type}

        # Build: Interface(address).func
        attr_node = ast.Attribute(value=iface_cast, attr=func.name)
        attr_node._metadata = {"type": func}

        ret = self._finalize_call(attr_node, args, func.return_type, func)
        assert isinstance(ret, (ast.StaticCall, ast.ExtCall))
        return ret

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

    def _generate_builtin_call(
        self,
        name: str,
        builtin,
        target_type: VyperType,
        context: GenerationContext,
        depth: int,
    ) -> Optional[Union[ast.Call, ast.StaticCall, ast.ExtCall]]:
        # func node
        func_node = ast.Name(id=name)
        func_node._metadata = getattr(func_node, "_metadata", {})
        func_node._metadata["type"] = builtin

        handler = self._builtin_handlers.get(name)
        if handler is None:
            return None

        return handler(
            func_node=func_node,
            name=name,
            builtin=builtin,
            target_type=target_type,
            context=context,
            depth=depth,
        )

    def _builtin_min_max(
        self,
        *,
        func_node: ast.Name,
        target_type: VyperType,
        context: GenerationContext,
        depth: int,
        **_,
    ) -> Optional[Union[ast.Call, ast.StaticCall, ast.ExtCall]]:
        if not isinstance(target_type, (IntegerT, DecimalT)):
            return None
        arg_t = target_type
        a0 = self.generate(arg_t, context, max(0, depth))
        a1 = self.generate(arg_t, context, max(0, depth))
        call_node = ast.Call(func=func_node, args=[a0, a1], keywords=[])
        call_node._metadata = {"type": arg_t}
        return call_node

    def _builtin_abs(
        self,
        *,
        func_node: ast.Name,
        target_type: VyperType,
        context: GenerationContext,
        depth: int,
        **_,
    ) -> Optional[Union[ast.Call, ast.StaticCall, ast.ExtCall]]:
        if isinstance(target_type, IntegerT) and not target_type.is_signed:
            return None
        if not isinstance(target_type, (IntegerT, DecimalT)):
            return None
        arg_t = target_type
        a0 = self.generate(arg_t, context, max(0, depth))
        return self._finalize_call(func_node, [a0], arg_t)

    def _builtin_floor_ceil(
        self,
        *,
        func_node: ast.Name,
        builtin,
        context: GenerationContext,
        depth: int,
        **_,
    ) -> Optional[Union[ast.Call, ast.StaticCall, ast.ExtCall]]:
        # Expect decimal arg, return concrete integer type from builtin
        a0 = self.generate(DecimalT(), context, max(0, depth))
        ret_t = getattr(builtin, "_return_type", None) or IntegerT(True, 256)
        return self._finalize_call(func_node, [a0], ret_t)

    def _builtin_len(
        self,
        *,
        func_node: ast.Name,
        builtin,
        context: GenerationContext,
        depth: int,
        **_,
    ) -> Optional[Union[ast.Call, ast.StaticCall, ast.ExtCall]]:
        # Only support BytesT/StringT to start
        max_len = self.rng.randint(1, 128)

        arg_t = self.rng.choice([BytesT(max_len), StringT(max_len)])
        a0 = self.generate(arg_t, context, max(0, depth))
        ret_t = getattr(builtin, "_return_type", IntegerT(False, 256))
        return self._finalize_call(func_node, [a0], ret_t)

    def _builtin_concat(
        self,
        *,
        func_node: ast.Name,
        target_type: VyperType,
        context: GenerationContext,
        depth: int,
        **_,
    ) -> Optional[Union[ast.Call, ast.StaticCall, ast.ExtCall]]:
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

    def _builtin_slice(
        self,
        *,
        func_node: ast.Name,
        target_type: VyperType,
        context: GenerationContext,
        depth: int,
        **_,
    ) -> Optional[Union[ast.Call, ast.StaticCall, ast.ExtCall]]:
        # Generate slice with dynamic expressions for start/length.
        # Bias towards valid slices, but still produce some invalid
        # ones to ensure compiler/runtime checks are exercised.
        if not isinstance(target_type, (BytesT, StringT)):
            return None

        cfg = self.cfg
        builtins = self.function_registry.builtins if self.function_registry else {}
        ret_len = target_type.length

        # Choose source type:
        # - For String target, must slice a String
        # - For Bytes target, pick Bytes[..] or sometimes bytes32
        if isinstance(target_type, StringT):
            arg_len = ret_len + self.rng.randint(0, 32)
            src_t = StringT(arg_len)
        else:
            if self.rng.random() < cfg.slice_use_bytes32_source_prob:
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
            len_call = ast_builder.builtin_call("len", [len_arg], len_ret_t, builtins)
        else:
            # bytes32 has fixed length 32
            len_call = ast_builder.uint256_literal(32)

        # Random uint expressions to feed into min/max
        rand_u = self.generate(IntegerT(False, 256), context, max(0, depth))
        rand_v = self.generate(IntegerT(False, 256), context, max(0, depth))

        # Valid vs invalid selection
        make_valid = self.rng.random() < cfg.slice_valid_prob

        if make_valid:
            # Start: if len == 0 -> 0 else rand % len
            one = ast_builder.uint256_literal(1)
            zero = ast_builder.uint256_literal(0)

            len_is_zero = ast_builder.compare(len_call, ast.Eq(), zero)
            start_else = ast_builder.uint256_binop(rand_u, ast.Mod(), len_call)
            a1 = ast_builder.ifexp(len_is_zero, zero, start_else, IntegerT(False, 256))

            # remaining = len - start
            remaining = ast_builder.uint256_binop(len_call, ast.Sub(), a1)

            # length: if remaining == 0 -> 1 (will be invalid but rare)
            # else min((rand_v % bound)+1, remaining) where bound = ret_len (if >0) else remaining
            rem_is_zero = ast_builder.compare(remaining, ast.Eq(), zero)
            if ret_len > 0:
                bound = ast_builder.uint256_literal(ret_len)
            else:
                bound = remaining
            # Avoid modulo by 0 by ensuring bound >= 1 when it's a literal
            rand_mod = ast_builder.uint256_binop(rand_v, ast.Mod(), bound)
            plus_one = ast_builder.uint256_binop(rand_mod, ast.Add(), one)
            len_else = ast_builder.builtin_call(
                "min", [plus_one, remaining], len_ret_t, builtins
            )
            a2 = ast_builder.ifexp(rem_is_zero, one, len_else, IntegerT(False, 256))
        else:
            # Intentionally produce out-of-bounds in a dynamic way
            one = ast_builder.uint256_literal(1)
            two = ast_builder.uint256_literal(2)
            ten = ast_builder.uint256_literal(10)

            choice = self.rng.random()
            if choice < cfg.slice_invalid_start_at_len_prob:
                # start = len(arg0) (or more); length = 1
                a1 = len_call
                a2 = one
            elif choice < cfg.slice_invalid_start_plus_rand_prob:
                # start = len(arg0) + (rand % 10); length = (rand_v % (ret_len+1)) + 1
                a1 = ast_builder.uint256_binop(
                    len_call, ast.Add(), ast_builder.uint256_binop(rand_u, ast.Mod(), ten)
                )
                a2 = ast_builder.uint256_binop(
                    ast_builder.uint256_binop(
                        rand_v,
                        ast.Mod(),
                        ast_builder.uint256_literal(max(1, ret_len + 1)),
                    ),
                    ast.Add(),
                    one,
                )
            else:
                # start = (len(arg0) * 2); length = 1
                a1 = ast_builder.uint256_binop(len_call, ast.Mult(), two)
                a2 = one

        return self._finalize_call(func_node, [arg0, a1, a2], target_type)

import random

from vyper.ast import nodes as ast
from vyper.semantics.types import VyperType, IntegerT, BoolT, AddressT, StructT

from .value_mutator import ValueMutator
from .context import Context


class LiteralValue(ast.VyperNode):
    __slots__ = ("value", "typ")


class ExprGenerator:
    def __init__(self, value_mutator: ValueMutator, rng: random.Random):
        self.value_mutator = value_mutator
        self.rng = rng

    def generate(
        self, target_type: VyperType, context: Context, depth: int = 3
    ) -> ast.VyperNode:
        # TODO we probably need a special case also for tuples
        if isinstance(target_type, StructT):
            return self._generate_struct(target_type, context, depth)

        if depth == 0:
            return self._generate_terminal(target_type, context)

        strategies = []

        strategies.append(lambda: self._generate_literal(target_type, context))

        matching_vars = self._find_matching_variables(target_type, context)
        if matching_vars:
            strategies.append(
                lambda: self._generate_variable_ref(
                    self.rng.choice(matching_vars), context
                )
            )

        if isinstance(target_type, IntegerT):
            strategies.append(
                lambda: self._generate_arithmetic(target_type, context, depth - 1)
            )
            if target_type.is_signed:
                strategies.append(
                    lambda: self._generate_unary_minus(target_type, context, depth - 1)
                )

        if isinstance(target_type, BoolT):
            strategies.append(lambda: self._generate_comparison(context, depth - 1))
            strategies.append(lambda: self._generate_boolean_op(context, depth - 1))
            strategies.append(lambda: self._generate_not(context, depth - 1))

        strategy = self.rng.choice(strategies)
        return strategy()

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
    ) -> LiteralValue:
        value = self.value_mutator.generate_value_for_type(target_type)
        node = LiteralValue()
        node.value = value
        node.typ = target_type
        node._metadata["type"] = target_type
        return node

    def _find_matching_variables(
        self, target_type: VyperType, context: Context
    ) -> list[str]:
        matches = []
        for name, var_info in context.all_vars.items():
            if target_type.compare_type(var_info.typ) or var_info.typ.compare_type(
                target_type
            ):
                matches.append(name)
        return matches

    def _generate_variable_ref(self, name: str, context: Context) -> ast.Name:
        node = ast.Name(id=name)
        node._metadata["type"] = context.all_vars[name].typ
        return node

    def _generate_arithmetic(
        self, target_type: IntegerT, context: Context, depth: int
    ) -> ast.BinOp:
        op_classes = [ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod]
        op_class = self.rng.choice(op_classes)

        left = self.generate(target_type, context, depth)
        right = self.generate(target_type, context, depth)

        node = ast.BinOp(left=left, op=op_class(), right=right)
        node._metadata["type"] = target_type
        return node

    def _generate_comparison(self, context: Context, depth: int) -> ast.Compare:
        op_classes = [ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq]
        op_class = self.rng.choice(op_classes)

        comparable_types = [IntegerT(True, 256), IntegerT(True, 128), AddressT()]
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

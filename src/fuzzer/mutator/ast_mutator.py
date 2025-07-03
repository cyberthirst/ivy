import copy
import random
from typing import Optional, Type

from vyper.ast import nodes as ast
from vyper.semantics.types import VyperType, IntegerT
from vyper.compiler.phases import CompilerData

from .value_mutator import ValueMutator
from src.unparser.unparser import unparse


class FreshNameGenerator:
    """Generates unique variable names with a consistent prefix."""

    def __init__(self, prefix: str = "ivy_internal"):
        self.prefix = prefix
        self.counter = 0

    def generate(self) -> str:
        name = f"{self.prefix}{self.counter}"
        self.counter += 1
        return name


class Scope:
    def __init__(self):
        self.params: dict[str, VyperType] = {}
        self.locals: dict[str, VyperType] = {}


class AstMutator:
    PROB = {
        ast.Int: 0.4,
        ast.BinOp: 0.3,
        ast.If: 0.2,
        ast.Assign: 0.2,
        ast.Return: 0.1,
        ast.UnaryOp: 0.1,
        ast.BoolOp: 0.1,
        ast.Attribute: 0.2,
        ast.Subscript: 0.2,
        ast.For: 0.2,
        ast.Compare: 0.3,
    }

    def __init__(
        self, rng: random.Random, *, max_mutations: int = 8, mutate_prob: float = 0.3
    ):
        self.rng = rng
        self.max_mutations = max_mutations
        self.mutate_prob = mutate_prob
        self.mutations_done = 0
        self.current_scope = None
        self.scope_stack = []
        self.name_generator = FreshNameGenerator()
        self.value_mutator = ValueMutator(rng)

    def visit(self, node):
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        if hasattr(node, "body"):
            for child in node.body:
                self.visit(child)
        elif hasattr(node, "children"):
            for child in node.children:
                self.visit(child)

    def mutate(self, root: ast.Module) -> ast.Module:
        self.mutations_done = 0
        self.current_scope = None
        self.scope_stack = []

        # Deep copy the root to avoid modifying the original
        new_root = copy.deepcopy(root)

        self.visit(new_root)

        return new_root

    def push_scope(self):
        self.scope_stack.append(self.current_scope)
        self.current_scope = Scope()

    def pop_scope(self):
        self.current_scope = self.scope_stack.pop()

    def add_local(self, name: str, typ: VyperType):
        if self.current_scope:
            self.current_scope.locals[name] = typ

    def pick_var(self, want_type: Optional[VyperType] = None) -> Optional[ast.Name]:
        if not self.current_scope:
            return None

        vars_pool = []

        for name, typ in self.current_scope.params.items():
            if want_type is None or typ == want_type:
                vars_pool.append(name)

        for name, typ in self.current_scope.locals.items():
            if want_type is None or typ == want_type:
                vars_pool.append(name)

        if not vars_pool:
            return None

        selected_name = self.rng.choice(vars_pool)

        return ast.Name(id=selected_name)

    def should_mutate(self, node_type: Type) -> bool:
        node_prob = self.PROB.get(node_type, 0)
        return (
            self.mutations_done < self.max_mutations
            and self.rng.random() < self.mutate_prob * node_prob
        )

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.push_scope()

        for arg in node.args.args:
            if hasattr(arg, "annotation") and hasattr(arg.annotation, "_metadata"):
                typ = arg.annotation._metadata.get("type")
                if typ:
                    self.current_scope.params[arg.arg] = typ

        for stmt in node.body:
            self.visit(stmt)

        self.pop_scope()

    def visit_Int(self, node: ast.Int):
        if not self.should_mutate(ast.Int):
            return

        # Get the type if available
        node_type = getattr(node, "_metadata", {}).get("type")

        if node_type and isinstance(node_type, IntegerT):
            # Use the value mutator with proper type
            node.value = self.value_mutator.mutate_value(node.value, node_type)
        else:
            # Fallback to simple mutations
            mutation_type = self.rng.choice(["add_one", "subtract_one", "bit_flip"])

            if mutation_type == "add_one":
                node.value += 1
            elif mutation_type == "subtract_one":
                node.value -= 1
            elif mutation_type == "bit_flip":
                bit_position = self.rng.randint(0, 63)  # Limit to 64 bits for safety
                node.value ^= 1 << bit_position

        self.mutations_done += 1

    def visit_BinOp(self, node: ast.BinOp):
        self.visit(node.left)
        self.visit(node.right)

        if not self.should_mutate(ast.BinOp):
            return

        left_type = getattr(node.left, "_metadata", {}).get("type")
        right_type = getattr(node.right, "_metadata", {}).get("type")

        if left_type is not None and right_type is not None and left_type != right_type:
            return

        mutation_type = self.rng.choice(["swap_operands", "change_operator"])

        if mutation_type == "swap_operands":
            node.left, node.right = node.right, node.left

        elif mutation_type == "change_operator":
            if isinstance(node.op, (ast.Add, ast.Sub)):
                # Swap + and -
                if isinstance(node.op, ast.Add):
                    node.op = ast.Sub()
                else:
                    node.op = ast.Add()
            elif isinstance(node.op, (ast.Mult, ast.FloorDiv)):
                # Swap * and //
                if isinstance(node.op, ast.Mult):
                    node.op = ast.FloorDiv()
                else:
                    node.op = ast.Mult()

        self.mutations_done += 1

    def visit_If(self, node: ast.If):
        self.visit(node.test)

        for stmt in node.body:
            self.visit(stmt)

        for stmt in node.orelse:
            self.visit(stmt)

        if not self.should_mutate(ast.If):
            return

        if isinstance(node.test, ast.Call):
            return

        mutation_type = self.rng.choice(["negate_condition", "swap_branches"])

        if mutation_type == "negate_condition":
            # Negate the condition
            # For a simple comparison, invert the operator
            if isinstance(node.test, ast.Compare):
                # Map operators to their negation
                op_map = {
                    ast.Lt: ast.GtE(),
                    ast.LtE: ast.Gt(),
                    ast.Gt: ast.LtE(),
                    ast.GtE: ast.Lt(),
                    ast.Eq: ast.NotEq(),
                    ast.NotEq: ast.Eq(),
                    ast.In: ast.NotIn(),
                    ast.NotIn: ast.In(),
                }

                if type(node.test.op) in op_map:
                    node.test.op = op_map[type(node.test.op)]
            else:
                # Wrap in a Not operation for other expressions
                node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)

        elif mutation_type == "swap_branches":
            # Swap body and orelse
            node.body, node.orelse = node.orelse, node.body

            # If we're swapping branches, also negate the condition
            if isinstance(node.test, ast.Compare):
                # Map operators to their negation
                op_map = {
                    ast.Lt: ast.GtE(),
                    ast.LtE: ast.Gt(),
                    ast.Gt: ast.LtE(),
                    ast.GtE: ast.Lt(),
                    ast.Eq: ast.NotEq(),
                    ast.NotEq: ast.Eq(),
                    ast.In: ast.NotIn(),
                    ast.NotIn: ast.In(),
                }

                if type(node.test.op) in op_map:
                    node.test.op = op_map[type(node.test.op)]
            else:
                # Wrap in a Not operation for other expressions
                node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)

        self.mutations_done += 1

    def visit_Return(self, node: ast.Return):
        self.visit(node.value)

        if not self.should_mutate(ast.Return):
            return

        if hasattr(node.value, "is_literal_value") and node.value.is_literal_value:
            # TODO
            # In a real implementation, we would use vyper.eval_fold here
            # For this example, we just skip this part as we don't have the actual fold helper
            pass

        # self.mutations_done += 1

    def visit_Assign(self, node: ast.Assign):
        self.visit(node.target)
        self.visit(node.value)

        if not self.should_mutate(ast.Assign):
            return

        rhs_type = getattr(node.value, "_metadata", {}).get("type")

        mutation_type = self.rng.choice(["swap_rhs", "insert_new_local"])

        if mutation_type == "swap_rhs" and rhs_type is not None:
            other_var = self.pick_var(rhs_type)
            if other_var:
                node.value = other_var

        elif mutation_type == "insert_new_local" and isinstance(node.target, ast.Name):
            new_name = self.name_generator.generate()

            # For simplicity, we're assuming uint256 type with value 0
            if rhs_type:
                self.add_local(new_name, rhs_type)
                node.target.id = new_name
                # For uint256, initialize to 0
                if "uint" in str(rhs_type):
                    node.value = ast.Int(value=0)

        self.mutations_done += 1

    def visit_UnaryOp(self, node: ast.UnaryOp):
        self.visit(node.operand)

        if not self.should_mutate(ast.UnaryOp):
            return

        # Only apply to Int operands that are not already negative
        if isinstance(node.operand, ast.Int) and (
            isinstance(node.op, (ast.USub, ast.Invert)) or node.operand.value >= 0
        ):
            # Choose to either drop the unary operator or add one
            if isinstance(node.op, (ast.USub, ast.Invert)):
                # TODO
                # Remove the unary operator by replacing with the operand
                # In a real implementation, we would need to properly replace the node in the parent
                # For this example, we're just showing the concept
                pass
            else:
                # Add a unary operator
                if node.operand.value >= 0:
                    # Add a negation or bitwise not
                    op_choice = self.rng.choice([ast.USub(), ast.Invert()])
                    node.op = op_choice

        self.mutations_done += 1

    def visit_BoolOp(self, node: ast.BoolOp):
        # Visit all values
        for value in node.values:
            self.visit(value)

        if not self.should_mutate(ast.BoolOp):
            return

        # Check if all operands are boolean typed
        all_bool = True
        for value in node.values:
            value_type = getattr(value, "_metadata", {}).get("type")
            if value_type is not None and "bool" not in str(value_type):
                all_bool = False
                break

        if all_bool and len(node.values) > 0:
            # Duplicate one input (a and b → a and a)
            duplicate_idx = self.rng.randint(0, len(node.values) - 1)
            node.values[1 - duplicate_idx] = node.values[duplicate_idx]

        self.mutations_done += 1

    def visit_Attribute(self, node: ast.Attribute):
        """Mutate attribute access (e.g., self.foo)"""
        self.visit(node.value)

        if not self.should_mutate(ast.Attribute):
            return

        # For now, we can swap attribute names if we have a mapping
        # This is a simplified implementation
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            # Could implement storage variable reordering here
            pass

        self.mutations_done += 1

    def visit_Subscript(self, node: ast.Subscript):
        """Mutate subscript access (e.g., arr[i])"""
        self.visit(node.value)
        self.visit(node.slice)

        if not self.should_mutate(ast.Subscript):
            return

        # Could mutate the index
        if isinstance(node.slice, ast.Int):
            node.slice.value = self.rng.choice(
                [0, 1, node.slice.value + 1, node.slice.value - 1]
            )

        self.mutations_done += 1

    def visit_For(self, node: ast.For):
        """Mutate for loops"""
        self.visit(node.target)
        self.visit(node.iter)

        for stmt in node.body:
            self.visit(stmt)

        if not self.should_mutate(ast.For):
            return

        # Could swap loop bounds or modify iteration
        # This is a simplified implementation
        # self.mutations_done += 1

    def visit_Compare(self, node: ast.Compare):
        """Mutate comparison operations"""
        self.visit(node.left)
        self.visit(node.right)

        if not self.should_mutate(ast.Compare):
            return

        # Swap comparison operators
        op_swaps = {
            ast.Lt: [ast.LtE, ast.Gt, ast.Eq],
            ast.LtE: [ast.Lt, ast.GtE, ast.Eq],
            ast.Gt: [ast.GtE, ast.Lt, ast.Eq],
            ast.GtE: [ast.Gt, ast.LtE, ast.Eq],
            ast.Eq: [ast.NotEq, ast.Lt, ast.Gt],
            ast.NotEq: [ast.Eq, ast.Lt, ast.Gt],
        }

        op_type = type(node.op)
        if op_type in op_swaps:
            new_op_type = self.rng.choice(op_swaps[op_type])
            node.op = new_op_type()

        self.mutations_done += 1

    def generate_pragma_lines(self, settings) -> list[str]:
        pragma_lines = []

        if settings.compiler_version:
            pragma_lines.append(f"# pragma version {settings.compiler_version}")

        if settings.evm_version:
            pragma_lines.append(f"# pragma evm-version {settings.evm_version}")

        if settings.enable_decimals or settings.experimental_codegen:
            pragma_lines.append("# pragma experimental-codegen")

        return pragma_lines

    def mutate_source_with_compiler_data(
        self, compiler_data: CompilerData
    ) -> Optional[str]:
        """Mutate source using annotated AST from CompilerData."""
        try:
            annotated_module = compiler_data.annotated_vyper_module

            mutated_ast = self.mutate(annotated_module)

            result = unparse(mutated_ast)

            pragma_lines = self.generate_pragma_lines(compiler_data.settings)

            if pragma_lines:
                result = "\n".join(pragma_lines) + "\n\n" + result

            return result
        except Exception as e:
            import logging

            logging.info(f"Failed to mutate source: {e}")
            logging.debug("Traceback:", exc_info=True)
            return None

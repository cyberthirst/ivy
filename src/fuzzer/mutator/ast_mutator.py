import copy
import random
from typing import Optional, Type

from vyper.ast import nodes as ast
from vyper.semantics.types import VyperType, IntegerT
from vyper.semantics.analysis.base import VarInfo, DataLocation
from vyper.compiler.phases import CompilerData

from .value_mutator import ValueMutator
from .context import Context
from .expr_generator import ExprGenerator
from .stmt_generator import StatementGenerator
from src.unparser.unparser import unparse
from src.fuzzer.type_generator import TypeGenerator


class VyperNodeTransformer:
    """Base class for transforming Vyper AST nodes.

    Similar to Python's ast.NodeTransformer but adapted for Vyper AST.
    Visit methods should return the node (possibly modified or replaced).
    """

    def visit(self, node):
        """Visit a node and return the transformed node."""
        if node is None:
            return None

        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        """Called if no explicit visitor method exists for a node."""
        # Iterate through all fields of this node
        for field_name in node.get_fields():
            if not hasattr(node, field_name):
                continue

            old_value = getattr(node, field_name)

            if isinstance(old_value, list):
                # Handle list fields (like body, args, etc.)
                new_values = []
                for item in old_value:
                    if isinstance(item, ast.VyperNode):
                        new_item = self.visit(item)
                        if new_item is not None:
                            new_values.append(new_item)
                    else:
                        # Not an AST node
                        # Examples in lists: strings in bases, decorator_list
                        new_values.append(item)
                setattr(node, field_name, new_values)

            elif isinstance(old_value, ast.VyperNode):
                # Handle single node fields - this includes ast.Int, ast.Name, etc.
                new_value = self.visit(old_value)
                setattr(node, field_name, new_value)

            # Skip non-AST values like:
            # - node.id (ast.Name, ast.FunctionDef - identifier string)
            # - node.attr (ast.Attribute - attribute name string)
            # - node.value (ast.Int, ast.Str - raw Python int/str value)
            # - node.op (ast.BinOp, ast.UnaryOp - operator singleton)
            # - node.name (ast.ImportFrom - module name string)

        return node


class FreshNameGenerator:
    """Generates unique variable names with a consistent prefix."""

    def __init__(self, prefix: str = "ivy_internal"):
        self.prefix = prefix
        self.counter = 0

    def generate(self) -> str:
        name = f"{self.prefix}{self.counter}"
        self.counter += 1
        return name


class AstMutator(VyperNodeTransformer):
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
        self.context = Context()
        self.name_generator = FreshNameGenerator()
        self.value_mutator = ValueMutator(rng)
        # Type generator for random types
        self.type_generator = TypeGenerator(rng)
        # Control how many variables to generate per scope
        self.min_vars_per_scope = 0
        self.max_vars_per_scope = 6
        # Probability of injecting variables into a scope
        self.inject_vars_prob = 0.5
        # Expression generator
        self.expr_generator = ExprGenerator(self.value_mutator, self.rng)
        # Statement generator
        self.stmt_generator = StatementGenerator(
            self.expr_generator, self.type_generator, self.rng
        )
        # Control statement injection
        self.inject_statements_prob = 0.3
        self.max_injection_depth = 4

    def mutate(self, root: ast.Module) -> ast.Module:
        self.mutations_done = 0
        # Reset context
        self.context = Context()
        # Initialize with module-level scope
        self.context.scope_stack.append(
            self.context.current_scope
        )  # Module scope stays on stack
        self.name_generator.counter = 0
        self.type_generator.struct_counter = 0
        self.stmt_generator.source_fragments = []
        self.stmt_generator.name_generator.counter = 0

        # Deep copy the root to avoid modifying the original
        new_root = copy.deepcopy(root)

        # Visit and get the potentially transformed root
        new_root = self.visit(new_root)

        # Handle immutables initialization after mutation
        self._ensure_init_with_immutables(new_root)

        return new_root

    def push_scope(self):
        self.context.push_scope()

    def pop_scope(self):
        self.context.pop_scope()

    def add_variable(self, name: str, var_info: VarInfo):
        self.context.add_variable(name, var_info)

    def add_local(self, name: str, typ: VyperType):
        """Convenience method to add a local variable."""
        self.context.add_local(name, typ)

    def pick_var(self, want_type: Optional[VyperType] = None) -> Optional[ast.Name]:
        """Pick a variable from all accessible variables matching the type."""
        if not self.context.all_vars:
            return None

        vars_pool = []

        # Check all variables in the global pool
        for name, var_info in self.context.all_vars.items():
            if want_type is None or var_info.typ == want_type:
                vars_pool.append((name, var_info))

        if not vars_pool:
            return None

        selected_name, selected_var_info = self.rng.choice(vars_pool)

        node = ast.Name(id=selected_name)
        node._metadata = {"type": selected_var_info.typ, "varinfo": selected_var_info}
        return node

    @property
    def is_module_scope(self) -> bool:
        """Check if we're in module scope (only one scope on the stack)."""
        return self.context.is_module_scope

    def should_mutate(self, node_type: Type) -> bool:
        node_prob = self.PROB.get(node_type, 0)
        return (
            self.mutations_done < self.max_mutations
            and self.rng.random() < self.mutate_prob * node_prob
        )

    def generate_random_expr(self, target_type: VyperType) -> ast.VyperNode:
        return self.expr_generator.generate(target_type, self.context, depth=4)

    def visit_Module(self, node: ast.Module):
        self.stmt_generator.inject_statements(node.body, self.context, node, depth=0)

        return super().generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Interface functions are just signatures so we skip them
        parent = node.get_ancestor()
        if parent and isinstance(parent, ast.InterfaceDef):
            return node

        self.push_scope()

        # Get function arguments from the function type
        func_type = node._metadata["func_type"]

        for arg in func_type.arguments:
            if func_type.is_external:
                location = DataLocation.CALLDATA
            else:
                assert func_type.is_internal, (
                    f"Expected internal function, got {func_type}"
                )
                location = DataLocation.MEMORY

            var_info = VarInfo(
                typ=arg.typ,
                location=location,
                decl_node=arg.ast_source if hasattr(arg, "ast_source") else None,
            )
            self.add_variable(arg.name, var_info)

        # Use statement generator to inject statements
        self.stmt_generator.inject_statements(node.body, self.context, node, depth=0)

        # Let the base class handle visiting children
        node = super().generic_visit(node)

        self.pop_scope()
        return node

    def visit_Int(self, node: ast.Int):
        if not self.should_mutate(ast.Int):
            return node

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
        return node

    def visit_BinOp(self, node: ast.BinOp):
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)

        if not self.should_mutate(ast.BinOp):
            return node

        left_type = getattr(node.left, "_metadata", {}).get("type")
        right_type = getattr(node.right, "_metadata", {}).get("type")

        if left_type is not None and right_type is not None and left_type != right_type:
            return node

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
        return node

    def visit_If(self, node: ast.If):
        node.test = self.visit(node.test)

        # Push scope for if body
        self.push_scope()
        self.stmt_generator.inject_statements(node.body, self.context, node, depth=1)
        self.pop_scope()

        if node.orelse:
            # Push scope for else body
            self.push_scope()
            self.stmt_generator.inject_statements(
                node.orelse, self.context, node, depth=1
            )
            self.pop_scope()

        node = super().generic_visit(node)

        # Visit body and orelse using generic_visit to handle lists properly
        if not self.should_mutate(ast.If):
            return node

        if isinstance(node.test, ast.Call):
            return node

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
        return node

    def visit_Return(self, node: ast.Return):
        if node.value:
            node.value = self.visit(node.value)

        if not self.should_mutate(ast.Return):
            return node

        if (
            node.value
            and hasattr(node.value, "is_literal_value")
            and node.value.is_literal_value
        ):
            # TODO
            # In a real implementation, we would use vyper.eval_fold here
            # For this example, we just skip this part as we don't have the actual fold helper
            pass

        # self.mutations_done += 1
        return node

    def visit_Assign(self, node: ast.Assign):
        node.target = self.visit(node.target)
        node.value = self.visit(node.value)

        if not self.should_mutate(ast.Assign):
            return node

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
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp):
        node.operand = self.visit(node.operand)

        if not self.should_mutate(ast.UnaryOp):
            return node

        # Only apply to Int operands that are not already negative
        if isinstance(node.operand, ast.Int) and (
            isinstance(node.op, (ast.USub, ast.Invert)) or node.operand.value >= 0
        ):
            # Choose to either drop the unary operator or add one
            if isinstance(node.op, (ast.USub, ast.Invert)):
                # Just return the operand, effectively removing the UnaryOp node
                self.mutations_done += 1
                return node.operand
            else:
                # Add a unary operator
                if node.operand.value >= 0:
                    # Add a negation or bitwise not
                    op_choice = self.rng.choice([ast.USub(), ast.Invert()])
                    node.op = op_choice

        self.mutations_done += 1
        return node

    def visit_BoolOp(self, node: ast.BoolOp):
        # Visit all values
        node.values = [self.visit(value) for value in node.values]

        if not self.should_mutate(ast.BoolOp):
            return node

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
        return node

    def visit_Attribute(self, node: ast.Attribute):
        """Mutate attribute access (e.g., self.foo)"""
        node.value = self.visit(node.value)

        if not self.should_mutate(ast.Attribute):
            return node

        # For now, we can swap attribute names if we have a mapping
        # This is a simplified implementation
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            # Could implement storage variable reordering here
            pass

        self.mutations_done += 1
        return node

    def visit_Subscript(self, node: ast.Subscript):
        """Mutate subscript access (e.g., arr[i])"""
        node.value = self.visit(node.value)
        node.slice = self.visit(node.slice)

        if not self.should_mutate(ast.Subscript):
            return node

        # Could mutate the index
        if isinstance(node.slice, ast.Int):
            node.slice.value = self.rng.choice(
                [0, 1, node.slice.value + 1, node.slice.value - 1]
            )

        self.mutations_done += 1
        return node

    def visit_For(self, node: ast.For):
        """Mutate for loops"""
        node.target = self.visit(node.target)
        node.iter = self.visit(node.iter)

        self.push_scope()
        self.stmt_generator.inject_statements(node.body, self.context, node, depth=1)

        # Use generic_visit to handle body list
        node = super().generic_visit(node)

        self.pop_scope()

        if not self.should_mutate(ast.For):
            return node

        # Could swap loop bounds or modify iteration
        # This is a simplified implementation
        # self.mutations_done += 1
        return node

    def visit_Compare(self, node: ast.Compare):
        """Mutate comparison operations"""
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)

        if not self.should_mutate(ast.Compare):
            return node

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
        return node

    def _ensure_init_with_immutables(self, module: ast.Module):
        if not self.context.immutables_to_init:
            return

        # Find existing __init__ function
        init_func = None
        for node in module.body:
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                init_func = node
                break

        # Create __init__ if it doesn't exist
        if init_func is None:
            deploy_decorator = ast.Name(id="deploy")

            init_func = ast.FunctionDef(
                name="__init__",
                args=ast.arguments(args=[], defaults=[]),
                body=[],
                decorator_list=[deploy_decorator],
                returns=None,
            )
            module.body.append(init_func)

        # Generate assignment statements for each immutable
        for name, var_info in self.context.immutables_to_init:
            value_expr = self.expr_generator.generate(
                var_info.typ, self.context, depth=3
            )

            assign = ast.Assign(targets=[ast.Name(id=name)], value=value_expr)

            init_func.body.append(assign)

        assert init_func.body

    def generate_pragma_lines(self, settings) -> list[str]:
        pragma_lines = []

        if settings.compiler_version:
            pragma_lines.append(f"# pragma version {settings.compiler_version}")

        if settings.evm_version:
            pragma_lines.append(f"# pragma evm-version {settings.evm_version}")

        if settings.experimental_codegen:
            pragma_lines.append("# pragma experimental-codegen")

        # TODO we should probably dump settings in traces (maybe just read from solc_json)
        pragma_lines.append("# pragma enable-decimals")

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

            # Build final source with pragmas, type declarations, and code
            parts = []
            if pragma_lines:
                parts.append("\n".join(pragma_lines))

            # Add all source fragments (struct declarations, etc.)
            if self.stmt_generator.source_fragments:
                parts.append("\n\n".join(self.stmt_generator.source_fragments))

            parts.append(result)

            final_result = "\n\n".join(parts)

            print(final_result)
            print("===============")
            return final_result
        except Exception as e:
            import logging
            import traceback

            logging.info(f"Failed to mutate source: {e}")
            logging.debug("Traceback:", exc_info=True)
            traceback.print_exc()
            return None

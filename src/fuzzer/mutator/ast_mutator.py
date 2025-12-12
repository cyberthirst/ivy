import copy
import random
from typing import List, Optional
from dataclasses import dataclass, field

from vyper.ast import nodes as ast
from vyper.semantics.types import VyperType, IntegerT, ContractFunctionT, TupleT
from vyper.semantics.types.primitives import NumericT
from vyper.semantics.analysis.base import VarInfo, DataLocation, Modifiability
from vyper.compiler.phases import CompilerData

from .literal_generator import LiteralGenerator
from .value_mutator import ValueMutator
from .candidate_selector import CandidateSelector
from src.fuzzer.mutator.function_registry import FunctionRegistry
from .context import Context, ScopeType, state_to_expr_mutability
from .expr_generator import ExprGenerator
from .stmt_generator import StatementGenerator
from src.unparser.unparser import unparse
from src.fuzzer.type_generator import TypeGenerator
from src.fuzzer.xfail import XFailExpectation


@dataclass
class MutationResult:
    source: str
    compilation_xfails: List[XFailExpectation] = field(default_factory=list)
    runtime_xfails: List[XFailExpectation] = field(default_factory=list)


from .mode import MutationMode  # noqa: E402


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
        self,
        rng: random.Random,
        *,
        mode: MutationMode,
        max_mutations: int = 5,
    ):
        self.rng = rng
        self.mode = mode
        self.max_mutations = max_mutations
        self.mutations_done = 0
        self._mutation_targets: set[int] = set()
        self._candidate_selector = CandidateSelector(rng, self.PROB)
        self.context = Context()
        self.name_generator = FreshNameGenerator()
        self.literal_generator = LiteralGenerator(rng)
        self.value_mutator = ValueMutator(rng)
        # Type generator for random types
        self.type_generator = TypeGenerator(rng)
        # Control how many variables to generate per scope
        self.min_vars_per_scope = 0
        self.max_vars_per_scope = 6
        # Probability of injecting variables into a scope
        self.inject_vars_prob = 0.5
        # Function registry for tracking and generating functions
        self.function_registry = FunctionRegistry(self.rng, max_generated_functions=5)
        # Expression generator with function registry
        self.expr_generator = ExprGenerator(
            self.literal_generator,
            self.rng,
            self.function_registry,
            self.type_generator,
            mode=self.mode,
        )
        # Statement generator
        self.stmt_generator = StatementGenerator(
            self.expr_generator, self.type_generator, self.rng
        )

    @property
    def is_generate_mode(self) -> bool:
        return self.mode == MutationMode.GENERATE

    def _negate_condition(self, expr: ast.VyperNode) -> ast.VyperNode:
        """Return a logically negated version of the test expression.

        - For Compare nodes, invert the operator directly.
        - For all other expressions, wrap with a boolean Not.
        """
        if isinstance(expr, ast.Compare):
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

            if type(expr.op) in op_map:
                expr.op = op_map[type(expr.op)]
                return expr

        return ast.UnaryOp(op=ast.Not(), operand=expr)

    def mutate(self, root: ast.Module) -> ast.Module:
        self.reset_state()

        # Deep copy the root to avoid modifying the original
        new_root = copy.deepcopy(root)

        # Pass 1: select mutation targets
        self._mutation_targets = self._candidate_selector.select(
            new_root, self.max_mutations
        )

        # Pass 2: visit and mutate selected nodes
        new_root = self.visit(new_root)
        assert isinstance(new_root, ast.Module)

        # Handle immutables initialization after mutation
        self._ensure_init_with_immutables(new_root)

        return new_root

    def reset_state(self) -> None:
        """Reset all per-mutation internal state to a clean baseline."""
        self.mutations_done = 0
        self._mutation_targets = set()
        self.context = Context()
        self.context.scope_stack.append(self.context.current_scope)  # keep module scope
        self.name_generator.counter = 0
        self.type_generator.struct_counter = 0
        self.stmt_generator.source_fragments = []
        self.stmt_generator.name_generator.counter = 0
        self.function_registry.reset()

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

    def should_mutate(self, node: ast.VyperNode) -> bool:
        """Check if node was selected for mutation. Consumes the target."""
        node_id = id(node)
        if node_id not in self._mutation_targets:
            return False
        self._mutation_targets.discard(node_id)
        return True

    def _type_of(self, node: ast.VyperNode):
        """Safely extract the inferred type from AST metadata."""
        return getattr(node, "_metadata", {}).get("type")

    def generate_random_expr(self, target_type: VyperType) -> ast.VyperNode:
        return self.expr_generator.generate(target_type, self.context, depth=4)

    def visit_Module(self, node: ast.Module):
        # Preprocess: register all existing functions and module-level variables
        self._preprocess_module(node)

        if self.is_generate_mode:
            self.stmt_generator.inject_statements(
                node.body, self.context, node, depth=0
            )

        # Visit all existing nodes first
        node = super().generic_visit(node)

        if self.is_generate_mode:
            while True:
                pending_funcs = self.function_registry.get_pending_implementations()
                if not pending_funcs:
                    break

                for func in pending_funcs:
                    if func.ast_def:
                        assert isinstance(func.ast_def, ast.FunctionDef)
                        # Visit the function to fill its body
                        self.visit_FunctionDef(func.ast_def)
                        assert func.ast_def.body
                        node.body.append(func.ast_def)

        return node

    def _preprocess_module(self, node: ast.Module):
        """Register all existing functions and module-level variables."""
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                # Register existing function in the registry
                func_type = item._metadata.get("type")
                if func_type and isinstance(func_type, ContractFunctionT):
                    self.function_registry.register_function(func_type)

            elif isinstance(item, ast.VariableDecl):
                # Register module-level variable
                var_info = item._metadata.get("varinfo")
                if (
                    var_info
                    and hasattr(item, "target")
                    and isinstance(item.target, ast.Name)
                ):
                    self.context.add_variable(item.target.id, var_info)

    def visit_VariableDecl(self, node: ast.VariableDecl):
        """Skip processing variable declarations - they're handled in preprocessing."""
        # Don't visit children or mutate - already registered in preprocessing
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Interface functions are just signatures so we skip them
        parent = node.get_ancestor()
        if parent and isinstance(parent, ast.InterfaceDef):
            return node

        # Set current function for cycle detection
        self.function_registry.set_current_function(node.name)

        with self.context.new_scope(ScopeType.FUNCTION):
            # Get function arguments from the function type
            func_type = node._metadata["func_type"]
            func_state_mut = func_type.mutability
            expr_mut = state_to_expr_mutability(func_state_mut)

            with (
                self.context.mutability(expr_mut),
                self.context.function_mutability(func_state_mut),
            ):
                for arg in func_type.arguments:
                    if func_type.is_external:
                        location = DataLocation.CALLDATA
                        modifiability = Modifiability.RUNTIME_CONSTANT
                    else:
                        assert func_type.is_internal, (
                            f"Expected internal function, got {func_type}"
                        )
                        location = DataLocation.MEMORY
                        modifiability = Modifiability.MODIFIABLE

                    var_info = VarInfo(
                        typ=arg.typ,
                        location=location,
                        modifiability=modifiability,
                        decl_node=arg.ast_source
                        if hasattr(arg, "ast_source")
                        else None,
                    )
                    self.add_variable(arg.name, var_info)

                # Use statement generator to inject statements
                # For generated functions with empty bodies, ensure they get statements
                if self.is_generate_mode:
                    if not node.body:
                        n_stmts = self.rng.randint(1, 5)
                        self.stmt_generator.inject_statements(
                            node.body, self.context, node, depth=0, n_stmts=n_stmts
                        )
                    else:
                        self.stmt_generator.inject_statements(
                            node.body, self.context, node, depth=0
                        )

            # Let the base class handle visiting children
            node = super().generic_visit(node)

        self.function_registry.set_current_function(None)

        return node

    def visit_Int(self, node: ast.Int):
        return node
        if not self.should_mutate(ast.Int):
            return node

        # Get the type if available
        node_type = self._type_of(node)

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
        left = self.visit(node.left)
        right = self.visit(node.right)
        assert isinstance(left, ast.ExprNode) and isinstance(right, ast.ExprNode)
        node.left = left
        node.right = right

        if not self.should_mutate(node):
            return node

        left_type = self._type_of(node.left)
        right_type = self._type_of(node.right)

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

        if self.is_generate_mode:
            with self.context.new_scope(ScopeType.IF):
                self.stmt_generator.inject_statements(
                    node.body, self.context, node, depth=1
                )

            if node.orelse:
                with self.context.new_scope(ScopeType.IF):
                    self.stmt_generator.inject_statements(
                        node.orelse, self.context, node, depth=1
                    )

        node = super().generic_visit(node)

        # Visit body and orelse using generic_visit to handle lists properly
        if not self.should_mutate(node):
            return node

        if isinstance(node.test, ast.Call):
            return node

        mutation_type = self.rng.choice(["negate_condition", "swap_branches"])
        mutated = False

        if mutation_type == "negate_condition":
            node.test = self._negate_condition(node.test)
            mutated = True

        elif mutation_type == "swap_branches":
            # Swap body and orelse
            # Only swap when both branches are non-empty to avoid empty if bodies
            if node.body and node.orelse:
                node.body, node.orelse = node.orelse, node.body
                # If we're swapping branches, also negate the condition
                node.test = self._negate_condition(node.test)
                mutated = True

        if mutated:
            self.mutations_done += 1
        return node

    def visit_Return(self, node: ast.Return):
        if node.value:
            node.value = self.visit(node.value)

        if not self.should_mutate(node):
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

        if not self.should_mutate(node):
            return node

        rhs_type = self._type_of(node.value)

        if self.is_generate_mode:
            mutation_type = self.rng.choice(["use_var_as_rhs", "generate_new_expr"])
        else:
            mutation_type = "use_var_as_rhs"

        if mutation_type == "use_var_as_rhs" and rhs_type is not None:
            other_var = self.pick_var(rhs_type)
            if other_var:
                node.value = other_var

        elif mutation_type == "generate_new_expr" and rhs_type is not None:
            new_expr = self.expr_generator.generate(rhs_type, self.context, depth=2)
            node.value = new_expr

        self.mutations_done += 1
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp):
        node.operand = self.visit(node.operand)

        if not self.should_mutate(node):
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

        if not self.should_mutate(node):
            return node

        # Check if all operands are boolean typed
        all_bool = True
        for value in node.values:
            value_type = self._type_of(value)
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

        if not self.should_mutate(node):
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

        if not self.should_mutate(node):
            return node

        # Mutate the index; skip tuples entirely to avoid changing element types.
        base_type = self._type_of(node.value)
        if isinstance(node.slice, ast.Int):
            if not isinstance(base_type, TupleT):
                node.slice.value = self.rng.choice(
                    [0, 1, node.slice.value + 1, node.slice.value - 1]
                )

        self.mutations_done += 1
        return node

    def visit_For(self, node: ast.For):
        """Mutate for loops"""
        node.target = self.visit(node.target)
        node.iter = self.visit(node.iter)

        with self.context.new_scope(ScopeType.FOR):
            if self.is_generate_mode:
                self.stmt_generator.inject_statements(
                    node.body, self.context, node, depth=1
                )

            node = super().generic_visit(node)

        if not self.should_mutate(node):
            return node

        # Could swap loop bounds or modify iteration
        # This is a simplified implementation
        # self.mutations_done += 1
        return node

    def visit_Compare(self, node: ast.Compare):
        """Mutate comparison operations"""
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)

        if not self.should_mutate(node):
            return node

        typ = self._type_of(node) or self._type_of(node.left)

        if typ and isinstance(typ, NumericT):
            # For numeric types, any comparison operator is valid
            # TODO add mutations for other operators (shifting etc)
            ops = [ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq]
            new_op_type = self.rng.choice(ops)
            node.op = new_op_type()
        else:
            # For non-numeric types, only eq/neq are valid
            # TODO add mutations for other operators
            if isinstance(node.op, ast.Eq):
                node.op = ast.NotEq()
            elif isinstance(node.op, ast.NotEq):
                node.op = ast.Eq()
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

    def mutate_source_with_compiler_data(
        self, compiler_data: CompilerData
    ) -> Optional[MutationResult]:
        """Mutate source using annotated AST from CompilerData."""
        try:
            annotated_module = compiler_data.annotated_vyper_module

            mutated_ast = self.mutate(annotated_module)

            result = unparse(mutated_ast)

            # Build final source with type declarations and code.
            # Note: Pragmas are no longer embedded in source - compiler settings
            # are passed directly to loaders via DeploymentTrace.compiler_settings.
            parts = []

            # Add all source fragments (struct declarations, etc.)
            if self.stmt_generator.source_fragments:
                parts.append("\n\n".join(self.stmt_generator.source_fragments))

            parts.append(result)

            final_result = "\n\n".join(parts)

            print(final_result)
            print("===============")
            return MutationResult(
                source=final_result,
                compilation_xfails=list(self.context.compilation_xfails),
                runtime_xfails=list(self.context.runtime_xfails),
            )
        except Exception as e:
            import logging
            import traceback

            logging.info(f"Failed to mutate source: {e}")
            logging.debug("Traceback:", exc_info=True)
            traceback.print_exc()
            return None

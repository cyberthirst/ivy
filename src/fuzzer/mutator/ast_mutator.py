import copy
import random
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from contextlib import nullcontext

from vyper.ast import nodes as ast
from vyper.semantics.types import VyperType, ContractFunctionT, HashMapT
from vyper.semantics.types.function import FunctionVisibility
from vyper.semantics.analysis.base import VarInfo, DataLocation, Modifiability
from vyper.compiler.phases import CompilerData

from fuzzer.mutator.literal_generator import LiteralGenerator
from fuzzer.mutator.value_mutator import ValueMutator
from fuzzer.mutator.candidate_selector import CandidateSelector
from fuzzer.mutator.function_registry import FunctionRegistry, KWARG_PLACEHOLDER_TAG
from fuzzer.mutator.interface_registry import InterfaceRegistry
from fuzzer.mutator.context import (
    GenerationContext,
    ScopeType,
    ExprMutability,
    state_to_expr_mutability,
)
from fuzzer.mutator.expr_generator import ExprGenerator
from fuzzer.mutator.stmt_generator import StatementGenerator
from fuzzer.mutator.ast_utils import body_is_terminated, expr_type, hoist_prelude_decls
from fuzzer.mutator.strategy import StrategyRegistry
from fuzzer.mutator.constant_folding import evaluate_constant_expression
from fuzzer.mutator.mutation_engine import MutationEngine
from fuzzer.mutator.mutations import register_all as register_all_mutations
from fuzzer.mutator.mutations.base import MutationCtx
from fuzzer.mutator.config import MutatorConfig
from fuzzer.mutator.name_generator import FreshNameGenerator
from unparser.unparser import unparse
from fuzzer.type_generator import TypeGenerator
from fuzzer.xfail import XFailExpectation


@dataclass
class MutationResult:
    sources: Dict[str, str]
    compilation_xfails: List[XFailExpectation] = field(default_factory=list)
    runtime_xfails: List[XFailExpectation] = field(default_factory=list)


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


class AstMutator(VyperNodeTransformer):
    def __init__(
        self,
        rng: random.Random,
        max_mutations: int = 5,
        generate: bool = False,
        cfg: Optional[MutatorConfig] = None,
    ):
        self.rng = rng
        self.max_mutations = max_mutations
        self.generate = generate
        self.cfg = cfg or MutatorConfig()
        self._mutation_targets: set[int] = set()
        self._candidate_selector = CandidateSelector(rng)
        self.context = GenerationContext()
        self.name_generator = FreshNameGenerator(prefix="gen_var")
        self.literal_generator = LiteralGenerator(rng)
        self.value_mutator = ValueMutator(rng)
        # Type generator for random types
        self.type_generator = TypeGenerator(rng)
        # Function registry for tracking and generating functions
        self.function_registry = FunctionRegistry(
            self.rng, max_initial_functions=5, max_dynamic_functions=5
        )
        # Interface registry for external calls
        self.interface_registry = InterfaceRegistry(self.rng)
        # Expression generator with function and interface registries
        self.expr_generator = ExprGenerator(
            self.literal_generator,
            self.rng,
            self.interface_registry,
            self.function_registry,
            self.type_generator,
            cfg=self.cfg.expr,
            depth_cfg=self.cfg.expr_depth,
            name_generator=self.name_generator,
        )
        # Statement generator
        self.stmt_generator = StatementGenerator(
            self.expr_generator,
            self.type_generator,
            self.rng,
            cfg=self.cfg.stmt,
            depth_cfg=self.cfg.stmt_depth,
            name_generator=self.name_generator,
        )

        # Mutation engine
        self._mutation_registry = StrategyRegistry()
        register_all_mutations(self._mutation_registry)
        self._mutation_engine = MutationEngine(self._mutation_registry, self.rng)

    def _build_ctx(self, node: ast.VyperNode, **kwargs) -> MutationCtx:
        return MutationCtx(
            node=node,
            rng=self.rng,
            context=self.context,
            expr_gen=self.expr_generator,
            stmt_gen=self.stmt_generator,
            function_registry=self.function_registry,
            value_mutator=self.value_mutator,
            **kwargs,
        )

    def generic_visit(self, node):
        if self.generate:
            return node
        return super().generic_visit(node)

    def _try_mutate(self, node: ast.VyperNode, **ctx_kwargs) -> ast.VyperNode:
        if self.generate:
            return node
        if not self.should_mutate(node):
            return node
        return self._mutation_engine.mutate(self._build_ctx(node, **ctx_kwargs))

    def mutate(self, root: ast.Module) -> ast.Module:
        self.reset_state()

        if self.generate:
            new_root = self._generate_module()
            self._maybe_create_init(new_root)
            self._maybe_create_default()
            self._dispatch_pending_functions(new_root)
            self._add_interface_definitions(new_root)
            hoist_prelude_decls(new_root)
            return new_root
        else:
            # Deep copy the root to avoid modifying the original
            new_root = copy.deepcopy(root)

            # Pass 1: select mutation targets
            self._mutation_targets = self._candidate_selector.select(
                new_root, self.max_mutations
            )

        # Pass 2: visit and mutate selected nodes
        new_root = self.visit(new_root)
        assert isinstance(new_root, ast.Module)

        hoist_prelude_decls(new_root)
        return new_root

    def _generate_module(self) -> ast.Module:
        module = ast.Module(body=[])

        # Module scope only supports variable declarations.
        self.stmt_generator.inject_variable_decls(
            module.body,
            self.context,
            parent=module,
            depth=self.stmt_generator.root_depth(),
            min_stmts=0,
            max_stmts=15,
        )

        max_funcs = self.function_registry.max_initial_functions
        if max_funcs <= 0:
            return module

        num_funcs = self.rng.randint(1, max_funcs)
        functions_added = 0

        return_type = self.type_generator.generate_type(nesting=2, skip={HashMapT})
        func_def = self.function_registry.create_new_function(
            return_type=return_type,
            type_generator=self.type_generator,
            max_args=6,
            initial=True,
            visibility=FunctionVisibility.EXTERNAL,
            max_kwargs=3,
        )
        if func_def is not None:
            functions_added += 1

        for _ in range(num_funcs - functions_added):
            return_type = self.type_generator.generate_type(nesting=2, skip={HashMapT})
            func_def = self.function_registry.create_new_function(
                return_type=return_type,
                type_generator=self.type_generator,
                max_args=6,
                initial=True,
                max_kwargs=3,
            )
            if func_def is not None:
                functions_added += 1

        return module

    def reset_state(self) -> None:
        """Reset all per-mutation internal state to a clean baseline."""
        self._mutation_targets = set()
        self.context = GenerationContext()
        self.context.scope_stack.append(self.context.current_scope)  # keep module scope
        self.name_generator.reset()
        self.type_generator.struct_counter = 0
        self.type_generator.source_fragments = []
        self.expr_generator.reset_state()
        self.function_registry.reset()
        self.interface_registry.reset()

    def add_variable(self, name: str, var_info: VarInfo):
        self.context.add_variable(name, var_info)

    def should_mutate(self, node: ast.VyperNode) -> bool:
        """Check if node was selected for mutation. Consumes the target."""
        node_id = id(node)
        if node_id not in self._mutation_targets:
            return False
        self._mutation_targets.discard(node_id)
        return True

    def _type_of(self, node: ast.VyperNode):
        """Safely extract the inferred type from AST metadata."""
        return expr_type(node)

    def generate_random_expr(self, target_type: VyperType) -> ast.VyperNode:
        return self.expr_generator.generate(
            target_type,
            self.context,
            depth=self.expr_generator.root_depth(),
            allow_tuple_literal=False,
        )

    def visit_Module(self, node: ast.Module):
        assert not self.generate
        self._preprocess_module(node)
        node = self._try_mutate(node)
        node = super().generic_visit(node)
        self._maybe_create_init(node)
        self._dispatch_pending_functions(node)
        self._add_interface_definitions(node)
        return node

    def _dispatch_pending_functions(self, node: ast.Module):
        while True:
            pending_funcs = self.function_registry.get_pending_implementations()
            if not pending_funcs:
                break

            for func in pending_funcs:
                if func.ast_def:
                    assert isinstance(func.ast_def, ast.FunctionDef)
                    self.visit_FunctionDef(func.ast_def)
                    assert func.ast_def.body
                    node.body.append(func.ast_def)

    def _add_interface_definitions(self, node: ast.Module):
        """Add generated interface definitions to the module body."""
        interface_defs = self.interface_registry.get_interface_defs()
        if interface_defs:
            # Insert at beginning of module body
            node.body = interface_defs + node.body

    def _module_has_init(self, node: ast.Module) -> bool:
        if "__init__" in self.function_registry.functions:
            return True
        return any(
            isinstance(item, ast.FunctionDef) and item.name == "__init__"
            for item in node.body
        )

    def _maybe_create_init(self, node: ast.Module) -> None:
        has_immutables = bool(self.context.immutables_to_init)
        if self.generate:
            if not has_immutables and self.rng.random() >= 0.5:
                return
            self.function_registry.create_init(
                type_generator=self.type_generator,
                max_args=6,
            )
            return

        if not has_immutables or self._module_has_init(node):
            return
        self.function_registry.create_init(
            type_generator=self.type_generator,
            max_args=6,
            payable=False,
        )

    def _maybe_create_default(self) -> None:
        if "__default__" in self.function_registry.functions:
            return
        if self.rng.random() >= 0.5:
            return
        self.function_registry.create_default()

    def _preprocess_module(self, node: ast.Module):
        """Register all existing functions and module-level variables."""
        assert not self.generate
        settings = getattr(node, "settings", None)
        self.function_registry.set_nonreentrancy_by_default(
            bool(getattr(settings, "nonreentrancy_by_default", False))
        )
        function_types: List[ContractFunctionT] = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                # Register existing function in the registry
                func_type = item._metadata.get("func_type")
                if func_type and isinstance(func_type, ContractFunctionT):
                    self.function_registry.register_function(func_type)
                    function_types.append(func_type)

            elif isinstance(item, ast.VariableDecl):
                # Register module-level variable
                var_info = item._metadata.get("varinfo")
                if (
                    var_info
                    and hasattr(item, "target")
                    and isinstance(item.target, ast.Name)
                ):
                    self.context.add_variable(item.target.id, var_info)
                    if (
                        var_info.modifiability == Modifiability.CONSTANT
                        and item.value is not None
                    ):
                        try:
                            const_value = evaluate_constant_expression(
                                item.value, self.context.constants
                            )
                        except Exception:
                            pass
                        else:
                            self.context.add_constant(item.target.id, const_value)

        # Seed call graph with calls already present in the annotated module.
        for fn_t in function_types:
            for callee_t in fn_t.called_functions:
                self.function_registry.add_call(
                    fn_t.name,
                    callee_t.name,
                    internal=True,
                )

    def visit_VariableDecl(self, node: ast.VariableDecl):
        # VariableDecl is module-level only (storage, immutables, transient, constants).
        # Per Vyper's VariableDecl.validate(): only constants have a value,
        # and non-constants cannot have one. Constants are out of scope for mutation for now.
        return node

    def visit_StructDef(self, node: ast.StructDef):
        # Struct member AnnAssign nodes must not be visited — their field names
        # are not local variables and must not be registered in all_vars.
        return node

    def visit_EventDef(self, node: ast.EventDef):
        return node

    def visit_FlagDef(self, node: ast.FlagDef):
        return node

    def visit_InterfaceDef(self, node: ast.InterfaceDef):
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign):
        """Handle local variable declaration."""
        node.value = self.visit(node.value)

        # Local variables don't have VarInfo persisted in metadata after analysis,
        # so we construct it from the type info
        assert isinstance(node.target, ast.Name)
        var_type = self._type_of(node.target)
        if var_type:
            var_info = VarInfo(
                typ=var_type,
                location=DataLocation.MEMORY,
                modifiability=Modifiability.MODIFIABLE,
            )
            self.context.add_variable(node.target.id, var_info)

        return node

    def _fill_kwarg_defaults(
        self, node: ast.FunctionDef, func_type: ContractFunctionT
    ) -> None:
        if not func_type.keyword_args:
            return
        for i, kw_arg in enumerate(func_type.keyword_args):
            if not getattr(kw_arg.default_value, "_metadata", {}).get(
                KWARG_PLACEHOLDER_TAG
            ):
                continue
            with self.context.mutability(ExprMutability.CONST):
                default_expr = self.expr_generator.generate(
                    kw_arg.typ,
                    self.context,
                    depth=self.expr_generator.root_depth(),
                )
            kw_arg.default_value = default_expr
            node.args.defaults[i] = default_expr

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Interface functions are just signatures so we skip them
        parent = node.get_ancestor()
        if parent and isinstance(parent, ast.InterfaceDef):
            return node

        # Set current function for cycle detection
        self.function_registry.set_current_function(node.name)
        is_init = node.name == "__init__"
        init_ctx = self.context.init_assignments() if is_init else nullcontext()

        with self.context.new_scope(ScopeType.FUNCTION):
            # Get function arguments from the function type
            func_type = node._metadata["func_type"]
            func_state_mut = func_type.mutability
            expr_mut = state_to_expr_mutability(func_state_mut)

            with (
                self.context.mutability(expr_mut),
                self.context.function_mutability(func_state_mut),
                init_ctx,
            ):
                # Fill kwarg defaults before registering args —
                # defaults are evaluated at module scope, not function scope.
                self._fill_kwarg_defaults(node, func_type)

                for arg in func_type.arguments:
                    if func_type.is_internal:
                        location = DataLocation.MEMORY
                        modifiability = Modifiability.MODIFIABLE
                    else:
                        # Both external and deploy functions have immutable args
                        location = DataLocation.CALLDATA
                        modifiability = Modifiability.RUNTIME_CONSTANT

                    var_info = VarInfo(
                        typ=arg.typ,
                        location=location,
                        modifiability=modifiability,
                        decl_node=arg.ast_source
                        if hasattr(arg, "ast_source")
                        else None,
                    )
                    self.add_variable(arg.name, var_info)

                # Generated functions with empty bodies need statements
                if not node.body:
                    self.stmt_generator.inject_statements(
                        node.body,
                        self.context,
                        node,
                        depth=self.stmt_generator.root_depth(),
                        min_stmts=2,
                        max_stmts=5,
                    )
                else:
                    node = self._try_mutate(node)

                # Visit children within mutability context so body mutations
                # respect @pure/@view constraints
                node = super().generic_visit(node)
                if is_init:
                    self._inject_missing_immutable_assignments(node)

        self.function_registry.set_current_function(None)

        return node

    def visit_Int(self, node: ast.Int):
        return self._try_mutate(node, inferred_type=self._type_of(node))

    def visit_BinOp(self, node: ast.BinOp):
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        return self._try_mutate(node)

    def visit_If(self, node: ast.If):
        node.test = self.visit(node.test)
        node = self._try_mutate(node)

        # Vyper has block scoping - vars declared in if/else are not visible outside
        with self.context.new_scope(ScopeType.IF):
            node.body = [self.visit(stmt) for stmt in node.body]
        with self.context.new_scope(ScopeType.IF):
            node.orelse = [self.visit(stmt) for stmt in node.orelse]

        return node

    def visit_Return(self, node: ast.Return):
        if node.value:
            node.value = self.visit(node.value)
        cur = self.function_registry.current_function
        if cur and node.value:
            func_type = self.function_registry.functions.get(cur)
            if func_type and func_type.return_type:
                return self._try_mutate(node, inferred_type=func_type.return_type)
        return node

    def visit_Assign(self, node: ast.Assign):
        if hasattr(node, "targets"):
            node.targets = [self.visit(target) for target in node.targets]
        elif hasattr(node, "target"):
            node.target = self.visit(node.target)
        node.value = self.visit(node.value)
        return self._try_mutate(node, inferred_type=self._type_of(node.value))

    def visit_UnaryOp(self, node: ast.UnaryOp):
        node.operand = self.visit(node.operand)
        return self._try_mutate(node)

    def visit_BoolOp(self, node: ast.BoolOp):
        node.values = [self.visit(value) for value in node.values]
        return self._try_mutate(node)

    def visit_Attribute(self, node: ast.Attribute):
        node.value = self.visit(node.value)
        # No mutations for attribute access yet
        return node

    def visit_Subscript(self, node: ast.Subscript):
        node.value = self.visit(node.value)
        node.slice = self.visit(node.slice)
        return self._try_mutate(node)

    def visit_For(self, node: ast.For):
        node.iter = self.visit(node.iter)

        # Mutation happens inside scope context so injected statements have access to loop var
        with self.context.new_scope(ScopeType.FOR):
            # Register the loop variable (Vyper loop vars are RUNTIME_CONSTANT - can't reassign)
            # node.target is AnnAssign, node.target.target is the Name
            loop_annassign = node.target
            assert isinstance(loop_annassign, ast.AnnAssign)
            loop_var_name = loop_annassign.target
            assert isinstance(loop_var_name, ast.Name)
            target_type = self._type_of(loop_var_name)
            if target_type:
                var_info = VarInfo(
                    typ=target_type,
                    modifiability=Modifiability.RUNTIME_CONSTANT,
                )
                self.context.add_variable(loop_var_name.id, var_info)

            node = self._try_mutate(node)
            node.body = [self.visit(stmt) for stmt in node.body]

        return node

    def visit_Continue(self, node: ast.Continue):
        return self._try_mutate(node)

    def visit_Break(self, node: ast.Break):
        return self._try_mutate(node)

    def visit_Compare(self, node: ast.Compare):
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        # Pass operand type for mutation decisions (result type is always BoolT)
        return self._try_mutate(node, inferred_type=self._type_of(node.left))

    def _inject_missing_immutable_assignments(self, node: ast.FunctionDef) -> None:
        missing = self.context.remaining_immutable_items()
        if not missing:
            return
        assignments = self.stmt_generator.generate_immutable_assignments(
            self.context, missing
        )
        if not assignments:
            return
        insert_pos = len(node.body)
        if body_is_terminated(node.body):
            insert_pos = max(0, insert_pos - 1)
        node.body[insert_pos:insert_pos] = assignments

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
            if self.type_generator.source_fragments:
                parts.append("\n\n".join(self.type_generator.source_fragments))

            parts.append(result)

            final_result = "\n\n".join(parts)

            file_input = compiler_data.file_input
            resolved_path = getattr(file_input, "resolved_path", None) or getattr(
                file_input, "path", None
            )
            if resolved_path is None:
                raise ValueError("CompilerData missing file path")
            source_key = str(resolved_path)

            return MutationResult(
                sources={source_key: final_result},
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

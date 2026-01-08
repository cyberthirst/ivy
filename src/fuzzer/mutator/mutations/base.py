from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
import random

from vyper.ast import nodes as ast
from vyper.semantics.types import VyperType

from fuzzer.mutator.context import Context
from fuzzer.mutator.expr_generator import ExprGenerator
from fuzzer.mutator.stmt_generator import StatementGenerator
from fuzzer.mutator.function_registry import FunctionRegistry

if TYPE_CHECKING:
    from ..value_mutator import ValueMutator


@dataclass
class MutationCtx:
    node: ast.VyperNode
    rng: random.Random
    context: Context
    expr_gen: ExprGenerator
    stmt_gen: StatementGenerator
    function_registry: FunctionRegistry
    value_mutator: ValueMutator
    inferred_type: Optional[VyperType] = None
    parent: Optional[ast.VyperNode] = None
    depth: int = 0

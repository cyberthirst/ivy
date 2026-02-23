import random
from typing import Type

from vyper.ast import nodes as ast
from vyper.semantics.types import TYPE_T, TupleT

from fuzzer.mutator.ast_utils import expr_type


class CandidateSelector:
    """
    Selects mutation targets via weighted sampling.
    Without replacement: each node can be selected at most once.
    """

    PROB: dict[Type, float] = {
        # ─────────────────────────────────────────────
        # Top-level
        # ─────────────────────────────────────────────
        ast.Module: 0.01,
        ast.FunctionDef: 0.01,
        # ─────────────────────────────────────────────
        # Literals
        # ─────────────────────────────────────────────
        ast.Int: 0.15,
        # TODO
        ast.Decimal: 0.0,
        # TODO
        ast.Hex: 0.0,
        # ─────────────────────────────────────────────
        # Operators
        # ─────────────────────────────────────────────
        ast.BinOp: 0.35,
        ast.UnaryOp: 0.15,
        ast.BoolOp: 0.15,
        ast.Compare: 0.35,
        # ─────────────────────────────────────────────
        # Control flow - safe, negate/swap branches
        # ─────────────────────────────────────────────
        ast.If: 0.25,
        # TODO
        ast.IfExp: 0.0,
        ast.For: 0.15,
        ast.Continue: 0.35,
        ast.Break: 0.35,
        # TODO
        ast.Assert: 0.0,
        # ─────────────────────────────────────────────
        # Assignments - safe with type-aware RHS swap
        # ─────────────────────────────────────────────
        ast.Assign: 0.25,
        # TODO
        ast.AugAssign: 0.0,
        # TODO
        ast.VariableDecl: 0.0,
        ast.Return: 0.25,
        # ─────────────────────────────────────────────
        # Access patterns - moderate risk, keep lower
        # ─────────────────────────────────────────────
        ast.Subscript: 0.2,
        # TODO
        ast.Attribute: 0.0,
    }

    def __init__(self, rng: random.Random, prob_map: dict[Type, float] | None = None):
        self.rng = rng
        self.prob_map = prob_map if prob_map is not None else self.PROB

    def select(self, root: ast.VyperNode, max_mutations: int) -> set[int]:
        """Walk tree, collect candidates, return set of node IDs to mutate."""
        candidates: list[ast.VyperNode] = []
        weights: list[float] = []

        for node in self._walk(root):
            if not self.filter(node):
                continue

            w = self.prob_map.get(type(node), 0.0)
            if w > 0:
                candidates.append(node)
                weights.append(w)

        chosen = self._weighted_sample_without_replacement(
            candidates,
            weights,
            k=min(max_mutations, len(candidates)),
        )
        return {id(n) for n in chosen}

    def filter(self, node: ast.VyperNode) -> bool:
        """Return True if node is a valid mutation candidate."""
        if isinstance(self._type_of(node), TYPE_T):
            return False

        # Tuple subscript indices - changing index changes result type
        if isinstance(node, ast.Subscript):
            base_type = self._type_of(node.value)
            if isinstance(base_type, TupleT) and isinstance(node.slice, ast.Int):
                return False

        return True

    def _type_of(self, node: ast.VyperNode):
        """Safely extract the inferred type from AST metadata."""
        return expr_type(node)

    # Type definition nodes whose subtrees should never be mutated.
    _SKIP_SUBTREE = (ast.StructDef, ast.EventDef, ast.FlagDef, ast.InterfaceDef)

    _SKIP_FUNC = frozenset({"range", "empty"})

    def _should_skip_subtree(self, node: ast.VyperNode, field_name: str) -> bool:
        if isinstance(self._type_of(node), TYPE_T):
            return True

        if field_name in ("annotation", "returns"):
            return True

        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in self._SKIP_FUNC
            and field_name in ("args", "keywords")
        ):
            return True

        return False

    def _walk(self, node: ast.VyperNode):
        """Yield all nodes in the AST (preorder), skipping unmutatable subtrees."""
        if node is None:
            return

        if isinstance(node, self._SKIP_SUBTREE):
            return

        yield node

        for field_name in node.get_fields():
            if not hasattr(node, field_name):
                continue

            if self._should_skip_subtree(node, field_name):
                continue

            val = getattr(node, field_name)
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, ast.VyperNode):
                        yield from self._walk(item)
            elif isinstance(val, ast.VyperNode):
                yield from self._walk(val)

    def _weighted_sample_without_replacement(
        self,
        items: list[ast.VyperNode],
        weights: list[float],
        k: int,
    ) -> list[ast.VyperNode]:
        """
        Weighted sampling without replacement (Efraimidis-Spirakis).

        For each item with weight w > 0:
            key = U^(1/w), U ~ Uniform(0,1]
        Pick top-k keys.
        """
        if k <= 0 or not items:
            return []

        keyed: list[tuple[float, ast.VyperNode]] = []
        for item, w in zip(items, weights):
            if w <= 0:
                continue
            u = self.rng.random()
            if u == 0.0:
                u = 1e-12
            key = u ** (1.0 / w)
            keyed.append((key, item))

        keyed.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in keyed[:k]]

import random
from typing import Type

from vyper.ast import nodes as ast
from vyper.semantics.types import TupleT


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
        # TODO
        ast.Return: 0.0,
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
        # Tuple subscript indices - changing index changes result type
        if isinstance(node, ast.Subscript):
            base_type = self._type_of(node.value)
            if isinstance(base_type, TupleT) and isinstance(node.slice, ast.Int):
                return False

        return True

    def _type_of(self, node: ast.VyperNode):
        """Safely extract the inferred type from AST metadata."""
        return getattr(node, "_metadata", {}).get("type")

    def _walk(self, node: ast.VyperNode):
        """Yield all nodes in the AST (preorder), skipping unmutatable subtrees."""
        if node is None:
            return

        yield node

        for field_name in node.get_fields():
            if not hasattr(node, field_name):
                continue

            # Skip type annotation subtrees - never mutate type definitions
            if field_name in ("annotation", "returns"):
                continue

            # Skip range() arguments - bounds define loop semantics
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "range"
                and field_name in ("args", "keywords")
            ):
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

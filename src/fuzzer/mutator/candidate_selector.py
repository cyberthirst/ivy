import random
from typing import Type

from vyper.ast import nodes as ast


class CandidateSelector:
    """
    Selects mutation targets via weighted sampling.
    Without replacement:  each node can be selected at most once.
    """

    def __init__(self, rng: random.Random, prob_map: dict[Type, float]):
        self.rng = rng
        self.prob_map = prob_map

    def select(self, root: ast.VyperNode, max_mutations: int) -> set[int]:
        """Walk tree, collect candidates, return set of node IDs to mutate."""
        candidates: list[ast.VyperNode] = []
        weights: list[float] = []

        for node in self._walk(root):
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

    def _walk(self, node: ast.VyperNode):
        """Yield all nodes in the AST (preorder)."""
        if node is None:
            return
        yield node
        for field_name in node.get_fields():
            if not hasattr(node, field_name):
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

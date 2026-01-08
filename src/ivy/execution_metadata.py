from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ivy.types import Address

BoolOp = Literal["and", "or"]


@dataclass
class ExecutionMetadata:
    coverage: dict[Address, set[int]] = field(default_factory=dict)
    edges: set[tuple[Address, int, int]] = field(default_factory=set)
    branches: set[tuple[Address, int, bool]] = field(default_factory=set)
    boolops: set[tuple[Address, int, BoolOp, int, bool]] = field(default_factory=set)
    loops: set[tuple[Address, int, int]] = field(default_factory=set)
    state_modified: bool = False

    def reset(self) -> None:
        self.coverage.clear()
        self.edges.clear()
        self.branches.clear()
        self.boolops.clear()
        self.loops.clear()
        self.state_modified = False

    def merge(self, other: ExecutionMetadata) -> None:
        for addr, node_ids in other.coverage.items():
            self.coverage.setdefault(addr, set()).update(node_ids)
        self.edges |= other.edges
        self.branches |= other.branches
        self.boolops |= other.boolops
        self.loops |= other.loops
        self.state_modified |= other.state_modified

    def coverage_signature(self) -> int:
        coverage_items = frozenset(
            (addr, frozenset(node_ids)) for addr, node_ids in self.coverage.items()
        )
        signature = (
            coverage_items,
            frozenset(self.edges),
            frozenset(self.branches),
            frozenset(self.boolops),
            frozenset(self.loops),
            self.state_modified,
        )
        return hash(signature)

    def record_node(self, addr: Address, node_id: int) -> None:
        self.coverage.setdefault(addr, set()).add(node_id)

    def record_edge(self, addr: Address, prev_node_id: int, node_id: int) -> None:
        self.edges.add((addr, prev_node_id, node_id))

    def record_branch(self, addr: Address, node_id: int, taken: bool) -> None:
        self.branches.add((addr, node_id, taken))

    def record_boolop(
        self,
        addr: Address,
        node_id: int,
        op: BoolOp,
        evaluated_count: int,
        result: bool,
    ) -> None:
        self.boolops.add((addr, node_id, op, evaluated_count, result))

    def record_loop(self, addr: Address, node_id: int, bucket: int) -> None:
        self.loops.add((addr, node_id, bucket))

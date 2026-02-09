from __future__ import annotations

from typing import TYPE_CHECKING

from ivy.execution_metadata import ExecutionMetadata

if TYPE_CHECKING:
    from vyper.ast import nodes as ast

    from ivy.types import Address


class Tracer:
    def on_node(self, addr: Address, node: ast.VyperNode) -> None:
        pass

    def on_edge(self, addr: Address, prev_node_id: int, node_id: int) -> None:
        pass

    def on_branch(self, addr: Address, node: ast.VyperNode, taken: bool) -> None:
        pass

    def on_boolop(
        self,
        addr: Address,
        node: ast.BoolOp,
        op: str,
        evaluated_count: int,
        result: bool,
    ) -> None:
        pass

    def on_loop(self, addr: Address, node: ast.For, iteration_count: int) -> None:
        pass

    def on_state_modified(self) -> None:
        pass


def iteration_bucket(count: int) -> int:
    if count <= 0:
        return 0
    if count <= 3:
        return count
    if count < 8:
        return 4
    if count < 16:
        return 5
    if count < 32:
        return 6
    if count < 64:
        return 7
    if count < 128:
        return 8
    return 9


class CoverageTracer(Tracer):
    def __init__(self):
        self.metadata = ExecutionMetadata()

    def reset(self) -> None:
        self.metadata.reset()

    @staticmethod
    def _source_id_for_node(node: ast.VyperNode) -> int:
        module_node = node.module_node
        if module_node is None:
            return -1
        source_id = getattr(module_node, "source_id", None)
        if source_id is None:
            return -1
        return int(source_id)

    def on_node(self, addr, node) -> None:
        self.metadata.record_node(addr, self._source_id_for_node(node), node.node_id)

    def on_edge(self, addr, prev_node_id, node_id) -> None:
        self.metadata.record_edge(addr, prev_node_id, node_id)

    def on_branch(self, addr, node, taken) -> None:
        self.metadata.record_branch(
            addr,
            self._source_id_for_node(node),
            node.node_id,
            taken,
        )

    def on_boolop(self, addr, node, op, evaluated_count, result) -> None:
        self.metadata.record_boolop(addr, node.node_id, op, evaluated_count, result)

    def on_loop(self, addr, node, iteration_count) -> None:
        self.metadata.record_loop(addr, node.node_id, iteration_bucket(iteration_count))

    def on_state_modified(self) -> None:
        self.metadata.state_modified = True

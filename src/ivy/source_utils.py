from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vyper.ast import nodes as ast


def source_id_for_node(node: ast.VyperNode) -> int:
    module_node = node.module_node
    if module_node is None:
        return -1
    source_id = getattr(module_node, "source_id", None)
    if source_id is None:
        return -1
    return int(source_id)

from enum import Enum

import vyper.ast.nodes as ast
from vyper.semantics.types.module import ModuleT
from vyper.semantics.types import BoolT

from ivy.constants import REENTRANT_KEY

OFFSET = 2**16


class Allocator:
    def __init__(self):
        self.immutables = 0 * OFFSET
        self.constants = 1 * OFFSET
        self.storage = 2 * OFFSET
        self.transient = 3 * OFFSET
        # TODO refactor this into write once dict
        self.addresses = {}

    def _get_allocatable(self, vyper_module: ast.Module) -> list[ast.VyperNode]:
        allocable = (ast.InitializesDecl, ast.VariableDecl)
        return [node for node in vyper_module.body if isinstance(node, allocable)]

    def allocate_nonreentrant_key(self):
        address = self.transient
        self.transient += 1
        return address

    def allocate_r(self, mod: ast.Module):
        nodes = self._get_allocatable(mod)
        for node in nodes:
            if isinstance(node, ast.InitializesDecl):
                module_info = node._metadata["initializes_info"].module_info
                self.allocate_r(module_info.module_node)
                continue

            assert isinstance(node, ast.VariableDecl)
            varinfo = node.target._metadata["varinfo"]

            if varinfo.is_constant:
                self.addresses[varinfo] = self.constants
                self.constants += 1
            elif varinfo.is_immutable:
                self.addresses[varinfo] = self.immutables
                self.immutables += 1
            elif varinfo.is_transient:
                self.addresses[varinfo] = self.transient
                self.transient += 1
            else:
                assert varinfo.is_storage
                varinfo.position = self.storage
                self.addresses[varinfo] = self.storage
                self.storage += 1

    def allocate_addresses(self, module_t: ModuleT):
        # allocate a nonreentrant key for all contracts, although they might not use it
        nonreentrant = self.allocate_nonreentrant_key()
        self.allocate_r(module_t.decl_node)
        self.verify_gaps()
        return nonreentrant, self.addresses

    # assert that all of the locations have at most 2**16 allocations
    def verify_gaps(self):
        assert self.immutables < 1 * OFFSET
        assert self.constants < 2 * OFFSET
        assert self.storage < 3 * OFFSET
        assert self.transient < 4 * OFFSET

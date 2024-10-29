from enum import Enum
import vyper.ast.nodes as ast
from vyper.semantics.types.module import ModuleT

OFFSET = 2**16


class StorageType(Enum):
    IMMUTABLE = 0
    CONSTANT = 1
    STORAGE = 2
    TRANSIENT = 3


class Allocator:
    def __init__(self):
        # Initialize counters dictionary with base offsets
        self.counters = {
            StorageType.IMMUTABLE: StorageType.IMMUTABLE.value * OFFSET,
            StorageType.CONSTANT: StorageType.CONSTANT.value * OFFSET,
            StorageType.STORAGE: StorageType.STORAGE.value * OFFSET,
            StorageType.TRANSIENT: StorageType.TRANSIENT.value * OFFSET,
        }
        self.visited = set()

    def _get_allocatable(self, vyper_module: ast.Module) -> list[ast.VyperNode]:
        allocable = (ast.InitializesDecl, ast.VariableDecl)
        return [node for node in vyper_module.body if isinstance(node, allocable)]

    def _increment_counter(self, storage_type: StorageType) -> int:
        current = self.counters[storage_type]
        self.counters[storage_type] += 1
        return current

    def allocate_nonreentrant_key(self):
        return self._increment_counter(StorageType.TRANSIENT)

    def allocate_r(self, mod: ast.Module):
        nodes = self._get_allocatable(mod)
        for node in nodes:
            if isinstance(node, ast.InitializesDecl):
                module_info = node._metadata["initializes_info"].module_info
                self.allocate_r(module_info.module_node)
                continue

            assert isinstance(node, ast.VariableDecl)
            varinfo = node.target._metadata["varinfo"]

            # sanity check
            assert varinfo not in self.visited

            if varinfo.is_constant:
                varinfo.position = self._increment_counter(StorageType.CONSTANT)
            elif varinfo.is_immutable:
                varinfo.position = self._increment_counter(StorageType.IMMUTABLE)
            elif varinfo.is_transient:
                varinfo.position = self._increment_counter(StorageType.TRANSIENT)
            else:
                assert varinfo.is_storage
                varinfo.position = self._increment_counter(StorageType.STORAGE)

            self.visited.add(varinfo)

    def allocate_addresses(self, module_t: ModuleT):
        # allocate a nonreentrant key for all contracts, although they might not use it
        nonreentrant = self.allocate_nonreentrant_key()
        self.allocate_r(module_t.decl_node)
        self.verify_gaps()
        return nonreentrant, self.visited

    def verify_gaps(self):
        for storage_type in StorageType:
            assert self.counters[storage_type] < (storage_type.value + 1) * OFFSET

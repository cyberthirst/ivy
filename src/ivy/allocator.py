import vyper.ast.nodes as ast
from vyper.semantics.types.module import ModuleT
from vyper.semantics.data_locations import DataLocation

IGNORED_LOCATIONS = (DataLocation.CALLDATA,)


class Allocator:
    def __init__(self):
        # NOTE: maybe from the delegatecall persepective it would be better to make
        # constants and immutables affect the same counter
        self.counters = {
            location: 0
            for location in DataLocation
            if location not in IGNORED_LOCATIONS
        }
        self.visited = set()

    def _get_allocatable(self, vyper_module: ast.Module) -> list[ast.VyperNode]:
        allocable = (ast.InitializesDecl, ast.VariableDecl)
        return [node for node in vyper_module.body if isinstance(node, allocable)]

    def _increment_counter(self, location: DataLocation) -> int:
        current = self.counters[location]
        self.counters[location] += 1
        return current

    def allocate_nonreentrant_key(self):
        return self._increment_counter(DataLocation.TRANSIENT)

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
            assert varinfo.is_state_variable or varinfo.is_constant

            varinfo.position = self._increment_counter(varinfo.location)
            self.visited.add(varinfo)

    def allocate_addresses(self, module_t: ModuleT):
        nonreentrant = self.allocate_nonreentrant_key()
        self.allocate_r(module_t.decl_node)
        return nonreentrant, self.visited

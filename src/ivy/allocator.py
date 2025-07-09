import vyper.ast.nodes as ast
from vyper.semantics.analysis.base import ModuleInfo
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
        self.visited = {}

    def _get_allocatable(self, vyper_module: ast.Module) -> list[ast.VyperNode]:
        allocable = (ast.InitializesDecl, ast.VariableDecl)
        return [node for node in vyper_module.body if isinstance(node, allocable)]

    def _increment_counter(self, location: DataLocation) -> int:
        current = self.counters[location]
        self.counters[location] += 1
        return current

    def allocate_nonreentrant_key(self):
        return self._increment_counter(DataLocation.TRANSIENT)

    def _allocate_var(self, node, allocate_constants=False):
        assert isinstance(node, ast.VariableDecl)
        varinfo = node.target._metadata["varinfo"]
        if varinfo.is_constant and not allocate_constants:
            return
        if not varinfo.is_constant and allocate_constants:
            return

        # sanity check
        assert varinfo not in self.visited
        assert varinfo.is_state_variable

        varinfo.position = self._increment_counter(varinfo.location)
        self.visited[varinfo] = True

    def _allocate_r(self, mod: ast.Module):
        nodes = self._get_allocatable(mod)
        for node in nodes:
            if isinstance(node, ast.InitializesDecl):
                module_info = node._metadata["initializes_info"].module_info
                self._allocate_r(module_info.module_node)
                continue

            assert isinstance(node, ast.VariableDecl)
            self._allocate_var(node, allocate_constants=False)

    def _allocate_constants_r(self, vyper_module: ast.Module, visited: set):
        """
         Constants must be allocated even in modules which aren't initialized.
         Thus, for cleanliness, we do it in a separate function.

        Consider:
         # mod1.vy
         X: constant(uint256) = empty(uint256)

         # main.vy
         import mod1

         @external
         def foo() -> uint256:
             return mod1.X
        """
        if vyper_module in visited:
            return
        visited.add(vyper_module)

        decls = [
            node for node in vyper_module.body if isinstance(node, ast.VariableDecl)
        ]

        for d in decls:
            assert isinstance(d, ast.VariableDecl)
            self._allocate_var(d, allocate_constants=True)

        import_nodes = (ast.Import, ast.ImportFrom)
        imports = [node for node in vyper_module.body if isinstance(node, import_nodes)]

        for node in imports:
            import_info = node._metadata["import_info"]
            # could be an interface
            if isinstance(import_info.typ, ModuleInfo):
                typ = import_info.typ.module_t
                self._allocate_constants_r(typ.decl_node, visited)

    def allocate_addresses(self, module_t: ModuleT):
        nonreentrant = self.allocate_nonreentrant_key()
        self._allocate_r(module_t.decl_node)
        self._allocate_constants_r(module_t.decl_node, set())
        return nonreentrant, self.visited

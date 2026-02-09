from __future__ import annotations

from types import SimpleNamespace

import vyper.ast as vy_ast
from vyper.compiler.input_bundle import BUILTIN

from fuzzer.runtime_engine.runtime_fuzz_engine import HarnessConfig, RuntimeFuzzEngine


class _CompilerInputKey:
    def __init__(self, source_id: int):
        self.source_id = source_id


def _make_module(source: str, *, source_id: int, resolved_path: str):
    module = vy_ast.parse_to_ast(source)
    module.source_id = source_id
    module.resolved_path = resolved_path
    return module


def _offset_node_ids(engine: RuntimeFuzzEngine, module, offset: int) -> None:
    for node in engine._iter_nodes(module):
        node_id = getattr(node, "node_id", None)
        if node_id is not None:
            node.node_id = node_id + offset


def _collect_sites(engine: RuntimeFuzzEngine, root_module, compiler_inputs):
    contract = SimpleNamespace(
        address="0x123",
        compiler_data=SimpleNamespace(
            annotated_vyper_module=root_module,
            resolved_imports=SimpleNamespace(compiler_inputs=compiler_inputs),
        ),
    )
    return engine._collect_static_coverage_sites_for_contract(contract)


def test_static_coverage_sites_include_non_interface_non_builtin_imports():
    engine = RuntimeFuzzEngine(HarnessConfig(), seed=1)

    root_module = _make_module(
        """
@external
def root_fn(x: uint256) -> uint256:
    if x > 0:
        return x
    assert x == 0
    return 1
        """,
        source_id=100,
        resolved_path="main.vy",
    )
    import_module = _make_module(
        """
@internal
def lib_fn(y: uint256) -> uint256:
    z: uint256 = y
    return z
        """,
        source_id=101,
        resolved_path="lib.vy",
    )
    interface_module = _make_module(
        """
@external
def ping() -> uint256:
    ...
        """,
        source_id=102,
        resolved_path="iface.vyi",
    )
    interface_module.is_interface = True
    builtin_module = _make_module(
        """
@internal
def helper(a: uint256) -> uint256:
    return a
        """,
        source_id=103,
        resolved_path="builtin.vy",
    )

    # Keep node IDs disjoint between modules so set membership reflects module inclusion.
    _offset_node_ids(engine, import_module, 1000)
    _offset_node_ids(engine, interface_module, 2000)
    _offset_node_ids(engine, builtin_module, 3000)

    root_stmt, root_branch = _collect_sites(engine, root_module, {})
    import_stmt, import_branch = _collect_sites(engine, import_module, {})

    combined_stmt, combined_branch = _collect_sites(
        engine,
        root_module,
        {
            _CompilerInputKey(source_id=101): import_module,
            # Duplicate module should be de-duplicated via source_id/resolved_path.
            _CompilerInputKey(source_id=999): import_module,
            _CompilerInputKey(source_id=102): interface_module,
            _CompilerInputKey(source_id=BUILTIN): builtin_module,
        },
    )

    assert combined_stmt == root_stmt | import_stmt
    assert combined_branch == root_branch | import_branch


def test_static_coverage_sites_use_source_id_with_node_id():
    engine = RuntimeFuzzEngine(HarnessConfig(), seed=1)

    root_module = _make_module(
        """
@external
def root_fn(x: uint256) -> uint256:
    if x > 0:
        return x
    return 0
        """,
        source_id=200,
        resolved_path="main.vy",
    )
    import_module = _make_module(
        """
@internal
def lib_fn(y: uint256) -> uint256:
    if y > 0:
        return y
    return 1
        """,
        source_id=201,
        resolved_path="lib.vy",
    )

    # Keep node IDs overlapping to verify source_id disambiguation.
    root_stmt, root_branch = _collect_sites(engine, root_module, {})
    import_stmt, import_branch = _collect_sites(engine, import_module, {})
    combined_stmt, combined_branch = _collect_sites(
        engine,
        root_module,
        {_CompilerInputKey(source_id=201): import_module},
    )

    root_stmt_ids = {(source_id, node_id) for _, source_id, node_id in root_stmt}
    import_stmt_ids = {(source_id, node_id) for _, source_id, node_id in import_stmt}
    assert root_stmt_ids
    assert import_stmt_ids
    # node_id overlap can exist; identity must still differ by source_id.
    overlapping_node_ids = {node_id for _, node_id in root_stmt_ids} & {
        node_id for _, node_id in import_stmt_ids
    }
    assert overlapping_node_ids
    assert root_stmt_ids.isdisjoint(import_stmt_ids)

    assert combined_stmt == root_stmt | import_stmt
    assert combined_branch == root_branch | import_branch

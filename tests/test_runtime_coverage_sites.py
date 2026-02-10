from __future__ import annotations

import vyper.ast as vy_ast

from fuzzer.runtime_engine.runtime_fuzz_engine import HarnessConfig, RuntimeFuzzEngine


_STMT_NODE_TYPE = getattr(vy_ast, "Stmt")


def _source_id_for_node(node: vy_ast.VyperNode) -> int:
    module_node = node.module_node
    if module_node is None:
        return -1
    source_id = getattr(module_node, "source_id", None)
    if source_id is None:
        return -1
    return int(source_id)


def _stmt_sites_for_function(
    address: str,
    fn_t,
) -> set[tuple[str, int, int]]:
    decl_node = getattr(fn_t, "decl_node", None)
    if decl_node is None:
        return set()

    stmt_sites: set[tuple[str, int, int]] = set()
    for node in decl_node.get_descendants(_STMT_NODE_TYPE):
        node_id = getattr(node, "node_id", None)
        if node_id is None:
            continue
        stmt_sites.add((address, _source_id_for_node(node), int(node_id)))
    return stmt_sites


def _branch_outcomes_for_function(
    address: str,
    fn_t,
) -> set[tuple[str, int, int, bool]]:
    decl_node = getattr(fn_t, "decl_node", None)
    if decl_node is None:
        return set()

    branch_outcomes: set[tuple[str, int, int, bool]] = set()
    for node in decl_node.get_descendants((vy_ast.If, vy_ast.Assert, vy_ast.IfExp)):
        node_id = getattr(node, "node_id", None)
        if node_id is None:
            continue
        source_id = _source_id_for_node(node)
        branch_outcomes.add((address, source_id, int(node_id), True))
        branch_outcomes.add((address, source_id, int(node_id), False))
    return branch_outcomes


def test_static_coverage_sites_skip_constructors_and_init_only_functions(get_contract):
    source = """
x: uint256

@deploy
def __init__(a: uint256):
    self.x = a
    self.init_only(a)

@internal
def init_only(a: uint256):
    if a > 0:
        self.x = a + 1

@external
def run(a: uint256):
    self.runtime_fn(a)

@internal
def runtime_fn(a: uint256):
    if a > 1:
        self.x = a
    """
    contract = get_contract(source, 1)
    engine = RuntimeFuzzEngine(HarnessConfig(), seed=1)

    stmt_sites, branch_outcomes = engine._collect_static_coverage_sites_for_contract(
        contract
    )
    address = str(contract.address)
    module_t = contract.compiler_data.global_ctx

    run_fn = next(fn for fn in module_t.exposed_functions if fn.name == "run")
    runtime_fn = module_t.functions["runtime_fn"]
    init_fn = module_t.init_function
    init_only_fn = module_t.functions["init_only"]

    expected_runtime_stmt = _stmt_sites_for_function(
        address, run_fn
    ) | _stmt_sites_for_function(address, runtime_fn)
    expected_runtime_branches = _branch_outcomes_for_function(
        address, run_fn
    ) | _branch_outcomes_for_function(address, runtime_fn)

    assert expected_runtime_stmt <= stmt_sites
    assert expected_runtime_branches <= branch_outcomes

    assert _stmt_sites_for_function(address, init_fn).isdisjoint(stmt_sites)
    assert _branch_outcomes_for_function(address, init_fn).isdisjoint(branch_outcomes)
    assert _stmt_sites_for_function(address, init_only_fn).isdisjoint(stmt_sites)
    assert _branch_outcomes_for_function(address, init_only_fn).isdisjoint(
        branch_outcomes
    )


def test_static_coverage_sites_include_reachable_internal_imports_only(
    get_contract, make_input_bundle
):
    main = """
import lib1
initializes: lib1

@external
def run(a: uint256) -> uint256:
    return lib1.used(a)
    """
    lib1 = """
@internal
def used(a: uint256) -> uint256:
    if a > 0:
        return a
    return 0

@internal
def dead(a: uint256) -> uint256:
    if a == 1:
        return 1
    return 2
    """
    contract = get_contract(main, input_bundle=make_input_bundle({"lib1.vy": lib1}))
    engine = RuntimeFuzzEngine(HarnessConfig(), seed=1)

    stmt_sites, branch_outcomes = engine._collect_static_coverage_sites_for_contract(
        contract
    )
    address = str(contract.address)
    root_module_t = contract.compiler_data.global_ctx
    lib_module_t = root_module_t.imported_modules["lib1"].module_t

    run_fn = next(fn for fn in root_module_t.exposed_functions if fn.name == "run")
    used_fn = lib_module_t.functions["used"]
    dead_fn = lib_module_t.functions["dead"]

    assert _stmt_sites_for_function(address, run_fn) <= stmt_sites
    assert _stmt_sites_for_function(address, used_fn) <= stmt_sites
    assert _branch_outcomes_for_function(address, run_fn) <= branch_outcomes
    assert _branch_outcomes_for_function(address, used_fn) <= branch_outcomes

    assert _stmt_sites_for_function(address, dead_fn).isdisjoint(stmt_sites)
    assert _branch_outcomes_for_function(address, dead_fn).isdisjoint(branch_outcomes)


def test_static_coverage_sites_track_source_id_per_module(
    get_contract, make_input_bundle
):
    main = """
import lib1
initializes: lib1

@external
def run(a: uint256) -> uint256:
    if a > 10:
        return lib1.used(a)
    return 0
    """
    lib1 = """
@internal
def used(a: uint256) -> uint256:
    if a > 0:
        return a
    return 1
    """
    contract = get_contract(main, input_bundle=make_input_bundle({"lib1.vy": lib1}))
    engine = RuntimeFuzzEngine(HarnessConfig(), seed=1)

    stmt_sites, branch_outcomes = engine._collect_static_coverage_sites_for_contract(
        contract
    )
    root_module_t = contract.compiler_data.global_ctx
    run_fn = next(fn for fn in root_module_t.exposed_functions if fn.name == "run")
    lib_fn = root_module_t.imported_modules["lib1"].module_t.functions["used"]

    run_source_id = _source_id_for_node(run_fn.decl_node)
    lib_source_id = _source_id_for_node(lib_fn.decl_node)

    stmt_source_ids = {source_id for _, source_id, _ in stmt_sites}
    branch_source_ids = {source_id for _, source_id, _, _ in branch_outcomes}

    assert run_source_id in stmt_source_ids
    assert lib_source_id in stmt_source_ids
    assert run_source_id in branch_source_ids
    assert lib_source_id in branch_source_ids

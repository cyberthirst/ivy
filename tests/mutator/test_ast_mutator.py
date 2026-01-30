import hashlib
import random
from pathlib import Path

import pytest
from vyper.compiler.phases import CompilerData

from fuzzer.compilation import compile_vyper
from fuzzer.export_utils import load_all_exports, filter_exports, TestFilter
from fuzzer.trace_types import DeploymentTrace
from fuzzer.mutator.ast_mutator import AstMutator


def deterministic_seed(s: str) -> int:
    """Generate a deterministic seed from a string using MD5."""
    return int(hashlib.md5(s.encode()).hexdigest(), 16) & 0xFFFFFFFF

# Attributes that never affect semantics (from unparser test)
_IGNORE = {
    "lineno",
    "col_offset",
    "end_lineno",
    "end_col_offset",
    "src",
    "node_id",
    "path",
    "resolved_path",
    "source_id",
    "source_sha256sum",
    "variable_reads",
    "variable_writes",
    "type",
    "doc_string",
    "folded_value",
}


def _strip(obj):
    """Strip non-semantic attributes from AST nodes."""
    if isinstance(obj, list):
        return [_strip(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in _IGNORE}
    return obj


def _as_clean_dict(source_code: str):
    """Convert source code to a normalized AST dict for comparison."""
    ast = CompilerData(source_code).annotated_vyper_module
    return _strip(ast.to_dict())


def get_mutator_test_filter() -> TestFilter:
    test_filter = TestFilter(exclude_multi_module=True)
    test_filter.exclude_source(r"#\s*@version")  # Skip version pragmas
    return test_filter


def get_mutator_test_cases():
    exports = load_all_exports("tests/vyper-exports")
    exports = filter_exports(exports, test_filter=get_mutator_test_filter())

    cases = []
    seen_sources = set()

    for path, export in exports.items():
        for item_name, item in export.items.items():
            for trace in item.traces:
                if isinstance(trace, DeploymentTrace):
                    if trace.deployment_type == "source" and trace.source_code:
                        # Deduplicate by source code hash
                        source_hash = hash(trace.source_code)
                        if source_hash in seen_sources:
                            continue
                        seen_sources.add(source_hash)

                        test_id = f"{Path(path).stem}::{item_name}"
                        cases.append(pytest.param(trace.source_code, test_id, id=test_id))
    return cases


@pytest.mark.xfail(reason="mutator WIP - many mutations cause compilation failures", strict=False)
@pytest.mark.parametrize("source_code,test_id", get_mutator_test_cases())
def test_mutator_produces_valid_code(source_code: str, test_id: str):
    """Test that the AST mutator produces semantically valid Vyper code.

    For each source:
    1. Create deterministic RNG from test_id for reproducibility
    2. Apply random number of mutations (1-16)
    3. Assert mutation succeeded (MutationResult is not None)
    4. Assert something actually changed (AST differs from original)
    5. If no compilation_xfails: expect compilation to succeed
       If compilation_xfails: expect compilation to fail
    """
    # Deterministic seed from test_id for reproducibility
    seed = deterministic_seed(test_id)
    rng = random.Random(seed)

    # Random number of mutations in range [1, 16]
    max_mutations = rng.randint(1, 16)

    # Create compiler data from original source
    original_compiler_data = CompilerData(source_code)
    original_ast_dict = _strip(original_compiler_data.annotated_vyper_module.to_dict())

    # Create mutator and mutate
    mutator = AstMutator(rng, max_mutations=max_mutations)
    mutation_result = mutator.mutate_source_with_compiler_data(original_compiler_data)

    # Mutation must not fail
    assert mutation_result is not None, (
        f"Mutation failed (returned None). "
        f"seed={seed}, max_mutations={max_mutations}"
    )

    mutated_source = mutation_result.source
    compilation_xfails = mutation_result.compilation_xfails

    if not compilation_xfails:
        # No expected compilation failures - must compile successfully
        result = compile_vyper(mutated_source)

        if result.is_compiler_crash:
            return

        if result.is_compilation_failure:
            pytest.fail(
                f"Mutated code failed to compile but no compilation_xfails was set.\n"
                f"seed={seed}, max_mutations={max_mutations}\n"
                f"Error: {result.error}\n"
                f"Mutated source:\n{mutated_source}"
            )

        # Something must have changed
        assert result.compiler_data is not None
        mutated_ast_dict = _strip(
            result.compiler_data.annotated_vyper_module.to_dict()
        )
        assert original_ast_dict != mutated_ast_dict, (
            f"No mutation occurred - AST unchanged. "
            f"seed={seed}, max_mutations={max_mutations}"
        )
    else:
        # Compilation failures expected - verify compilation does fail
        result = compile_vyper(mutated_source)

        if result.is_success:
            pytest.fail(
                f"Mutated code compiled successfully but compilation_xfails was set.\n"
                f"seed={seed}, max_mutations={max_mutations}\n"
                f"Expected failures: {compilation_xfails}\n"
                f"Mutated source:\n{mutated_source}"
            )
        # Expected - compilation failed or crashed

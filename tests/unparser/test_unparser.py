from pathlib import Path

import pytest
import vyper
from vyper.compiler.phases import CompilerData

from unparser.unparser import unparse
from fuzzer.export_utils import load_all_exports, filter_exports, TestFilter
from fuzzer.trace_types import DeploymentTrace

# attributes that never affect semantics
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
    # analysis caches
    "variable_reads",
    "variable_writes",
    "type",
    "doc_string",
    "settings",  # compiler pragmas
    "folded_value",  # constant folding cache
}


def _strip(obj):
    """Strip non-semantic attributes from AST nodes."""
    if isinstance(obj, list):
        return [_strip(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in _IGNORE}
    return obj


def _as_clean_dict(code: str) -> dict:
    """Convert source code to a normalized AST dict for comparison."""
    ast = CompilerData(code).annotated_vyper_module
    return _strip(ast.to_dict())


def get_unparser_test_filter() -> TestFilter:
    test_filter = TestFilter(exclude_multi_module=True)
    test_filter.exclude_source(r"#\s*@version")  # Skip version pragmas
    test_filter.exclude_source(r"@nonreentrant")  # Skip nonreentrant for now
    test_filter.exclude_source(r"@reentrant")  # Skip @reentrant decorator (requires pragma)
    test_filter.exclude_source(r"reentrant\(")  # Skip reentrant() flag (requires pragma)
    return test_filter


def get_unparser_test_cases():
    exports = load_all_exports("tests/vyper-exports")
    exports = filter_exports(exports, test_filter=get_unparser_test_filter())

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
                        cases.append(pytest.param(trace.source_code, id=test_id))
    return cases


@pytest.mark.parametrize("source_code", get_unparser_test_cases())
def test_unparser(source_code):
    """Test that unparsing Vyper source produces semantically equivalent code."""
    # Parse the original source
    original_ast = vyper.ast.parse_to_ast(source_code)

    # Unparse it back to source
    roundtrip_source = unparse(original_ast)

    # Compare normalized ASTs
    original_dict = _as_clean_dict(source_code)
    roundtrip_dict = _as_clean_dict(roundtrip_source)

    assert original_dict == roundtrip_dict, (
        f"AST mismatch after roundtrip.\n"
        f"Original:\n{source_code[:500]}...\n"
        f"Roundtrip:\n{roundtrip_source[:500]}..."
    )

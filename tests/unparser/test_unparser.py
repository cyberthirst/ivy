from pathlib import Path

import pytest
from vyper.compiler.phases import CompilerData
from vyper.compiler.input_bundle import FileInput

from unparser.unparser import unparse
from ivy.frontend.loader import loads_from_solc_json
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
    "folded_value",  # constant folding cache
}


def _strip(obj):
    """Strip non-semantic attributes from AST nodes."""
    if isinstance(obj, list):
        return [_strip(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in _IGNORE}
    return obj


def _as_clean_dict(
    code: str,
    *,
    file_input: FileInput | None = None,
    input_bundle=None,
) -> dict:
    """Convert source code to a normalized AST dict for comparison."""
    if file_input is None:
        ast = CompilerData(code).annotated_vyper_module
    else:
        roundtrip_input = FileInput(
            source_id=file_input.source_id,
            path=file_input.path,
            resolved_path=file_input.resolved_path,
            contents=code,
        )
        ast = CompilerData(roundtrip_input, input_bundle).annotated_vyper_module
    return _strip(ast.to_dict())


def get_unparser_test_filter() -> TestFilter:
    test_filter = TestFilter()
    test_filter.exclude_source(r"#\s*@version")  # Skip version pragmas
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
                    if trace.deployment_type == "source" and trace.solc_json:
                        integrity = trace.solc_json.get("integrity")
                        if not isinstance(integrity, str):
                            raise ValueError("solc_json missing integrity")
                        if integrity in seen_sources:
                            continue
                        seen_sources.add(integrity)

                        test_id = f"{Path(path).stem}::{item_name}"
                        cases.append(pytest.param(trace.solc_json, id=test_id))
    return cases


@pytest.mark.parametrize("solc_json", get_unparser_test_cases())
def test_unparser(solc_json):
    """Test that unparsing Vyper source produces semantically equivalent code."""
    compiler_data = loads_from_solc_json(solc_json, get_compiler_data=True)
    original_ast = compiler_data.annotated_vyper_module

    # Unparse it back to source
    roundtrip_source = unparse(original_ast)

    # Compare normalized ASTs
    original_dict = _strip(original_ast.to_dict())
    roundtrip_dict = _as_clean_dict(
        roundtrip_source,
        file_input=compiler_data.file_input,
        input_bundle=compiler_data.input_bundle,
    )

    assert original_dict == roundtrip_dict, (
        f"AST mismatch after roundtrip.\n"
        f"Original:\n{compiler_data.file_input.contents[:500]}...\n"
        f"Roundtrip:\n{roundtrip_source[:500]}..."
    )

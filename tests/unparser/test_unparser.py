from pathlib import Path

import vyper
from vyper.compiler.phases import CompilerData

from unparser.unparser import unparse
from fuzzer.export_utils import load_all_exports, extract_test_cases, TestFilter

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


def test_unparser():
    """Test that unparsing Vyper test exports produces semantically equivalent code."""
    # Load test exports
    exports_dir = Path("tests/vyper-exports")
    if not exports_dir.exists():
        print(f"Warning: Test exports directory not found at {exports_dir}")
        return

    # Create filter to only test source deployments
    test_filter = TestFilter()
    # Skip tests with features that might not roundtrip perfectly
    test_filter.exclude_source(r"#\s*@version")  # Skip version pragmas
    test_filter.exclude_source(r"@nonreentrant")  # Skip nonreentrant for now

    # Load and extract test cases
    exports = load_all_exports(exports_dir)
    test_cases = extract_test_cases(exports)

    print(f"Testing unparser roundtrip on {len(test_cases)} test cases...")

    passed = 0
    failed = 0

    for i, (source_code, _) in enumerate(test_cases):
        try:
            # Parse the original source
            original_ast = vyper.ast.parse_to_ast(source_code)

            # Unparse it back to source
            roundtrip_source = unparse(original_ast)

            # Compare normalized ASTs
            original_dict = _as_clean_dict(source_code)
            roundtrip_dict = _as_clean_dict(roundtrip_source)

            if original_dict == roundtrip_dict:
                passed += 1
            else:
                failed += 1
                print(f"\nFailed roundtrip for test case {i + 1}:")
                print(f"Original source:\n{source_code[:200]}...")
                print(f"Roundtrip source:\n{roundtrip_source[:200]}...")

        except Exception as e:
            failed += 1
            print(f"\nError in test case {i + 1}: {e}")
            print(f"Source:\n{source_code[:200]}...")

    print(f"\n=== Unparser Test Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")

    # Assert all tests passed
    assert failed == 0, f"{failed} tests failed"


if __name__ == "__main__":
    test_unparser()

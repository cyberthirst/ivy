from vyper.compiler.phases import CompilerData
from unparser.unparser import unparse

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
    if isinstance(obj, list):
        return [_strip(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in _IGNORE}
    return obj


def _as_clean_dict(code: str) -> dict:
    ast = CompilerData(code).annotated_vyper_module
    return _strip(ast.to_dict())


def test_unparser():
    counter = 0
    #for rec in contract_dumper.load_records():
    #    original = rec["source"]
    #    roundtrip = unparse(CompilerData(original).annotated_vyper_module)
    #    print(original)
    #    print("============================")
    #    print(roundtrip)
    #    assert _as_clean_dict(original) == _as_clean_dict(roundtrip)
    #    counter += 1
    #    print("passed: ", counter)

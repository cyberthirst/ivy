import copy

from vyper.ast import nodes as ast
from vyper.semantics.types.shortcuts import UINT256_T
from vyper.semantics.analysis.base import VarInfo, DataLocation, Modifiability

from fuzzer.compilation import compile_vyper
from fuzzer.mutator.ast_utils import hoist_prelude_decls
from unparser.unparser import unparse


def _hoist_local(expr, name, hoist_seq):
    """Create a hoisted local variable, mimicking ExprGenerator.hoist_to_tmp_var."""
    expr_type = getattr(expr, "_metadata", {}).get("type", UINT256_T)
    var_info = VarInfo(
        typ=expr_type,
        location=DataLocation.MEMORY,
        modifiability=Modifiability.MODIFIABLE,
    )
    decl = ast.AnnAssign(
        target=ast.Name(id=name),
        annotation=ast.Name(id=str(expr_type)),
        value=expr,
    )
    ref = ast.Name(id=name)
    ref._metadata = {
        "type": expr_type,
        "varinfo": var_info,
        "hoisted_prelude": decl,
        "hoist_seq": hoist_seq,
    }
    return ref


def _hoist_constant(expr, name, hoist_seq):
    """Create a hoisted constant, mimicking ExprGenerator.hoist_to_tmp_var for module scope."""
    expr_type = getattr(expr, "_metadata", {}).get("type", UINT256_T)
    var_info = VarInfo(
        typ=expr_type,
        location=DataLocation.UNSET,
        modifiability=Modifiability.CONSTANT,
    )
    annotation = ast.Call(
        func=ast.Name(id="constant"),
        args=[ast.Name(id=str(expr_type))],
    )
    decl = ast.VariableDecl(
        target=ast.Name(id=name),
        annotation=annotation,
        value=expr,
    )
    ref = ast.Name(id=name)
    ref._metadata = {
        "type": expr_type,
        "varinfo": var_info,
        "hoisted_prelude": decl,
        "hoist_seq": hoist_seq,
    }
    return ref


def test_hoist_prelude_multiple_locations():
    source = """
CONST_VAL: constant(uint256) = 1 + 2

@external
def foo(x: uint256, y: uint256):
    for i: uint256 in range(10):
        a: uint256 = x + y
    if x > y:
        b: uint256 = x
    else:
        c: uint256 = x + y
"""
    result = compile_vyper(source)
    assert result.is_success
    module = copy.deepcopy(result.compiler_data.annotated_vyper_module)

    seq = 0

    # 1. Constant expr: hoist the right operand of `1 + 2`
    const_decl = module.body[0]
    assert isinstance(const_decl, ast.VariableDecl)
    assert isinstance(const_decl.value, ast.BinOp), (
        f"Expected BinOp, got {type(const_decl.value).__name__}"
    )
    ref = _hoist_constant(const_decl.value.right, "gen_var_0", seq)
    seq += 1
    const_decl.value.right = ref

    # 2. For loop iterable: hoist `10` from range(10)
    func_def = module.body[1]
    assert isinstance(func_def, ast.FunctionDef)
    for_stmt = func_def.body[0]
    assert isinstance(for_stmt, ast.For)
    range_call = for_stmt.iter
    assert isinstance(range_call, ast.Call)
    ref = _hoist_local(range_call.args[0], "gen_var_1", seq)
    seq += 1
    range_call.args[0] = ref

    # 3. Loop body: hoist `y` from `x + y`
    ann_in_loop = for_stmt.body[0]
    assert isinstance(ann_in_loop, ast.AnnAssign)
    assert isinstance(ann_in_loop.value, ast.BinOp)
    ref = _hoist_local(ann_in_loop.value.right, "gen_var_2", seq)
    seq += 1
    ann_in_loop.value.right = ref

    # 4. If test: hoist `y` from `x > y`
    if_stmt = func_def.body[1]
    assert isinstance(if_stmt, ast.If)
    assert isinstance(if_stmt.test, ast.Compare)
    ref = _hoist_local(if_stmt.test.right, "gen_var_3", seq)
    seq += 1
    if_stmt.test.right = ref

    # 5. Else body: hoist `y` from `x + y`
    ann_in_else = if_stmt.orelse[0]
    assert isinstance(ann_in_else, ast.AnnAssign)
    assert isinstance(ann_in_else.value, ast.BinOp)
    ref = _hoist_local(ann_in_else.value.right, "gen_var_4", seq)
    seq += 1
    ann_in_else.value.right = ref

    hoist_prelude_decls(module)
    result_source = unparse(module)

    expected = """\
gen_var_0: constant(uint256) = 2

CONST_VAL: constant(uint256) = 1 + gen_var_0

@external
def foo(x: uint256, y: uint256):
    gen_var_1: uint256 = 10
    for i: uint256 in range(gen_var_1):
        gen_var_2: uint256 = y
        a: uint256 = x + gen_var_2
    gen_var_3: uint256 = y
    if x > gen_var_3:
        b: uint256 = x
    else:
        gen_var_4: uint256 = y
        c: uint256 = x + gen_var_4
"""
    assert result_source == expected


def test_hoist_seq_ordering():
    """Two hoists on the same statement: right operand gets lower seq than left.
    Verifies the hoist_seq sort produces correct declaration order regardless
    of tree traversal order."""
    source = """
@external
def foo(x: uint256, y: uint256) -> uint256:
    return x + y
"""
    result = compile_vyper(source)
    assert result.is_success
    module = copy.deepcopy(result.compiler_data.annotated_vyper_module)

    func_def = module.body[0]
    assert isinstance(func_def, ast.FunctionDef)
    ret_stmt = func_def.body[0]
    assert isinstance(ret_stmt, ast.Return)
    binop = ret_stmt.value
    assert isinstance(binop, ast.BinOp)

    # Give right operand the LOWER seq to prove sorting wins over traversal order
    ref_right = _hoist_local(binop.right, "gen_var_0", hoist_seq=0)
    ref_left = _hoist_local(binop.left, "gen_var_1", hoist_seq=1)
    binop.right = ref_right
    binop.left = ref_left

    hoist_prelude_decls(module)
    result_source = unparse(module)

    expected = """\
@external
def foo(x: uint256, y: uint256) -> uint256:
    gen_var_0: uint256 = y
    gen_var_1: uint256 = x
    return gen_var_1 + gen_var_0
"""
    assert result_source == expected


def test_hoist_nested_control_flow():
    """Hoists at three nesting levels: function body → if body → for body.
    Each decl stays scoped to its own body and doesn't bubble up."""
    source = """
@external
def foo(x: uint256, y: uint256):
    if x > y:
        for i: uint256 in range(10):
            a: uint256 = x + y
"""
    result = compile_vyper(source)
    assert result.is_success
    module = copy.deepcopy(result.compiler_data.annotated_vyper_module)

    seq = 0

    func_def = module.body[0]
    assert isinstance(func_def, ast.FunctionDef)

    # Level 1 — function body: hoist `y` from if test `x > y`
    if_stmt = func_def.body[0]
    assert isinstance(if_stmt, ast.If)
    assert isinstance(if_stmt.test, ast.Compare)
    ref = _hoist_local(if_stmt.test.right, "gen_var_0", seq)
    seq += 1
    if_stmt.test.right = ref

    # Level 2 — if body: hoist `10` from range(10)
    for_stmt = if_stmt.body[0]
    assert isinstance(for_stmt, ast.For)
    range_call = for_stmt.iter
    assert isinstance(range_call, ast.Call)
    ref = _hoist_local(range_call.args[0], "gen_var_1", seq)
    seq += 1
    range_call.args[0] = ref

    # Level 3 — for body: hoist `y` from `x + y`
    ann_in_loop = for_stmt.body[0]
    assert isinstance(ann_in_loop, ast.AnnAssign)
    assert isinstance(ann_in_loop.value, ast.BinOp)
    ref = _hoist_local(ann_in_loop.value.right, "gen_var_2", seq)
    seq += 1
    ann_in_loop.value.right = ref

    hoist_prelude_decls(module)
    result_source = unparse(module)

    expected = """\
@external
def foo(x: uint256, y: uint256):
    gen_var_0: uint256 = y
    if x > gen_var_0:
        gen_var_1: uint256 = 10
        for i: uint256 in range(gen_var_1):
            gen_var_2: uint256 = y
            a: uint256 = x + gen_var_2
"""
    assert result_source == expected


def test_hoist_from_assert_and_return():
    """Hoisting from assert condition and return value — statement types
    not covered by the main test."""
    source = """
@external
def foo(x: uint256, y: uint256) -> uint256:
    assert x > y
    return x + y
"""
    result = compile_vyper(source)
    assert result.is_success
    module = copy.deepcopy(result.compiler_data.annotated_vyper_module)

    seq = 0

    func_def = module.body[0]
    assert isinstance(func_def, ast.FunctionDef)

    # 1. Assert: hoist `y` from `x > y`
    assert_stmt = func_def.body[0]
    assert isinstance(assert_stmt, ast.Assert)
    compare = assert_stmt.test
    assert isinstance(compare, ast.Compare)
    ref = _hoist_local(compare.right, "gen_var_0", seq)
    seq += 1
    compare.right = ref

    # 2. Return: hoist `y` from `x + y`
    ret_stmt = func_def.body[1]
    assert isinstance(ret_stmt, ast.Return)
    binop = ret_stmt.value
    assert isinstance(binop, ast.BinOp)
    ref = _hoist_local(binop.right, "gen_var_1", seq)
    seq += 1
    binop.right = ref

    hoist_prelude_decls(module)
    result_source = unparse(module)

    expected = """\
@external
def foo(x: uint256, y: uint256) -> uint256:
    gen_var_0: uint256 = y
    assert x > gen_var_0
    gen_var_1: uint256 = y
    return x + gen_var_1
"""
    assert result_source == expected

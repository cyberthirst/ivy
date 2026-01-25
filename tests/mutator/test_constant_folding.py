from decimal import Decimal, localcontext

import pytest
from vyper.ast import nodes as ast
from vyper.compiler.phases import CompilerData
from vyper.utils import keccak256

from fuzzer.mutator.constant_folding import (
    ConstEvalError,
    _unbox_value,
    evaluate_constant_expression,
    fold_constant_expression,
)


def _get_module(source: str) -> ast.Module:
    return CompilerData(source).annotated_vyper_module


def _collect_constant_values(module: ast.Module) -> dict[str, object]:
    constants: dict[str, object] = {}
    for node in module.body:
        if isinstance(node, ast.VariableDecl) and node.value is not None:
            constants[node.target.id] = evaluate_constant_expression(
                node.value, constants
            )
    return constants


def _get_function_assigns(module: ast.Module, func_name: str) -> dict[str, ast.VyperNode]:
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            assigns: dict[str, ast.VyperNode] = {}
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(
                    stmt.target, ast.Name
                ):
                    assigns[stmt.target.id] = stmt.value
            return assigns
    raise AssertionError(f"missing function {func_name}")


def _trunc_div_int(n: int, d: int) -> int:
    sign = -1 if (n * d) < 0 else 1
    return sign * (abs(n) // abs(d))


def _trunc_mod_int(n: int, d: int) -> int:
    sign = -1 if n < 0 else 1
    return sign * (abs(n) % abs(d))


def _decimal_bounds() -> tuple[Decimal, Decimal]:
    with localcontext() as ctx:
        ctx.prec = 80
        scale = Decimal(10) ** 10
        dec_min = Decimal(-(2**167)) / scale
        dec_max = Decimal(2**167 - 1) / scale
    return dec_min, dec_max


def _build_cases() -> list[tuple[str, str, object]]:
    cases: list[tuple[str, str, object]] = []
    add = cases.append

    i256_max = 2**255 - 1
    i256_min = -(2**255)
    u256_max = 2**256 - 1
    dec_min, dec_max = _decimal_bounds()

    add(("int256", "1 + 2 * 3 - 4", 1 + 2 * 3 - 4))
    add(("int256", "(1 + 2) * (3 - 4)", (1 + 2) * (3 - 4)))
    add(("int256", "(-5) + 2 * 8", (-5) + 2 * 8))
    add(("int256", "-5 - (2 * 8)", -5 - (2 * 8)))
    add(("int256", "(7 - 10) * (3 + 2)", (7 - 10) * (3 + 2)))
    add(("int256", "-(2 ** 5)", -(2**5)))
    add(("int256", "2 ** 6", 2**6))
    add(("int256", "min_value(int256)", i256_min))
    add(("int256", "max_value(int256)", i256_max))
    add(("int256", "min_value(int256) + 1", i256_min + 1))
    add(("int256", "max_value(int256) - 1", i256_max - 1))
    add(("int256", "(min_value(int256) + 2) // 2", _trunc_div_int(i256_min + 2, 2)))
    add(("int256", "(-7) // 3", _trunc_div_int(-7, 3)))
    add(("int256", "(-7) % 3", _trunc_mod_int(-7, 3)))
    add(("int256", "(-7) + (3 ** 3)", (-7) + (3**3)))
    add(("int256", "((5 << 3) - 7)", (5 << 3) - 7))
    add(("int256", "(1024 >> 5)", 1024 >> 5))
    add(("int256", "(1 << 5) + (1 << 3)", (1 << 5) + (1 << 3)))
    add(("int256", "(12345 & 255)", 12345 & 255))
    add(("int256", "(12345 | 256)", 12345 | 256))
    add(("int256", "(12345 ^ 32)", 12345 ^ 32))
    add(("int256", "(-1) & 255", (-1) & 255))
    add(("int256", "(-1) | 0", (-1) | 0))
    add(("int256", "(-5) * (7 - 3)", (-5) * (7 - 3)))
    add(("int256", "(1 if (3 > 2) else -1)", 1 if (3 > 2) else -1))

    add(("uint256", "1 + 2 * 3 + 4", 1 + 2 * 3 + 4))
    add(("uint256", "(1 + 2) * (3 + 4)", (1 + 2) * (3 + 4)))
    add(("uint256", "(7 * 6) - 5", (7 * 6) - 5))
    add(("uint256", "(2 ** 8) + 1", (2**8) + 1))
    add(("uint256", "max_value(uint256)", u256_max))
    add(("uint256", "max_value(uint256) - 1", u256_max - 1))
    add(("uint256", "max_value(uint256) // 2", u256_max // 2))
    add(("uint256", "(1 << 0)", 1 << 0))
    add(("uint256", "(1 << 1)", 1 << 1))
    add(("uint256", "(1 << 8)", 1 << 8))
    add(("uint256", "(1 << 16) + (1 << 8)", (1 << 16) + (1 << 8)))
    add(("uint256", "(1024 >> 5)", 1024 >> 5))
    add(("uint256", "(123456 & 65535)", 123456 & 65535))
    add(("uint256", "(123456 | 65535)", 123456 | 65535))
    add(("uint256", "(123456 ^ 65535)", 123456 ^ 65535))
    add(("uint256", "((5 + 3) * (7 - 2))", (5 + 3) * (7 - 2)))
    add(("uint256", "((5 + 3) * (7 - 2)) % 11", ((5 + 3) * (7 - 2)) % 11))
    add(("uint256", "(100 // 3)", 100 // 3))
    add(("uint256", "(100 % 3)", 100 % 3))
    add(("uint256", "(2 ** 10) // (2 ** 5)", (2**10) // (2**5)))
    add(("uint256", "(2 ** 10) % (2 ** 5)", (2**10) % (2**5)))
    add(("uint256", "min(7, 4)", min(7, 4)))
    add(("uint256", "max(7, 4)", max(7, 4)))
    add(("uint256", "shift(1, 8)", 1 << 8))
    add(("uint256", "shift(1024, -5)", 1024 >> 5))

    add(("bool", "1 < 2", True))
    add(("bool", "2 < 1", False))
    add(("bool", "(1 < 2) and (3 > 2)", (1 < 2) and (3 > 2)))
    add(("bool", "(1 < 2) and (3 < 2)", (1 < 2) and (3 < 2)))
    add(("bool", "(1 < 2) or (3 < 2)", (1 < 2) or (3 < 2)))
    add(("bool", "not False", True))
    add(("bool", "not (1 == 1)", not (1 == 1)))
    add(("bool", "(max_value(uint256) > 0)", u256_max > 0))
    add(("bool", "(min_value(int256) < 0)", i256_min < 0))
    add(("bool", "(max_value(int256) == max_value(int256))", i256_max == i256_max))
    add(("bool", "(max_value(int256) != min_value(int256))", i256_max != i256_min))
    add(("bool", "(5 >= 5) and (4 <= 4)", (5 >= 5) and (4 <= 4)))
    add(("bool", "(5 >= 6) or (4 <= 3)", (5 >= 6) or (4 <= 3)))
    add(("bool", "(1 if True else 0) == 1", (1 if True else 0) == 1))
    add(
        (
            "bool",
            "((1 < 2) and (3 > 2)) or ((2 < 1) and (1 == 1))",
            ((1 < 2) and (3 > 2)) or ((2 < 1) and (1 == 1)),
        )
    )

    add(("decimal", "1.5 + 2.25", Decimal("3.75")))
    add(("decimal", "5.0 - 2.125", Decimal("2.875")))
    add(("decimal", "-1.5 + 2.0", Decimal("0.5")))
    add(("decimal", "-1.5 * 2.0", Decimal("-3.0")))
    add(("decimal", "3.75 / 1.5", Decimal("2.5")))
    add(("decimal", "3.0 / 2.0", Decimal("1.5")))
    add(("decimal", "-3.0 / 2.0", Decimal("-1.5")))
    add(("decimal", "7.25 % 2.0", Decimal("1.25")))
    add(("decimal", "min_value(decimal)", dec_min))
    add(("decimal", "max_value(decimal)", dec_max))
    add(("int256", "floor(1.9)", 1))
    add(("int256", "floor(-1.1)", -2))
    add(("int256", "ceil(1.1)", 2))
    add(("int256", "ceil(-1.1)", -1))

    add(("uint256", "len(BYTES)", 6))
    add(("uint256", "len(STR)", 6))
    add(("uint256", "len(BYTES) + len(STR)", 12))
    add(("uint256", "(len(BYTES) * 3) + 1", (6 * 3) + 1))
    add(("uint256", "len(STR) * len(BYTES)", 6 * 6))
    add(("uint256", "len(BYTES) << 2", 6 << 2))
    add(("uint256", "max(len(BYTES), len(STR))", max(6, 6)))
    add(("uint256", "min(len(BYTES), len(STR))", min(6, 6)))
    add(("bytes32", "keccak256(b\"ivy\")", keccak256(b"ivy")))
    add(("bytes32", "keccak256(STR)", keccak256(b"world!")))
    add(("bytes32", "keccak256(BYTES)", keccak256(b"hello!")))
    add(("Bytes[4]", "method_id(\"foo(uint256)\")", keccak256(b"foo(uint256)")[:4]))

    add(("uint256", "ARR[0] + ARR[4]", 0 + 4))
    add(("uint256", "ARR[2] * ARR[3]", 2 * 3))
    add(("uint256", "NEST[0][0] + NEST[1][2]", 1 + 6))
    add(("uint256", "NEST[1][0] * NEST[0][2]", 4 * 3))
    add(("uint256", "FOO.a + FOO.c", 7 + 42))
    add(("int256", "FOO.b - 5", -5 - 5))
    add(("uint256", "TUP[0] + ARR[1]", 7 + 1))
    add(("int256", "TUP[1] * -2", -8 * -2))
    add(("bool", "TUP[2] and (ARR[3] == 3)", True and (3 == 3)))

    assert len(cases) == 100
    return cases


def _build_source(cases: list[tuple[str, str, object]]) -> str:
    lines = [
        "struct Foo:",
        "    a: uint256",
        "    b: int256",
        "    c: uint256",
        "",
        "FOO: constant(Foo) = Foo({a: 7, b: -5, c: 42})",
        "ARR: constant(uint256[5]) = [0, 1, 2, 3, 4]",
        "NEST: constant(uint256[3][2]) = [[1, 2, 3], [4, 5, 6]]",
        "TUP: constant((uint256, int256, bool)) = (7, -8, True)",
        "BYTES: constant(Bytes[6]) = b\"hello!\"",
        "STR: constant(String[6]) = \"world!\"",
        "",
        "@external",
        "def f():",
    ]
    for i, (typ, expr, _) in enumerate(cases):
        lines.append(f"    v{i}: {typ} = {expr}")
    return "\n".join(lines) + "\n"


CASES = _build_cases()
SOURCE = _build_source(CASES)


@pytest.fixture(scope="module")
def const_eval_context():
    module = _get_module(SOURCE)
    constants = _collect_constant_values(module)
    assigns = _get_function_assigns(module, "f")
    return constants, assigns


def test_constant_eval_supported_exprs():
    source = """
struct Foo:
    a: uint256
    b: int256

FOO: constant(Foo) = Foo({a: 2, b: -3})
ARR: constant(uint256[3]) = [1, 2, 3]
TUP: constant((uint256, bool)) = (4, True)
BYTES: constant(Bytes[5]) = b"hello"
STR: constant(String[5]) = "hello"

@external
def f() -> bool:
    a: uint256 = 1 + 2 * 3
    b: int256 = -5
    c: bool = (1 < 2) and (3 >= 3)
    d: uint256 = ARR[1]
    e: uint256 = TUP[0]
    g: bool = TUP[1]
    f: uint256 = FOO.a
    h: int256 = FOO.b
    i: uint256 = len(BYTES)
    j: uint256 = len(STR)
    k: bytes32 = keccak256(b"hi")
    l: Bytes[4] = method_id("foo()")
    m: int256 = abs(-5)
    n: uint256 = min(4, 7)
    o: uint256 = max(4, 7)
    p: int256 = min_value(int256)
    q: int256 = max_value(int256)
    r: uint256 = shift(1, 2)
    s: uint256 = (1 if True else 2)
    t: bool = not False
    u: uint256 = 7 % 4
    v: bool = 1 == 1
    w: int256 = floor(1.5)
    x: int256 = ceil(1.1)
    return True
    """
    module = _get_module(source)
    constants = _collect_constant_values(module)
    assigns = _get_function_assigns(module, "f")

    assert int(constants["ARR"][1]) == 2
    assert int(constants["FOO"]["a"]) == 2
    assert int(constants["FOO"]["b"]) == -3
    assert int(constants["TUP"][0]) == 4
    assert bool(constants["TUP"][1]) is True

    assert int(evaluate_constant_expression(assigns["a"], constants)) == 7
    assert int(evaluate_constant_expression(assigns["b"], constants)) == -5
    assert bool(evaluate_constant_expression(assigns["c"], constants)) is True
    assert int(evaluate_constant_expression(assigns["d"], constants)) == 2
    assert int(evaluate_constant_expression(assigns["e"], constants)) == 4
    assert bool(evaluate_constant_expression(assigns["g"], constants)) is True
    assert int(evaluate_constant_expression(assigns["f"], constants)) == 2
    assert int(evaluate_constant_expression(assigns["h"], constants)) == -3
    assert int(evaluate_constant_expression(assigns["i"], constants)) == 5
    assert int(evaluate_constant_expression(assigns["j"], constants)) == 5

    k_val = evaluate_constant_expression(assigns["k"], constants)
    assert bytes(k_val) == keccak256(b"hi")

    l_val = evaluate_constant_expression(assigns["l"], constants)
    assert bytes(l_val) == keccak256(b"foo()")[:4]

    assert int(evaluate_constant_expression(assigns["m"], constants)) == 5
    assert int(evaluate_constant_expression(assigns["n"], constants)) == 4
    assert int(evaluate_constant_expression(assigns["o"], constants)) == 7
    assert int(evaluate_constant_expression(assigns["p"], constants)) == -(2**255)
    assert int(evaluate_constant_expression(assigns["q"], constants)) == 2**255 - 1
    assert int(evaluate_constant_expression(assigns["r"], constants)) == 4
    assert int(evaluate_constant_expression(assigns["s"], constants)) == 1
    assert bool(evaluate_constant_expression(assigns["t"], constants)) is True
    assert int(evaluate_constant_expression(assigns["u"], constants)) == 3
    assert bool(evaluate_constant_expression(assigns["v"], constants)) is True
    assert int(evaluate_constant_expression(assigns["w"], constants)) == 1
    assert int(evaluate_constant_expression(assigns["x"], constants)) == 2


def test_constant_eval_rejects_unsupported_exprs():
    source = """
a: uint256

@external
def g():
    bad_concat: Bytes[2] = concat(b"a", b"b")
    bad_slice: Bytes[2] = slice(b"abc", 0, 2)
    bad_env: address = msg.sender
    bad_self: uint256 = self.a
    """
    module = _get_module(source)
    assigns = _get_function_assigns(module, "g")

    for name in ["bad_concat", "bad_slice", "bad_env", "bad_self"]:
        with pytest.raises(ConstEvalError):
            evaluate_constant_expression(assigns[name], {})


def test_fold_constant_expression_returns_literal():
    source = """
@external
def h() -> uint256:
    a: uint256 = 2 + 3
    t: (uint256, bool) = (1, True)
    return a
    """
    module = _get_module(source)
    assigns = _get_function_assigns(module, "h")

    folded_int = fold_constant_expression(assigns["a"], {})
    assert isinstance(folded_int, ast.Int)
    assert folded_int.value == 5

    folded_tuple = fold_constant_expression(assigns["t"], {})
    assert isinstance(folded_tuple, ast.Tuple)
    assert len(folded_tuple.elements) == 2


@pytest.mark.parametrize("index, case", list(enumerate(CASES)))
def test_constant_eval_complex_cases(const_eval_context, index, case):
    constants, assigns = const_eval_context
    _, _, expected = case
    node = assigns[f"v{index}"]
    value = evaluate_constant_expression(node, constants)
    actual = _unbox_value(value, node._metadata["type"])
    assert actual == expected

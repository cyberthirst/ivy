---
name: testing
description: Writing tests for Ivy - e2e tests with Ivy interpreter, BOA-based bytecode tests, property-based testing principles, test hygiene
---

# Testing Skill

## Overview
Tests verify Ivy's AST interpreter matches Vyper's compiled bytecode behavior. Good tests are stable across time, test end-to-end pipelines, and verify properties rather than implementation details.

## Section 1: Ivy Tests (AST Interpreter)

Use `ivy.frontend.loader.loads` to test the Ivy interpreter directly.

```python
from ivy.frontend.loader import loads
from ivy.exceptions import Assert, Revert

def test_basic_call():
    src = """
@external
def foo(x: uint256) -> uint256:
    return x + 1
"""
    c = loads(src)
    assert c.foo(41) == 42

def test_storage():
    src = """
counter: uint256

@external
def inc() -> uint256:
    self.counter += 1
    return self.counter
"""
    c = loads(src)
    assert c.inc() == 1
    assert c.inc() == 2
```

Key fixtures (from `tests/conftest.py`):
- `get_contract` - loads contract via Ivy
- `tx_failed` - context manager for expected reverts

```python
def test_revert(get_contract, tx_failed):
    src = """
@external
def fail():
    assert False, "error"
"""
    c = get_contract(src)
    with tx_failed(Assert):
        c.fail()
```

## Section 2: BOA Tests (Bytecode Execution)

Use `boa.loads` when you need compiled bytecode behavior (ground truth).

```python
import boa

def test_with_boa():
    c = boa.loads("""
@external
def foo(x: uint256) -> uint256:
    return x + 1
""")
    assert c.foo(41) == 42

def test_revert_with_boa():
    c = boa.loads("""
@external
def fail():
    raise "error"
""")
    with boa.reverts():
        c.fail()
```

When to use BOA vs Ivy:
- **Ivy tests**: Testing the interpreter implementation
- **BOA tests**: Verifying ground truth, testing compiler behavior

## Section 3: Properties of Good Tests

Good tests verify **properties** rather than implementation details. When you encounter a function or system, ask: "What mathematical or logical properties does this have?" Then test those properties.

### Examples of properties

#### 1. Round-Trip (Encode/Decode)
`decode(encode(x)) == x`

```python
# Unparser: parse(unparse(ast)) == ast
def test_unparser_roundtrip(source_code):
    ast = parse(source_code)
    unparsed = unparse(ast)
    assert normalize(parse(unparsed)) == normalize(ast)

# Serialization: deserialize(serialize(obj)) == obj
def test_json_roundtrip(obj):
    assert from_json(to_json(obj)) == obj

# Compression, encryption, any codec
```

#### 2. Idempotence
`f(f(x)) == f(x)` — applying twice is same as applying once.

```python
# Normalization
def test_normalize_idempotent(text):
    once = normalize(text)
    twice = normalize(once)
    assert once == twice

# Formatting, canonicalization, deduplication, sorting
# Cache warming, garbage collection, state cleanup
```


### How to Find Properties

When writing tests, ask:
1. **Does this function have an inverse?** → Test invertibility
2. **What if I apply it twice?** → Test idempotence
3. **What quantities should be preserved?** → Test invariants
4. **Is there a reference implementation?** → Test oracle comparison
5. **Does order matter?** → Test commutativity
6. **What are the edge cases?** (empty, zero, max) → Test identity/bounds
7. **Can an external tool validate this?** → Test with validity oracle

### Comprehensive Coverage

Don't test 3 arbitrary cases. Cover the space systematically:

```python
# BAD: Random selection
@pytest.mark.parametrize("typ", ["uint256", "int128", "bool"])
def test_storage(typ): ...

# GOOD: All types, including nested
ALL_TYPES = (
    [f"uint{i}" for i in [8, 16, 32, 64, 128, 256]] +
    [f"int{i}" for i in [8, 16, 32, 64, 128, 256]] +
    ["bool", "address", "Bytes[32]", "String[32]"] +
    ["DynArray[uint256, 10]", "HashMap[address, uint256]"]
)

@pytest.mark.parametrize("typ", ALL_TYPES)
def test_storage(typ): ...
```

### E2E Over Unit Tests

Prefer testing full pipelines. They're more stable and catch integration issues:

```python
# FRAGILE: Tests internal parsing
def test_parse_literal():
    assert parse_literal("42").value == 42

# STABLE: Tests through the whole system
def test_arithmetic():
    c = loads("@external\ndef f() -> uint256: return 40 + 2")
    assert c.f() == 42
```

## Section 4: Test Hygiene

Follow these rules from `AGENTS.md`:

### Simple `test_` Functions
```python
# GOOD: Simple function
def test_slice_basic():
    c = loads(src)
    assert c.foo() == expected

# BAD: Class hierarchy
class TestSlice(BaseTestCase):
    def setUp(self):
        ...
```

### Inline Vyper Source
```python
# GOOD: Source visible in test
def test_foo():
    src = """
@external
def foo() -> uint256:
    return 42
"""
    c = loads(src)
    assert c.foo() == 42

# BAD: Source in separate file
def test_foo():
    c = loads(open("fixtures/foo.vy").read())
```

### Use `./ivytest.sh`, Not Raw pytest
```bash
# GOOD
./ivytest.sh tests/ivy/test_e2e.py

# BAD
pytest tests/ivy/test_e2e.py
```

### Run `./check.sh` Before Commits
```bash
./check.sh  # Runs ruff + pyright
./ivytest.sh tests/
```

### No Docstrings Unless Complex
```python
# GOOD: Self-explanatory name, no docstring
def test_slice_with_negative_start():
    ...

# BAD: Unnecessary docstring
def test_slice():
    """Test that slice works correctly with various inputs."""
    ...
```

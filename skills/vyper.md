---
name: vyper
description: Understanding Vyper compiler internals - use when debugging compiler issues, investigating how Vyper works, or tracing compilation from source to bytecode
---

# Vyper Compiler Skill

## Overview
Vyper is a Pythonic smart contract language for the EVM. When debugging compiler issues or understanding Vyper behavior, you need to trace how source code flows through parsing → AST → semantic analysis → IR → bytecode.

## Workflow

### 1. Locate Vyper Source
```bash
uv run python -c "import vyper; print(vyper.__file__)"
# Typically: .venv/lib/python3.x/site-packages/vyper/
```

### 2. Understand the Module Structure
| Module | Purpose |
|--------|---------|
| `vyper/ast/` | AST node definitions, parsing |
| `vyper/semantics/` | Type checking, semantic analysis |
| `vyper/semantics/types/` | Type definitions (IntegerT, BytesT, etc.) |
| `vyper/codegen/` | IR generation from AST |
| `vyper/venom/` | New IR pipeline (Venom) |
| `vyper/compiler/` | Orchestrates the compilation phases |

### 3. Inspect Compilation Outputs
```python
from vyper.compiler import compile_code

src = """
@external
def foo(x: uint256) -> uint256:
    return x + 1
"""

# Print IR (intermediate representation)
print(compile_code(src, output_formats=["ir"])["ir"])

# Print bytecode
print(compile_code(src, output_formats=["bytecode"])["bytecode"])

# Print annotated AST
from vyper.compiler.phases import CompilerData
data = CompilerData(src)
print(data.vyper_module)  # AST with type annotations
```

### 4. Test Vyper Behavior
Vyper tests aren't included in the pip package. To test behavior, write a quick script:
```python
import boa

# Deploy and test
c = boa.loads("""
@external
def foo(x: uint256) -> uint256:
    return x + 1
""")
assert c.foo(41) == 42
```

## Key References

### AST Nodes (`vyper.ast.nodes`)
```python
from vyper.ast import nodes as ast

# Statements
ast.Assign, ast.AnnAssign, ast.If, ast.For, ast.Return
ast.Assert, ast.Raise, ast.Pass, ast.Break, ast.Continue

# Expressions
ast.BinOp, ast.Compare, ast.BoolOp, ast.UnaryOp
ast.Call, ast.Subscript, ast.Attribute
ast.Name, ast.Int, ast.Str, ast.Bytes
```

### Type System (`vyper.semantics.types`)
```python
from vyper.semantics.types import (
    IntegerT,      # int128, int256, uint8, uint256, etc.
    BoolT,         # bool
    AddressT,      # address
    BytesT,        # Bytes[N]
    StringT,       # String[N]
    DArrayT,       # DynArray[T, N]
    SArrayT,       # T[N] (static array)
    HashMapT,      # HashMap[K, V]
    StructT,       # struct definitions
    TupleT,        # tuples
)
```

## Example: Debugging a Codegen Issue

Suspect `slice()` generates wrong bytecode:

```bash
# 1. Find the builtin implementation
rg "class Slice" .venv/lib/python3.*/site-packages/vyper/builtins/functions.py

# 2. Print IR for a minimal reproducer
uv run python -c "
from vyper.compiler import compile_code
src = '''
@external
def f(x: Bytes[32]) -> Bytes[10]:
    return slice(x, 0, 10)
'''
print(compile_code(src, output_formats=['ir'])['ir'])
"

# 3. Test actual behavior vs expected
uv run python -c "
import boa
c = boa.loads('''
@external
def f(x: Bytes[32]) -> Bytes[10]:
    return slice(x, 0, 10)
''')
result = c.f(b'hello world' + b'\x00' * 21)
print('Result:', result)
"
```

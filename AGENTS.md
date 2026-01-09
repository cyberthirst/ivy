# AGENTS.md - Ivy Development Guide

## Project Overview
Ivy is an AST interpreter for Vyper (EVM smart contract language) built for differential fuzzing.
The goal is to find semantic bugs (miscompilations) by comparing Vyper-compiled bytecode execution
against Ivy's source-level interpretation.

## Build & Test Commands

### Environment Setup
This project uses `uv` for dependency management.
```bash
uv sync --all-extras                      # Install all dependencies (creates .venv/ automatically)
```
To run commands, use `uv run` (no manual activation needed):
```bash
uv run pytest tests/ivy/ -v -s
uv run python -m src.fuzzer.generative_fuzzer
```

### Critical Tests (MUST PASS after any interpreter change)
```bash
# 1. AST interpretation & EVM semantics test
uv run pytest tests/ivy/ -v -s

# 2. Vyper export replay - thousands of tests from Vyper's test suite
uv run pytest tests/test_replay.py -v -s -n auto
```

### Linting & Type Checking
```bash
uv run ruff check src/                    # Lint
uv run ruff format src/                   # Format
uv run ruff check src/ --fix              # Auto-fix lint issues
uv run pyright src/                       # Type check
./check.sh                                # Run all checks (ruff + pyright)
```

### Running the Fuzzer
```bash
uv run python -m src.fuzzer.generative_fuzzer
```

## Code Style Guidelines

### Imports
Use absolute imports, not relative ones.

```python
# Standard library first
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import copy

# Third-party
from vyper.ast import nodes as ast
from vyper.semantics.types import VyperType

# Local imports
from ivy.types import Address
from ivy.exceptions import Revert
```

### Type Annotations
- Use type hints for function signatures
- Use `Optional[T]` for nullable types
- Use `Union[A, B]` for multiple types (or `A | B` syntax)
- Type checking mode: basic (pyright)
- **Use `TYPE_CHECKING` guard** for import-only types to avoid circular imports
- **Use `from __future__ import annotations`** for forward references
```python
from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Dict, Any

if TYPE_CHECKING:
    from ivy.frontend.vyper_contract import VyperContract

def foo(contract: VyperContract, b: Optional[str] = None) -> Dict[str, Any]:
    ...
```

### Testing Patterns
- **Prefer simple `test_` functions** - nothing fancy, no complex class hierarchies
- **New interpreter features MUST have e2e tests** in `tests/ivy/`
- **Avoid unit tests** for unstable/evolving behavior (e.g., specific AST mutation internals)
- Only write unit tests for stable, well-defined behavior unlikely to change
- Fixtures defined in `tests/conftest.py`
- Key fixtures: `get_contract`, `tx_failed`, `env`, `make_input_bundle`
- Inline Vyper source in tests as multi-line strings
- Tests auto-rollback state via `env.anchor()` hook
```python
def test_example(get_contract):
    src = """
@external
def foo() -> uint256:
    return 42
    """
    c = get_contract(src)
    assert c.foo() == 42

def test_revert(tx_failed):
    with tx_failed(Assert):
        c.foo()  # Should revert
```

## Project Structure

```
src/
  ivy/                    # Core interpreter
    abi/                  # ABI encoding/decoding
    builtins/             # Vyper builtin functions
    evm/                  # Custom EVM implementation
    expr/                 # Expression evaluation
    frontend/             # Contract loading, Vyper integration
    stmt.py               # Statement visitor
    visitor.py            # Base visitor pattern
    types.py              # Type representations (Address, Flag, etc.)
    exceptions.py         # Custom exceptions
  fuzzer/                 # Differential fuzzing infrastructure
    mutator/              # AST and value mutation
    runner/               # Scenario execution
  unparser/               # AST to source code

tests/
  ivy/                    # Interpreter tests
  conftest.py             # Pytest fixtures
  test_replay.py          # Export replay validation
```

## Key Concepts

### Exports
JSON traces from Vyper's test suite. See `trace-format.json` for the schema. Load/filter with:
```python
from fuzzer.export_utils import load_all_exports, filter_exports, TestFilter
exports = load_all_exports(Path("tests/vyper-exports"))
exports = filter_exports(exports, test_filter=TestFilter().include_path("test_slice"))
```

### Contract Loading
```python
from ivy.frontend.loader import loads
c = loads(vyper_source_code)
result = c.foo()  # Call external function
```

### Environment
Singleton state management:
```python
from ivy.frontend.env import Env
env = Env().get_singleton()
env.clear_state()
with env.anchor():
    # Changes rolled back after block
```

## Guidelines for AI Agents

1. **Minimal changes**: No unnecessary abstractions. Reuse existing code.
2. **No docs**: Don't write docstrings for modules/functions unless they're complex.
3. **Comments**: Only for non-obvious logic.
4. **Focus**: Compiler correctness bugs only, not lexer/parser bugs.
5. **Testing**: Use `pytest -n1` when debugging. Run `./check.sh` before commits.
6. **PYTHONPATH**: Already configured in `pytest.ini`. For scripts, use `uv run` which handles this automatically.

## Available Skills

Skills are located in `.opencode/skill/<name>/SKILL.md`:

| Skill | Purpose |
|-------|---------|
| **boa** | Titanoboa testing framework - source in .venv, writing tests, debugging |
| **vyper** | Vyper compiler internals - AST, types, compilation pipeline |
| **replay** | Replaying divergences, finding root cause, debugging workflow |
| **dedup-divergences** | Batch analysis of fuzzer divergences, parallel subagent workflow |

Use these skills when working on divergence analysis or debugging interpreter issues.
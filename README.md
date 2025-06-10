
# Ivy

An AST interpreter for Vyper with a custom EVM backend.

## Overview

Ivy is an experimental interpreter that executes Vyper smart contracts by interpreting their Abstract Syntax Tree (AST) directly, rather than compiling to bytecode. This approach enables powerful debugging, testing, and analysis capabilities.

### Key Features

- Direct AST interpretation without bytecode compilation
- Custom EVM implementation
- Integration with Vyper test exports for differential testing
- AST mutation capabilities for fuzzing
- Compatible with most Vyper language features

### How it Works

- Uses the Vyper compiler as a library to parse and analyze contracts
- Obtains the annotated AST from the compiler
- Stores the contract's annotated AST (instead of bytecode) in EVM accounts
- Interprets the AST when transactions or calls are made to contracts
- Maintains global state (storage, balances, etc.) like a regular EVM

## Setup and Installation

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/cyberthirst/ivy.git
   cd ivy
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install the package and dependencies:**
   ```bash
   pip install -e .
   ```

   Or if you're using `uv`:
   ```bash
   uv pip install -e .
   ```

### Development Setup

For development, you'll need additional dependencies:

1. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

2. **Configure your environment:**
   - The project uses `PYTHONPATH=src` for imports
   - This is already configured in `pytest.ini` for tests
   - For running scripts, you may need to set: `export PYTHONPATH=src`

### Testing

Run the test suite to ensure everything is working:

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/ivy/test_e2e.py
```

### Using Vyper Test Exports

Ivy supports loading and executing Vyper test exports for differential testing:

1. **Obtain Vyper test exports:**
   ```bash
   # From your Vyper repository, export tests:
   pytest -n 1 --export=/path/to/exports
   ```

2. **Copy exports to Ivy:**
   ```bash
   cp -r /path/to/exports tests/vyper-exports
   ```

3. **Run differential testing:**
   ```bash
   python examples/differential_testing.py
   ```

## Quick Start

### Basic Usage

```python
from ivy.frontend.loader import loads

# Deploy a simple contract
src = """
@external
def foo(a: uint256=42) -> uint256:
    return a
"""

c = loads(src)
assert c.foo() == 42
```

### Command Line Usage

```bash
# Run a Vyper contract directly
python -m ivy tests/example_contracts/example.vy

# With uv
uv run python -m ivy tests/example_contracts/example.vy
```

### Limitations

- **Gas metering**: Gas costs are not supported as they can't be accurately mapped to AST interpretation
- **Delegatecall**: Basic support only - proper variable mapping between caller and callee requires compatible storage layouts
- **Bytecode operations**: Some low-level operations that depend on bytecode are not supported

### Design Decisions

- The interpreter frontend is heavily inspired by [Titanoboa](https://github.com/vyperlang/titanoboa)
- State changes are journaled for proper rollback support
- The EVM implementation focuses on correctness over performance

### Development Tips

- The `PYTHONPATH=src` environment variable may be needed for some scripts
- Check `examples/` directory for usage patterns
- Run `ruff` for linting: `ruff check src tests`

## Acknowledgments

- The Vyper team for the compiler and Titanoboa

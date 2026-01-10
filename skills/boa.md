---
name: boa
description: Titanoboa testing framework for Vyper - read source in .venv, write tests, debug execution differences against Ivy interpreter
---

# Titanoboa (boa) Skill

## Overview
Titanoboa is a Vyper interpreter/testing framework. It compiles Vyper source to bytecode and executes it in a Python-based EVM (py-evm). In Ivy's differential fuzzing, Boa serves as the reference implementation against which Ivy's AST interpreter is compared.

## Reading Boa Source Code
Boa is installed in the project's `.venv/`. To find the source:
```bash
# Find boa installation path
uv run python -c "import boa; print(boa.__file__)"

# Typically at: .venv/lib/python3.x/site-packages/boa/
```

Key modules to understand:
- `boa/contracts/vyper/` - Contract compilation and deployment
- `boa/environment.py` - EVM environment management
- `boa/vm/` - VM execution internals

## Writing Tests with Boa
```python
import boa

# Load and deploy a contract
contract = boa.loads("""
@external
def foo(x: uint256) -> uint256:
    return x + 1
""")

# Call functions
result = contract.foo(41)
assert result == 42

# With constructor args
contract = boa.loads(source_code, arg1, arg2)

# Check for reverts
with boa.reverts():
    contract.will_revert()

# With specific error message
with boa.reverts("error message"):
    contract.will_revert()
```

## Environment Control
```python
import boa

# Set block timestamp
boa.env.evm.patch.timestamp = 1234567890

# Time travel (advance by seconds)
boa.env.time_travel(seconds=100)

# Set msg.sender
with boa.env.prank(some_address):
    contract.foo()

# Set msg.value
contract.payable_func(value=1000)

# Snapshot and revert state
with boa.env.anchor():
    # Changes in this block are rolled back after
```

## Debugging Tips
- Boa executes compiled bytecode, so divergences may indicate:
  - Vyper compiler bugs (bytecode doesn't match source semantics)
  - Ivy interpreter bugs (AST interpretation differs from bytecode)
  - Boa/py-evm bugs (rare, but possible)
- Compare storage dumps between Ivy and Boa to pinpoint state differences
- Use `boa.env.get_storage()` to inspect contract storage


## Ivy
- an AST interpreter for Vyper with a custom EVM backend

### How it works
- uses the Vyper compiler as a library
- calls the compiler to get the annotated AST
- an Account in the EVM doesn't contain the code, but the contract's annotated AST
- when a tx/call is made to the contract, the AST (instead of bytecode) is interpreted
- as in `boa` contracts can be deployed using `load` function
  - it returns a user-facing contract representation for user interaction

## Run & Test
- run: ` uv run python -m ivy tests/example_contracts/example.vy`
- test: `uv run pytest tests`

```python
from ivy.frontend.loader import loads


def test_deploy_and_call():
    src = """
@external
def foo(a: uint256=42) -> uint256:
    return a
    """

    c = loads(src)
    assert c.foo() == 42
```
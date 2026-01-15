# mutations/

AST node mutations for the fuzzer. Each module registers strategies that transform existing Vyper AST nodes into semantically different but syntactically valid variants.

## How It Works

Mutations operate on nodes selected by `candidate_selector.py`. When a node is visited during AST traversal, the `MutationEngine` collects applicable strategies based on node type and executes one via weighted random selection.

Each mutation receives a `MutationCtx` containing:
- The target node
- Expression/statement generators for synthesizing new code
- Scope context for type-aware generation

## Adding a Mutation

1. Create a new file with `@strategy`-decorated run functions
2. Tag each strategy with `"mutation"`
3. Add the module to `__init__.py` so `register_all` picks it up

```python
# Example: mutations/example.py
from __future__ import annotations

from vyper.ast import nodes as ast

from fuzzer.mutator.strategy import strategy
from fuzzer.mutator.mutations.base import MutationCtx


def _is_applicable(*, ctx: MutationCtx, **_) -> bool:
    return some_check(ctx.node)


@strategy(
    name="example.my_mutation",
    type_classes=(ast.SomeNode,),
    tags=frozenset({"mutation"}),
    is_applicable="_is_applicable",
)
def _run_mutation(*, ctx: MutationCtx, **_) -> ast.VyperNode:
    # Transform ctx.node and return it
    ...
```

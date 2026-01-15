# mutations/

AST node mutations for the fuzzer. Each module registers strategies that transform existing Vyper AST nodes into semantically different but syntactically valid variants.

## How It Works

Mutations operate on nodes selected by `candidate_selector.py`. When a node is visited during AST traversal, the `MutationEngine` collects applicable strategies based on node type and executes one via weighted random selection.

Each mutation receives a `MutationCtx` containing:
- The target node
- Expression/statement generators for synthesizing new code
- Scope context for type-aware generation

## Adding a Mutation

1. Create a new file with a `register(registry: StrategyRegistry)` function
2. Define strategies with `type_classes` matching target AST node types
3. Tag with `"mutation"` so the engine can find it
4. Add to `__init__.py`

```python
# Example: mutations/example.py
from fuzzer.mutator.strategy import Strategy, StrategyRegistry
from fuzzer.mutator.mutations.base import MutationCtx

def register(registry: StrategyRegistry) -> None:
    registry.register(Strategy(
        name="example.my_mutation",
        type_classes=(ast.SomeNode,),
        tags=frozenset({"mutation"}),
        is_applicable=lambda *, ctx, **_: some_check(ctx.node),
        weight=lambda **_: 1.0,
        run=_run_mutation,
    ))

def _run_mutation(*, ctx: MutationCtx, **_) -> ast.VyperNode:
    # Transform ctx.node and return it
    ...
```

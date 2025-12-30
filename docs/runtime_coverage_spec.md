# Runtime Coverage (Ivy) – Design + Implementation Plan

## Goal
Add runtime coverage signals for ABI fuzzing of generated contracts, with good “gradient” for short-circuit predicates and low overhead in Ivy.

Scope: deploy once, then do a few thousand ABI calls; then mutate contract and repeat.

## Current State (codebase)
**IMPLEMENTED** - The coverage infrastructure is now in place:
- `src/ivy/execution_metadata.py` - data model for coverage signals
- `src/ivy/tracer.py` - `Tracer` base class and `CoverageTracer` implementation
- `src/ivy/visitor.py` - `_on_node`, `_on_branch`, `_on_boolop`, `_on_loop` hooks
- `src/ivy/stmt.py` - emits branch/loop events from `visit_If`, `visit_Assert`, `visit_For`
- `src/ivy/expr/expr.py` - emits branch/boolop events from `visit_IfExp`, `visit_BoolOp`
- `src/ivy/vyper_interpreter.py` - implements hooks, dispatches to tracers, manages edge state
- `src/ivy/frontend/env.py` - wires up `CoverageTracer`, exposes `execution_metadata`
- `tests/ivy/test_runtime_coverage.py` - smoke tests for branches, edges, boolops, loops

## Design Decisions
- **No `coverage.py`**: Python interpreter coverage saturates quickly and adds high overhead; we want per-contract-AST signals.
- **Hook methods in visitors**: add `_on_node()`, `_on_branch()`, `_on_boolop()`, `_on_loop()` no-op hooks that `VyperInterpreter` overrides.
- **Tracer interface**: interpreter owns `self.tracers: list[Tracer]` and dispatches events to them.
- **No tracer = minimal overhead**: hooks should be a single `if self.tracers: ...` fast path.
- **No storage touch tracking (v1)**: focus on control-flow coverage first.
- **No entrypoint/outcome tracking**: see rationale below.

## Coverage Signals (v1)
All signals are tracked **per contract address**.

- **Branches**: `(addr, node_id, taken)` for `if`, `assert`, `ifexp`
- **BoolOps**: `(addr, node_id, op, evaluated_count, result)`
  - `op ∈ {"and", "or"}`
  - `evaluated_count ∈ [1..N]` (gradient for short-circuit depth)
- **Loops**: `(addr, node_id, bucket)` where `bucket = log2(iteration_count)` style
- **Edges (edge-lite)**: `(addr, prev_node_id, node_id)` for statement/control nodes only

Optional debug-only:
- **Node hits**: `(addr, node_id)` (useful for inspection, but saturates quickly)

### Why No Entrypoint/Outcome Tracking

Originally planned: `(addr, selector, outcome)` with `outcome ∈ {"success", "decode_error", "revert", "error"}`.

**Not implemented because it's redundant for our use case:**

1. **We supply ABI-compliant inputs**: A separate ABI-decoder fuzzer handles malformed calldata. This fuzzer sends valid, well-formed ABI inputs only.

2. **New selector → new edges**: Calling a different function executes different code, which produces new edges. Edge coverage already captures "which function was called".

3. **Reverts are branch-covered**: Reverts come from `assert` or `raise` statements, which are already captured in branch coverage (the condition that led to the revert).

4. **Decode errors don't apply**: Since inputs are ABI-compliant, we never hit decode errors, short calldata, or unknown selectors.

**When entrypoint tracking WOULD matter:**
- Fuzzing the ABI decoder itself (malformed inputs)
- Testing fallback behavior with garbage selectors
- Needing to distinguish success vs revert for identical control flow

For these cases, entrypoint tracking can be added later.

### Edge node whitelist (initial)
Start with statement/control-flow nodes (avoid expression nodes to reduce noise):
- `ast.AnnAssign`, `ast.Assign`, `ast.AugAssign`, `ast.Expr`, `ast.Log`
- `ast.If`, `ast.Assert`, `ast.Raise`, `ast.For`
- `ast.Return`, `ast.Break`, `ast.Continue`, `ast.Pass`

## Semantics / Scoping
- **Edge state is per-call (stack-scoped)**: reset `prev_edge_node_id` at message-call entry and restore at exit so edges never span ABI calls or nested calls.
- **Loop recording is exception-safe**: emit loop coverage from a `finally` so partial iteration counts are still recorded on revert/error.
- **Deployment vs ABI fuzzing**: reset coverage after deployment to avoid constructor/init dominating.

## Integration Point (fuzzer)
Plumb the coverage signal out via `ivy.env` so `src/fuzzer/runner/ivy_scenario_runner.py` can:
- reset coverage after deployments, then
- after each call, decide whether the input is “interesting” based on new coverage.

The simplest “new coverage” heuristic is: treat any new tuple in `branches/boolops/loops/edges` as interesting.

## Implementation Plan

### 1) Add a minimal data model for collected coverage
**New file**: `src/ivy/execution_metadata.py`

- Store the sets listed above.
- Provide:
  - `reset()`
  - `merge(other)`
  - `coverage_signature()`:
    - OK to use Python `hash(...)` for **in-process** dedup.
    - Use a stable digest (`hashlib.blake2b`) if you want to persist across runs.

### 2) Add a tracer interface + CoverageTracer
**New file**: `src/ivy/tracer.py`

Tracer API should be raw-object based (node objects + address), e.g.:
- `on_node(addr, node)` (optional)
- `on_edge(addr, prev_node_id, node_id)`
- `on_branch(addr, node, taken)`
- `on_boolop(addr, node, op, evaluated_count, result)`
- `on_loop(addr, node, iteration_count)`

`CoverageTracer` writes into `ExecutionMetadata`.

### 3) Add hook points to visitors
**File**: `src/ivy/visitor.py`

- Add a no-op `BaseVisitor._on_node(self, node)` and call it at the start of `BaseVisitor.visit()`.

**File**: `src/ivy/stmt.py`

- Add no-op hooks on `StmtVisitor`:
  - `_on_branch(self, node, taken: bool)`
  - `_on_loop(self, node, iteration_count: int)`
- Emit branch events from `visit_If` and `visit_Assert`.
- Wrap `visit_For` with a `try/finally` and emit loop count in `finally`.

**File**: `src/ivy/expr/expr.py`

- Add no-op hooks on `ExprVisitor`:
  - `_on_branch(self, node, taken: bool)` (for `IfExp`)
  - `_on_boolop(self, node, op: str, evaluated_count: int, result: bool)`
- Update `visit_BoolOp` to compute:
  - `op = "and"|"or"`
  - `evaluated_count`
  - `result`
  - and emit exactly once per BoolOp evaluation.

### 4) Implement hooks in `VyperInterpreter` and dispatch to tracers
**File**: `src/ivy/vyper_interpreter.py`

- Add:
  - `self.tracers: list[Tracer]`
  - edge tracking fields: `self._prev_edge_node_id`, `self._edge_stack`
- Implement:
  - `_on_node` to drive edge-lite collection for a whitelisted set of statement/control node types.
  - `_on_branch`, `_on_boolop`, `_on_loop` as tracer fan-out.

### 5) Wire tracer into `Env`
**File**: `src/ivy/frontend/env.py`

- Create and attach a `CoverageTracer` to the interpreter.
- Expose:
  - `Env.execution_metadata` (or `Env.coverage`) for consumers.
  - `Env.reset_execution_metadata()`.
- Ensure `Env.clear_state()` also re-attaches the tracer (it re-instantiates the interpreter).

### 6) Minimal smoke tests (recommended)
**New test**: `tests/test_runtime_coverage.py`

Create tiny contracts and assert that:
- calling the same function with inputs that flip a branch adds new `branches` elements
- BoolOp short-circuit produces different `evaluated_count` buckets
- loop iteration buckets change with different inputs

## Open Questions / Missing Pieces
- **Interestingness heuristic**: should the fuzzer look at raw set growth, a weighted score, or an AFL-style bitmap?
- **Performance knobs**: if overhead is too high, skip `on_node` entirely and emit only statement-level events.
- **Env singleton interactions**: ensure coverage is reset between scenarios/tests (pytest `-n` runs).
- **Fuzzer integration**: coverage infrastructure exists but is not yet consumed by `generative_fuzzer.py` for corpus guidance.

## Files to Add / Modify
- Add: `src/ivy/execution_metadata.py`
- Add: `src/ivy/tracer.py`
- Modify: `src/ivy/visitor.py`
- Modify: `src/ivy/stmt.py`
- Modify: `src/ivy/expr/expr.py`
- Modify: `src/ivy/vyper_interpreter.py`
- Modify: `src/ivy/frontend/env.py`

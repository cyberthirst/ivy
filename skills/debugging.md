---
name: debugging
description: Debugging bugs in Ivy/Vyper/Titanoboa - find root cause, write bug report to BUGS/
---

# Debugging Skill

## Overview
Debug bugs and document findings. Bugs can originate from:
- **Ivy** - AST interpreter bug
- **Vyper** - compiler bug (codegen doesn't match source semantics, compiler crash, etc.)
- **Titanoboa** - Boa/py-evm execution bug

For fuzzer divergences, use the **replay** skill first.

## Workflow

### 1. Create a Worktree
Always debug in an isolated worktree:
```bash
./wt-start.sh debug-<bug-name>
cd <worktree-path>  # path printed by script
```

### 2. Investigate
Inside the worktree, do whatever helps find the root cause:
- Add debug prints anywhere
- Hardcode assumptions to isolate behavior
- Write throwaway tests
- Compare Ivy vs Boa outputs

### 3. Differential Comparison
Compare Ivy vs Boa to pinpoint where behavior diverges:

```python
# tests/debug_comparison.py
def test_differential():
    src = """
@external
def foo() -> uint256:
    x: DynArray[uint256, 10] = [1, 2, 3, 4, 5]
    return x[2]
    """

    from ivy.frontend.loader import loads
    import boa

    ivy_result = loads(src).foo()
    boa_result = boa.loads(src).foo()

    print(f"Ivy: {ivy_result}, Boa: {boa_result}")
    assert ivy_result == boa_result
```

```bash
./ivytest.sh tests/debug_comparison.py -v -s
```

### 4. Write Bug Report
Create `BUGS/<descriptive-name>.md` with your findings.

**Important**: A fix agent will pick up this bug report. Include all context needed to implement a fix without additional investigation.

### 5. Report Back
```
Worktree: debug-<bug-name>
TLDR: <1-2 sentence summary>
Bug report: BUGS/<name>.md
```

## Bug Report Template

```markdown
# Bug Title

## Description
What's happening vs what's expected. Be specific about the symptoms.

## Component
Ivy | Vyper | Titanoboa

## POC
(Optional if complex)

```python
@external
def foo() -> uint256:
    return 42
```

Or: `./ivytest.sh tests/path/to/test.py::test_name`

## Root Cause
Detailed explanation of why the bug occurs. Include:
- The flawed logic or incorrect assumption
- Code path that triggers it
- Why the current behavior is wrong

## Related Files
Files the fix agent should examine:
- `src/ivy/path/to/file.py` - <what's relevant here>
- `src/ivy/other/file.py` - <what's relevant here>

## Fix
(Optional - only if immediately obvious)

```diff
-        if end <= len(arr):
+        if end < len(arr):
```
```

## Example Bug Reports

**Complete report (ready for fix agent):**
```markdown
# Off-by-one in slice bounds

## Description
`x[1:3]` returns 3 elements instead of 2 when x is a DynArray.
Ivy returns `[2, 3, 4]`, Boa returns `[2, 3]`.

## Component
Ivy

## POC
```python
@external
def foo() -> DynArray[uint256, 10]:
    x: DynArray[uint256, 10] = [1, 2, 3, 4, 5]
    return x[1:3]  # Returns [2, 3, 4] instead of [2, 3]
```

## Root Cause
In `visit_Slice`, the end bound calculation uses `<=` instead of `<`.

Python slice semantics are `[start, end)` (end-exclusive), but the current
implementation treats it as `[start, end]` (end-inclusive).

The bug is on line 234:
```python
end_idx = min(end, len(arr))  # Should be: end (already exclusive)
```

The `min()` call is correct for bounds checking, but the slice itself
includes one extra element because the loop uses `<=` comparison.

## Related Files
- `src/ivy/expr/expr.py:230-250` - `visit_Slice` method, contains the bug
- `src/ivy/expr/subscript.py` - related subscript handling, may need review
- `tests/ivy/test_slice.py` - existing slice tests, add regression test here

## Fix
```diff
-        for i in range(start, end_idx + 1):
+        for i in range(start, end_idx):
```
```

**Partial understanding (needs more investigation):**
```markdown
# Storage corruption after reentrancy

## Description
Contract storage shows stale values after reentrant call sequence.
Specifically, a state variable updated in the reentrant call reverts
to its pre-call value when control returns to the outer call.

## Component
Ivy

## POC
See divergence: `divergences/2024-01-15-abc123.json`

Minimal repro not yet isolated - the bug requires a specific call sequence
with reentrancy.

## Root Cause
The bug is in state snapshot handling, but exact cause unclear.

Hypothesis: `env.anchor()` creates a snapshot before external calls, but
the snapshot restoration doesn't account for state changes made during
reentrant calls back into the same contract.

Debug traces show:
1. Contract A calls external contract B
2. `anchor()` saves state at this point
3. B calls back into A (reentrancy)
4. A updates storage variable `x` to new value
5. B returns, A's outer call continues
6. At this point, `x` has reverted to pre-call value

## Related Files
- `src/ivy/frontend/env.py` - `anchor()` context manager, snapshot logic
- `src/ivy/context.py` - execution context, may interact with snapshots
- `src/ivy/stmt.py:external_call` - where anchor is used

## Fix
Needs deeper investigation of snapshot/restore logic.
```

## Quick Commands

```bash
./ivytest.sh tests/path/to/test.py::test_name -v -s   # Run with output
uv run python -m src.fuzzer.replay_divergence path/to/divergence.json
```

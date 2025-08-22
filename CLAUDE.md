# Purpose
- Ivy is an AST interpreter for Vyper (EVM smart contract language).
- The project’s purpose is differential fuzzing: compare Vyper-compiled bytecode (executed in Titanoboa/PyEVM) against Ivy.
- Goal: uncover semantic bugs (miscompilations) where compiled bytecode diverges from source-level semantics.

# Workflow
- Interpreter: executes Vyper AST.
- Contract generator/mutator: produces new inputs by mutating or generating code.
- Exports: JSON traces from Vyper’s test suite, used to validate interpreter correctness and serve as the base corpus for the generator. See `exports-readme.md` if necessary.
- Tests: run with pytest (multicore enabled by default; use 1 core when debugging).
- Key test: `test_replay.py` (replays exports, checks e2e interpreter correctness).
- Run the fuzzer: `python -m src.fuzzer.differential_fuzzer`.
- Filter exports with `TestFilter` (e.g. `include_path("functional/builtins/codegen/test_slice")`).

# Coding Guidelines
- Minimal changes, no unnecessary abstractions. Reuse existing code where possible.
- Write concise, readable code. Comments only for non-obvious logic. Don't write docs.
- Prioritize throughput but not at the expense of readability.
- Focus strictly on compiler correctness bugs, not lexer/parser bugs.

# Research Direction
- Apply knowledge from compiler testing (e.g. EMI, Csmith).
- Suggest 2–3 best approaches (with pros/cons) when exploring new techniques/features.
- Do not drift the project outside differential fuzzing.

# Goals
- Build a continuous fuzzing tool for Vyper dev workflow.
- Aim for a working MVP, not a perfect system.
- Use test modules and debug prints/logging to experiment with new ideas.

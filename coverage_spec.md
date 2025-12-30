# Coverage-Guided Fuzzer Specification

## 1. Core Objective
Implement a feedback-driven fuzzing loop that prioritizes mutations exploring new or rare code paths within the Vyper compiler (`vyper` package). The system shifts from "Brute Force Random Testing" to "Coverage-Guided Hill Climbing".

Currently, the fuzzer is single threaded. But we plan to use the `multiprocessing` library and parallelize it. Keep this in mind when designing the coverage guidance. E.g. we'll need to collect the run and coverage results from all the parallel workers.

## 2. Architecture & Components

### 2.1 The Corpus (Data Structure)
*   **Unit:** `Scenario` object (encapsulates Source Code, Runtime Arguments, and Metadata).
*   **Storage:** In-Memory `Corpus` class (List/Queue).
    *   *Future:* Pluggable backend for disk/database persistence.
*   **Deduplication:** Plumbed but **currently a no-op**. The loop will call a `Deduper` hook post-execution; it will eventually deduplicate on behavior (coverage fingerprint + failure/divergence signature). The acceptance policy should also retain a small set of "best representatives" (smallest/fastest) for each coverage region to prevent corpus bloat.

### 2.2 Coverage Metric
*   **Guidance Target:** The `vyper` python package, **scoring only** arcs in `vyper/codegen`, `vyper/ir`, `vyper/venom`, `vyper/evm` (optionally `vyper/builtins`).
    *   Arcs outside these modules may execute during compilation but are ignored for guidance.
*   **Type:** **Arc/Branch Coverage** (via `coverage.py` arcs).
    *   Arcs (edges) distinguish branch decisions more precisely than line hits.
*   **Collection boundary:** Enable coverage only around **Vyper compilation**. Deployment/execution must run with coverage disabled to keep overhead and noise low. If a helper (e.g. `boa.loads`) does compile+deploy, split those phases explicitly.
*   **Edge IDs:** Arcs are hashed into a fixed-size, shared counter map (AFL-style). Counters must be **saturating** to avoid wraparound (no 255 -> 0).
*   **Metric:** **"Rare Edge Priority"**.
    *   Global counter map: `GlobalEdgeCounts[edge_id] = total_hits`.
    *   Scenario Score: $\sum_{edge \in Scenario} \frac{1}{GlobalEdgeCounts[edge] + 1}$ (computed pre-merge).
    *   Normalize by **compile_time** for compiler guidance; keep runtime scoring separate if/when runtime coverage is added.
*   **Introspection (optional):** Scenario-local coverage resets should not be the only source for human-readable reports. If reports are needed, accumulate a separate "ever-hit arcs" set or replay a corpus subset offline.

### 2.3 Collection & Multi-Config Compilation
*   Coverage is collected only around compilation call sites, not the full scenario run.
*   `MultiRunner` compiles multiple configurations; record arcs for each **successful** compile.
    *   A failure in one config should not discard coverage from others; record the failure as metadata.
    *   Optionally tag edge IDs with the compiler config to preserve config-specific signals.

#### 2.3.1 Boa-Only Coverage Collection
*   Coverage is enabled **only** in Boa runners (Ivy runners never enable it).
*   Use a compile-then-deploy split to keep coverage compile-only:
    *   Start coverage, call `boa.loads_partial(..., compiler_args=merged_args, no_vvm=True)`, stop coverage.
    *   Deploy with `deployer.deploy(...)` after coverage is stopped.
*   Filter arcs by filename before hashing:
    *   Keep only arcs whose filenames contain one of the guidance targets (`vyper/codegen`, `vyper/ir`, `vyper/venom`, `vyper/evm`, optional `vyper/builtins`).
    *   Drop arcs from parser/import/semantics to avoid front-end bias.

### 2.5 Parallel Execution & Shared State
*   **Shared edge map:** created by the coordinator process as a shared-memory array (e.g., `multiprocessing.Array`) and passed to workers; workers merge edges with saturating, lock-free increments.
*   **Corpus synchronization:** each worker maintains a local index of corpus metadata and periodically scans the corpus directory for new entries (no full reload per sample). Payloads are loaded only when selected.
*   **Work distribution:** workers sample from their local indices; a central scheduler is optional and not required for correctness.
*   **Bounding growth:** enforce caps to avoid unbounded corpus size:
    *   Keep only (a) new-edge discoverers, (b) best-known representatives per coverage fingerprint (smallest/fastest), and (c) divergences/crashes.
    *   Apply global caps (max entries or disk size) and evict dominated entries (larger/slower with same coverage), preserving sole representatives and divergences/crashes.

### 2.4 The Fuzzing Loop
1.  **Bootstrap:**
    *   Load all valid test exports from `tests/vyper-exports`. The `include_path` and `exclude_path` from TestFilter should be respected.
    *   Compile each to establish the baseline `GlobalEdgeCounts`.
    *   Populate the initial Corpus.

2.  **Selection:**
    *   **Weighted Random Selection** using `rare_edge_score = sum(1 / (GlobalEdgeCounts[edge] + 1))`.

3.  **Mutation:**
    *   Use the `@src/fuzzer/mutator/ast_mutator.py`

4.  **Execution (Differential Testing):**
    *   Just plug into the existing `MultiRunner` from `@src/fuzzer/runner/multi_runner.py`.
    *   Ensure compiler coverage collection wraps only the compilation steps.
    *   Track `compile_time` separately from runtime so compiler guidance is not penalized by slow execution.

5.  **Evaluation (The Gatekeeper):**
    *   **Inputs:** Mutated `Scenario`, coverage run results (edges hit + counts), execution outcome (success/exception/crash/divergence).
    *   **Step 1: Filter out invalids.**
        *   If **all** compiler configs fail to compile, record the failure reason and drop the scenario.
        *   If only some configs fail, keep coverage from successful configs and record the failures in metadata.
        *   If coverage data is missing/incomplete, drop the scenario.
        *   Divergences/crashes are always retained regardless of coverage novelty.
    *   **Step 2: Score the scenario.**
        *   Compute `rare_edge_score = sum(1 / (GlobalEdgeCounts[edge] + 1))` using pre-run counts.
        *   Set `selection_weight = rare_edge_score / max(compile_time, eps)`.
    *   **Step 3: Deduplication hook.**
        *   Call `Deduper` with a behavior fingerprint (coverage set + exception/crash/divergence signature). **No-op for now**; always passes.
    *   **Step 4: Decide if it enters the Corpus.**
        *   Accept if it hits **any new edge id** not yet in `GlobalEdgeCounts`.
        *   Otherwise accept if it improves a **best-known representative** for its coverage fingerprint (smaller source or faster compile).
        *   Divergence/crash inputs always enter the corpus.
    *   **Step 5: Update global state.**
        *   Update `GlobalEdgeCounts` with the observed edge hits (even if the scenario is rejected).
        *   If accepted, append the scenario to the Corpus with its measured metadata.

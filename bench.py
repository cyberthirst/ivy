"""Unified compilation + coverage benchmark for the Vyper program generator.

Generates N random Vyper programs, compiles each under coverage instrumentation,
and reports success rate + unique arc counts as JSON to stdout.
Uses all available cores via multiprocessing.

Usage:
    uv run python bench.py --n 2000
    uv run python bench.py --n 2000 --save-failures bench_failures/
    uv run python bench.py --n 2000 -j 4   # limit to 4 workers
"""

import argparse
import json
import multiprocessing
import os
import random
import sys
import traceback
from collections import defaultdict
from pathlib import Path

from fuzzer.coverage.collector import DEFAULT_GUIDANCE_TARGETS

OUTCOME_SUCCESS = "success"
OUTCOME_XFAIL = "xfail"
OUTCOME_FAILED = "failed"
OUTCOME_ICE = "ice"


def _run_one(_):
    """Worker: generate one random program, compile under coverage, return results."""
    seed_i = random.randint(0, 2**32 - 1)

    # imports inside worker to avoid issues with forked coverage state
    from vyper.ast import nodes as ast
    from fuzzer.compilation import compile_vyper
    from fuzzer.coverage.collector import ArcCoverageCollector
    from fuzzer.mutator.ast_mutator import AstMutator
    from unparser.unparser import unparse

    rng = random.Random(seed_i)
    mutator = AstMutator(rng, generate=True)
    module = mutator.mutate(ast.Module(body=[]))
    source = unparse(module)

    if mutator.type_generator.source_fragments:
        source = "\n\n".join(mutator.type_generator.source_fragments) + "\n\n" + source

    collector = ArcCoverageCollector()
    collector.start_scenario()
    with collector.collect_compile():
        result = compile_vyper(source)
    arcs = collector.get_scenario_arcs()

    failure_text = None
    if result.is_success:
        outcome = OUTCOME_SUCCESS
    elif result.is_compiler_crash:
        outcome = OUTCOME_ICE
    elif mutator.context.compilation_xfails:
        outcome = OUTCOME_XFAIL
    else:
        outcome = OUTCOME_FAILED
        tb = "".join(
            traceback.format_exception(
                type(result.error), result.error, result.error.__traceback__
            )
        )
        failure_text = f"{source}\n\n{tb}"

    return outcome, arcs, seed_i, failure_text


def arcs_by_target(arcs, targets=DEFAULT_GUIDANCE_TARGETS):
    counts = defaultdict(int)
    for filename, _from, _to in arcs:
        for target in targets:
            if target in filename:
                counts[target] += 1
                break
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description="Generator compilation + coverage benchmark")
    parser.add_argument("--n", type=int, default=2000, help="Number of programs (default: 2000)")
    parser.add_argument("--save-failures", type=str, default=None, metavar="DIR",
                        help="Directory to save compilation failures")
    parser.add_argument("-j", "--jobs", type=int, default=None,
                        help="Number of parallel workers (default: all cores)")
    args = parser.parse_args()

    failures_dir = None
    if args.save_failures:
        failures_dir = Path(args.save_failures)
        failures_dir.mkdir(parents=True, exist_ok=True)

    total = args.n
    n_workers = args.jobs or os.cpu_count() or 1

    all_arcs = set()
    successful = 0
    successful_xfail = 0
    failed = 0
    ice = 0
    done = 0

    with multiprocessing.Pool(n_workers) as pool:
        for outcome, arcs, seed_i, failure_text in pool.imap_unordered(_run_one, range(total), chunksize=16):
            all_arcs |= arcs

            if outcome == OUTCOME_SUCCESS:
                successful += 1
            elif outcome == OUTCOME_ICE:
                ice += 1
            elif outcome == OUTCOME_XFAIL:
                successful_xfail += 1
            else:
                failed += 1
                if failures_dir and failure_text:
                    failure_path = failures_dir / f"seed_{seed_i}.txt"
                    failure_path.write_text(failure_text)

            done += 1
            if done % 100 == 0:
                print(f"Progress: {done}/{total}", file=sys.stderr)

    success_rate = successful / total if total > 0 else 0.0
    target_counts = arcs_by_target(all_arcs)

    output = {
        "n": total,
        "success": successful,
        "successful_xfail": successful_xfail,
        "failed": failed,
        "ice": ice,
        "success_rate": round(success_rate, 4),
        "total_unique_arcs": len(all_arcs),
        "arcs_by_target": target_counts,
    }

    json.dump(output, sys.stdout, indent=2)
    print(file=sys.stdout)


if __name__ == "__main__":
    main()

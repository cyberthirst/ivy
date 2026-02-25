"""Unified compilation + coverage benchmark for the Vyper program generator.

Generates N random Vyper programs, compiles each under coverage instrumentation,
and reports success rate + unique arc counts as JSON to stdout.
Uses all available cores via multiprocessing.

Usage:
    uv run python covbench.py --n 2000
    uv run python covbench.py --n 2000 --save-failures bench_failures/
    uv run python covbench.py --n 2000 -j 4   # limit to 4 workers
    uv run python covbench.py --n 2000 --mutate   # mutate existing exports
"""

import argparse
import json
import multiprocessing
import os
import random
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from fuzzer.coverage.collector import DEFAULT_GUIDANCE_TARGETS

OUTCOME_SUCCESS = "success"
OUTCOME_XFAIL = "xfail"
OUTCOME_FAILED = "failed"
OUTCOME_ICE = "ice"


@dataclass(frozen=True)
class FocusRegion:
    file: str
    start: int
    stop: int

    @property
    def key(self) -> str:
        return f"{self.file}:{self.start}:{self.stop}"

    @property
    def interval_lines(self) -> int:
        return self.stop - self.start + 1


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/")


def _parse_focus_region(value: str) -> FocusRegion:
    try:
        file_name, start_raw, stop_raw = value.rsplit(":", 2)
    except ValueError as exc:
        raise ValueError("expected FILE:START:STOP format") from exc

    if not file_name:
        raise ValueError("file name cannot be empty")

    try:
        start = int(start_raw)
        stop = int(stop_raw)
    except ValueError as exc:
        raise ValueError("START and STOP must be integers") from exc

    if start <= 0 or stop <= 0:
        raise ValueError("START and STOP must be positive")
    if stop < start:
        raise ValueError("STOP must be >= START")

    return FocusRegion(file=_normalize_path(file_name), start=start, stop=stop)


def _focus_file_matches(arc_file: str, focus_file: str) -> bool:
    norm_arc = _normalize_path(arc_file)
    if norm_arc == focus_file:
        return True
    if norm_arc.endswith(f"/{focus_file}"):
        return True
    return focus_file in norm_arc


def _new_focus_accumulator(region: FocusRegion) -> dict:
    return {
        "region": region,
        "programs_touching": 0,
        "arcs_touching": set(),
        "arcs_inside": set(),
        "arcs_entering": set(),
        "arcs_exiting": set(),
        "executable_lines": set(),
        "lines_hit": set(),
    }


def _collect_focus_line_sets(
    focus_regions: tuple[FocusRegion, ...], file_analysis: dict
) -> dict:
    region_line_sets = {}
    for region in focus_regions:
        region_hit_lines = set()
        region_executable_lines = set()

        for analysis_file, (executable_lines, missing_lines) in file_analysis.items():
            if not _focus_file_matches(analysis_file, region.file):
                continue

            for line in executable_lines:
                if not (region.start <= line <= region.stop):
                    continue
                line_id = (analysis_file, line)
                region_executable_lines.add(line_id)
                if line not in missing_lines:
                    region_hit_lines.add(line_id)

        region_line_sets[region.key] = (region_hit_lines, region_executable_lines)

    return region_line_sets


def _update_focus_accumulators(
    focus_accumulators: dict, arcs: set, line_sets_by_region: dict
) -> None:
    for key, acc in focus_accumulators.items():
        touched = False
        if key in line_sets_by_region:
            hit_lines, executable_lines = line_sets_by_region[key]
            acc["lines_hit"].update(hit_lines)
            acc["executable_lines"].update(executable_lines)
            touched = bool(hit_lines)

        region = acc["region"]

        for arc_file, from_line, to_line in arcs:
            if not _focus_file_matches(arc_file, region.file):
                continue

            from_in = region.start <= from_line <= region.stop
            to_in = region.start <= to_line <= region.stop
            if not (from_in or to_in):
                continue

            touched = True
            arc = (arc_file, from_line, to_line)
            acc["arcs_touching"].add(arc)

            if from_in and to_in:
                acc["arcs_inside"].add(arc)
            elif not from_in and to_in:
                acc["arcs_entering"].add(arc)
            elif from_in and not to_in:
                acc["arcs_exiting"].add(arc)

        if touched:
            acc["programs_touching"] += 1


def _build_focus_output(focus_accumulators: dict, total_programs: int) -> dict:
    output = {}
    for key, acc in focus_accumulators.items():
        region = acc["region"]
        interval_lines = region.interval_lines
        unique_executable_lines = len(acc["executable_lines"])
        unique_lines_hit = len(acc["lines_hit"])
        programs_touching = acc["programs_touching"]

        output[key] = {
            "file": region.file,
            "start": region.start,
            "stop": region.stop,
            "interval_lines": interval_lines,
            "unique_executable_lines": unique_executable_lines,
            "unique_lines_hit": unique_lines_hit,
            "line_coverage_pct": round(
                (
                    unique_lines_hit / unique_executable_lines
                    if unique_executable_lines > 0
                    else 0.0
                ),
                4,
            ),
            "programs_touching": programs_touching,
            "touch_rate": round(
                programs_touching / total_programs if total_programs > 0 else 0.0, 4
            ),
            "unique_arcs_touching": len(acc["arcs_touching"]),
            "unique_arcs_inside": len(acc["arcs_inside"]),
            "unique_arcs_entering": len(acc["arcs_entering"]),
            "unique_arcs_exiting": len(acc["arcs_exiting"]),
        }
    return output


def _compile_with_coverage(source, focus_regions, settings=None):
    from fuzzer.compilation import compile_vyper
    from fuzzer.coverage.collector import ArcCoverageCollector

    collector = ArcCoverageCollector()
    collector.start_scenario()
    with collector.collect_compile():
        result = compile_vyper(source, settings=settings)
    arcs = collector.get_scenario_arcs()
    focus_line_sets = {}
    if focus_regions:
        file_analysis = collector.get_scenario_line_analysis()
        focus_line_sets = _collect_focus_line_sets(focus_regions, file_analysis)
    return result, arcs, focus_line_sets


def _classify_outcome(result, source, compilation_xfails):
    failure_text = None
    if result.is_success:
        outcome = OUTCOME_SUCCESS
    elif result.is_compiler_crash:
        outcome = OUTCOME_ICE
        tb = "".join(
            traceback.format_exception(
                type(result.error), result.error, result.error.__traceback__
            )
        )
        failure_text = f"{source}\n\n{tb}"
    elif compilation_xfails:
        outcome = OUTCOME_XFAIL
    else:
        outcome = OUTCOME_FAILED
        tb = "".join(
            traceback.format_exception(
                type(result.error), result.error, result.error.__traceback__
            )
        )
        failure_text = f"{source}\n\n{tb}"
    return outcome, failure_text


def _load_export_solc_jsons() -> list[dict]:
    """Load unique solc_json dicts from test exports."""
    from fuzzer.export_utils import (
        load_all_exports,
        filter_exports,
        iter_unique_solc_jsons,
        TestFilter,
    )

    test_filter = TestFilter(exclude_multi_module=True, exclude_deps=True)
    exports = load_all_exports("tests/vyper-exports", include_compiler_settings=False)
    exports = filter_exports(exports, test_filter=test_filter)
    return [solc_json for solc_json, _, _ in iter_unique_solc_jsons(exports)]


# module-level storage for worker processes (set via pool initializer)
_WORKER_SOLC_JSONS: list[dict] = []


def _init_mutate_worker(solc_jsons: list[dict]) -> None:
    global _WORKER_SOLC_JSONS
    _WORKER_SOLC_JSONS = solc_jsons


def _run_one_mutate(_, focus_regions: tuple[FocusRegion, ...] = (), settings=None):
    """Worker: pick a random export, mutate it, compile under coverage, return results."""
    seed_i = random.randint(0, 2**32 - 1)

    from ivy.frontend.loader import loads_from_solc_json
    from fuzzer.mutator.ast_mutator import AstMutator

    rng = random.Random(seed_i)
    solc_json = rng.choice(_WORKER_SOLC_JSONS)

    # get annotated AST from the original source
    data = loads_from_solc_json(solc_json, get_compiler_data=True)

    n_mutations = rng.randint(1, 8)
    mutator = AstMutator(rng, max_mutations=n_mutations)
    mutation_result = mutator.mutate_source_with_compiler_data(data)

    if mutation_result is None:
        return (
            OUTCOME_FAILED,
            set(),
            {},
            seed_i,
            f"mutation returned None for seed {seed_i}",
        )

    mutated_source = next(iter(mutation_result.sources.values()))

    result, arcs, focus_line_sets = _compile_with_coverage(
        mutated_source, focus_regions, settings=settings
    )
    outcome, failure_text = _classify_outcome(
        result, mutated_source, mutation_result.compilation_xfails
    )
    return outcome, arcs, focus_line_sets, seed_i, failure_text


def _run_one(_, focus_regions: tuple[FocusRegion, ...] = (), settings=None):
    """Worker: generate one random program, compile under coverage, return results."""
    seed_i = random.randint(0, 2**32 - 1)

    # imports inside worker to avoid issues with forked coverage state
    from vyper.ast import nodes as ast
    from fuzzer.mutator.ast_mutator import AstMutator
    from unparser.unparser import unparse

    rng = random.Random(seed_i)
    mutator = AstMutator(rng, generate=True)
    module = mutator.mutate(ast.Module(body=[]))
    source = unparse(module)

    if mutator.type_generator.source_fragments:
        source = "\n\n".join(mutator.type_generator.source_fragments) + "\n\n" + source

    result, arcs, focus_line_sets = _compile_with_coverage(
        source, focus_regions, settings=settings
    )
    outcome, failure_text = _classify_outcome(
        result, source, mutator.context.compilation_xfails
    )
    return outcome, arcs, focus_line_sets, seed_i, failure_text


def arcs_by_target(arcs, targets=DEFAULT_GUIDANCE_TARGETS):
    counts = defaultdict(int)
    for filename, _from, _to in arcs:
        for target in targets:
            if target in filename:
                counts[target] += 1
                break
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(
        description="Generator compilation + coverage benchmark"
    )
    parser.add_argument(
        "--n", type=int, default=2000, help="Number of programs (default: 2000)"
    )
    parser.add_argument(
        "--save-failures",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory to save compilation failures",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help="Number of parallel workers (default: all cores)",
    )
    parser.add_argument(
        "--mutate",
        action="store_true",
        default=False,
        help="Mutate existing test exports instead of generating from scratch",
    )
    parser.add_argument(
        "--focus",
        action="append",
        default=[],
        metavar="FILE:START:STOP",
        help=(
            "Report detailed coverage stats for this line interval. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--venom",
        action="store_true",
        default=False,
        help="Compile using the Venom pipeline (experimental_codegen)",
    )
    args = parser.parse_args()

    failures_dir = None
    ice_dir = None
    compilation_dir = None
    if args.save_failures:
        failures_dir = Path(args.save_failures)
        ice_dir = failures_dir / "ice"
        compilation_dir = failures_dir / "compilation"
        ice_dir.mkdir(parents=True, exist_ok=True)
        compilation_dir.mkdir(parents=True, exist_ok=True)

    focus_regions = []
    for raw_region in args.focus:
        try:
            focus_regions.append(_parse_focus_region(raw_region))
        except ValueError as exc:
            parser.error(f"invalid --focus value {raw_region!r}: {exc}")
    focus_regions = tuple(dict.fromkeys(focus_regions))
    focus_accumulators = {
        region.key: _new_focus_accumulator(region) for region in focus_regions
    }

    from vyper.compiler.settings import Settings

    compiler_settings = Settings(
        enable_decimals=True,
        experimental_codegen=True if args.venom else None,
    )

    total = args.n
    n_workers = args.jobs or os.cpu_count() or 1

    all_arcs = set()
    successful = 0
    successful_xfail = 0
    failed = 0
    ice = 0
    done = 0

    if args.mutate:
        solc_jsons = _load_export_solc_jsons()
        if not solc_jsons:
            print("ERROR: no sources found in exports", file=sys.stderr)
            sys.exit(1)
        print(f"Loaded {len(solc_jsons)} unique sources from exports", file=sys.stderr)
        run_one = partial(
            _run_one_mutate,
            focus_regions=focus_regions,
            settings=compiler_settings,
        )
        pool = multiprocessing.Pool(
            n_workers, initializer=_init_mutate_worker, initargs=(solc_jsons,)
        )
    else:
        run_one = partial(
            _run_one, focus_regions=focus_regions, settings=compiler_settings
        )
        pool = multiprocessing.Pool(n_workers)

    with pool:
        for outcome, arcs, focus_line_sets, seed_i, failure_text in pool.imap_unordered(
            run_one, range(total), chunksize=16
        ):
            if outcome == OUTCOME_SUCCESS:
                successful += 1
                all_arcs |= arcs
                if focus_accumulators:
                    _update_focus_accumulators(
                        focus_accumulators, arcs, focus_line_sets
                    )
            elif outcome == OUTCOME_ICE:
                ice += 1
                if ice_dir and failure_text:
                    (ice_dir / f"seed_{seed_i}.txt").write_text(failure_text)
            elif outcome == OUTCOME_XFAIL:
                successful_xfail += 1
            else:
                failed += 1
                if compilation_dir and failure_text:
                    (compilation_dir / f"seed_{seed_i}.txt").write_text(failure_text)

            done += 1
            if done % 100 == 0:
                print(f"Progress: {done}/{total}", file=sys.stderr)

    success_rate = successful / total if total > 0 else 0.0
    target_counts = arcs_by_target(all_arcs)

    output = {
        "mode": "mutate" if args.mutate else "generate",
        "venom": args.venom,
        "n": total,
        "success": successful,
        "success_pct": round(successful / total if total > 0 else 0.0, 4),
        "successful_xfail": successful_xfail,
        "successful_xfail_pct": round(
            successful_xfail / total if total > 0 else 0.0, 4
        ),
        "failed": failed,
        "failed_pct": round(failed / total if total > 0 else 0.0, 4),
        "ice": ice,
        "ice_pct": round(ice / total if total > 0 else 0.0, 4),
        "success_rate": round(success_rate, 4),
        "total_unique_arcs": len(all_arcs),
        "arcs_by_target": target_counts,
    }
    if focus_accumulators:
        output["focus_stats"] = _build_focus_output(focus_accumulators, total)

    json.dump(output, sys.stdout, indent=2)
    print(file=sys.stdout)


if __name__ == "__main__":
    main()

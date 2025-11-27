"""
Replay a saved divergence by reconstructing the exact mutated scenario.

Usage:
  PYTHONPATH=src python -m fuzzer.replay_divergence path/to/divergence.json
"""

import json
import sys
from pathlib import Path

from .differential_fuzzer import DifferentialFuzzer
from .export_utils import TestFilter
from .runner.multi_runner import MultiRunner
from .divergence_detector import DivergenceDetector


def _load_divergence(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _find_item(fuzzer: DifferentialFuzzer, item_name: str):
    exports = fuzzer.load_filtered_exports(TestFilter(exclude_multi_module=True))
    for export in exports.values():
        if item_name in export.items:
            return export.items[item_name]
    raise ValueError(f"Test item '{item_name}' not found in exports")


def replay_divergence(divergence_path: Path) -> bool:
    data = _load_divergence(divergence_path)

    item_name = data.get("item_name")
    scenario_num = data.get("scenario_num")
    base_seed = data.get("seed")
    scenario_seed = data.get("scenario_seed")

    if item_name is None or scenario_num is None:
        raise ValueError("Divergence file missing 'item_name' or 'scenario_num'")

    if base_seed is None:
        raise ValueError("Divergence file missing 'seed'")

    if scenario_seed is None:
        raise ValueError("Divergence file missing 'scenario_seed'")

    # Initialize fuzzer with base seed (used for consistency when loading)
    fuzzer = DifferentialFuzzer(seed=base_seed)

    # Locate the item and reconstruct the mutated scenario
    item = _find_item(fuzzer, item_name)
    scenario = fuzzer.create_mutated_scenario(item, scenario_seed=scenario_seed)

    multi_runner = MultiRunner(collect_storage_dumps=True, no_solc_json=True)
    detector = DivergenceDetector()

    results = multi_runner.run(scenario)

    divergences = detector.compare_all_results(
        results.ivy_result, results.boa_results, scenario
    )
    return len(divergences) > 0


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        print("Usage: python -m fuzzer.replay_divergence path/to/divergence.json")
        raise SystemExit(2)

    path = Path(argv[1])
    ok = replay_divergence(path)
    if ok:
        print("Reproduced divergence")
        raise SystemExit(0)
    else:
        print("No divergence reproduced; Ivy and Boa matched")
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv)

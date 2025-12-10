"""
Differential fuzzer for Vyper using test exports.

This module loads test exports, mutates them, and compares
execution between Ivy and the Vyper compiler (via Boa).
"""

import logging
from pathlib import Path
from typing import Optional

from .base_fuzzer import BaseFuzzer
from .export_utils import TestFilter
from .trace_types import TestItem
from .runner.scenario import Scenario, create_scenario_from_item


class TestFuzzer(BaseFuzzer):
    """Fuzzer that uses Vyper test exports for differential testing."""

    def __init__(
        self,
        exports_dir: Path = Path("tests/vyper-exports"),
        seed: Optional[int] = None,
        debug_mode: bool = True,
    ):
        super().__init__(exports_dir=exports_dir, seed=seed, debug_mode=debug_mode)

    def create_mutated_scenario(
        self,
        item: TestItem,
        *,
        scenario_seed: Optional[int] = None,
    ) -> Scenario:
        """Create a scenario from a test item with mutations."""
        scenario = create_scenario_from_item(item, use_python_args=True)
        return self.mutate_scenario(scenario, scenario_seed=scenario_seed)

    def fuzz_exports(
        self,
        test_filter: Optional[TestFilter] = None,
        max_scenarios: int = 30,
    ):
        """Main fuzzing loop following the spec structure."""
        exports = self.load_filtered_exports(test_filter)
        logging.info(
            f"Loaded {sum(len(e.items) for e in exports.values())} test items "
            f"from {len(exports)} files"
        )

        self.reporter.start_timer()
        items_processed = 0

        for export in exports.values():
            for item_name, item in export.items.items():
                if item.item_type == "fixture":
                    continue

                items_processed += 1
                logging.info(f"Testing {item_name} ({items_processed})")

                self.reporter.set_context(item_name, 0, self.seed, scenario_seed=None)

                for scenario_num in range(max_scenarios):
                    scenario_seed = self.derive_scenario_seed(item_name, scenario_num)
                    self.reporter.set_context(
                        item_name, scenario_num, self.seed, scenario_seed
                    )

                    scenario = self.create_mutated_scenario(
                        item, scenario_seed=scenario_seed
                    )

                    analysis = self.run_scenario(scenario)
                    self.reporter.report(analysis, debug_mode=self.debug_mode)

        self.finalize()


def main():
    """Run differential fuzzing with test exports."""
    test_filter = TestFilter(exclude_multi_module=True)
    test_filter.include_path("functional/builtins/codegen/test_slice")
    test_filter.exclude_source(r"\.code")
    test_filter.exclude_name("zero_length_side_effects")

    fuzzer = TestFuzzer()
    fuzzer.fuzz_exports(test_filter=test_filter, max_scenarios=20)


if __name__ == "__main__":
    main()

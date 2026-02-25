"""
Generative fuzzer for Vyper.

Each iteration does a coin flip between generating a fresh contract
from scratch and mutating a random seed (8 AST mutations).
"""

import logging
from pathlib import Path
from typing import List, Optional

from fuzzer.base_fuzzer import BaseFuzzer
from fuzzer.export_utils import TestFilter, exclude_unsupported_patterns
from fuzzer.generator import generate_scenario
from fuzzer.issue_filter import IssueFilter, default_issue_filter
from fuzzer.runtime_engine import HarnessConfig
from fuzzer.runner.scenario import Scenario, create_scenario_from_item


class GenerativeFuzzer(BaseFuzzer):
    """
    Simplified generative fuzzer.

    Each iteration: coin flip between generating a fresh contract
    and mutating a random seed with 8 AST mutations.
    """

    def __init__(
        self,
        exports_dir: Path = Path("tests/vyper-exports"),
        seed: Optional[int] = None,
        debug_mode: bool = True,
        generate_prob: float = 0.5,
        harness_config: HarnessConfig = HarnessConfig(),
        issue_filter: Optional[IssueFilter] = None,
    ):
        super().__init__(
            exports_dir=exports_dir,
            seed=seed,
            debug_mode=debug_mode,
            issue_filter=issue_filter,
            harness_config=harness_config,
        )

        self.seeds: List[Scenario] = []
        self.generate_prob = generate_prob
        self._iteration = 0

    def load_tests_as_seeds(self, test_filter: Optional[TestFilter] = None) -> int:
        """Load test exports as seeds. Returns number of seeds loaded."""
        exports = self.load_filtered_exports(test_filter)

        count = 0
        for export in exports.values():
            for item in export.items.values():
                if item.item_type == "fixture":
                    continue

                scenario = create_scenario_from_item(
                    item, use_python_args=True, include_compiler_settings=False
                )
                self.seeds.append(scenario)
                count += 1

        logging.info(f"Loaded {count} seed scenarios")
        return count

    def fuzz_loop(
        self,
        test_filter: Optional[TestFilter] = None,
        max_iterations: Optional[int] = None,
        log_interval: int = 100,
    ):
        seed_count = self.load_tests_as_seeds(test_filter)
        if seed_count == 0:
            logging.error("No seeds loaded, cannot fuzz")
            return

        self.reporter.start_timer()
        self.reporter.start_metrics_stream(self.harness_config.enable_interval_metrics)

        try:
            while max_iterations is None or self._iteration < max_iterations:
                self._iteration += 1

                scenario_seed = self.derive_scenario_seed("gen", self._iteration)

                if self.rng.random() < self.generate_prob:
                    scenario = generate_scenario(seed=scenario_seed)
                    if scenario is None:
                        continue
                else:
                    base = self.rng.choice(self.seeds)
                    scenario = self.mutate_scenario(
                        base, scenario_seed=scenario_seed, n_mutations=8
                    )

                self.reporter.set_context(
                    f"gen_{self._iteration}",
                    self._iteration,
                    self.seed,
                    scenario_seed,
                )

                artifacts = self.run_scenario_with_artifacts(
                    scenario, seed=scenario_seed
                )
                self.reporter.ingest_run(
                    artifacts.analysis,
                    artifacts,
                    debug_mode=self.debug_mode,
                )

                if self._iteration % log_interval == 0:
                    snapshot = self.reporter.record_interval_metrics(
                        iteration=self._iteration,
                        corpus_seed_count=seed_count,
                        corpus_evolved_count=0,
                        corpus_max_evolved=0,
                        debug_mode=self.debug_mode,
                    )
                    self.reporter.log_generative_progress(
                        iteration=self._iteration,
                        corpus_seed_count=seed_count,
                        corpus_evolved_count=0,
                        snapshot=snapshot,
                        debug_mode=self.debug_mode,
                    )

        except KeyboardInterrupt:
            logging.info("Interrupted by user")

        final_snapshot = self.reporter.record_interval_metrics(
            iteration=self._iteration,
            corpus_seed_count=seed_count,
            corpus_evolved_count=0,
            corpus_max_evolved=0,
            debug_mode=self.debug_mode,
        )
        self.reporter.log_generative_progress(
            iteration=self._iteration,
            corpus_seed_count=seed_count,
            corpus_evolved_count=0,
            snapshot=final_snapshot,
            debug_mode=self.debug_mode,
        )
        self.finalize()


def main():
    """Run generative fuzzing."""
    import boa

    boa.interpret.disable_cache()  # pyright: ignore[reportAttributeAccessIssue]

    test_filter = TestFilter(exclude_multi_module=True, exclude_deps=True)
    exclude_unsupported_patterns(test_filter)
    # Exclude `send(...)` because its success depends on EVM gas stipend semantics.
    # The EVM adds a 2300 gas stipend only when `value > 0` for CALL, so
    # `send(to, 0)` executes the callee with *zero* gas. Whether the call succeeds
    # then depends on the callee's gas needs (e.g., precompiles require gas to run).
    # Ivy does not model gas, so it cannot decide if 2300 is enough (or if 0 is too
    # little) for the recipient to execute, leading to systematic divergences.
    test_filter.exclude_source(r"\bsend\s*\(")
    test_filter.include_path("functional/builtins/codegen/")
    test_filter.exclude_name("zero_length_side_effects")

    issue_filter = default_issue_filter()

    fuzzer = GenerativeFuzzer(
        generate_prob=0.5,
        issue_filter=issue_filter,
        harness_config=HarnessConfig(),
    )
    fuzzer.fuzz_loop(test_filter=test_filter, max_iterations=None)


if __name__ == "__main__":
    main()

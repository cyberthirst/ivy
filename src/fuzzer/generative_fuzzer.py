"""
Generative fuzzer for Vyper using corpus evolution.

Starts with test exports as seeds, then continuously mutates
and evolves the corpus based on successful compilations.
"""

import logging
from pathlib import Path
from typing import Optional

from fuzzer.base_fuzzer import BaseFuzzer
from fuzzer.corpus import FuzzCorpus
from fuzzer.export_utils import TestFilter, exclude_unsupported_patterns
from fuzzer.issue_filter import IssueFilter, default_issue_filter
from fuzzer.runtime_engine import HarnessConfig
from fuzzer.runner.scenario import create_scenario_from_item


class GenerativeFuzzer(BaseFuzzer):
    """
    Queue-based fuzzer with corpus evolution.

    Seeds: original test exports (never removed)
    Evolved: successfully mutated scenarios (bounded, O(1) replacement)
    """

    def __init__(
        self,
        exports_dir: Path = Path("tests/vyper-exports"),
        seed: Optional[int] = None,
        debug_mode: bool = False,
        seed_selection_prob: float = 0.3,
        generate_prob: float = 1e-4,
        harness_config: Optional[HarnessConfig] = None,
        issue_filter: Optional[IssueFilter] = None,
    ):
        super().__init__(
            exports_dir=exports_dir,
            seed=seed,
            debug_mode=debug_mode,
            issue_filter=issue_filter,
            harness_config=harness_config,
        )

        self.corpus: FuzzCorpus = FuzzCorpus(
            rng=self.rng,
            seed_selection_prob=seed_selection_prob,
        )

        self.generate_prob = generate_prob
        self._iteration = 0

    def load_tests_as_seeds(self, test_filter: Optional[TestFilter] = None) -> int:
        """Load test exports as seed corpus. Returns number of seeds loaded."""
        exports = self.load_filtered_exports(test_filter)

        count = 0
        for export in exports.values():
            for item in export.items.values():
                if item.item_type == "fixture":
                    continue

                scenario = create_scenario_from_item(
                    item, use_python_args=True, include_compiler_settings=False
                )
                self.corpus.add_seed(scenario)
                count += 1

        logging.info(f"Loaded {count} seed scenarios into corpus")
        return count

    def bootstrap_corpus(self) -> int:
        """Generate aggressive mutations of each seed to bootstrap the corpus."""
        count = 0

        for i, seed_scenario in enumerate(self.corpus.seeds):
            scenario_seed = self.derive_scenario_seed("bootstrap", i)

            mutated = self.mutate_scenario(
                seed_scenario,
                scenario_seed=scenario_seed,
                n_mutations=8,
            )

            self.reporter.set_context(
                f"bootstrap_{i}",
                i,
                self.seed,
                scenario_seed,
            )

            analysis = self.run_scenario(mutated, seed=scenario_seed)
            self.reporter.report(analysis, debug_mode=self.debug_mode)

            if not analysis.compile_failures and not analysis.crashes:
                self.corpus.add_evolved(mutated)
                count += 1

        logging.info(
            f"Bootstrap: added {count} evolved scenarios from {len(self.corpus.seeds)} seeds"
        )
        return count

    def fuzz_loop(
        self,
        test_filter: Optional[TestFilter] = None,
        max_iterations: Optional[int] = None,
        log_interval: int = 100,
    ):
        """
        Main fuzzing loop.

        Infinite loop (or until max_iterations) that:
        1. Picks a scenario from corpus
        2. Mutates it
        3. Runs differential comparison
        4. Adds successful mutations back to corpus
        """
        seed_count = self.load_tests_as_seeds(test_filter)
        if seed_count == 0:
            logging.error("No seeds loaded, cannot fuzz")
            return

        self.reporter.start_timer()

        try:
            self.bootstrap_corpus()

            # Set max_evolved to 2x the initial corpus size (seeds + bootstrapped)
            self.corpus.max_evolved = 2 * seed_count

            while max_iterations is None or self._iteration < max_iterations:
                self._iteration += 1

                base_scenario = self.corpus.pick()
                if base_scenario is None:
                    logging.error("Corpus empty, stopping")
                    break

                scenario_seed = self.derive_scenario_seed("gen", self._iteration)

                # Rarely use 8 mutations to help escape local optima
                if self.rng.random() < self.generate_prob:
                    n_mutations = 8
                else:
                    n_mutations = 1

                mutated_scenario = self.mutate_scenario(
                    base_scenario, scenario_seed=scenario_seed, n_mutations=n_mutations
                )

                self.reporter.set_context(
                    f"gen_{self._iteration}",
                    self._iteration,
                    self.seed,
                    scenario_seed,
                )

                analysis = self.run_scenario(mutated_scenario, seed=scenario_seed)

                self.reporter.report(analysis, debug_mode=self.debug_mode)

                # Add to evolved corpus if no compilation failures
                # (divergences are fine - we want to keep exploring that space)
                if not analysis.compile_failures and not analysis.crashes:
                    self.corpus.add_evolved(mutated_scenario)

                if self._iteration % log_interval == 0:
                    self._log_progress()

        except KeyboardInterrupt:
            logging.info("Interrupted by user")

        self.finalize()

    def _log_progress(self):
        """Log fuzzing progress."""
        elapsed = self.reporter.get_elapsed_time()
        rate = self._iteration / elapsed if elapsed > 0 else 0

        logging.info(
            f"iter={self._iteration} | "
            f"seeds={self.corpus.seed_count} | "
            f"evolved={self.corpus.evolved_count} | "
            f"divergences={self.reporter.divergences} | "
            f"rate={rate:.1f}/s"
        )


def main():
    """Run generative fuzzing."""
    import boa

    boa.interpret.disable_cache()

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
        seed_selection_prob=0.3,
        issue_filter=issue_filter,
        harness_config=HarnessConfig(),
    )
    fuzzer.fuzz_loop(test_filter=test_filter, max_iterations=None)


if __name__ == "__main__":
    main()

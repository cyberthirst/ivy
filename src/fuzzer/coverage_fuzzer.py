"""
Coverage-guided fuzzing loop for the Vyper compiler (Boa-only coverage).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .base_fuzzer import BaseFuzzer
from .coverage.collector import ArcCoverageCollector
from .coverage.corpus import Corpus, CorpusEntry, scenario_source_size
from .coverage.edge_map import EdgeMap
from .coverage.gatekeeper import Gatekeeper, all_boa_configs_failed_to_compile
from .coverage.tracker import GlobalEdgeTracker
from .export_utils import TestFilter
from .issue_filter import IssueFilter
from .runner.scenario import create_scenario_from_item

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class CoverageGuidedFuzzer(BaseFuzzer):
    def __init__(
        self,
        exports_dir: Path = Path("tests/vyper-exports"),
        *,
        seed: Optional[int] = None,
        debug_mode: bool = False,
        map_size: int = 1 << 20,
        tag_edges_with_config: bool = False,
        issue_filter: Optional[IssueFilter] = None,
    ):
        from boa.interpret import disable_cache

        disable_cache()

        super().__init__(
            exports_dir=exports_dir,
            seed=seed,
            debug_mode=debug_mode,
            issue_filter=issue_filter,
        )

        self.collector = ArcCoverageCollector()
        self.edge_map = EdgeMap(map_size, tag_with_config=tag_edges_with_config)
        self.tracker = GlobalEdgeTracker(map_size)
        self.corpus = Corpus(rng=self.rng)
        self.gatekeeper = Gatekeeper(self.tracker)

    def bootstrap(self, test_filter: Optional[TestFilter] = None) -> int:
        exports = self.load_filtered_exports(test_filter)

        total_items = sum(
            1
            for export in exports.values()
            for item in export.items.values()
            if item.item_type != "fixture"
        )
        logging.info(f"Bootstrap: processing {total_items} seed scenarios...")

        added = 0
        processed = 0
        for export in exports.values():
            for item in export.items.values():
                if item.item_type == "fixture":
                    continue

                processed += 1
                scenario = create_scenario_from_item(item, use_python_args=True)

                self.collector.start_scenario()
                results = self.multi_runner.run(
                    scenario, coverage_collector=self.collector
                )

                analysis = self.result_analyzer.analyze_run(
                    scenario, results.ivy_result, results.boa_results
                )

                if all_boa_configs_failed_to_compile(results.boa_results) and not (
                    analysis.crashes or analysis.divergences
                ):
                    continue

                arcs_by_config = self.collector.get_arcs_by_config()
                if self.edge_map.tag_with_config:
                    edge_ids = self.edge_map.hash_arcs_by_config(arcs_by_config)
                else:
                    edge_ids = self.edge_map.hash_arcs(
                        self.collector.get_scenario_arcs()
                    )

                self.tracker.merge(edge_ids)

                entry = CorpusEntry(
                    scenario=scenario,
                    edge_ids=edge_ids,
                    compile_time_s=max(self.collector.compile_time_s, 1e-9),
                    generation=0,
                    parent_score=None,
                    source_size=scenario_source_size(scenario),
                    keep_forever=True,
                )
                self.corpus.add(
                    entry, coverage_fp=self.gatekeeper_decision_fp(edge_ids)
                )
                added += 1

                if processed % 50 == 0:
                    logging.info(
                        f"Bootstrap: {processed}/{total_items} processed, {added} added"
                    )

        logging.info(f"Bootstrap complete: {added}/{total_items} seeds added")
        return added

    def gatekeeper_decision_fp(self, edge_ids: set[int]) -> str:
        # Avoid importing fingerprint helper at module import time.
        from .coverage.gatekeeper import coverage_fingerprint

        return coverage_fingerprint(edge_ids)

    def fuzz_loop(
        self,
        *,
        test_filter: Optional[TestFilter] = None,
        max_iterations: Optional[int] = None,
        log_interval: int = 100,
    ) -> None:
        if len(self.corpus) == 0:
            if self.bootstrap(test_filter) == 0:
                logging.error("No seeds loaded, cannot fuzz")
                return

        logging.info(
            f"Starting coverage-guided fuzzing with {len(self.corpus)} seeds..."
        )
        self.reporter.start_timer()

        iteration = 0
        try:
            while max_iterations is None or iteration < max_iterations:
                iteration += 1

                parent = self.corpus.select_weighted(self.tracker)
                if parent is None:
                    logging.error("Corpus empty, stopping")
                    break

                scenario_seed = self.derive_scenario_seed("cov", iteration)
                mutated = self.mutate_scenario(
                    parent.scenario, scenario_seed=scenario_seed
                )

                self.collector.start_scenario()
                results = self.multi_runner.run(
                    mutated, coverage_collector=self.collector
                )

                self.reporter.set_context(
                    f"cov_{iteration}", iteration, self.seed, scenario_seed
                )
                analysis = self.result_analyzer.analyze_run(
                    mutated, results.ivy_result, results.boa_results
                )
                self.reporter.report(analysis, debug_mode=self.debug_mode)

                arcs_by_config = self.collector.get_arcs_by_config()
                if self.edge_map.tag_with_config:
                    edge_ids = self.edge_map.hash_arcs_by_config(arcs_by_config)
                else:
                    edge_ids = self.edge_map.hash_arcs(
                        self.collector.get_scenario_arcs()
                    )

                compile_time_s = self.collector.compile_time_s
                coverage_fp = self.gatekeeper_decision_fp(edge_ids)
                improves_rep = self.corpus.improves_representative(
                    coverage_fp=coverage_fp,
                    source_size=scenario_source_size(mutated),
                    compile_time_s=compile_time_s,
                )

                decision = self.gatekeeper.decide_and_update(
                    edge_ids=edge_ids,
                    compile_time_s=compile_time_s,
                    analysis=analysis,
                    boa_results=results.boa_results,
                    improves_representative=improves_rep,
                    coverage_fp=coverage_fp,
                )

                if decision.accept:
                    parent_score = self.tracker.compute_rare_score(
                        parent.edge_ids
                    ) / max(parent.compile_time_s, 1e-9)
                    keep_forever = decision.reason in ("issue", "new_edge")
                    entry = CorpusEntry(
                        scenario=mutated,
                        edge_ids=edge_ids,
                        compile_time_s=max(compile_time_s, 1e-9),
                        generation=parent.generation + 1,
                        parent_score=parent_score,
                        source_size=scenario_source_size(mutated),
                        keep_forever=keep_forever,
                    )
                    self.corpus.add(entry, coverage_fp=decision.coverage_fingerprint)

                if iteration % log_interval == 0:
                    logging.info(
                        f"iter={iteration} | corpus={len(self.corpus)} | "
                        f"unique_divergences={analysis.unique_divergence_count()} | "
                        f"unique_crashes={analysis.unique_crash_count()}"
                    )

        except KeyboardInterrupt:
            logging.info("Interrupted by user")
        finally:
            self.finalize()


def main():
    from .export_utils import apply_unsupported_exclusions
    from .issue_filter import default_issue_filter

    test_filter = TestFilter(exclude_multi_module=True)
    apply_unsupported_exclusions(test_filter)

    issue_filter = default_issue_filter()

    fuzzer = CoverageGuidedFuzzer(issue_filter=issue_filter)
    fuzzer.fuzz_loop(test_filter=test_filter, max_iterations=None)


if __name__ == "__main__":
    main()

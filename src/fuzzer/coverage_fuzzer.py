"""
Coverage-guided fuzzing loop for the Vyper compiler.

Uses the RuntimeFuzzEngine for scenario execution (deployment retries,
call generation/mutation) and ArcCoverageCollector for compiler arc
coverage to guide the scenario-level corpus.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fuzzer.base_fuzzer import BaseFuzzer
from fuzzer.coverage.collector import ArcCoverageCollector
from fuzzer.coverage.corpus import Corpus, CorpusEntry, scenario_source_size
from fuzzer.coverage.edge_map import EdgeMap
from fuzzer.coverage.gatekeeper import Gatekeeper, had_any_successful_compile
from fuzzer.coverage.tracker import GlobalEdgeTracker
from fuzzer.export_utils import TestFilter, exclude_unsupported_patterns
from fuzzer.issue_filter import IssueFilter, default_issue_filter
from fuzzer.runtime_engine import HarnessConfig
from fuzzer.runner.scenario import create_scenario_from_item

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def build_default_coverage_test_filter() -> TestFilter:
    test_filter = TestFilter(exclude_multi_module=True, exclude_deps=True)
    exclude_unsupported_patterns(test_filter)
    test_filter.exclude_source(r"\bsend\s*\(")
    test_filter.include_path("functional/codegen/")
    test_filter.exclude_name("zero_length_side_effects")
    return test_filter


def build_default_coverage_issue_filter() -> IssueFilter:
    return default_issue_filter()


class CoverageGuidedFuzzer(BaseFuzzer):
    def __init__(
        self,
        exports_dir: Path = Path("tests/vyper-exports"),
        *,
        seed: Optional[int] = None,
        debug_mode: bool = True,
        map_size: int = 1 << 20,
        tag_edges_with_config: bool = False,
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

        self.collector = ArcCoverageCollector()
        self.edge_map = EdgeMap(map_size, tag_with_config=tag_edges_with_config)
        self.tracker = GlobalEdgeTracker(map_size)
        self.corpus = Corpus(rng=self.rng)
        self.gatekeeper = Gatekeeper(self.tracker)

    def _hash_collected_arcs(self) -> set[int]:
        if self.edge_map.tag_with_config:
            return self.edge_map.hash_arcs_by_config(
                self.collector.get_arcs_by_config()
            )
        return self.edge_map.hash_arcs(self.collector.get_scenario_arcs())

    def gatekeeper_decision_fp(self, edge_ids: set[int]) -> str:
        from fuzzer.coverage.gatekeeper import coverage_fingerprint

        return coverage_fingerprint(edge_ids)

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
                scenario_seed = self.derive_scenario_seed("bootstrap", processed)

                self.reporter.set_context(
                    f"bootstrap_{processed}", processed, self.seed, scenario_seed
                )

                self.collector.start_scenario()
                artifacts = self.run_scenario_with_artifacts(
                    scenario,
                    seed=scenario_seed,
                    coverage_collector=self.collector,
                )

                self.reporter.ingest_run(
                    artifacts.analysis, artifacts, debug_mode=self.debug_mode
                )

                if not had_any_successful_compile(artifacts.ivy_result):
                    continue

                edge_ids = self._hash_collected_arcs()
                if not edge_ids and not (
                    artifacts.analysis.crashes or artifacts.analysis.divergences
                ):
                    continue

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

    def fuzz_loop(
        self,
        *,
        test_filter: Optional[TestFilter] = None,
        max_iterations: Optional[int] = None,
        log_interval: int = 100,
    ) -> None:
        self.reporter.start_timer()
        self.reporter.start_metrics_stream(self.harness_config.enable_interval_metrics)

        if len(self.corpus) == 0:
            if self.bootstrap(test_filter) == 0:
                logging.error("No seeds loaded, cannot fuzz")
                self.finalize()
                return

        logging.info(
            f"Starting coverage-guided fuzzing with {len(self.corpus)} seeds..."
        )

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

                self.reporter.set_context(
                    f"cov_{iteration}", iteration, self.seed, scenario_seed
                )

                self.collector.start_scenario()
                artifacts = self.run_scenario_with_artifacts(
                    mutated,
                    seed=scenario_seed,
                    coverage_collector=self.collector,
                )

                self.reporter.ingest_run(
                    artifacts.analysis, artifacts, debug_mode=self.debug_mode
                )

                edge_ids = self._hash_collected_arcs()
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
                    analysis=artifacts.analysis,
                    ivy_result=artifacts.ivy_result,
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
                    snapshot = self.reporter.record_interval_metrics(
                        iteration=iteration,
                        corpus_seed_count=len(self.corpus),
                        corpus_evolved_count=0,
                        corpus_max_evolved=0,
                        debug_mode=self.debug_mode,
                    )
                    self.reporter.log_generative_progress(
                        iteration=iteration,
                        corpus_seed_count=len(self.corpus),
                        corpus_evolved_count=0,
                        snapshot=snapshot,
                        debug_mode=self.debug_mode,
                    )

        except KeyboardInterrupt:
            logging.info("Interrupted by user")
        finally:
            self.finalize()


def main():
    import boa  # pyright: ignore[reportMissingImports]

    boa.interpret.disable_cache()  # pyright: ignore[reportAttributeAccessIssue]

    test_filter = build_default_coverage_test_filter()
    issue_filter = build_default_coverage_issue_filter()

    fuzzer = CoverageGuidedFuzzer(
        issue_filter=issue_filter,
        harness_config=HarnessConfig(),
    )
    fuzzer.fuzz_loop(test_filter=test_filter, max_iterations=None)


if __name__ == "__main__":
    main()

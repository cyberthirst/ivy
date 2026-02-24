"""
Base fuzzer infrastructure shared between DifferentialFuzzer and GenerativeFuzzer.
"""

from __future__ import annotations

import logging
import random
import hashlib
import secrets
import typing
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Set

if TYPE_CHECKING:
    from fuzzer.coverage.collector import ArcCoverageCollector

from fuzzer.coverage_types import RuntimeBranchOutcome, RuntimeStmtSite
from fuzzer.runtime_engine.runtime_fuzz_engine import HarnessConfig

from fuzzer.mutator.ast_mutator import AstMutator
from fuzzer.export_utils import (
    load_all_exports,
    filter_exports,
    TestFilter,
    settings_to_kwargs,
)
from fuzzer.trace_types import (
    DeploymentTrace,
    TestExport,
)
from ivy.frontend.loader import loads_from_solc_json

from fuzzer.runner.scenario import Scenario
from fuzzer.runner.multi_runner import MultiRunner
from fuzzer.deduper import Deduper
from fuzzer.result_analyzer import ResultAnalyzer
from fuzzer.reporter import FuzzerReporter
from fuzzer.issue_filter import IssueFilter

from vyper.compiler.phases import CompilerData


DEFAULT_AST_MUTATIONS = 8

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@dataclass
class ScenarioRunArtifacts:
    analysis: Any
    harness_stats: Any
    runtime_edge_ids: Set[int]
    runtime_stmt_sites_seen: Set[RuntimeStmtSite]
    runtime_branch_outcomes_seen: Set[RuntimeBranchOutcome]
    runtime_stmt_sites_total: Set[RuntimeStmtSite]
    runtime_branch_outcomes_total: Set[RuntimeBranchOutcome]
    contract_fingerprints: Set[str]
    finalized_scenario: Scenario


class BaseFuzzer:
    """Base class with shared fuzzer infrastructure."""

    def __init__(
        self,
        exports_dir: Path = Path("tests/vyper-exports"),
        seed: Optional[int] = None,
        debug_mode: bool = True,
        issue_filter: Optional[IssueFilter] = None,
        harness_config: HarnessConfig = HarnessConfig(),
    ):
        self.exports_dir = exports_dir
        self.seed = seed if seed is not None else secrets.randbits(64)
        self.debug_mode = debug_mode
        self.rng = random.Random(self.seed)
        self.harness_config = harness_config

        self.deduper = Deduper()
        self.reporter = FuzzerReporter(seed=self.seed)
        self.issue_filter = issue_filter
        self.result_analyzer = ResultAnalyzer(self.deduper, issue_filter)

        # Multi-runner (created lazily or by subclass)
        self._multi_runner: Optional[MultiRunner] = None

    @property
    def multi_runner(self) -> MultiRunner:
        if self._multi_runner is None:
            self._multi_runner = MultiRunner(
                collect_storage_dumps=self.harness_config.collect_storage_dumps,
            )
        return self._multi_runner

    def load_filtered_exports(
        self, test_filter: Optional[TestFilter] = None
    ) -> Dict[Path, TestExport]:
        exports = load_all_exports(self.exports_dir, include_compiler_settings=False)
        if test_filter:
            exports = filter_exports(exports, test_filter=test_filter)
        return exports

    def get_compiler_data(self, trace: DeploymentTrace) -> Optional[CompilerData]:
        """Get CompilerData for a deployment trace."""
        if not trace.solc_json:
            return None

        try:
            return typing.cast(
                CompilerData,
                loads_from_solc_json(trace.solc_json, get_compiler_data=True),
            )
        except Exception as e:
            logging.debug(f"Failed to load compiler data ({type(e).__name__}): {e}")
            return None

    def derive_scenario_seed(self, base: str, num: int) -> int:
        """Derive a deterministic per-scenario seed."""
        h = hashlib.blake2b(
            f"{self.seed}|{base}|{num}".encode("utf-8"), digest_size=16
        ).digest()
        return int.from_bytes(h, byteorder="big", signed=False)

    def mutate_scenario(
        self,
        scenario: Scenario,
        scenario_seed: Optional[int] = None,
        *,
        n_mutations: int = DEFAULT_AST_MUTATIONS,
    ) -> Scenario:
        """
        Apply AST source mutations to deployment traces in a scenario.

        Non-deployment traces (calls, set-balance, etc.) pass through unchanged;
        call-level mutations are handled by RuntimeFuzzEngine.
        """
        rng = random.Random(scenario_seed) if scenario_seed else self.rng
        ast_mutator = AstMutator(rng, max_mutations=n_mutations)

        new_scenario = deepcopy(scenario)

        for trace in new_scenario.traces:
            if not (
                isinstance(trace, DeploymentTrace) and trace.deployment_type == "source"
            ):
                continue

            compiler_data = self.get_compiler_data(trace)

            if trace.solc_json and compiler_data:
                mutation_result = ast_mutator.mutate_source_with_compiler_data(
                    compiler_data
                )
                if mutation_result:
                    sources = trace.solc_json.get("sources")
                    if sources is None:
                        sources = {}
                        trace.solc_json["sources"] = sources
                    changed = False
                    for filename, content in mutation_result.sources.items():
                        entry = sources.get(filename)
                        if isinstance(entry, dict):
                            if entry.get("content") != content:
                                entry["content"] = content
                                changed = True
                        else:
                            if entry != content:
                                sources[filename] = {"content": content}
                                changed = True

                    if changed:
                        trace.compilation_xfails = list(
                            trace.compilation_xfails
                        ) + list(mutation_result.compilation_xfails)
                        trace.runtime_xfails = list(trace.runtime_xfails) + list(
                            mutation_result.runtime_xfails
                        )

        return new_scenario

    def run_scenario_with_artifacts(
        self,
        scenario: Scenario,
        *,
        seed: Optional[int] = None,
        coverage_collector: Optional["ArcCoverageCollector"] = None,
    ) -> ScenarioRunArtifacts:
        """Run a scenario and return analysis plus runtime metrics/artifacts."""
        from fuzzer.runtime_engine.runtime_fuzz_engine import RuntimeFuzzEngine

        harness = RuntimeFuzzEngine(self.harness_config, seed)
        harness_result = harness.run(scenario)

        results = self.multi_runner.run_boa_only(
            harness_result.finalized_scenario,
            harness_result.ivy_result,
            coverage_collector=coverage_collector,
        )

        analysis = self.result_analyzer.analyze_run(
            harness_result.finalized_scenario,
            harness_result.ivy_result,
            results.boa_results,
        )

        return ScenarioRunArtifacts(
            analysis=analysis,
            harness_stats=harness_result.stats,
            runtime_edge_ids=set(harness_result.runtime_edge_ids),
            runtime_stmt_sites_seen=set(harness_result.runtime_stmt_sites_seen),
            runtime_branch_outcomes_seen=set(
                harness_result.runtime_branch_outcomes_seen
            ),
            runtime_stmt_sites_total=set(harness_result.runtime_stmt_sites_total),
            runtime_branch_outcomes_total=set(
                harness_result.runtime_branch_outcomes_total
            ),
            contract_fingerprints=set(harness_result.contract_fingerprints),
            finalized_scenario=harness_result.finalized_scenario,
        )

    def run_scenario(
        self,
        scenario: Scenario,
        *,
        seed: Optional[int] = None,
    ):
        """Run a scenario and analyze results."""
        return self.run_scenario_with_artifacts(scenario, seed=seed).analysis

    def finalize(self):
        """Stop timer and output final reports."""
        self.reporter.stop_timer()
        self.reporter.print_summary()
        self.reporter.save_statistics()  #

"""
Arc coverage collector for Vyper compiler guidance.

Uses coverage.py's C tracer (not sys.settrace) to record line-to-line arcs during
the compilation window.

Arcs are kept in-memory only, and we don't read any coverage configuration files.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import DefaultDict, Dict, Iterable, Iterator, Optional, Set, Tuple

import coverage

Arc = Tuple[str, int, int]
FileLineAnalysis = Tuple[Set[int], Set[int]]

DEFAULT_GUIDANCE_TARGETS: Tuple[str, ...] = (
    "vyper/codegen",
    "vyper/ir",
    "vyper/venom",
    "vyper/evm",
    "vyper/builtins",
    "vyper/semantics",
)


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/")


def _unique_preserve_order(items: Iterable[str]) -> Tuple[str, ...]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return tuple(unique)


class ArcCoverageCollector:
    """
    Collects executed arcs (line-to-line edges) during compiler compilation.

    Call `start_scenario()` once per scenario, and wrap compilation call sites
    in `collect_compile(...)`.
    """

    def __init__(
        self,
        *,
        guidance_targets: Iterable[str] = DEFAULT_GUIDANCE_TARGETS,
    ):
        self._targets = tuple(guidance_targets)

        self._scenario_arcs: Set[Arc] = set()
        self._scenario_arcs_by_config: DefaultDict[str, Set[Arc]] = defaultdict(set)

        self._compile_time_s: float = 0.0
        self._compile_time_by_config_s: DefaultDict[str, float] = defaultdict(float)

        self._active: bool = False

        # coverage.py uses glob patterns where `*` does not match directory
        # separators. Use `**/` patterns to match the target subpath anywhere.
        include_patterns: list[str] = []
        for target in self._targets:
            norm_target = _normalize_path(target).strip("/")
            if not norm_target:
                continue

            include_patterns.append(f"**/{norm_target}")
            include_patterns.append(f"**/{norm_target}/**")

            parts = [p for p in norm_target.split("/") if p]
            fuzzy = "/".join(f"*{p}*" for p in parts)
            include_patterns.append(f"**/{fuzzy}")
            include_patterns.append(f"**/{fuzzy}/**")

        self._include_patterns = _unique_preserve_order(include_patterns)

        # Reusable Coverage instance (created lazily on first use).
        self._cov: Optional[coverage.Coverage] = None

    def start_scenario(self) -> None:
        self._scenario_arcs.clear()
        self._scenario_arcs_by_config.clear()
        self._compile_time_s = 0.0
        self._compile_time_by_config_s.clear()

    @property
    def compile_time_s(self) -> float:
        return self._compile_time_s

    @property
    def compile_time_by_config_s(self) -> Dict[str, float]:
        return dict(self._compile_time_by_config_s)

    def get_scenario_arcs(self) -> Set[Arc]:
        return set(self._scenario_arcs)

    def get_arcs_by_config(self) -> Dict[str, Set[Arc]]:
        return {k: set(v) for k, v in self._scenario_arcs_by_config.items()}

    def get_scenario_line_analysis(self) -> Dict[str, FileLineAnalysis]:
        """
        Return per-file executable and missing line numbers from the latest run.

        The returned mapping is:
            filename -> (executable_lines, missing_lines)
        """
        if self._cov is None:
            return {}

        data = self._cov.get_data()
        analysis: Dict[str, FileLineAnalysis] = {}
        for filename in data.measured_files():
            if not self._should_trace_file(filename):
                continue

            try:
                canonical, executable, _excluded, missing, _missing_fmt = (
                    self._cov.analysis2(filename)
                )
            except coverage.CoverageException:
                continue

            analysis[canonical] = (
                set(executable),
                set(missing),
            )

        return analysis

    def _should_trace_file(self, filename: str) -> bool:
        norm = _normalize_path(filename)
        return any(target in norm for target in self._targets)

    def _get_coverage(self) -> coverage.Coverage:
        """Get or create the reusable Coverage instance."""
        if self._cov is not None:
            return self._cov

        cov = coverage.Coverage(
            branch=True,
            timid=False,
            config_file=False,
            data_file=None,
            include=self._include_patterns if self._include_patterns else None,
        )
        cov.config.disable_warnings = ["no-data-collected"]
        self._cov = cov
        return cov

    def _extract_arcs(self, cov: coverage.Coverage) -> Set[Arc]:
        data = cov.get_data()
        if not data.has_arcs():
            return set()

        arcs: Set[Arc] = set()
        for filename in data.measured_files():
            if not self._should_trace_file(filename):
                continue

            file_arcs = data.arcs(filename)
            if not file_arcs:
                continue

            for from_line, to_line in file_arcs:
                # coverage.py uses negative line numbers for synthetic arcs
                # (function entry/exit). Filter those out.
                if from_line <= 0 or to_line <= 0:
                    continue
                if from_line == to_line:
                    continue
                arcs.add((filename, from_line, to_line))

        return arcs

    @contextmanager
    def collect_compile(self, *, config_name: Optional[str] = None) -> Iterator[None]:
        if self._active:
            raise RuntimeError("ArcCoverageCollector is already collecting")

        prev_trace = sys.gettrace()
        cov = self._get_coverage()

        # Clear data from previous collection (reusing the Coverage instance).
        # Workaround for https://github.com/coveragepy/coveragepy/issues/2138:
        # erase() skips closing in-memory SQLite connections in no_disk mode,
        # leaking ~130 KB per cycle.
        # See also https://github.com/coveragepy/coveragepy/issues/2139 for
        # a clear_data() API that would avoid needing erase() entirely.
        if cov._data is not None:
            cov._data.close(force=True)
        cov.erase()

        self._active = True

        t0: Optional[float] = None
        success = False
        started = False
        try:
            cov.start()
            started = True

            # Match the old sys.settrace collector: start timing after tracing
            # is enabled, so compile_time measures just the compilation window.
            t0 = time.perf_counter()
            yield
            success = True
        finally:
            dt = (time.perf_counter() - t0) if t0 is not None else 0.0

            try:
                if started:
                    cov.stop()
            finally:
                # Restore any pre-existing tracer (debuggers, pytest-cov, etc).
                sys.settrace(prev_trace)

            self._compile_time_s += dt
            if config_name is not None:
                self._compile_time_by_config_s[config_name] += dt

            if success:
                arcs = self._extract_arcs(cov)
                self._scenario_arcs.update(arcs)
                if config_name is not None:
                    self._scenario_arcs_by_config[config_name].update(arcs)

            self._active = False

"""
Deduplication for fuzzer scenarios.

Filters duplicate scenarios based on behavior fingerprints to avoid
redundant output (crashes, compilation failures, divergences).

Fingerprinting Strategy
-----------------------

| Category                              | Dedup? | Fingerprint                                            |
|---------------------------------------|--------|--------------------------------------------------------|
| Divergence: both ok, different result | No     | -                                                      |
| Divergence: one ok, one failed        | Yes    | (type, runner, error_type, msg[:20], last_3_frames)    |
| Divergence: xfail violation           | Yes    | (xfail_expected, xfail_actual, xfail_reasons)          |
| Compiler crash                        | Yes    | (error_type, msg[:20], last_3_frames)                  |
| Compilation failure                   | Yes    | (error_type, msg[:20], last_5_frames)                  |
| Compilation timeout                   | Yes    | (error_type, msg[:20], last_5_frames)                  |

Error fingerprinting uses:
- error_type: Exception class name (e.g., "CompilerPanic")
- msg[:20]: First 20 chars of first line of error message
- last_N_frames: Last N stack frames as (filename:lineno:funcname)
"""

import hashlib
import traceback
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from fuzzer.divergence_detector import Divergence, DivergenceType


def extract_stack_frames(error: Optional[Exception], n: int = 3) -> Tuple[str, ...]:
    """
    Extract last N frames from exception traceback.

    Normalizes frame info to (filename, lineno, funcname) for stability.
    """
    if error is None:
        return ()

    tb = getattr(error, "__traceback__", None)
    if tb is None:
        return ()

    # Extract all frames
    frames = traceback.extract_tb(tb)
    if not frames:
        return ()

    # Take last N frames
    last_frames = frames[-n:] if len(frames) >= n else frames

    # Normalize: use just filename (not full path), lineno, and function name
    normalized = []
    for frame in last_frames:
        # Use basename of filename for stability across machines
        filename = frame.filename.split("/")[-1]
        normalized.append(f"{filename}:{frame.lineno}:{frame.name}")

    return tuple(normalized)


def fingerprint_error(
    error: Optional[Exception], n_frames: int = 3
) -> Tuple[str, str, Tuple[str, ...]]:
    """
    Build fingerprint tuple from an exception.

    Returns (error_type, msg_prefix, stack_frames).
    """
    if error is None:
        return ("", "", ())

    error_type = type(error).__name__
    try:
        msg = str(error).split("\n")[0][:20]  # First line, first 20 chars
    except Exception:
        msg = repr(error).split("\n")[0][:20]
    frames = extract_stack_frames(error, n_frames)

    return (error_type, msg, frames)


@dataclass
class KeepDecision:
    """Result of deduplication check."""

    keep: bool
    reason: str
    fingerprint: str


class Deduper:
    """
    Deduplicates scenarios based on behavior fingerprints.

    Tracks seen fingerprints for: compiler crashes, compilation failures,
    and divergences. Each category maintains its own seen set.
    """

    # Default stack frame counts
    CRASH_FRAMES = 3
    COMPILE_FAILURE_FRAMES = 5
    DIVERGENCE_FRAMES = 3

    def __init__(
        self,
        *,
        seen_crashes: Optional[Dict[str, bool]] = None,
        seen_compile_failures: Optional[Dict[str, bool]] = None,
        seen_compilation_timeouts: Optional[Dict[str, bool]] = None,
        seen_divergences: Optional[Dict[str, bool]] = None,
    ):
        self._seen_crashes: Dict[str, bool] = (
            seen_crashes if seen_crashes is not None else {}
        )
        self._seen_compile_failures: Dict[str, bool] = (
            seen_compile_failures if seen_compile_failures is not None else {}
        )
        self._seen_compilation_timeouts: Dict[str, bool] = (
            seen_compilation_timeouts if seen_compilation_timeouts is not None else {}
        )
        self._seen_divergences: Dict[str, bool] = (
            seen_divergences if seen_divergences is not None else {}
        )

    def _compute_fingerprint(self, sig: Tuple) -> str:
        """Compute blake2b fingerprint from a signature tuple."""
        h = hashlib.blake2b(repr(sig).encode("utf-8"), digest_size=16)
        return h.hexdigest()

    def _check_seen(
        self, sig: Tuple, seen: Dict[str, bool], category: str
    ) -> KeepDecision:
        fingerprint = self._compute_fingerprint(sig)
        if fingerprint in seen:
            return KeepDecision(
                keep=False, reason=f"duplicate_{category}", fingerprint=fingerprint
            )
        seen[fingerprint] = True
        return KeepDecision(
            keep=True, reason=f"new_{category}", fingerprint=fingerprint
        )

    def check_divergence(self, divergence: Divergence) -> KeepDecision:
        if divergence.type == DivergenceType.XFAIL:
            return self._check_xfail(divergence)
        elif divergence.type in (DivergenceType.DEPLOYMENT, DivergenceType.EXECUTION):
            return self._check_result_divergence(divergence)
        else:
            return KeepDecision(keep=True, reason="unknown_type", fingerprint="")

    def _check_xfail(self, divergence: Divergence) -> KeepDecision:
        sig = (
            "xfail",
            divergence.xfail_expected,
            divergence.xfail_actual,
            tuple(divergence.xfail_reasons),
        )
        return self._check_seen(sig, self._seen_divergences, "divergence")

    def _check_result_divergence(self, divergence: Divergence) -> KeepDecision:
        ivy_success = divergence.ivy_result.success if divergence.ivy_result else None
        boa_success = divergence.boa_result.success if divergence.boa_result else None

        if ivy_success == boa_success:
            return KeepDecision(
                keep=True, reason="both_succeeded_different_results", fingerprint=""
            )

        # One succeeded, one failed - fingerprint on the error
        if ivy_success and not boa_success:
            failing_runner = divergence.divergent_runner
            error = divergence.boa_result.error if divergence.boa_result else None
        else:
            failing_runner = "ivy"
            error = divergence.ivy_result.error if divergence.ivy_result else None

        error_fp = fingerprint_error(error, self.DIVERGENCE_FRAMES)
        sig = (str(divergence.type), failing_runner, error_fp)
        return self._check_seen(sig, self._seen_divergences, "divergence")

    def check_compiler_crash(self, error: Exception) -> KeepDecision:
        sig = ("crash", fingerprint_error(error, self.CRASH_FRAMES))
        return self._check_seen(sig, self._seen_crashes, "crash")

    def check_compilation_failure(self, error: Exception) -> KeepDecision:
        sig = ("compile_fail", fingerprint_error(error, self.COMPILE_FAILURE_FRAMES))
        return self._check_seen(sig, self._seen_compile_failures, "compile_failure")

    def check_compilation_timeout(self, error: Exception) -> KeepDecision:
        sig = ("compile_timeout", fingerprint_error(error, self.COMPILE_FAILURE_FRAMES))
        return self._check_seen(
            sig, self._seen_compilation_timeouts, "compilation_timeout"
        )

    def get_stats(self) -> Dict[str, int]:
        """Return statistics about seen fingerprints."""
        return {
            "unique_crashes": len(self._seen_crashes),
            "unique_compile_failures": len(self._seen_compile_failures),
            "unique_compilation_timeouts": len(self._seen_compilation_timeouts),
            "unique_divergences": len(self._seen_divergences),
        }

    def reset(self):
        """Clear all seen fingerprints."""
        self._seen_crashes.clear()
        self._seen_compile_failures.clear()
        self._seen_compilation_timeouts.clear()
        self._seen_divergences.clear()

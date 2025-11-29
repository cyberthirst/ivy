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

Error fingerprinting uses:
- error_type: Exception class name (e.g., "CompilerPanic")
- msg[:20]: First 20 chars of first line of error message
- last_N_frames: Last N stack frames as (filename:lineno:funcname)
"""

import hashlib
import traceback
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

from .divergence_detector import Divergence, DivergenceType


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
    msg = str(error).split("\n")[0][:20]  # First line, first 20 chars
    frames = extract_stack_frames(error, n_frames)

    return (error_type, msg, frames)


@dataclass
class DedupDecision:
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

    def __init__(self):
        self._seen_crashes: Set[str] = set()
        self._seen_compile_failures: Set[str] = set()
        self._seen_divergences: Set[str] = set()

    def _compute_fingerprint(self, sig: Tuple) -> str:
        """Compute blake2b fingerprint from a signature tuple."""
        h = hashlib.blake2b(repr(sig).encode("utf-8"), digest_size=16)
        return h.hexdigest()

    def check_divergence(self, divergence: Divergence) -> DedupDecision:
        """
        Check if a divergence should be kept or dropped.

        Returns DedupDecision with keep=True for new divergences.
        """
        # Build fingerprint based on divergence type
        if divergence.type == DivergenceType.XFAIL:
            sig = (
                "xfail",
                divergence.xfail_expected,
                divergence.xfail_actual,
                tuple(divergence.xfail_reasons),
            )
        elif divergence.type in (DivergenceType.DEPLOYMENT, DivergenceType.EXECUTION):
            # Check if one succeeded and one failed
            ivy_success = divergence.ivy_result.success if divergence.ivy_result else None
            boa_success = divergence.boa_result.success if divergence.boa_result else None

            if ivy_success == boa_success:
                # Both succeeded with different results - don't dedup
                return DedupDecision(
                    keep=True,
                    reason="both_succeeded_different_results",
                    fingerprint="",
                )

            # One succeeded, one failed - fingerprint on the error
            if ivy_success and not boa_success:
                failing_runner = divergence.divergent_runner
                error = divergence.boa_result.error if divergence.boa_result else None
            else:
                failing_runner = "ivy"
                error = divergence.ivy_result.error if divergence.ivy_result else None

            error_fp = fingerprint_error(error, self.DIVERGENCE_FRAMES)
            sig = (
                divergence.type.value,
                failing_runner,
                error_fp,
            )
        else:
            # Unknown type - don't dedup
            return DedupDecision(keep=True, reason="unknown_type", fingerprint="")

        fingerprint = self._compute_fingerprint(sig)

        if fingerprint in self._seen_divergences:
            return DedupDecision(
                keep=False,
                reason=f"duplicate_divergence",
                fingerprint=fingerprint,
            )

        self._seen_divergences.add(fingerprint)
        return DedupDecision(
            keep=True,
            reason="new_divergence",
            fingerprint=fingerprint,
        )

    def check_compiler_crash(self, error: Exception) -> DedupDecision:
        """
        Check if a compiler crash should be kept or dropped.
        """
        error_fp = fingerprint_error(error, self.CRASH_FRAMES)
        sig = ("crash", error_fp)
        fingerprint = self._compute_fingerprint(sig)

        if fingerprint in self._seen_crashes:
            return DedupDecision(
                keep=False,
                reason="duplicate_crash",
                fingerprint=fingerprint,
            )

        self._seen_crashes.add(fingerprint)
        return DedupDecision(
            keep=True,
            reason="new_crash",
            fingerprint=fingerprint,
        )

    def check_compilation_failure(self, error: Exception) -> DedupDecision:
        """
        Check if a compilation failure should be kept or dropped.
        """
        error_fp = fingerprint_error(error, self.COMPILE_FAILURE_FRAMES)
        sig = ("compile_fail", error_fp)
        fingerprint = self._compute_fingerprint(sig)

        if fingerprint in self._seen_compile_failures:
            return DedupDecision(
                keep=False,
                reason="duplicate_compile_failure",
                fingerprint=fingerprint,
            )

        self._seen_compile_failures.add(fingerprint)
        return DedupDecision(
            keep=True,
            reason="new_compile_failure",
            fingerprint=fingerprint,
        )

    def get_stats(self) -> Dict[str, int]:
        """Return statistics about seen fingerprints."""
        return {
            "unique_crashes": len(self._seen_crashes),
            "unique_compile_failures": len(self._seen_compile_failures),
            "unique_divergences": len(self._seen_divergences),
        }

    def reset(self):
        """Clear all seen fingerprints."""
        self._seen_crashes.clear()
        self._seen_compile_failures.clear()
        self._seen_divergences.clear()

"""
Deduplication for fuzzer scenarios.

Filters duplicate scenarios based on behavior fingerprints to avoid
redundant output (crashes, compilation failures, divergences).

Fingerprinting Strategy
-----------------------

| Category                              | Dedup? | Fingerprint                                            |
|---------------------------------------|--------|--------------------------------------------------------|
| Divergence: both ok, different result | No     | -                                                      |
| Divergence: one ok, one failed        | Yes    | (type, reason, runner, error_fp)                       |
| Divergence: xfail violation           | Yes    | (reason, xfail_expected, xfail_actual, xfail_reasons)  |
| Compiler crash                        | Yes    | (error_type, last_3_frames)                            |
| Compilation failure                   | Yes    | (error_type, last_5_frames)                            |

Error fingerprinting (error_fp) returns (error_type, detail, frames):
- Generic exceptions: (class_name, msg[:20], last_N_stack_frames)
- BoaError: (class_name, "", frame_fingerprints) where each frame fingerprint
  is:
  - ("unknown",) for plain string stack frames
  - (fingerprint(ast_source), error_detail, pretty_vm_reason) for ErrorDetail
    frames, using AST fingerprint depth 3 and up to the first 3 frames
    (innermost first)

Compiler crash and compilation-failure dedup intentionally ignore message
detail and key only on error type + stack frames to avoid over-splitting on
volatile SSA names in exception text.
"""

import hashlib
import traceback
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

from boa.contracts.base_evm_contract import BoaError  # type: ignore[import-not-found]
from vyper.ast.nodes import Constant, Name, VyperNode

from fuzzer.divergence_detector import Divergence, DivergenceType


def fingerprint_ast_node(node: VyperNode, depth: int = 3) -> str:
    """Structural fingerprint of a Vyper AST node for dedup."""
    name = type(node).__name__

    if depth <= 0:
        return name

    # Leaves: type only, no values.
    if isinstance(node, Constant):
        return name
    if any(base.__name__ == "Operator" for base in type(node).__mro__):
        return name
    if isinstance(node, Name):
        if node.id in _builtin_names():
            return f"Name({node.id})"
        return "Name"

    # Compound nodes: recurse into fields.
    parts = []
    for field in sorted(node.get_fields()):
        if field.startswith("_"):
            continue
        child = getattr(node, field, None)
        if child is None:
            continue
        if isinstance(child, VyperNode):
            parts.append(fingerprint_ast_node(child, depth - 1))
        elif isinstance(child, list):
            sub = [
                fingerprint_ast_node(c, depth - 1)
                for c in child
                if isinstance(c, VyperNode)
            ]
            parts.append(f"[{','.join(sub)}]")

    return f"{name}({','.join(parts)})"


@lru_cache(maxsize=1)
def _builtin_names() -> frozenset[str]:
    from vyper.builtins.functions import get_builtin_functions

    return frozenset(get_builtin_functions()) | frozenset({"self", "msg", "block"})


def _boa_frame_pretty_vm_reason(frame: Any) -> str:
    try:
        pretty_vm_reason = frame.pretty_vm_reason
    except Exception:
        pretty_vm_reason = None

    if pretty_vm_reason is not None:
        return str(pretty_vm_reason)

    return repr(getattr(frame, "vm_error", None))


def _fingerprint_boa_frame(frame: Any, ast_depth: int = 3) -> Tuple[Any, ...]:
    if isinstance(frame, str):
        return ("unknown",)

    ast_source = getattr(frame, "ast_source", None)
    ast_fingerprint = (
        fingerprint_ast_node(ast_source, ast_depth)
        if isinstance(ast_source, VyperNode)
        else None
    )
    error_detail = getattr(frame, "error_detail", None)
    pretty_vm_reason = _boa_frame_pretty_vm_reason(frame)
    return (ast_fingerprint, error_detail, pretty_vm_reason)


def _fingerprint_boa_error(
    error: BoaError, n_frames: int = 3, ast_depth: int = 3
) -> Tuple[Tuple[Any, ...], ...]:
    stack_trace = getattr(error, "stack_trace", None)
    if stack_trace is None:
        return ()
    frames = list(stack_trace)[:n_frames]
    return tuple(_fingerprint_boa_frame(frame, ast_depth) for frame in frames)


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
) -> Tuple[str, str, Tuple[Any, ...]]:
    """
    Build fingerprint tuple from an exception.

    Returns (error_type, detail, stack_frames).
    For BoaError, returns ("BoaError", "", boa_frame_fingerprints), where
    each frame fingerprint is either ("unknown",) or
    (ast_fingerprint, error_detail, pretty_vm_reason).
    For other exceptions, uses (class_name, msg[:20], last_N_frames).
    """
    if error is None:
        return ("", "", ())

    if isinstance(error, BoaError):
        return (type(error).__name__, "", _fingerprint_boa_error(error, n_frames, 3))

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
        seen_divergences: Optional[Dict[str, bool]] = None,
    ):
        self._seen_crashes: Dict[str, bool] = (
            seen_crashes if seen_crashes is not None else {}
        )
        self._seen_compile_failures: Dict[str, bool] = (
            seen_compile_failures if seen_compile_failures is not None else {}
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
            divergence.reason,
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
        sig = (str(divergence.type), divergence.reason, failing_runner, error_fp)
        return self._check_seen(sig, self._seen_divergences, "divergence")

    def check_compiler_crash(self, error: Exception) -> KeepDecision:
        error_type, _detail, frames = fingerprint_error(error, self.CRASH_FRAMES)
        sig = ("crash", error_type, frames)
        return self._check_seen(sig, self._seen_crashes, "crash")

    def check_compilation_failure(self, error: Exception) -> KeepDecision:
        error_type, _detail, frames = fingerprint_error(error, self.COMPILE_FAILURE_FRAMES)
        sig = ("compile_fail", error_type, frames)
        return self._check_seen(sig, self._seen_compile_failures, "compile_failure")

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

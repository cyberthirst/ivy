"""
Deduplication for fuzzer scenarios.

Filters duplicate scenarios based on behavior fingerprints to avoid
redundant output (crashes, compilation failures, divergences).

Fingerprinting Strategy
-----------------------

| Category                              | Dedup? | Fingerprint                                           |
|---------------------------------------|--------|-------------------------------------------------------|
| Divergence: both ok, different result | No     | -                                                     |
| Divergence: one ok, one failed        | Yes    | DivergenceSig(type, reason, runner, error_fp)         |
| Divergence: xfail violation           | Yes    | XfailSig(reason, expected, actual, reasons)           |
| Compiler crash                        | Yes    | CrashSig(error_type, frames)                         |
| Compilation failure                   | Yes    | CompileFailSig(error_type, frames)                   |

Error fingerprinting (ErrorFP) has fields (error_type, frames):
- Generic exceptions: ErrorFP(class_name, last_N_stack_frames)
- BoaError: ErrorFP(class_name, boa_frame_fingerprints) where each frame is:
  - UnknownFrameFP() for plain string stack frames
  - BoaFrameFP(ast_fingerprint, error_detail, pretty_vm_reason) for ErrorDetail
    frames, using AST fingerprint depth 3 and up to the first 3 frames
    (innermost first)

Error messages are intentionally excluded from all fingerprints to avoid
over-splitting on volatile content (SSA names, addresses, values).
"""

from __future__ import annotations

import hashlib
import traceback
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from boa.contracts.base_evm_contract import BoaError  # type: ignore[import-not-found]
from vyper.ast.nodes import Constant, Name, VyperNode

from fuzzer.divergence_detector import Divergence, DivergenceType


# --- Fingerprint dataclasses ---


@dataclass(frozen=True)
class Signature:
    def digest(self) -> str:
        h = hashlib.blake2b(repr(self).encode("utf-8"), digest_size=16)
        return h.hexdigest()


@dataclass(frozen=True)
class UnknownFrameFP(Signature):
    pass


@dataclass(frozen=True)
class BoaFrameFP(Signature):
    ast_fingerprint: str | None
    error_detail: str | None
    pretty_vm_reason: str


@dataclass(frozen=True)
class ErrorFP(Signature):
    error_type: str
    frames: tuple[BoaFrameFP | UnknownFrameFP, ...] | tuple[str, ...]


@dataclass(frozen=True)
class CrashSig(Signature):
    error_type: str
    frames: tuple[BoaFrameFP | UnknownFrameFP, ...] | tuple[str, ...]


@dataclass(frozen=True)
class CompileFailSig(Signature):
    error_type: str
    frames: tuple[BoaFrameFP | UnknownFrameFP, ...] | tuple[str, ...]


@dataclass(frozen=True)
class DivergenceSig(Signature):
    div_type: str
    reason: str | None
    failing_runner: str | None
    error_fp: ErrorFP


@dataclass(frozen=True)
class XfailSig(Signature):
    reason: str | None
    xfail_expected: str | None
    xfail_actual: str | None
    xfail_reasons: tuple[str, ...]


# --- AST fingerprinting ---


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


# --- Boa error fingerprinting ---


def _boa_frame_pretty_vm_reason(frame: Any) -> str:
    try:
        pretty_vm_reason = frame.pretty_vm_reason
    except Exception:
        pretty_vm_reason = None

    if pretty_vm_reason is not None:
        return str(pretty_vm_reason)

    return repr(getattr(frame, "vm_error", None))


def _fingerprint_boa_frame(frame: Any, ast_depth: int = 3) -> BoaFrameFP | UnknownFrameFP:
    if isinstance(frame, str):
        return UnknownFrameFP()

    ast_source = getattr(frame, "ast_source", None)
    ast_fp = (
        fingerprint_ast_node(ast_source, ast_depth)
        if isinstance(ast_source, VyperNode)
        else None
    )
    error_detail = getattr(frame, "error_detail", None)
    pretty_vm_reason = _boa_frame_pretty_vm_reason(frame)
    return BoaFrameFP(
        ast_fingerprint=ast_fp,
        error_detail=error_detail,
        pretty_vm_reason=pretty_vm_reason,
    )


def _fingerprint_boa_error(
    error: BoaError, n_frames: int = 3, ast_depth: int = 3
) -> tuple[BoaFrameFP | UnknownFrameFP, ...]:
    stack_trace = getattr(error, "stack_trace", None)
    if stack_trace is None:
        return ()
    frames = list(stack_trace)[:n_frames]
    return tuple(_fingerprint_boa_frame(frame, ast_depth) for frame in frames)


# --- Generic error fingerprinting ---


def extract_stack_frames(error: Exception | None, n: int = 3) -> tuple[str, ...]:
    """Extract last N frames from exception traceback.

    Normalizes frame info to (filename, lineno, funcname) for stability.
    """
    if error is None:
        return ()

    tb = getattr(error, "__traceback__", None)
    if tb is None:
        return ()

    frames = traceback.extract_tb(tb)
    if not frames:
        return ()

    last_frames = frames[-n:] if len(frames) >= n else frames

    normalized = []
    for frame in last_frames:
        filename = frame.filename.split("/")[-1]
        normalized.append(f"{filename}:{frame.lineno}:{frame.name}")

    return tuple(normalized)


def fingerprint_error(error: Exception | None, n_frames: int = 3) -> ErrorFP:
    """Build an ErrorFP from an exception.

    For BoaError: ErrorFP(error_type, boa_frame_fingerprints)
    For other exceptions: ErrorFP(class_name, last_N_frames)
    """
    if error is None:
        return ErrorFP(error_type="", frames=())

    if isinstance(error, BoaError):
        return ErrorFP(
            error_type=type(error).__name__,
            frames=_fingerprint_boa_error(error, n_frames, 3),
        )

    error_type = type(error).__name__
    frames = extract_stack_frames(error, n_frames)

    return ErrorFP(error_type=error_type, frames=frames)


# --- Deduper ---


@dataclass
class KeepDecision:
    keep: bool
    reason: str
    fingerprint: str


class Deduper:
    """Deduplicates scenarios based on behavior fingerprints.

    Tracks seen fingerprints for: compiler crashes, compilation failures,
    and divergences. Each category maintains its own seen set.
    """

    CRASH_FRAMES = 3
    COMPILE_FAILURE_FRAMES = 5
    DIVERGENCE_FRAMES = 3

    def __init__(
        self,
        *,
        seen_crashes: dict[str, bool] | None = None,
        seen_compile_failures: dict[str, bool] | None = None,
        seen_divergences: dict[str, bool] | None = None,
    ):
        self._seen_crashes: dict[str, bool] = (
            seen_crashes if seen_crashes is not None else {}
        )
        self._seen_compile_failures: dict[str, bool] = (
            seen_compile_failures if seen_compile_failures is not None else {}
        )
        self._seen_divergences: dict[str, bool] = (
            seen_divergences if seen_divergences is not None else {}
        )

    def _check_seen(
        self, sig: Signature, seen: dict[str, bool], category: str
    ) -> KeepDecision:
        fingerprint = sig.digest()
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
            assert False, "unreachable"

    def _check_xfail(self, divergence: Divergence) -> KeepDecision:
        sig = XfailSig(
            reason=divergence.reason,
            xfail_expected=divergence.xfail_expected,
            xfail_actual=divergence.xfail_actual,
            xfail_reasons=tuple(divergence.xfail_reasons),
        )
        divergence.fingerprint = repr(sig)
        return self._check_seen(sig, self._seen_divergences, "divergence")

    def _check_result_divergence(self, divergence: Divergence) -> KeepDecision:
        assert divergence.ivy_result is not None or divergence.boa_result is not None, (
            "divergence must have at least one result"
        )
        if divergence.ivy_result is None or divergence.boa_result is None:
            # TODO: add fingerprinting for divergences with a missing result
            return KeepDecision(
                keep=True, reason="missing_result", fingerprint=""
            )

        ivy_success = divergence.ivy_result.success
        boa_success = divergence.boa_result.success

        if ivy_success == boa_success:
            return KeepDecision(
                keep=True, reason="both_succeeded_different_results", fingerprint=""
            )

        if ivy_success and not boa_success:
            failing_runner = divergence.divergent_runner
            error = divergence.boa_result.error
        else:
            failing_runner = "ivy"
            error = divergence.ivy_result.error

        error_fp = fingerprint_error(error, self.DIVERGENCE_FRAMES)
        sig = DivergenceSig(
            div_type=str(divergence.type),
            reason=divergence.reason,
            failing_runner=failing_runner,
            error_fp=error_fp,
        )
        divergence.fingerprint = repr(sig)
        return self._check_seen(sig, self._seen_divergences, "divergence")

    def check_compiler_crash(self, error: Exception) -> KeepDecision:
        error_fp = fingerprint_error(error, self.CRASH_FRAMES)
        sig = CrashSig(error_type=error_fp.error_type, frames=error_fp.frames)
        return self._check_seen(sig, self._seen_crashes, "crash")

    def check_compilation_failure(self, error: Exception) -> KeepDecision:
        error_fp = fingerprint_error(error, self.COMPILE_FAILURE_FRAMES)
        sig = CompileFailSig(error_type=error_fp.error_type, frames=error_fp.frames)
        return self._check_seen(sig, self._seen_compile_failures, "compile_failure")

    def get_stats(self) -> dict[str, int]:
        return {
            "unique_crashes": len(self._seen_crashes),
            "unique_compile_failures": len(self._seen_compile_failures),
            "unique_divergences": len(self._seen_divergences),
        }

    def reset(self):
        self._seen_crashes.clear()
        self._seen_compile_failures.clear()
        self._seen_divergences.clear()

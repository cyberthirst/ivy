"""
Signal-based timeout for Ivy call execution.

Uses SIGALRM to enforce per-call wall-clock timeouts.
Timeout is delivered as a Python exception, triggering Ivy journal rollback.

Constraints:
- Must run in main thread (signal handlers only work there).
- Timeout may be delayed if execution is in long-running C code.
"""

from __future__ import annotations

import signal
from contextlib import contextmanager
from typing import Generator


class CallTimeout(Exception):
    pass


_call_active = False


def _timeout_handler(signum: int, frame) -> None:
    global _call_active
    if _call_active:
        raise CallTimeout("Call exceeded time budget")


@contextmanager
def call_with_timeout(timeout_seconds: float) -> Generator[None, None, None]:
    global _call_active

    if timeout_seconds <= 0:
        yield
        return

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)

    try:
        _call_active = True
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        _call_active = False
        signal.signal(signal.SIGALRM, old_handler)

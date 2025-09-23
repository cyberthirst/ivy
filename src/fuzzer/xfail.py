"""Shared types for expected failure annotations."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class XFailExpectation:
    """Records that a failure of ``kind`` is expected, with optional context."""

    kind: str
    reason: Optional[str] = None

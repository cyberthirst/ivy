from __future__ import annotations

from typing import TypeAlias

RuntimeStmtSite: TypeAlias = tuple[str, int, int]
RuntimeBranchOutcome: TypeAlias = tuple[str, int, int, bool]

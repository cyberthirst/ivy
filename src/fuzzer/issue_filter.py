"""
Rule-based filtering for known issues.

Filters out divergences, crashes, and compilation failures that match
configurable rules. Used to suppress known limitations or bugs that
don't need repeated reporting.

Rules search the serialized JSON representation of an issue, so they
can match on any field (error messages, storage dumps, source code, etc).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable, Optional


class IssueType(StrEnum):
    DIVERGENCE = "divergence"
    CRASH = "crash"
    COMPILE_FAILURE = "compile_failure"


@dataclass
class FilterRule:
    """A filter rule that matches issues based on their content."""

    name: str
    issue_types: set[IssueType] = field(default_factory=lambda: set(IssueType))
    # String matching (searches the full JSON representation)
    contains: Optional[str] = None
    pattern: Optional[re.Pattern] = None
    # Custom predicate for complex logic
    predicate: Optional[Callable[[dict], bool]] = None

    def matches(self, issue_type: IssueType, json_str: str, issue_dict: dict) -> bool:
        """Check if rule matches. All conditions are ANDed together."""
        if issue_type not in self.issue_types:
            return False

        if self.contains and self.contains not in json_str:
            return False

        if self.pattern and not self.pattern.search(json_str):
            return False

        if self.predicate and not self.predicate(issue_dict):
            return False

        return True


class IssueFilter:
    """
    Filters issues based on configurable rules.

    Rules are checked against the JSON-serialized issue dict, so they can
    match content anywhere in the structure (errors, storage dumps, etc).

    Usage:
        f = IssueFilter()

        # Filter all issues containing "Unknown builtin:"
        f.add_rule("ivy_unknown_builtin", contains="Unknown builtin:")

        # Filter only divergences
        f.add_rule("decoder_issue", contains="unimplemented decoder",
                   issue_types={IssueType.DIVERGENCE})

        # Check if issue should be filtered
        if f.should_filter(divergence.as_dict, IssueType.DIVERGENCE):
            pass  # Skip this divergence
    """

    def __init__(self):
        self.rules: list[FilterRule] = []

    def add_rule(
        self,
        name: str,
        contains: Optional[str] = None,
        pattern: Optional[str] = None,
        issue_types: Optional[set[IssueType]] = None,
        predicate: Optional[Callable[[dict], bool]] = None,
    ) -> IssueFilter:
        """Add a filter rule."""
        rule = FilterRule(
            name=name,
            issue_types=issue_types if issue_types is not None else set(IssueType),
            contains=contains,
            pattern=re.compile(pattern) if pattern else None,
            predicate=predicate,
        )
        self.rules.append(rule)
        return self

    def should_filter(self, issue_dict: dict, issue_type: IssueType) -> Optional[str]:
        """Check if issue should be filtered. Returns rule name if filtered, None otherwise."""
        json_str = json.dumps(issue_dict, default=str)
        for rule in self.rules:
            if rule.matches(issue_type, json_str, issue_dict):
                return rule.name
        return None


def default_issue_filter() -> IssueFilter:
    """Create filter with default rules for known limitations."""
    f = IssueFilter()

    # Ivy limitations - unsupported builtins
    f.add_rule(
        name="ivy_unknown_builtin",
        contains="Unknown builtin:",
        issue_types={IssueType.DIVERGENCE},
    )

    # Boa decoder limitations - appears in storage dumps
    f.add_rule(
        name="boa_unimplemented_decoder",
        contains="unimplemented decoder",
        issue_types={IssueType.DIVERGENCE},
    )

    # Contract too large for EVM
    f.add_rule(
        name="eip3860_code_size_limit",
        contains="Contract code size exceeds EIP-3860 limit",
    )

    # Boa feature not implemented
    f.add_rule(
        name="boa_not_supported",
        contains="TODO does boa support this",
    )

    # Static assertion failure (not a runtime bug)
    f.add_rule(
        name="static_assertion_failure",
        contains="assertion found to fail at compile time",
    )

    # Known compiler bugs
    f.add_rule(
        name="known_bug_dereference_non_pointer",
        contains="cannot dereference non-pointer type",
        issue_types={IssueType.CRASH},
    )

    f.add_rule(
        name="known_bug_tuple_value_type",
        contains="'TupleT' object has no attribute 'value_type'",
        issue_types={IssueType.CRASH},
    )

    f.add_rule(
        name="zero_array_indexing",
        contains="indexing into zero array not allowed",
    )

    return f

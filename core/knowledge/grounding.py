"""Rule grounding checks before promotion."""
from __future__ import annotations

from .rule_extractor import ExtractedRule


def rule_is_grounded(rule: ExtractedRule, passing_tests: set[str]) -> bool:
    return bool(rule.grounding_tests) and all(test in passing_tests for test in rule.grounding_tests)


__all__ = ["rule_is_grounded"]

"""core/self_improvement/discrepancy_attributor.py — Root-cause classification.

Classifies each test failure into interpretable categories adapted from
the paper's error taxonomy. Over 75% of the paper's divergences were
traced to a specific interpretable source — Aura needs the same.

Categories:
  AGENT_ERROR      — candidate code is incorrect
  SPEC_UNDERSPEC   — spec/docs were ambiguous or incomplete
  TEST_DEFICIENCY  — test itself is flawed
  DATA_MISMATCH    — trace data format issue
  ENVIRONMENT_ERROR — sandbox/runtime issue
  UNKNOWN          — unclassifiable
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List

from core.self_improvement.interface_contract import (
    ComparisonReport,
    DiscrepancyCategory,
    DiscrepancyItem,
    DiscrepancyReport,
    ModuleSpec,
    TestVerdict,
)

logger = logging.getLogger("Aura.DiscrepancyAttributor")

# Heuristic patterns for attribution
_ENVIRONMENT_PATTERNS = [
    r"ModuleNotFoundError",
    r"No module named",
    r"Permission denied",
    r"TimeoutError",
    r"OSError",
    r"MemoryError",
    r"timed out",
    r"Connection refused",
]

_SPEC_PATTERNS = [
    r"NotImplementedError.*reimplementation required",
    r"stub",
    r"TODO",
    r"not specified",
    r"ambiguous",
]

_TEST_DEFICIENCY_PATTERNS = [
    r"fixture.*not found",
    r"conftest",
    r"ImportError.*test",
    r"collection error",
    r"setup error",
]


class DiscrepancyAttributor:
    """Classifies test failures into root-cause categories."""

    def attribute(
        self, comparison: ComparisonReport, spec: ModuleSpec
    ) -> DiscrepancyReport:
        """Attribute all failures in a comparison report.

        Returns a DiscrepancyReport with per-failure classification
        and an aggregate summary.
        """
        items: List[DiscrepancyItem] = []

        for verdict in comparison.verdicts:
            if verdict.passed:
                continue
            category = self._classify(verdict, spec)
            severity = self._severity(verdict, category)
            items.append(DiscrepancyItem(
                test_name=verdict.test_name,
                category=category,
                description=self._describe(verdict, category),
                severity=severity,
            ))

        # Structural checks
        if not comparison.syntax_valid:
            items.append(DiscrepancyItem(
                test_name="__syntax__",
                category=DiscrepancyCategory.AGENT_ERROR,
                description="Generated code has syntax errors",
                severity="critical",
            ))
        if not comparison.public_surface_preserved:
            items.append(DiscrepancyItem(
                test_name="__public_surface__",
                category=DiscrepancyCategory.AGENT_ERROR,
                description="Public interface not preserved — missing exports",
                severity="critical",
            ))

        # Build summary
        summary: Dict[str, int] = {}
        for item in items:
            key = item.category.value
            summary[key] = summary.get(key, 0) + 1

        report = DiscrepancyReport(items=items, summary=summary)

        logger.info(
            "Attribution complete: %d items — %s",
            len(items),
            ", ".join(f"{k}={v}" for k, v in sorted(summary.items())),
        )
        return report

    def _classify(self, verdict: TestVerdict, spec: ModuleSpec) -> DiscrepancyCategory:
        """Classify a single test failure."""
        error = (verdict.error_message + " " + verdict.stderr).lower()

        # Environment errors (sandbox/import/permission issues)
        for pattern in _ENVIRONMENT_PATTERNS:
            if re.search(pattern, error, re.IGNORECASE):
                return DiscrepancyCategory.ENVIRONMENT_ERROR

        # Test deficiency (fixture/conftest issues)
        for pattern in _TEST_DEFICIENCY_PATTERNS:
            if re.search(pattern, error, re.IGNORECASE):
                return DiscrepancyCategory.TEST_DEFICIENCY

        # Spec underspecification (NotImplementedError from stubs)
        for pattern in _SPEC_PATTERNS:
            if re.search(pattern, error, re.IGNORECASE):
                return DiscrepancyCategory.SPEC_UNDERSPECIFICATION

        # System-level test names indicate internal checks
        if verdict.test_name.startswith("__"):
            return DiscrepancyCategory.AGENT_ERROR

        # Default: if tests fail with assertion errors, it's agent error
        if "assertionerror" in error or "assert" in error:
            return DiscrepancyCategory.AGENT_ERROR

        # Type/attribute/name errors are agent errors
        if any(e in error for e in ["typeerror", "attributeerror", "nameerror", "valueerror"]):
            return DiscrepancyCategory.AGENT_ERROR

        return DiscrepancyCategory.UNKNOWN

    def _severity(self, verdict: TestVerdict, category: DiscrepancyCategory) -> str:
        """Determine severity of a discrepancy."""
        if category == DiscrepancyCategory.AGENT_ERROR:
            if verdict.test_name.startswith("__"):
                return "critical"
            return "high"
        if category == DiscrepancyCategory.ENVIRONMENT_ERROR:
            return "medium"
        if category == DiscrepancyCategory.TEST_DEFICIENCY:
            return "low"
        if category == DiscrepancyCategory.SPEC_UNDERSPECIFICATION:
            return "medium"
        return "medium"

    def _describe(self, verdict: TestVerdict, category: DiscrepancyCategory) -> str:
        """Build human-readable description of the discrepancy."""
        prefix = {
            DiscrepancyCategory.AGENT_ERROR: "Candidate code error",
            DiscrepancyCategory.SPEC_UNDERSPECIFICATION: "Spec incomplete",
            DiscrepancyCategory.TEST_DEFICIENCY: "Test infrastructure issue",
            DiscrepancyCategory.DATA_MISMATCH: "Data format mismatch",
            DiscrepancyCategory.ENVIRONMENT_ERROR: "Sandbox/environment issue",
            DiscrepancyCategory.UNKNOWN: "Unclassified failure",
        }.get(category, "Failure")

        detail = verdict.error_message[:200] if verdict.error_message else "no details"
        return f"{prefix} in {verdict.test_name}: {detail}"


__all__ = ["DiscrepancyAttributor"]

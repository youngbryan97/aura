"""core/science/redteam_ledger.py — a fixed finding that cannot come back.

A security finding gets fixed and then, months later, the same class of thing
appears somewhere else. Aura has strong governance and receipts and no ledger
that says which adversarial findings have been closed and what stops each one
recurring. Without that, the honest answer to "are we getting safer" is a
feeling.

Two rules, and the second is the one that does the work:

* **A finding is open until it names a fix.** Not "addressed", not "mitigated
  by design" - the commit or the module.
* **A fix is unpinned until it names a test that fails without it.** A fix
  with no regression test is a fix that will be undone by someone who does not
  know it was one, and this is the difference between a closed finding and a
  finding that will be reopened.

:meth:`RedTeamLedger.trend` is the number the review asks for: findings per
release, and the fraction that came back. A programme where findings fall and
recurrence is zero is one that is working; findings falling with recurrence
rising means they are being closed rather than fixed.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

__all__ = ["Severity", "Finding", "RedTeamLedger"]

ROOT = Path(__file__).resolve().parent.parent.parent


class Severity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Finding:
    """One adversarial finding, and what stops it coming back."""

    finding_id: str
    title: str
    severity: Severity
    release: str
    found_by: str
    fix: str = ""
    regression_test: str = ""
    recurred_in: tuple[str, ...] = ()

    @property
    def fixed(self) -> bool:
        return bool(self.fix)

    @property
    def pinned(self) -> bool:
        """A fix with no test is a fix somebody will undo without knowing."""
        return self.fixed and bool(self.regression_test) and (ROOT / self.regression_test.split("::")[0]).exists()

    @property
    def state(self) -> str:
        if not self.fixed:
            return "open"
        if not self.pinned:
            return "fixed_unpinned"
        return "recurred" if self.recurred_in else "closed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "title": self.title,
            "severity": self.severity.value,
            "release": self.release,
            "found_by": self.found_by,
            "fix": self.fix,
            "regression_test": self.regression_test,
            "state": self.state,
            "recurred_in": list(self.recurred_in),
        }


class RedTeamLedger:
    """Every finding, its fix, and the test that stops it returning."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._findings: dict[str, Finding] = {}

    def record(
        self, finding_id: str, title: str, severity: Severity, *, release: str, found_by: str
    ) -> Finding:
        with self._lock:
            finding = Finding(finding_id, title, severity, release, found_by)
            self._findings[finding_id] = finding
            return finding

    def fix(self, finding_id: str, *, fix: str, regression_test: str = "") -> Finding:
        with self._lock:
            finding = self._findings[finding_id]
            finding.fix = fix
            finding.regression_test = regression_test
            return finding

    def recurred(self, finding_id: str, release: str) -> Finding:
        with self._lock:
            finding = self._findings[finding_id]
            finding.recurred_in = (*finding.recurred_in, release)
            return finding

    def trend(self) -> dict[str, Any]:
        with self._lock:
            findings = list(self._findings.values())
        by_release: dict[str, int] = {}
        for finding in findings:
            by_release[finding.release] = by_release.get(finding.release, 0) + 1
        recurred = [f for f in findings if f.recurred_in]
        releases = sorted(by_release)
        falling = (
            all(by_release[a] >= by_release[b] for a, b in zip(releases, releases[1:], strict=False))
            if len(releases) > 1 else None
        )
        return {
            "findings": len(findings),
            "by_release": dict(sorted(by_release.items())),
            "by_state": {
                state: sorted(f.finding_id for f in findings if f.state == state)
                for state in sorted({f.state for f in findings})
            },
            "unpinned_fixes": sorted(f.finding_id for f in findings if f.state == "fixed_unpinned"),
            "recurrence_rate": len(recurred) / len(findings) if findings else 0.0,
            "falling": falling,
            "verdict": (
                "not yet measurable"
                if falling is None
                else "findings are falling and none has come back"
                if falling and not recurred
                else "findings are falling but some have come back; they are being closed "
                "rather than fixed"
                if falling
                else "findings are not falling release over release"
            ),
        }

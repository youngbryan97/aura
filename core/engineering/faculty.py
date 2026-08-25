"""What Aura can measure about her own ability to design something.

Metacognition here means one thing: a number she can read about herself,
taken from what actually happened rather than from a claim. So this declares
engineering design as a faculty with three probes, and each one reads a real
record.

Whether the formulas still reproduce their published answers is the first,
because every other number depends on it. What fraction of computed results
carried their working is the second, since a result that cannot say where it
came from is not a result. And what fraction of finished designs could
actually be ordered and built is the third, because a design nobody can
source is a picture.

The record the probes read is kept here, in memory, bounded, and written to
by :mod:`core.engineering.studio` as designs finish. A faculty whose metrics
have never been exercised reports as unmeasured rather than as healthy,
which is the honest answer to "can you do this?" before she has tried.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DesignRecord",
    "record_design",
    "design_history",
    "engineering_report",
    "declare_engineering_faculty",
    "capability_statement",
]

#: How many finished designs to keep. Enough to make a rate meaningful and
#: small enough that it costs nothing.
_HISTORY = 64

_LOCK = threading.Lock()
_RECORDS: deque[DesignRecord] = deque(maxlen=_HISTORY)


@dataclass(frozen=True, slots=True)
class DesignRecord:
    """One finished design, reduced to what can be measured about it."""

    name: str
    fingerprint: str
    at: float
    findings: int
    grounded: int
    dropped: int
    blocking: int
    warnings: int
    buildable: bool
    validation_ok: bool
    sheets: int
    files: int
    seconds: float
    disciplines: tuple[str, ...] = ()

    @property
    def grounded_fraction(self) -> float:
        total = self.grounded + self.dropped
        return self.grounded / total if total else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fingerprint": self.fingerprint,
            "at": self.at,
            "findings": self.findings,
            "grounded": self.grounded,
            "dropped": self.dropped,
            "grounded_fraction": self.grounded_fraction,
            "blocking": self.blocking,
            "warnings": self.warnings,
            "buildable": self.buildable,
            "validation_ok": self.validation_ok,
            "sheets": self.sheets,
            "files": self.files,
            "seconds": self.seconds,
            "disciplines": list(self.disciplines),
        }


def record_design(result, *, seconds: float = 0.0) -> DesignRecord:
    """Note what one finished design did, for the faculty probes to read."""
    verdict = result.verdict
    disciplines = sorted({
        f.id.split(".")[0] for f in result.findings
    })
    record = DesignRecord(
        name=result.design.name,
        fingerprint=result.design.fingerprint(),
        at=time.time(),
        findings=len(result.findings) + len(verdict.dropped),
        grounded=len(verdict.grounded),
        dropped=len(verdict.dropped),
        blocking=len(verdict.blocking),
        warnings=len(verdict.warnings),
        buildable=bool(verdict.buildable),
        validation_ok=bool(verdict.validation_ok),
        sheets=len(result.sheets),
        files=len(result.bundle.files) if result.bundle is not None else 0,
        seconds=float(seconds),
        disciplines=tuple(disciplines),
    )
    with _LOCK:
        _RECORDS.append(record)
    return record


def design_history() -> tuple[DesignRecord, ...]:
    with _LOCK:
        return tuple(_RECORDS)


# ── probes ────────────────────────────────────────────────────────────────


def _validation_pass_rate() -> float | None:
    """Whether the published-answer battery is currently green.

    Read live rather than from history, because it is a property of the code
    right now and not of anything she has done.
    """
    from core.engineering.validation import run_validation

    report = run_validation()
    total = report.passed + report.failed
    return report.passed / total if total else None


def _grounded_result_rate() -> float | None:
    """The share of computed results that carried their own working."""
    records = design_history()
    if not records:
        return None
    grounded = sum(r.grounded for r in records)
    total = sum(r.grounded + r.dropped for r in records)
    return grounded / total if total else None


def _buildable_rate() -> float | None:
    """The share of finished designs that could actually be ordered."""
    records = design_history()
    if not records:
        return None
    return sum(1 for r in records if r.buildable) / len(records)


def _blocking_defects() -> float | None:
    """Blocking faults per design, over the designs actually produced."""
    records = design_history()
    if not records:
        return None
    return sum(r.blocking for r in records) / len(records)


def engineering_report() -> dict[str, Any]:
    """What she can say about her own engineering, with the numbers behind it."""
    from core.engineering.analysis import ANALYSES
    from core.engineering.draw.symbols import SYMBOLS
    from core.engineering.materials import FLUIDS, MATERIALS
    from core.engineering.validation import coverage, run_validation

    report = run_validation()
    records = design_history()
    return {
        "analyses": len(ANALYSES),
        "analysis_keys": sorted(ANALYSES),
        "materials": len(MATERIALS),
        "fluids": len(FLUIDS),
        "symbols": len(SYMBOLS),
        "validation": {
            "passed": report.passed,
            "failed": report.failed,
            "ok": report.ok,
            "coverage": coverage(),
            "plain": report.plain(),
        },
        "designs_made": len(records),
        "grounded_result_rate": _grounded_result_rate(),
        "buildable_rate": _buildable_rate(),
        "recent": [r.to_dict() for r in records[-8:]],
    }


def capability_statement() -> str:
    """One paragraph she can read back about what she can and cannot do here.

    Written from the live registry rather than fixed, so it stays true when
    an analysis is added and stops claiming a discipline that is removed.
    """
    from core.engineering.analysis import ANALYSES
    from core.engineering.draw.sheet import SHEET_KINDS
    from core.engineering.draw.symbols import STANDARDS
    from core.engineering.export import FORMATS
    from core.engineering.materials import MATERIALS
    from core.engineering.validation import coverage, run_validation

    report = run_validation()
    disciplines = sorted({a.discipline for a in ANALYSES.values()})
    records = design_history()
    lines = [
        f"I can take a design brief and produce checked engineering drawings from it. "
        f"{len(ANALYSES)} analyses across {len(disciplines)} disciplines "
        f"({', '.join(disciplines)}), against {len(MATERIALS)} materials with real "
        f"property data.",
        f"I draw {len(SHEET_KINDS)} kinds of sheet and use the standard symbol sets "
        f"({', '.join(sorted(STANDARDS))}), and export in "
        f"{len(FORMATS)} formats including printable mesh and editable CAD source.",
        f"Every number on a drawing carries its formula, its inputs and its reference, "
        f"and anything that cannot is dropped before rendering rather than shown.",
        report.plain(),
    ]
    if records:
        buildable = _buildable_rate() or 0.0
        grounded = _grounded_result_rate() or 0.0
        lines.append(
            f"I have finished {len(records)} designs in this run. "
            f"{grounded * 100:.0f}% of computed results carried their working, and "
            f"{buildable * 100:.0f}% of the designs could be ordered and built as drawn."
        )
    else:
        lines.append(
            "I have not designed anything yet in this run, so I have no record of how "
            "well it goes and will not claim one."
        )
    lines.append(
        "What I cannot do: finite-element or computational-fluid analysis, anything "
        "needing a mesh solver, and anything in a discipline with no validated case "
        f"behind it. The validated disciplines are {', '.join(coverage())}."
    )
    return " ".join(lines)


def declare_engineering_faculty(registry=None):
    """Declare engineering design so metacognition can see and score it."""
    from core.metacognition.faculty_model import (
        Faculty,
        ImprovementMetric,
        get_faculty_registry,
    )

    target = registry if registry is not None else get_faculty_registry()
    return target.declare(
        Faculty(
            faculty_id="engineering_design",
            description=(
                "Designing a physical thing and drawing it: computing what it weighs, "
                "whether it holds, and what it takes to build."
            ),
            owner="core.engineering",
            gates=("reasoning", "memory"),
            metrics=(
                ImprovementMetric(
                    metric_id="validation_pass_rate",
                    unit="",
                    direction="higher_is_better",
                    probe=_validation_pass_rate,
                    floor=0.0,
                    target=1.0,
                    ceiling=1.0,
                    weight=4.0,
                    description=(
                        "Share of textbook problems with published answers that this "
                        "engine still reproduces. Every other number rests on it."
                    ),
                ),
                ImprovementMetric(
                    metric_id="grounded_result_rate",
                    unit="",
                    direction="higher_is_better",
                    probe=_grounded_result_rate,
                    floor=0.0,
                    target=1.0,
                    ceiling=1.0,
                    weight=3.0,
                    description=(
                        "Share of computed results that carried their formula, inputs "
                        "and reference. The rest were dropped before rendering."
                    ),
                ),
                ImprovementMetric(
                    metric_id="buildable_rate",
                    unit="",
                    direction="higher_is_better",
                    probe=_buildable_rate,
                    floor=0.0,
                    target=0.8,
                    ceiling=1.0,
                    weight=2.0,
                    description=(
                        "Share of finished designs whose parts all say how they would "
                        "be obtained, so an order could be placed."
                    ),
                ),
                ImprovementMetric(
                    metric_id="blocking_defects_per_design",
                    unit=" faults",
                    direction="lower_is_better",
                    probe=_blocking_defects,
                    floor=6.0,
                    target=0.0,
                    ceiling=0.0,
                    weight=2.0,
                    description=(
                        "Faults per design serious enough to stop it being drawn: a "
                        "clash, a domain mismatch, an impossible quantity."
                    ),
                ),
            ),
        )
    )

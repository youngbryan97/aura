"""Engineering truth engine — check a stated figure against the computed one.

A design produces its numbers by arithmetic. A reply about that design is
written by a language model, and the gap between the two is where a right
answer turns into a wrong sentence: the hull mass is 33.7 kg in the model
and "about forty kilos" in the paragraph, and nobody notices because both
are plausible.

So every quantity in a candidate answer is read out with its unit, matched
against the findings that measure the same thing, and compared. A figure
that contradicts one the runtime computed is a hard failure, because the
computed one is checkable and the written one is not.

Two rules keep it from being a nuisance. A number whose dimension matches
nothing computed is not checked — most numbers in a sentence are not claims
about the design. And a figure inside the tolerance of ANY finding with that
dimension passes, since a paragraph mentioning several masses should not
fail for putting them in a different order than the model did.
"""
from __future__ import annotations

import re
from typing import Any

from core.runtime.errors import record_degradation

from .base import VerificationResult

#: A number with a unit attached. The unit pattern is deliberately loose;
#: whether it means anything is settled by trying to parse it.
_QUANTITY_RE = re.compile(
    r"(?<![\w.])(-?\d[\d,]*(?:\.\d+)?)\s*"
    r"(k|M|G|T|m|µ|u|n|c|d)?"
    r"(N·m|N m|Nm|W|V|A|J|Pa|bar|psi|Hz|kg|g|t|m/s|m|mm|cm|km|in|ft|L/s|L|s|h|"
    r"min|K|degC|°C|rpm|ohm|Ω|F|H|mol|%)"
    r"(?![\w])"
)

#: How far a written figure may sit from the computed one. Two per cent
#: covers the rounding a sentence does — "about 34 kg" for 33.7 — and does
#: not cover getting it wrong.
_TOLERANCE = 0.02

#: Below this a comparison is noise: two figures near zero differ by a large
#: fraction while meaning the same thing.
_FLOOR = 1e-9


class EngineeringTruthEngine:
    """Checks stated engineering quantities against the computed findings."""

    name = "engineering"
    domains = ("engineering", "design", "mechanical", "electrical", "thermal", "fluid")

    def handles(self, task_type: str) -> bool:
        return task_type in self.domains

    async def verify(
        self, candidate: str, *, context: dict[str, Any] | None = None
    ) -> VerificationResult:
        findings = _findings_from(context)
        if not findings:
            # Nothing computed to check against. Not a failure, and not a
            # pass either: saying "verified" here would be the exact thing
            # this engine exists to stop, one level up.
            return VerificationResult(
                domain="engineering", ok=True, checked=False, engine=self.name,
                evidence=["no computed findings were supplied to check against"],
            )
        try:
            return self._compare(candidate, findings)
        except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
            record_degradation("verifier:engineering", exc, action="comparing stated figures")
            return VerificationResult(
                domain="engineering", ok=True, checked=False, engine=self.name,
                infrastructure_failed=True,
            )

    def _compare(self, candidate: str, findings: list[dict[str, Any]]) -> VerificationResult:
        from core.engineering.units import parse_quantity

        # Group the computed values by what they measure, so a mass is only
        # ever compared against a mass. The finding's own written form is
        # kept alongside: quoting a power back as "163 m^2 kg s^-3" is
        # correct and unreadable.
        computed: dict[str, list[tuple[str, float, str]]] = {}
        for finding in findings:
            value = finding.get("value") or {}
            si = str(value.get("si") or "")
            magnitude = value.get("value")
            if not si or magnitude is None:
                continue
            computed.setdefault(si, []).append((
                str(finding.get("name") or finding.get("id") or "?"),
                float(magnitude),
                str(value.get("text") or ""),
            ))
        if not computed:
            return VerificationResult(
                domain="engineering", ok=True, checked=False, engine=self.name,
                evidence=["the findings carried no comparable values"],
            )

        issues: list[str] = []
        evidence: list[str] = []
        checked = 0
        for match in _QUANTITY_RE.finditer(candidate):
            text = f"{match.group(1).replace(',', '')} {(match.group(2) or '')}{match.group(3)}"
            try:
                stated = parse_quantity(_normalised(text))
            except (ValueError, KeyError):
                continue
            si = stated.dimension.symbol()
            candidates = computed.get(si)
            if not candidates:
                continue
            checked += 1
            magnitude = float(stated.value)
            best_name = ""
            best_error = float("inf")
            for name, value, _written in candidates:
                reference = max(abs(value), abs(magnitude), _FLOOR)
                error = abs(value - magnitude) / reference
                if error < best_error:
                    best_error = error
                    best_name = name
            if best_error <= _TOLERANCE:
                evidence.append(f"{stated.text()} matches {best_name}")
            else:
                nearest = min(candidates, key=lambda entry: abs(entry[1] - magnitude))
                issues.append(
                    f"the reply states {stated.text()}, and the closest computed figure "
                    f"is {nearest[2] or _si_text(nearest[1], si)} for {nearest[0]}. The "
                    "computed one carries its formula and its inputs; the written one "
                    "does not."
                )

        if checked == 0:
            return VerificationResult(
                domain="engineering", ok=True, checked=False, engine=self.name,
                evidence=["no stated figure measured anything the model computed"],
            )
        ok = not issues
        return VerificationResult(
            domain="engineering",
            ok=ok,
            checked=True,
            score=1.0 if ok else max(0.0, 1.0 - len(issues) / max(checked, 1)),
            engine=self.name,
            issues=issues,
            evidence=evidence[:8],
            detail={"quantities_checked": checked, "contradictions": len(issues)},
        )


def _si_text(value: float, si: str) -> str:
    """A base-unit magnitude written the way an engineer says it."""
    from core.engineering.units import dimension_of, engineering_text, si_symbol

    try:
        return engineering_text(value, si_symbol(dimension_of(si)))
    except (KeyError, ValueError):
        return f"{value:g} {si}"


def _normalised(text: str) -> str:
    """Spellings a person writes, in the spellings the unit parser takes."""
    return (
        text.replace("°C", "degC")
        .replace("Ω", "ohm")
        .replace("N·m", "N m")
        .replace("Nm", "N m")
    )


def _findings_from(context: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The computed findings this answer should agree with.

    Taken from the context the caller passed, and never fetched from disk:
    an engine that goes looking for a design to check against would end up
    checking the answer against the wrong one.
    """
    if not isinstance(context, dict):
        return []
    for key in ("engineering_findings", "design_findings", "findings"):
        value = context.get(key)
        if isinstance(value, (list, tuple)) and value:
            return [entry for entry in value if isinstance(entry, dict)]
    design = context.get("design_result") or context.get("engineering")
    if isinstance(design, dict):
        value = design.get("findings")
        if isinstance(value, (list, tuple)):
            return [entry for entry in value if isinstance(entry, dict)]
    return []

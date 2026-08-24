"""The gate: nothing reaches a drawing that cannot say where it came from.

The failure this exists to prevent is a good-looking schematic covered in
technical words and numbers that mean nothing — a picture that borrows the
authority of engineering without doing any. The defence is mechanical rather
than editorial. A number may appear on a drawing only if it is a
:class:`~core.engineering.analysis.Finding` carrying a formula, its inputs,
and the reference the formula came from. A finding with any of those
missing is dropped before rendering and reported as dropped.

Four further checks stand behind that one.

The physics has to be possible: no negative mass, no efficiency above one,
no temperature below absolute zero, no infinities. A number that is
arithmetically derived and physically impossible means an input was wrong.

The units have to agree with the domain: a port that says it carries current
has to carry something measured in amperes, and a connection between a
48 volt port and a 5 volt port is a fault, not a wire.

Every requirement has to name a check that actually ran. A requirement whose
check does not exist is reported as unverified, never as met.

And the formulas themselves have to still reproduce their published answers,
which is :mod:`core.engineering.validation`. If that battery is failing, this
gate says so on the face of the drawing rather than letting the drawing
imply otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from core.engineering.domains import DomainError, domain as get_domain
from core.engineering.units import Q, Quantity

__all__ = [
    "Problem",
    "DesignVerdict",
    "verify_design",
    "grounded_findings",
    "ungrounded",
]

#: A finding must carry all of these before its number may be drawn.
_REQUIRED_PROVENANCE = ("formula", "method")

#: Ratios that cannot exceed one, whatever the arithmetic said.
_BOUNDED_RATIOS = (
    "efficiency", "usable_fraction", "conversion", "duty_cycle", "emissivity",
)


@dataclass(frozen=True, slots=True)
class Problem:
    """One thing wrong, and whether it stops the drawing."""

    code: str
    severity: str
    subject: str
    message: str
    advice: str = ""

    @property
    def blocking(self) -> bool:
        return self.severity == "blocking"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "subject": self.subject,
            "message": self.message,
            "advice": self.advice,
        }


@dataclass(frozen=True, slots=True)
class DesignVerdict:
    """Whether this design may be drawn, built, or neither."""

    #: Findings that carry their provenance and may be quoted.
    grounded: tuple = ()
    #: Findings dropped for having no formula, method or inputs.
    dropped: tuple = ()
    problems: tuple[Problem, ...] = ()
    checks_run: tuple[str, ...] = ()
    validation_ok: bool = True
    validation_note: str = ""
    buildable: bool = False
    interference: tuple[dict[str, Any], ...] = ()

    @property
    def blocking(self) -> tuple[Problem, ...]:
        return tuple(p for p in self.problems if p.blocking)

    @property
    def warnings(self) -> tuple[Problem, ...]:
        return tuple(p for p in self.problems if not p.blocking)

    @property
    def ok(self) -> bool:
        return not self.blocking and self.validation_ok

    def plain(self) -> str:
        if not self.validation_ok:
            return (
                "This design was not drawn. The engine's own formulas are failing their "
                f"validation problems: {self.validation_note}"
            )
        if self.blocking:
            first = self.blocking[0]
            count = len(self.blocking)
            head = (
                f"One thing stops this being a usable drawing: {first.message}"
                if count == 1
                else f"{count} things stop this being a usable drawing. The first is "
                f"{first.message}"
            )
            return head + (f" To fix it: {first.advice}" if first.advice else "")
        head = (
            f"{len(self.grounded)} results were computed and every one of them carries "
            "the formula, the inputs and the reference behind it."
        )
        if self.dropped:
            head += (
                f" {len(self.dropped)} were dropped for having no working behind them, "
                "and are not on the drawing."
            )
        if self.warnings:
            head += f" {len(self.warnings)} things are worth attention."
        head += (
            " Every part says how it would be obtained, so this can be ordered and built."
            if self.buildable
            else " Some parts do not say how they would be obtained, so this cannot be "
            "ordered as it stands."
        )
        return head

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "buildable": self.buildable,
            "grounded": len(self.grounded),
            "dropped": [
                {"id": f.id, "name": f.name} for f in self.dropped
            ],
            "problems": [p.to_dict() for p in self.problems],
            "blocking": len(self.blocking),
            "warnings": len(self.warnings),
            "checks_run": list(self.checks_run),
            "validation_ok": self.validation_ok,
            "validation_note": self.validation_note,
            "interference": list(self.interference),
            "plain": self.plain(),
        }


def ungrounded(finding) -> str:
    """Why this finding may not be drawn, or an empty string if it may.

    A number with no formula behind it is the exact thing this package
    exists to keep off a drawing, whatever produced it.
    """
    for attribute in _REQUIRED_PROVENANCE:
        if not str(getattr(finding, attribute, "") or "").strip():
            return f"no {attribute}"
    value = getattr(finding, "value", None)
    if not isinstance(value, Quantity):
        return "value is not a quantity"
    if not math.isfinite(float(value.value)):
        return "value is not a finite number"
    if not str(getattr(finding, "plain", "") or "").strip():
        return "no plain reading"
    return ""


def grounded_findings(findings) -> tuple[tuple, tuple]:
    """Split findings into those that may be drawn and those that may not."""
    keep = []
    drop = []
    for finding in findings:
        (drop if ungrounded(finding) else keep).append(finding)
    return (tuple(keep), tuple(drop))


def _check_physics(design, findings) -> list[Problem]:
    problems: list[Problem] = []
    for part in design.parts:
        mass = part.mass()
        if mass is not None and float(mass.value) < 0:
            problems.append(Problem(
                "negative_mass", "blocking", part.id,
                f"the {part.lay_name or part.name} has a negative mass, which cannot happen.",
                "Check the geometry: a wall thicker than its own radius does this.",
            ))
        volume = part.volume()
        if volume is not None and float(volume.value) <= 0:
            problems.append(Problem(
                "zero_volume", "blocking", part.id,
                f"the {part.lay_name or part.name} encloses no volume.",
                "Check its dimensions.",
            ))
        for name, rating in part.ratings.items():
            if not math.isfinite(float(rating.value)):
                problems.append(Problem(
                    "not_a_number", "blocking", part.id,
                    f"the {name} of the {part.lay_name or part.name} is not a number.",
                ))
                continue
            if any(word in name for word in _BOUNDED_RATIOS) and float(rating.value) > 1.0:
                problems.append(Problem(
                    "impossible_ratio", "blocking", part.id,
                    f"the {name} of the {part.lay_name or part.name} is "
                    f"{float(rating.value):.2f}, and a fraction cannot exceed one.",
                    "If it was meant as a percentage, write it as a percentage.",
                ))
            if "temperature" in name and rating.dimension == Q(1, "K").dimension:
                if float(rating.value) < 0:
                    problems.append(Problem(
                        "below_absolute_zero", "blocking", part.id,
                        f"the {name} of the {part.lay_name or part.name} is below "
                        "absolute zero.",
                    ))
    for finding in findings:
        if not math.isfinite(float(finding.value.value)):
            problems.append(Problem(
                "not_a_number", "blocking", finding.subject,
                f"the result named {finding.name} is not a finite number.",
                "One of its inputs is zero or missing.",
            ))
    return problems


def _check_units(design) -> list[Problem]:
    problems: list[Problem] = []
    for part in design.parts:
        for port in part.ports:
            try:
                spec = port.domain_spec
            except DomainError as exc:
                problems.append(Problem(
                    "unknown_domain", "blocking", part.id,
                    f"the {port.name} port of the {part.lay_name or part.name} names a "
                    f"domain nothing recognises. {exc}",
                ))
                continue
            if port.across is not None and spec.across_unit:
                if port.across.dimension != spec.across_dimension:
                    problems.append(Problem(
                        "wrong_unit", "blocking", part.id,
                        f"the {port.name} port of the {part.lay_name or part.name} is "
                        f"{spec.name.lower()}, so its {spec.across_name} must be in "
                        f"{spec.across_unit}, and it is given in {port.across.unit}.",
                    ))
            if port.through is not None and spec.through_unit:
                if port.through.dimension != spec.through_dimension:
                    problems.append(Problem(
                        "wrong_unit", "blocking", part.id,
                        f"the {port.name} port of the {part.lay_name or part.name} carries "
                        f"{spec.through_name}, which must be in {spec.through_unit}, and it "
                        f"is given in {port.through.unit}.",
                    ))

    for link in design.connections:
        source = design.find_port(link.source)
        target = design.find_port(link.target)
        for reference, found in ((link.source, source), (link.target, target)):
            if found is None:
                problems.append(Problem(
                    "dangling_connection", "blocking", link.id,
                    f"the connection {link.id} joins {reference}, and there is no such port.",
                    "Check the part id and port name.",
                ))
        if source is None or target is None:
            continue
        _sp, source_port = source
        _tp, target_port = target
        if source_port.domain != target_port.domain:
            problems.append(Problem(
                "domain_mismatch", "blocking", link.id,
                f"{link.id} joins a {get_domain(source_port.domain).name.lower()} port to a "
                f"{get_domain(target_port.domain).name.lower()} one. Those do not connect.",
                "Put a converter between them, or correct one of the domains.",
            ))
            continue
        if source_port.across is not None and target_port.across is not None:
            spec = source_port.domain_spec
            a = float(source_port.across.value)
            b = float(target_port.across.value)
            reference = max(abs(a), abs(b)) or 1.0
            if abs(a - b) / reference > 0.05:
                problems.append(Problem(
                    "potential_mismatch", "blocking", link.id,
                    f"{link.id} joins a port at {source_port.across.text()} to one at "
                    f"{target_port.across.text()}. Everything wired together sits at the "
                    f"same {spec.across_name}, so this connection cannot exist as drawn.",
                    "Add a converter, a regulator or a transformer between them.",
                ))
    return problems


def _check_requirements(design, findings) -> list[Problem]:
    problems: list[Problem] = []
    available = {f.id for f in findings}
    for requirement in design.requirements:
        if not requirement.check:
            problems.append(Problem(
                "unverified_requirement", "warning", requirement.id,
                f"{requirement.id} names no check, so nothing tested it.",
                "Point it at one of the computed results.",
            ))
            continue
        if requirement.check not in available:
            problems.append(Problem(
                "missing_check", "warning", requirement.id,
                f"{requirement.id} says it is verified by {requirement.check}, and that "
                "result was not produced.",
                "Either the check name is wrong or the analysis did not have its inputs.",
            ))
    return problems


def _check_completeness(design) -> list[Problem]:
    problems: list[Problem] = []
    if not design.parts:
        problems.append(Problem(
            "empty_design", "blocking", design.name,
            "this design has no parts in it.",
        ))
        return problems
    connected: set[str] = set()
    for link in design.connections:
        connected.add(link.source.split(".")[0])
        connected.add(link.target.split(".")[0])
    for part in design.parts:
        if part.solid is None:
            problems.append(Problem(
                "no_shape", "warning", part.id,
                f"the {part.lay_name or part.name} has no shape, so it has no mass, no "
                "size and cannot be drawn.",
                "Give it a shape and its dimensions.",
            ))
        if part.material is None and part.solid is not None:
            problems.append(Problem(
                "no_material", "warning", part.id,
                f"the {part.lay_name or part.name} does not say what it is made of, so "
                "nothing can be said about its weight or strength.",
                "Name a material.",
            ))
        if len(design.parts) > 1 and part.id not in connected and part.ports:
            problems.append(Problem(
                "orphan_part", "warning", part.id,
                f"the {part.lay_name or part.name} has ports and nothing is connected to "
                "them, so it does nothing in this design.",
                "Connect it, or remove it.",
            ))
        if not part.function:
            problems.append(Problem(
                "no_stated_function", "warning", part.id,
                f"the {part.lay_name or part.name} does not say what it is for.",
                "One sentence about what job it does.",
            ))
    return problems


def _check_sourcing(design) -> tuple[bool, list[Problem]]:
    problems: list[Problem] = []
    unsourced = [
        part for part in design.parts
        if part.sourcing.method == "unspecified" or not part.sourcing.specification
    ]
    for part in unsourced:
        problems.append(Problem(
            "no_source", "warning", part.id,
            f"the {part.lay_name or part.name} does not say whether it is bought or made, "
            "or to what specification.",
            "Name a supplier part, a standard, or the process that produces it.",
        ))
    return (not unsourced, problems)


def verify_design(design, findings: tuple = (), *, check_validation: bool = True) -> DesignVerdict:
    """Decide whether this design may be drawn, and say what is wrong if not.

    ``check_validation`` runs the whole published-answer battery, which takes
    a few milliseconds and is the difference between a drawing whose numbers
    are checked and one whose numbers merely look checked.
    """
    from core.engineering.layout import interference

    kept, dropped = grounded_findings(findings)
    problems: list[Problem] = []
    for finding in dropped:
        problems.append(Problem(
            "ungrounded_result", "warning", finding.subject or finding.id,
            f"the result named {finding.name} has {ungrounded(finding)}, so it is not "
            "shown on the drawing.",
            "A number without its working behind it is not evidence of anything.",
        ))

    problems.extend(_check_physics(design, kept))
    problems.extend(_check_units(design))
    problems.extend(_check_requirements(design, kept))
    problems.extend(_check_completeness(design))
    buildable, sourcing_problems = _check_sourcing(design)
    problems.extend(sourcing_problems)

    clashes = interference(design)
    for clash in clashes:
        problems.append(Problem(
            "interference", "blocking", "/".join(clash["parts"]),
            clash["plain"],
            "Move one of them, or declare one as enclosing the other.",
        ))

    validation_ok = True
    validation_note = ""
    if check_validation:
        from core.engineering.validation import run_validation

        report = run_validation()
        validation_ok = report.ok
        validation_note = report.plain()

    checks = (
        "provenance", "physical plausibility", "unit and domain consistency",
        "requirement coverage", "completeness", "sourcing", "interference",
    )
    if check_validation:
        checks = checks + ("published-answer validation",)

    return DesignVerdict(
        grounded=kept,
        dropped=dropped,
        problems=tuple(problems),
        checks_run=checks,
        validation_ok=validation_ok,
        validation_note=validation_note,
        buildable=buildable and not any(p.blocking for p in problems),
        interference=clashes,
    )

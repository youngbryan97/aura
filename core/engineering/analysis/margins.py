"""Margins of safety, mass growth, derating and single points of failure.

The checks a design review runs after the physics is done. Each one turns a
raw analysis result into the form a programme actually tracks: a margin
against a stated factor, a mass with its growth allowance, a part running
inside its derated limit, a failure mode with nothing behind it.
"""

from __future__ import annotations

from collections.abc import Iterable

from core.engineering.analysis import Finding, register
from core.engineering.assurance import (
    DERATING,
    FailureMode,
    criticality,
    derating_check,
    factor_set,
    margin_of_safety,
    mass_statement,
    readiness_note,
)
from core.engineering.units import Q


@register(
    "structural_margins",
    "Margin of safety",
    "Does every loaded part pass with the required design factor?",
    domains=("structural",),
    discipline="assurance",
)
def structural_margins(design) -> Iterable[Finding]:
    factors = factor_set(design)
    yield Finding(
        id="assurance.factor_set",
        name="Design factors in force",
        value=Q(factors.ultimate_factor, "count"),
        formula="the factor set the regime requires",
        inputs={
            "yield factor": Q(factors.yield_factor, "count"),
            "ultimate factor": Q(factors.ultimate_factor, "count"),
        },
        method=factors.reference,
        plain=(
            f"This is checked as {factors.applies_to or factors.name.lower()}, which means "
            f"every load is multiplied by {factors.yield_factor:g} before it is compared "
            f"against what the material can take without bending, and by "
            f"{factors.ultimate_factor:g} before it is compared against what breaks it. "
            f"{factors.notes}"
        ),
        subject="assurance",
    )

    for part in design.parts:
        if part.material is None:
            continue
        applied = None
        for key in ("working_stress", "applied_stress", "stress"):
            if key in part.ratings:
                applied = part.ratings[key]
                break
        if applied is None:
            continue
        yield_strength = part.material.yield_strength
        ultimate = part.material.ultimate_strength
        for label, allowable, factor in (
            ("yield", yield_strength, factors.yield_factor),
            ("ultimate", ultimate, factors.ultimate_factor),
        ):
            if allowable is None:
                continue
            result = margin_of_safety(
                allowable, applied, factor, basis=label, reference=factors.reference
            )
            yield Finding(
                id=f"assurance.margin.{label}.{part.id}",
                name=f"{label.title()} margin, {part.name}",
                value=Q(result.margin, "count"),
                formula="MS = allowable / (limit x factor) - 1",
                inputs={
                    "allowable": allowable,
                    "limit": applied,
                    "factor": Q(factor, "count"),
                },
                method=f"{factors.reference}, {label} case",
                plain=result.plain(),
                subject=part.id,
                verdict="pass" if result.passes else "fail",
                margin=result.margin,
                advice=(
                    ""
                    if result.passes
                    else f"Reduce the load, thicken the section, or move to a material with "
                    f"at least {Q(float(applied.value) * factor, 'Pa').as_('MPa').text()} "
                    "of strength."
                ),
            )


@register(
    "mass_growth",
    "Mass with growth allowance",
    "What will it actually weigh once the design is finished?",
    discipline="assurance",
)
def mass_growth(design) -> Iterable[Finding]:
    basic = design.total_mass()
    if basic is None:
        return
    maturity = str(design.environment.get("maturity", "")) or "estimated"
    if not isinstance(maturity, str):
        maturity = "estimated"
    budget = design.environment.get("mass_budget")
    statement = mass_statement(basic, maturity, budget)
    yield Finding(
        id="assurance.mass_growth",
        name="Predicted mass",
        value=statement.predicted,
        formula="predicted = basic x (1 + growth allowance)",
        inputs={"basic": statement.basic, "allowance": statement.growth},
        method=statement.reference,
        plain=statement.plain(),
        subject="assembly",
        verdict=(
            "pass"
            if statement.margin is None or statement.margin >= 0
            else "fail"
        ),
        margin=statement.margin,
        advice=(
            ""
            if statement.margin is None or statement.margin >= 0.05
            else "Find mass to remove now; it is far cheaper here than after drawings are released."
        ),
    )


@register(
    "part_derating",
    "Derating",
    "Is anything running too close to its own rating?",
    domains=("electrical",),
    discipline="assurance",
)
def part_derating(design) -> Iterable[Finding]:
    for part in design.parts:
        for kind in DERATING:
            applied_key = f"{kind}_applied"
            rated_key = f"{kind}_rated"
            applied = part.ratings.get(applied_key)
            rated = part.ratings.get(rated_key)
            if applied is None or rated is None:
                continue
            passes, used, sentence = derating_check(kind, applied, rated)
            yield Finding(
                id=f"assurance.derating.{kind}.{part.id}",
                name=f"Derating, {part.name}",
                value=Q(used, "count"),
                formula="usage = applied / (rated x derating fraction)",
                inputs={"applied": applied, "rated": rated},
                method="NASA EEE-INST-002 derating practice",
                plain=sentence,
                subject=part.id,
                verdict="pass" if passes else "fail",
                margin=1.0 - used,
                advice="" if passes else "Use a larger part, or share the load across two.",
            )


@register(
    "single_point_failures",
    "Single points of failure",
    "What stops the whole thing when it fails on its own?",
    discipline="assurance",
)
def single_point_failures(design) -> Iterable[Finding]:
    """Find the parts on a critical path with nothing in parallel.

    A part is a single point of failure when every path through the graph
    from a source to a load passes through it. That is a connectivity
    question the model can answer, and answering it is how redundancy gets
    designed in while it is still a line on a drawing.
    """
    modes: list[FailureMode] = []
    for part in design.parts:
        declared = part.ratings.get("failure_rate")
        mode_text = part.notes if "fail" in part.notes.lower() else ""
        essential = "redundant" not in part.tags and "spare" not in part.tags
        carrying = [
            link
            for link in design.connections
            if link.source.startswith(part.id) or link.target.startswith(part.id)
        ]
        if not carrying:
            continue
        # A part is bypassable if some other part serves the same two ends.
        endpoints = {
            (link.source.split(".")[0], link.target.split(".")[0], link.domain)
            for link in carrying
        }
        parallel = 0
        for other in design.parts:
            if other.id == part.id:
                continue
            other_endpoints = {
                (link.source.split(".")[0], link.target.split(".")[0], link.domain)
                for link in design.connections
                if link.source.startswith(other.id) or link.target.startswith(other.id)
            }
            if endpoints & other_endpoints and other.subsystem == part.subsystem:
                parallel += 1
        single = essential and parallel == 0 and len(carrying) >= 2
        if not single and not mode_text:
            continue
        modes.append(
            FailureMode(
                part_id=part.id,
                mode=mode_text or f"{part.lay_name or part.name} stops working",
                cause=part.notes or "wear, overload or a manufacturing fault",
                effect=(
                    f"everything downstream of the {part.lay_name or part.name.lower()} "
                    "stops"
                ),
                severity=2 if single else 3,
                occurrence=3 if declared is None else min(int(float(declared.value) * 10) + 1, 10),
                detection=4,
                single_point=single,
                mitigation=(
                    "Add a second one in parallel, or accept the stop and make it easy to swap."
                    if single
                    else ""
                ),
            )
        )
    if not modes:
        return
    summary = criticality(tuple(modes))
    worst = summary["worst"]
    yield Finding(
        id="assurance.fmeca",
        name="Failure modes",
        value=Q(len(summary["single_points"]), "count"),
        formula="a part with no parallel path in its own subsystem is a single point",
        inputs={"modes assessed": Q(len(modes), "count")},
        method="MIL-STD-1629A criticality analysis over the connection graph",
        plain=summary["plain"],
        subject="assurance",
        verdict="pass" if not summary["single_points"] else "watch",
        advice=(
            ""
            if not summary["single_points"]
            else f"Start with {worst['part']}: "
            + str(worst.get("mitigation") or "add a parallel path or a spare.")
        ),
    )
    for mode in modes:
        if not mode.single_point:
            continue
        yield Finding(
            id=f"assurance.spf.{mode.part_id}",
            name=f"Single point of failure: {mode.part_id}",
            value=Q(mode.risk_number, "count"),
            formula="risk = severity x likelihood x how hard it is to spot",
            inputs={
                "severity": Q(mode.severity, "count"),
                "likelihood": Q(mode.occurrence, "count"),
                "detectability": Q(mode.detection, "count"),
            },
            method="MIL-STD-1629A risk priority number",
            plain=mode.plain(),
            subject=mode.part_id,
            verdict="watch",
            advice=mode.mitigation,
        )


@register(
    "readiness",
    "How ready this is to build",
    "Is this a sketch, a prototype, or something proven?",
    discipline="assurance",
)
def readiness(design) -> Iterable[Finding]:
    known = [p for p in design.parts if p.sourcing.method != "unspecified"]
    if not design.parts:
        return
    buyable = [p for p in design.parts if p.sourcing.buyable]
    makeable = [p for p in design.parts if p.sourcing.makeable]
    covered = len(known) / len(design.parts)
    level = 2
    if covered >= 0.9 and len(buyable) + len(makeable) == len(design.parts):
        level = 4
    elif covered >= 0.5:
        level = 3
    yield Finding(
        id="assurance.readiness",
        name="Buildability",
        value=Q(covered, "count"),
        formula="the share of parts that say how they are obtained",
        inputs={
            "parts": Q(len(design.parts), "count"),
            "with a source": Q(len(known), "count"),
        },
        method="NASA technology readiness scale, applied to the sourcing record",
        plain=(
            f"{len(known)} of {len(design.parts)} parts say how they would be obtained: "
            f"{len(buyable)} bought off the shelf and {len(makeable)} made. "
            + readiness_note(level)
            + (
                ""
                if covered >= 0.9
                else " The parts with no source named are the ones that would stall an order."
            )
        ),
        subject="assurance",
        verdict="pass" if covered >= 0.9 else "watch",
        margin=covered,
        advice=(
            ""
            if covered >= 0.9
            else "Name a supplier, a standard part number or a process for each remaining part."
        ),
    )

"""Margins, maturity, failure modes and evidence: the aerospace discipline.

The difference between a sketch and a design that flies is not the drawing.
It is four habits, and all four are mechanical enough to enforce in code.

Every load is quoted against a factored allowable, and the answer is a
margin of safety that has to be at or above zero. NASA-STD-5001B defines it
as allowable over limit-load-times-factor, minus one, and a negative margin
is a design that has failed its own check whatever the drawing looks like.

Every mass carries a growth allowance sized by how mature the design is,
per ANSI/AIAA S-120A. A part still on a napkin grows; a part that has been
weighed does not. Reporting basic mass without the allowance is how
spacecraft arrive over budget.

Every requirement names how it will be shown to be met — test, analysis,
inspection or demonstration — because an unverified requirement is a wish
with a number on it.

Every part that can fail is asked how, how bad, and whether anybody would
notice, which is MIL-STD-1629A's FMECA in three columns. A single point of
failure that nothing detects is worth knowing about while it is still cheap
to design out.

Subsea work substitutes its own factors — collapse depth above operating
depth, an out-of-roundness allowance on any shell squeezed from outside —
and the factor sets are data here rather than assumptions in a formula.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from core.engineering.units import Q, Quantity

__all__ = [
    "FactorSet",
    "FACTOR_SETS",
    "factor_set",
    "margin_of_safety",
    "MarginResult",
    "MaturityLevel",
    "MASS_GROWTH",
    "mass_growth_allowance",
    "MassStatement",
    "mass_statement",
    "VerificationMethod",
    "VERIFICATION_METHODS",
    "FailureMode",
    "criticality",
    "DERATING",
    "derating_check",
    "TRL_LEVELS",
    "readiness_note",
]


# ---------------------------------------------------------------------------
# Factors of safety
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FactorSet:
    """One regime's design factors, named so the source can be checked."""

    key: str
    name: str
    yield_factor: float
    ultimate_factor: float
    proof_factor: float = 0.0
    burst_factor: float = 0.0
    reference: str = ""
    applies_to: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "yield_factor": self.yield_factor,
            "ultimate_factor": self.ultimate_factor,
            "proof_factor": self.proof_factor,
            "burst_factor": self.burst_factor,
            "reference": self.reference,
            "applies_to": self.applies_to,
            "notes": self.notes,
        }


FACTOR_SETS: dict[str, FactorSet] = {
    entry.key: entry
    for entry in (
        FactorSet(
            "spaceflight_metallic", "Spaceflight structure, metallic", 1.25, 1.40,
            reference="NASA-STD-5001B",
            applies_to="uncrewed spaceflight structure verified by analysis and test",
            notes="Yield 1.25 and ultimate 1.40 on limit load. Limit load is the "
            "worst load the structure will actually see in service.",
        ),
        FactorSet(
            "spaceflight_untested", "Spaceflight structure, no qualification test", 1.25, 2.00,
            reference="NASA-STD-5001B",
            applies_to="structure qualified by analysis alone",
            notes="The ultimate factor doubles when nothing was broken to prove the number.",
        ),
        FactorSet(
            "crewed_pressure_vessel", "Crewed pressure vessel", 1.50, 2.00,
            proof_factor=1.50, burst_factor=2.00,
            reference="MIL-STD-1522A and NASA-STD-5001B",
            applies_to="metallic pressure vessels with people nearby",
            notes="Proof at 1.5 times the maximum expected operating pressure, burst at 2.0.",
        ),
        FactorSet(
            "composite_overwrapped", "Composite overwrapped pressure vessel", 1.50, 3.50,
            proof_factor=1.50, burst_factor=3.50,
            reference="ANSI/AIAA S-081 practice for COPVs",
            applies_to="filament-wound vessels",
            notes="Composites carry a much larger burst factor because their failure "
            "is sudden and their scatter is wide.",
        ),
        FactorSet(
            "asme_viii_div1", "Unfired pressure vessel", 1.50, 3.50,
            proof_factor=1.30, burst_factor=3.50,
            reference="ASME BPVC Section VIII Division 1",
            applies_to="industrial pressure vessels",
            notes="Allowable stress is the tensile strength divided by 3.5, which is "
            "the design margin the code carries.",
        ),
        FactorSet(
            "submersible_hull", "Submersible pressure hull", 1.50, 1.50,
            proof_factor=1.25, burst_factor=1.50,
            reference="ABS Rules for Underwater Vehicles, Systems and Hyperbaric Facilities",
            applies_to="crewed and uncrewed pressure hulls",
            notes="Collapse depth is 1.5 times the maximum operating depth, and the "
            "shell is checked for buckling with an out-of-roundness allowance "
            "rather than for stress alone.",
        ),
        FactorSet(
            "lifting_equipment", "Lifting and rigging", 2.00, 5.00,
            reference="ASME B30 series practice",
            applies_to="anything suspended over a person",
            notes="Five to one on breaking strength is the usual rigging figure.",
        ),
        FactorSet(
            "civil_structure", "Building structure", 1.50, 1.60,
            reference="Eurocode partial factors, permanent 1.35 and variable 1.5",
            applies_to="buildings and static structures",
        ),
        FactorSet(
            "consumer_product", "Consumer product", 1.50, 2.00,
            reference="General product design practice",
            applies_to="handheld and household items",
            notes="No single standard governs; two to one on yield is the working default.",
        ),
        FactorSet(
            "prototype", "Prototype or bench rig", 1.20, 1.50,
            reference="Working practice for a rig nobody stands under",
            applies_to="laboratory and desk prototypes",
            notes="Thin on purpose. A prototype that never breaks teaches nothing about "
            "where its limits are, but it must still be safe to be near.",
        ),
    )
}

#: What kind of thing gets which factors, when the brief does not say.
_REGIME_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("spacecraft", "satellite", "orbital", "launch", "rocket"), "spaceflight_metallic"),
    (("submersible", "submarine", "subsea", "rov", "auv", "underwater", "deep"), "submersible_hull"),
    (("pressure vessel", "tank", "boiler", "receiver"), "asme_viii_div1"),
    (("crane", "hoist", "lifting", "rigging", "sling"), "lifting_equipment"),
    (("building", "bridge", "frame", "foundation"), "civil_structure"),
    (("prototype", "bench", "test rig", "breadboard"), "prototype"),
)


def factor_set(design_or_text: Any) -> FactorSet:
    """Pick the factor set the design falls under, from what it says it is."""
    if isinstance(design_or_text, str):
        text = design_or_text.lower()
    else:
        text = " ".join(
            str(value or "").lower()
            for value in (
                getattr(design_or_text, "name", ""),
                getattr(design_or_text, "purpose", ""),
                getattr(design_or_text, "discipline", ""),
                getattr(design_or_text, "rationale", ""),
            )
        )
        declared = (getattr(design_or_text, "environment", {}) or {}).get("factor_set")
        if isinstance(declared, str) and declared in FACTOR_SETS:
            return FACTOR_SETS[declared]
        if getattr(design_or_text, "environment", None) and (
            "depth" in design_or_text.environment
            or "water_depth" in design_or_text.environment
        ):
            return FACTOR_SETS["submersible_hull"]
    for keywords, key in _REGIME_HINTS:
        if any(word in text for word in keywords):
            return FACTOR_SETS[key]
    return FACTOR_SETS["consumer_product"]


@dataclass(frozen=True, slots=True)
class MarginResult:
    """A margin of safety with everything needed to defend it."""

    margin: float
    allowable: Quantity
    limit: Quantity
    factor: float
    basis: str
    reference: str

    @property
    def passes(self) -> bool:
        return self.margin >= 0.0

    def plain(self) -> str:
        if self.margin >= 0:
            return (
                f"It holds with {self.margin * 100:.0f}% to spare after the "
                f"{self.factor:g}x design factor is applied. Anything at or above zero "
                "passes; this one is not close to the edge."
                if self.margin > 0.25
                else f"It holds, with {self.margin * 100:.0f}% left after the "
                f"{self.factor:g}x design factor. That is a pass, and it is tight enough "
                "that a change anywhere nearby needs rechecking."
            )
        return (
            f"It fails its own check by {abs(self.margin) * 100:.0f}%. With the required "
            f"{self.factor:g}x factor applied, the load exceeds what the part can take. "
            "This has to change before anything is built."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "margin_of_safety": self.margin,
            "passes": self.passes,
            "allowable": self.allowable.to_dict(),
            "limit": self.limit.to_dict(),
            "factor": self.factor,
            "basis": self.basis,
            "reference": self.reference,
            "plain": self.plain(),
        }


def margin_of_safety(
    allowable: Quantity,
    limit: Quantity,
    factor: float,
    *,
    basis: str = "yield",
    reference: str = "NASA-STD-5001B",
) -> MarginResult:
    """Margin of safety: allowable over factored limit load, minus one.

    The definition is NASA-STD-5001B's. Reporting a bare ratio instead
    invites the reader to compare it against a factor they have in mind
    rather than the one the design was checked against, which is how two
    engineers agree on a number and disagree about whether it passes.
    """
    factored = float(limit.value) * float(factor)
    if factored <= 0:
        raise ValueError("a factored limit load must be positive")
    margin = float(allowable.value) / factored - 1.0
    return MarginResult(margin, allowable, limit, float(factor), basis, reference)


# ---------------------------------------------------------------------------
# Mass growth, per ANSI/AIAA S-120A
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MaturityLevel:
    """How well a mass is known, and what to add for what is not known yet."""

    key: str
    name: str
    allowance: float
    description: str


MASS_GROWTH: dict[str, MaturityLevel] = {
    entry.key: entry
    for entry in (
        MaturityLevel(
            "estimated", "Estimated", 0.30,
            "A number from a sketch or a similar past design. Thirty per cent is added, "
            "because designs at this stage grow.",
        ),
        MaturityLevel(
            "layout", "Layout", 0.20,
            "A number from a model with real shapes in it, before any drawing is released.",
        ),
        MaturityLevel(
            "preliminary", "Preliminary design", 0.15,
            "Drawings exist and have not been checked.",
        ),
        MaturityLevel(
            "released", "Released drawings", 0.08,
            "Drawings are checked and released, and nothing has been made yet.",
        ),
        MaturityLevel(
            "existing", "Existing hardware", 0.04,
            "The same part has been built before and weighed, on another programme.",
        ),
        MaturityLevel(
            "measured", "Measured", 0.00,
            "This part has been put on a scale. Nothing is added.",
        ),
    )
}


def mass_growth_allowance(maturity: str) -> MaturityLevel:
    key = str(maturity or "estimated").strip().lower()
    return MASS_GROWTH.get(key, MASS_GROWTH["estimated"])


@dataclass(frozen=True, slots=True)
class MassStatement:
    """Basic, predicted and allocated mass, the way a programme tracks it."""

    basic: Quantity
    growth: Quantity
    predicted: Quantity
    allocated: Quantity | None = None
    margin: float | None = None
    maturity: str = "estimated"
    reference: str = "ANSI/AIAA S-120A-2015"

    def plain(self) -> str:
        sentence = (
            f"The parts as drawn come to {self.basic.text()}. At this stage of design "
            f"({MASS_GROWTH.get(self.maturity, MASS_GROWTH['estimated']).name.lower()}) "
            f"the standard allowance for what is still missing is "
            f"{self.growth.text()}, so the number to plan against is "
            f"{self.predicted.text()}."
        )
        if self.allocated is not None and self.margin is not None:
            sentence += (
                f" Against a {self.allocated.text()} budget that leaves "
                f"{self.margin * 100:.0f}% in hand."
                if self.margin >= 0
                else f" That is {abs(self.margin) * 100:.0f}% over the "
                f"{self.allocated.text()} budget."
            )
        return sentence

    def to_dict(self) -> dict[str, Any]:
        out = {
            "basic": self.basic.to_dict(),
            "growth_allowance": self.growth.to_dict(),
            "predicted": self.predicted.to_dict(),
            "maturity": self.maturity,
            "reference": self.reference,
            "plain": self.plain(),
        }
        if self.allocated is not None:
            out["allocated"] = self.allocated.to_dict()
        if self.margin is not None:
            out["margin"] = self.margin
        return out


def mass_statement(
    basic: Quantity, maturity: str = "estimated", allocated: Quantity | None = None
) -> MassStatement:
    level = mass_growth_allowance(maturity)
    growth = basic * level.allowance
    predicted = basic + growth
    margin = None
    if allocated is not None and float(allocated.value) > 0:
        margin = (float(allocated.value) - float(predicted.value)) / float(allocated.value)
    return MassStatement(basic, growth, predicted, allocated, margin, level.key)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VerificationMethod:
    key: str
    name: str
    strength: int
    description: str
    when: str


VERIFICATION_METHODS: dict[str, VerificationMethod] = {
    entry.key: entry
    for entry in (
        VerificationMethod(
            "test", "Test", 4,
            "Build it and measure it doing the thing.",
            "The only method that catches what nobody thought to model. Required "
            "for anything whose failure hurts somebody.",
        ),
        VerificationMethod(
            "demonstration", "Demonstration", 3,
            "Operate it and watch the outcome, without instrumenting it.",
            "Good for a function that either happens or does not: a latch releases, "
            "a light comes on.",
        ),
        VerificationMethod(
            "analysis", "Analysis", 2,
            "Calculate it from established physics.",
            "What everything in this package does. Sound where the physics is "
            "well covered and the inputs are known, and no substitute for a test "
            "where either is in doubt.",
        ),
        VerificationMethod(
            "inspection", "Inspection", 1,
            "Look at it, measure it, read the certificate.",
            "For dimensions, materials, markings and workmanship.",
        ),
        VerificationMethod(
            "similarity", "Similarity", 1,
            "Point at the near-identical thing that was already qualified.",
            "Only valid when the differences are argued item by item.",
        ),
    )
}


# ---------------------------------------------------------------------------
# Failure modes, MIL-STD-1629A
# ---------------------------------------------------------------------------

#: Severity classes. The words are what a review board uses, and the number
#: is what the risk arithmetic uses.
SEVERITY: dict[int, tuple[str, str]] = {
    1: ("Catastrophic", "Someone is killed or the mission is lost."),
    2: ("Critical", "Someone is seriously hurt, or the mission is badly damaged."),
    3: ("Marginal", "Minor injury or degraded performance."),
    4: ("Negligible", "An inconvenience; nothing is damaged."),
}


@dataclass(frozen=True, slots=True)
class FailureMode:
    """One way a part can fail, and what it would take to catch it."""

    part_id: str
    mode: str
    cause: str
    effect: str
    severity: int = 3
    occurrence: int = 3
    detection: int = 3
    mitigation: str = ""
    single_point: bool = False
    detectable_by: str = ""

    @property
    def risk_number(self) -> int:
        """Severity x occurrence x detection, the FMECA risk priority number.

        The scale runs 1 to 10 on each axis in the usual practice; the
        defaults here are mid-scale, and a mode with no numbers on it gets
        reported as unassessed rather than as low risk.
        """
        return int(self.severity) * int(self.occurrence) * int(self.detection)

    def plain(self) -> str:
        label, meaning = SEVERITY.get(min(self.severity, 4), SEVERITY[3])
        sentence = (
            f"If the {self.mode.lower()} happens, {self.effect.lower().rstrip('.')}. "
            f"That is {label.lower()}: {meaning.lower()}"
        )
        if self.single_point:
            sentence += (
                " Nothing else covers this one, so the whole thing stops when it does."
            )
        if self.mitigation:
            sentence += f" The way to reduce it: {self.mitigation.rstrip('.')}."
        return sentence

    def to_dict(self) -> dict[str, Any]:
        label, meaning = SEVERITY.get(min(self.severity, 4), SEVERITY[3])
        return {
            "part": self.part_id,
            "mode": self.mode,
            "cause": self.cause,
            "effect": self.effect,
            "severity": self.severity,
            "severity_label": label,
            "severity_meaning": meaning,
            "occurrence": self.occurrence,
            "detection": self.detection,
            "risk_number": self.risk_number,
            "single_point": self.single_point,
            "mitigation": self.mitigation,
            "detectable_by": self.detectable_by,
            "plain": self.plain(),
        }


def criticality(modes: tuple[FailureMode, ...]) -> dict[str, Any]:
    """Rank the failure modes and say which ones a review would stop on."""
    if not modes:
        return {"modes": [], "worst": None, "single_points": []}
    ranked = sorted(modes, key=lambda m: (-m.risk_number, m.severity))
    single_points = [m for m in modes if m.single_point]
    return {
        "modes": [m.to_dict() for m in ranked],
        "worst": ranked[0].to_dict(),
        "single_points": [m.to_dict() for m in single_points],
        "plain": (
            f"{len(modes)} ways this can fail were assessed. The one to fix first is "
            f"{ranked[0].mode.lower()} on {ranked[0].part_id}. "
            + (
                f"{len(single_points)} of them have nothing backing them up."
                if single_points
                else "None of them are the only thing standing between working and not."
            )
        ),
    }


# ---------------------------------------------------------------------------
# Electrical derating
# ---------------------------------------------------------------------------

#: The fraction of a part's rating it should actually be run at, so that
#: heat, tolerance and ageing do not take it past its limit. Figures follow
#: NASA EEE-INST-002 practice for Level 2 hardware.
DERATING: dict[str, tuple[float, str, str]] = {
    "resistor_power": (0.50, "power", "Run a resistor at half its rating; the rest is its own heat."),
    "capacitor_voltage_ceramic": (0.50, "voltage", "Ceramic capacitors lose most of their capacitance near their rated voltage."),
    "capacitor_voltage_electrolytic": (0.80, "voltage", "Electrolytics dry out faster the closer they run to rated voltage."),
    "capacitor_voltage_film": (0.60, "voltage", "Film capacitors are derated for voltage transients."),
    "diode_current": (0.75, "current", "Diode forward current, to keep the junction cool."),
    "diode_voltage": (0.70, "voltage", "Reverse voltage, so a spike does not punch through."),
    "transistor_power": (0.50, "power", "Half the rated dissipation, measured at the actual case temperature."),
    "connector_current": (0.50, "current", "Connector pins heat each other when they are all loaded."),
    "wire_current": (0.80, "current", "Wire in a bundle cannot shed heat like wire in free air."),
    "relay_contact_current": (0.50, "current", "Contacts erode faster the harder they switch."),
    "inductor_current": (0.60, "current", "Above this the core saturates and the inductance collapses."),
    "battery_discharge": (0.80, "energy", "Emptying a cell completely shortens its life sharply."),
}


def derating_check(
    kind: str, applied: Quantity, rated: Quantity
) -> tuple[bool, float, str]:
    """Is this part being run within its derated limit?"""
    entry = DERATING.get(kind)
    if entry is None:
        raise KeyError(f"no derating rule named {kind!r}")
    fraction, _variable, reason = entry
    limit = float(rated.value) * fraction
    used = float(applied.value) / limit if limit > 0 else math.inf
    passes = used <= 1.0
    sentence = (
        f"It runs at {applied.text()} against a derated limit of "
        f"{Q(limit, rated.unit).text()}, which is {fraction * 100:.0f}% of the "
        f"{rated.text()} on the datasheet. {reason}"
    )
    if not passes:
        sentence += (
            f" It is over that limit by {(used - 1) * 100:.0f}%, so it will run hot and "
            "age fast even though the datasheet number has not been exceeded."
        )
    return (passes, used, sentence)


# ---------------------------------------------------------------------------
# Technology readiness
# ---------------------------------------------------------------------------

TRL_LEVELS: dict[int, tuple[str, str]] = {
    1: ("Basic principles observed", "Somebody has noticed the physics works."),
    2: ("Concept formulated", "There is an idea for how to use it, on paper."),
    3: ("Proof of concept", "A piece of it has been shown to work in a lab."),
    4: ("Validated in the lab", "The parts work together on a bench."),
    5: ("Validated in a relevant environment", "It works in conditions like the real ones."),
    6: ("Demonstrated in a relevant environment", "A full prototype has run in near-real conditions."),
    7: ("Demonstrated in the real environment", "A prototype has run where it will actually be used."),
    8: ("Qualified", "The final build has passed its qualification tests."),
    9: ("Proven in service", "It has done the job for real."),
}


def readiness_note(level: int) -> str:
    name, meaning = TRL_LEVELS.get(int(level), TRL_LEVELS[1])
    return f"Readiness level {level}: {name.lower()}. {meaning}"

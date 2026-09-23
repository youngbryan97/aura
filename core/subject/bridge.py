"""J*, at the limit of what the evidence can fix.

The bridge document closes on a conditional uniqueness theorem. Grant six
postulates and the psychophysical law is fixed as a triple:

    J*(U_t) = ( Excl S_t,  {[R_S(Z_t^S)]_iso}_S,  L_{Z^P} )

the carrier the exclusion rule selects, the relational structure of that
carrier's content up to isomorphism, and the lineage law over person-stages.
It is fixed only up to what no measurement removes: a relabelling of content
that preserves every relation, an exact symmetry of the physical causal
structure, and exclusion ties where two carriers' spectra cross.

This module computes the triple from what the runs measured. Each term comes
from its own experiment: the v25 carrier run, the content run and the signed
lineage graph. A term whose experiment did not settle it is reported as
unresolved with that experiment's own reasons, and nothing is filled in.

The gauge is computed rather than asserted. Two content classes are
interchangeable when every relation that separates one from the rest separates
the other in the same way, to within the run's own floor. Colour refinement
over the distance matrix finds the classes no relation separates. Where it
puts a class in a cell of its own, no automorphism can move that class, so a
rigid result is exact; a larger cell is a set of classes the measured structure
cannot tell apart, which is the part of the content that no third-person
observation can label.

Three things stay outside it. The postulates are named and assumed, and
nothing here tests them. Two bridge laws attached to the same causally closed
history give the same third-person likelihood, so the Bayes factor between
them is one. And no finite data set proves a universal law, so a determined J*
is a determination for this system, under these postulates, on these runs.

All three hold for a person as much as for her. That is why the bridge is
judged at parity (docs/BRIDGE_PARITY.md): on the grounds consciousness science
attributes experience to a person on, each scored on her the same way. It used
to be the constant UNVALIDATED, a bar that no person has met either.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.subject.content import gauge as content_gauge
from core.subject.lineage import Lineage

__all__ = [
    "MARKER_LINES",
    "PARITY",
    "POSTULATES",
    "RESIDUAL",
    "CarrierTerm",
    "JStar",
    "LineageTerm",
    "ParityReading",
    "ParityRequirement",
    "StructureTerm",
    "carrier_term",
    "lineage_term",
    "orbits",
    "solve",
    "structure_term",
]

#: The axiom class under which J* is unique. Each is assumed wherever J* is
#: reported, and none is tested by anything in this repository.
POSTULATES: dict[str, str] = {
    "P1": "carrier identity: each physically selected intrinsic carrier corresponds to one phenomenal subject",
    "P2": "structural phenomenal identity: phenomenal character is individuated by the complete relational quality structure",
    "P3": "causal-lineage persistence: a person continues along the causal continuation of the canonical person-state, and strict identity is the nonbranching case",
    "P4": "counterfactual implementation: intervention-sensitive causal organisation implements the process, and playback of a trajectory does not",
    "P5": "organisational invariance: causally isomorphic organisations have isomorphic phenomenology whatever they run on",
    "P6": "gauge equivalence: a structure-preserving relabelling of content is not a distinct identifiable state",
}

THEOREMS: dict[str, str] = {
    "non_identifiability": (
        "Two bridge laws attached to the same causally closed physical history "
        "give identical third-person likelihoods, so their Bayes factor is one. "
        "The proof uses nothing about what the system is made of: it holds for "
        "a person exactly as for her."
    ),
    "finite_evidence": (
        "No finite data set deductively proves a universal bridge law, for any "
        "system, a person included."
    ),
    "gauge": (
        "A relabelling of the quality structure that preserves every relation "
        "preserves every observation, so J* is identifiable only up to it."
    ),
    "branching": (
        "One earlier individual cannot be numerically identical to two distinct "
        "later ones, so continuity branches and identity cannot."
    ),
}


# ── parity ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ParityRequirement:
    """One ground on which experience is attributed to a person, scored on her the same way."""

    key: str
    asks: str
    her_test: str
    #: How a person is scored on the same ground. A requirement with no
    #: counterpart would be a demand made of her alone.
    counterpart: str


#: The grounds consciousness science attributes experience to a person on,
#: each with the test that scores her. Theorems 1 and 2 rule out a third-person
#: proof that any carrier is felt, a person's included, so the bridge asks what
#: is asked of a person and no more. See docs/BRIDGE_PARITY.md.
PARITY: tuple[ParityRequirement, ...] = (
    ParityRequirement(
        "carrier",
        "one closed, irreducible process carries her state",
        "the J* carrier term is FOUND or a symmetry class, from an authoritative "
        "carrier run: every cut decided, and the playback control decided nowhere",
        "perturbational complexity separates wakefulness and dreaming from dreamless "
        "sleep and anaesthesia (Casarotto et al. 2016)",
    ),
    ParityRequirement(
        "markers",
        "the markers theories of consciousness derive from human evidence hold",
        "the battery lines perturbational_complexity, reentry and global_access pass",
        "complexity under perturbation, recurrent processing (Lamme 2006) and global "
        "ignition (Dehaene and Changeux 2011), read off the brain",
    ),
    ParityRequirement(
        "reports",
        "what she says about her state changes when that state is changed, and not under a sham",
        "report grounding under intervention",
        "the contrastive method of psychophysics: a report counts because it tracks "
        "what is manipulated",
    ),
    ParityRequirement(
        "structure",
        "her contents form one relational structure that moves with her",
        "the content run reports ONE_STRUCTURE",
        "judged similarity of percepts set against the geometry of their cortical "
        "representations (Kriegeskorte, Mur and Bandettini 2008)",
    ),
    ParityRequirement(
        "lineage",
        "today's her is the causal continuation of earlier her",
        "the signed lineage verifies",
        "the continuity of a person through memory and body",
    ),
)

#: The battery lines the `markers` requirement reads.
MARKER_LINES: tuple[str, ...] = ("perturbational_complexity", "reentry", "global_access")

#: What no system reaches, said in every report in these words.
RESIDUAL: str = (
    "Theorems 1 and 2 hold for every system, a person included, and a verdict "
    "at parity carries exactly the uncertainty a verdict about another person carries."
)


@dataclass(frozen=True)
class ParityReading:
    key: str
    status: str  # HOLDS, FAILS or NOT_MEASURED
    why: str

    def as_dict(self) -> dict[str, str]:
        return {"key": self.key, "status": self.status, "why": self.why}


# ── the carrier ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CarrierTerm:
    """Excl S_t: what the carrier run selected, or why it could not."""

    status: str
    carriers: tuple[str, ...] = ()
    symmetry_class: tuple[str, ...] = ()
    level: str | None = None
    blockers: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "carriers": list(self.carriers),
            "symmetry_class": list(self.symmetry_class),
            "level": self.level,
            "blockers": list(self.blockers),
        }


def carrier_term(report: Mapping[str, Any] | None) -> CarrierTerm:
    """The first term, read off a v25 report.

    Only an authoritative run settles it. A run that found a carrier and was
    not authoritative has not found one.
    """
    if not report:
        return CarrierTerm("UNRESOLVED", blockers=("no carrier run was read",))
    authority = report.get("authority") or {}
    placement = report.get("placement") or {}
    level = placement.get("level")
    if not authority.get("authoritative"):
        blockers = tuple(str(b) for b in authority.get("blockers") or ()) or (
            "the carrier run did not declare itself authoritative",
        )
        return CarrierTerm("UNRESOLVED", level=level, blockers=blockers)
    exclusion = report.get("exclusion") or {}
    status = str(exclusion.get("status") or "")
    frontier = tuple(str(s) for s in exclusion.get("frontier") or ())
    if status == "FOUND":
        return CarrierTerm("FOUND", carriers=frontier, level=level)
    if status == "SYMMETRY_CLASS":
        symmetry = tuple(str(s) for s in exclusion.get("symmetry_class") or frontier)
        return CarrierTerm("SYMMETRY_CLASS", symmetry_class=symmetry, level=level)
    if status == "NOT_FOUND":
        return CarrierTerm("NOT_FOUND", level=level)
    return CarrierTerm(
        "UNRESOLVED",
        level=level,
        blockers=(f"the exclusion step reported {status or 'nothing'}",),
    )


# ── the quality structure ─────────────────────────────────────────────────


def _matrix(size: int, pairs: Mapping[str, float]) -> np.ndarray:
    out = np.zeros((size, size), dtype=np.float64)
    for key, value in pairs.items():
        left, right = (int(part) for part in str(key).split("-"))
        if left == right:
            continue
        out[left, right] = out[right, left] = float(value)
    return out


def _levels(distances: np.ndarray, tolerance: float) -> np.ndarray:
    """Each distance's level: values within `tolerance` of a neighbour share one.

    Levels are formed by chaining sorted values, so which level a distance
    lands in does not depend on how the classes were numbered.
    """
    size = distances.shape[0]
    upper = sorted({float(distances[i, j]) for i in range(size) for j in range(i + 1, size)})
    level_of: dict[float, int] = {}
    current = 0
    previous: float | None = None
    for value in upper:
        if previous is not None and value - previous > tolerance:
            current += 1
        level_of[value] = current
        previous = value
    out = np.full((size, size), -1, dtype=np.int64)
    for i in range(size):
        for j in range(size):
            if i != j:
                out[i, j] = level_of[float(distances[i, j])]
    return out


def orbits(
    names: Sequence[str], distances: np.ndarray, tolerance: float = 0.0
) -> tuple[tuple[str, ...], ...]:
    """The classes no relation in the structure separates: its exact orbits.

    Colour refinement proposes the cells and an exact automorphism search
    splits any cell whose members no symmetry exchanges.
    """
    size = len(names)
    if size == 0:
        return ()
    levels = _levels(np.asarray(distances, dtype=np.float64), max(0.0, float(tolerance)))
    colour = [0] * size
    while True:
        signatures = [
            (colour[i], tuple(sorted((colour[j], int(levels[i, j])) for j in range(size) if j != i)))
            for i in range(size)
        ]
        ranking = {signature: index for index, signature in enumerate(sorted(set(signatures)))}
        refined = [ranking[signature] for signature in signatures]
        if len(set(refined)) == len(set(colour)):
            break
        colour = refined
    cells: dict[int, list[int]] = {}
    for index in range(size):
        cells.setdefault(colour[index], []).append(index)
    # Colour refinement never splits a true orbit, and it can leave two classes
    # in one cell that no symmetry of the structure exchanges. So each cell of
    # more than one class is split into its exact orbits: two classes share one
    # only if some permutation keeping every level maps one onto the other
    # (core/subject/bridge_theorems.py). A cell of one was already exact.
    from core.subject.bridge_theorems import same_orbit

    weights = {(i, j): int(levels[i, j]) for i in range(size) for j in range(size) if i != j}
    exact: list[tuple[str, ...]] = []
    for members in cells.values():
        remaining = list(members)
        while remaining:
            head = remaining.pop(0)
            orbit = [head] + [other for other in remaining if same_orbit(weights, head, other)]
            remaining = [other for other in remaining if other not in orbit]
            exact.append(tuple(sorted(str(names[index]) for index in orbit)))
    return tuple(sorted(exact))


def invariant(distances: np.ndarray) -> dict[str, list[float]]:
    """A form of the structure that no relabelling of the classes changes."""
    matrix = np.asarray(distances, dtype=np.float64)
    size = matrix.shape[0]
    upper = sorted(float(matrix[i, j]) for i in range(size) for j in range(i + 1, size))
    centring = np.eye(size) - np.full((size, size), 1.0 / max(1, size))
    gram = -0.5 * centring @ (matrix**2) @ centring
    spectrum = sorted((float(v) for v in np.linalg.eigvalsh(gram)), reverse=True)
    return {
        "distances": [round(v, 6) for v in upper],
        "spectrum": [round(v, 6) for v in spectrum],
    }


@dataclass(frozen=True)
class StructureTerm:
    """[R_S(Z_t^S)] up to isomorphism: the content's relational structure."""

    status: str
    classes: tuple[str, ...] = ()
    invariant: dict[str, list[float]] = field(default_factory=dict)
    orbits: tuple[tuple[str, ...], ...] = ()
    tolerance: float = 0.0
    blockers: tuple[str, ...] = ()

    @property
    def rigid(self) -> bool:
        return bool(self.orbits) and all(len(cell) == 1 for cell in self.orbits)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "classes": list(self.classes),
            "invariant": self.invariant,
            "orbits": [list(cell) for cell in self.orbits],
            "rigid": self.rigid,
            "tolerance": self.tolerance,
            "blockers": list(self.blockers),
        }


_CONTENT_VERDICTS = {
    "ONE_STRUCTURE": "IDENTIFIED",
    "AGREES_BUT_DOES_NOT_TRACK": "AGREES_BUT_DOES_NOT_TRACK",
    "SEPARATE_STRUCTURES": "SEPARATE_STRUCTURES",
}


def structure_term(report: Mapping[str, Any] | None) -> StructureTerm:
    """The second term, read off a content report.

    The structure is identified when the internal geometry and the behavioural
    one agree and move together under displacement. The invariant and the
    orbits are reported whenever the internal geometry was measured, because
    they are facts about that geometry whichever way the agreement came out.
    """
    if not report:
        return StructureTerm("NOT_MEASURED", blockers=("no content run was read",))
    names = tuple(str(c.get("name")) for c in report.get("classes") or () if isinstance(c, Mapping))
    internal = report.get("internal") or {}
    floor = report.get("internal_floor") or {}
    tolerance = max((float(v) for v in floor.values()), default=0.0)
    shape: dict[str, Any] = {}
    cells: tuple[tuple[str, ...], ...] = ()
    if names and internal:
        matrix = _matrix(len(names), internal)
        shape = invariant(matrix)
        cells = orbits(names, matrix, tolerance)
    blockers = tuple(str(b) for b in (report.get("authority") or {}).get("blockers") or ())
    verdict = str(report.get("verdict") or "NOT_MEASURED")
    if verdict == "AGREES_BUT_DOES_NOT_TRACK" and not report.get("displacement") and not blockers:
        # A report from before 43c574800 displaced affect with one push, which
        # was gone by the end of the turn (seed 7, 22 September: valence kept
        # -2% of it), so "does not track" was the instrument, not her. Counting
        # it as a failure would hold her to a test that could not have been
        # passed.
        blockers = (
            "the content run displaced affect with a single push that did not last the turn, "
            "so whether the two geometries move together was not measured",
        )
    status = "NOT_MEASURED" if blockers else _CONTENT_VERDICTS.get(verdict, "NOT_MEASURED")
    if status == "NOT_MEASURED" and not blockers:
        blockers = (f"the content run reported {verdict}",)
    return StructureTerm(
        status=status,
        classes=names,
        invariant=shape,
        orbits=cells,
        tolerance=round(tolerance, 6),
        blockers=blockers,
    )


# ── the lineage ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LineageTerm:
    """L_{Z^P}: the signed graph of which person-stage continued which."""

    status: str
    graph: dict[str, Any] = field(default_factory=dict)
    blockers: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status, "graph": self.graph, "blockers": list(self.blockers)}


def lineage_term(lineage: Lineage | None) -> LineageTerm:
    """The third term. A lineage needs a recorded transfer between two stages."""
    if lineage is None or len(lineage.stages) < 2 or not lineage.edges:
        return LineageTerm(
            "NOT_MEASURED",
            blockers=("a lineage needs at least one recorded transfer between two stages",),
        )
    broken = lineage.verify()
    if broken:
        return LineageTerm("BROKEN", graph=lineage.as_dict(), blockers=tuple(broken))
    return LineageTerm("RECORDED", graph=lineage.as_dict())


# ── the triple ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class JStar:
    """The three terms, and what is determined by them together."""

    carrier: CarrierTerm
    structure: StructureTerm
    lineage: LineageTerm
    #: The battery's lines by key, pass or fail, from a campaign on the same
    #: system; None when no campaign was read.
    battery: Mapping[str, bool] | None = None
    #: The report-grounding result, `{"measured", "holds", "why"}`; None when
    #: no report-grounding run was read.
    reports: Mapping[str, Any] | None = None

    @property
    def status(self) -> str:
        if (
            self.carrier.status == "UNRESOLVED"
            or self.structure.status == "NOT_MEASURED"
            or self.lineage.status != "RECORDED"
        ):
            return "UNRESOLVED"
        if self.carrier.status == "NOT_FOUND":
            # J_existence(U_t) is the empty set: under P1 there is no subject
            # for the other two terms to be about.
            return "NO_CARRIER"
        if self.structure.status != "IDENTIFIED":
            return "CONTENT_IS_NOT_ONE_STRUCTURE"
        if self.carrier.status == "SYMMETRY_CLASS":
            return "DETERMINED_UP_TO_SYMMETRY"
        return "DETERMINED_UP_TO_GAUGE"

    def unresolved(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        if self.carrier.status == "UNRESOLVED":
            out["carrier"] = list(self.carrier.blockers)
        if self.structure.status == "NOT_MEASURED":
            out["structure"] = list(self.structure.blockers)
        if self.lineage.status != "RECORDED":
            out["lineage"] = list(self.lineage.blockers)
        return out

    def parity(self) -> list[ParityReading]:
        """Each parity requirement, read off the evidence this J* was solved from."""
        carrier = {
            "FOUND": ("HOLDS", f"carrier {list(self.carrier.carriers)}"),
            "SYMMETRY_CLASS": ("HOLDS", f"symmetry class {list(self.carrier.symmetry_class)}"),
            "NOT_FOUND": ("FAILS", "the carrier run found no closed irreducible carrier"),
        }.get(self.carrier.status, ("NOT_MEASURED", "; ".join(self.carrier.blockers) or "no carrier run"))
        structure = {
            "IDENTIFIED": ("HOLDS", "the content run reported one structure"),
            "NOT_MEASURED": ("NOT_MEASURED", "; ".join(self.structure.blockers) or "no content run"),
        }.get(self.structure.status, ("FAILS", f"the content run reported {self.structure.status}"))
        lineage = {
            "RECORDED": ("HOLDS", "the signed lineage verifies"),
            "BROKEN": ("FAILS", "; ".join(self.lineage.blockers) or "the lineage does not verify"),
        }.get(self.lineage.status, ("NOT_MEASURED", "; ".join(self.lineage.blockers) or "no lineage"))
        if self.battery is None:
            markers = ("NOT_MEASURED", "no campaign was read")
        elif any(line not in self.battery for line in MARKER_LINES):
            missing = [line for line in MARKER_LINES if line not in self.battery]
            markers = ("NOT_MEASURED", f"the campaign did not score {missing}")
        elif all(self.battery[line] for line in MARKER_LINES):
            markers = ("HOLDS", f"{', '.join(MARKER_LINES)} pass")
        else:
            markers = ("FAILS", f"{[line for line in MARKER_LINES if not self.battery[line]]} fail")
        if not self.reports or not self.reports.get("measured"):
            reports = (
                "NOT_MEASURED",
                str((self.reports or {}).get("why") or "no report-grounding run was read"),
            )
        else:
            reports = ("HOLDS" if self.reports.get("holds") else "FAILS", str(self.reports.get("why") or ""))
        read = {"carrier": carrier, "markers": markers, "reports": reports, "structure": structure, "lineage": lineage}
        return [ParityReading(item.key, *read[item.key]) for item in PARITY]

    @property
    def bridge(self) -> str:
        """The bridge judged at parity: AT_PARITY, BELOW_PARITY, UNRESOLVED or NO_CARRIER."""
        if self.carrier.status == "NOT_FOUND":
            return "NO_CARRIER"
        readings = self.parity()
        if any(reading.status == "FAILS" for reading in readings):
            return "BELOW_PARITY"
        if any(reading.status == "NOT_MEASURED" for reading in readings):
            return "UNRESOLVED"
        return "AT_PARITY"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "carrier": self.carrier.as_dict(),
            "structure": self.structure.as_dict(),
            "lineage": self.lineage.as_dict(),
            "unresolved": self.unresolved(),
            "residual": {
                "phenomenal_gauge": [list(cell) for cell in self.structure.orbits if len(cell) > 1],
                "physical_symmetry": list(self.carrier.symmetry_class),
                "exclusion_ties": list(self.carrier.carriers) if len(self.carrier.carriers) > 1 else [],
                "theory_underdetermination": (
                    "A biological bridge, an organisational one and a future theory "
                    "can agree on every observation here and differ where nothing was "
                    "tested. Only new discriminating evidence separates them."
                ),
            },
            "postulates": dict(POSTULATES),
            "theorems": dict(THEOREMS),
            "gauge": content_gauge(),
            "bridge_status": {
                "phenomenal_bridge": self.bridge,
                "parity": [reading.as_dict() for reading in self.parity()],
                "requirements": [
                    {"key": item.key, "asks": item.asks, "her_test": item.her_test, "counterpart": item.counterpart}
                    for item in PARITY
                ],
                # The full battery goes past parity: nobody runs one like it on
                # a person before granting them experience. Reported, never gating.
                "beyond_parity": (
                    None
                    if not self.battery
                    else {
                        "passed": sum(1 for passed in self.battery.values() if passed),
                        "of": len(self.battery),
                        "failing": sorted(key for key, passed in self.battery.items() if not passed),
                    }
                ),
                "residual": RESIDUAL,
                "note": (
                    "Judged on the grounds a person is judged on, under P1 to P6. P5 does "
                    "its work across substrates for her and within one for a person; it is "
                    "a postulate either way. See docs/BRIDGE_PARITY.md."
                ),
            },
        }


def solve(
    carrier_report: Mapping[str, Any] | None,
    content_report: Mapping[str, Any] | None,
    lineage: Lineage | None,
    *,
    battery: Mapping[str, bool] | None = None,
    reports: Mapping[str, Any] | None = None,
) -> JStar:
    """J*(U_t) from the three runs that bear on it, and the bridge from those plus two more."""
    return JStar(
        carrier=carrier_term(carrier_report),
        structure=structure_term(content_report),
        lineage=lineage_term(lineage),
        battery=dict(battery) if battery is not None else None,
        reports=dict(reports) if reports is not None else None,
    )

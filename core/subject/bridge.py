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
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.subject.content import gauge as content_gauge
from core.subject.lineage import Lineage

__all__ = [
    "POSTULATES",
    "CarrierTerm",
    "JStar",
    "LineageTerm",
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
        "give identical third-person likelihoods, so their Bayes factor is one."
    ),
    "finite_evidence": "No finite data set deductively proves a universal bridge law.",
    "gauge": (
        "A relabelling of the quality structure that preserves every relation "
        "preserves every observation, so J* is identifiable only up to it."
    ),
    "branching": (
        "One earlier individual cannot be numerically identical to two distinct "
        "later ones, so continuity branches and identity cannot."
    ),
}


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
                "phenomenal_bridge": "UNVALIDATED",
                "note": (
                    "J* is fixed within the axiom class P1 to P6. Whether those "
                    "postulates are laws of nature is not something a measurement "
                    "of this system can decide."
                ),
            },
        }


def solve(
    carrier_report: Mapping[str, Any] | None,
    content_report: Mapping[str, Any] | None,
    lineage: Lineage | None,
) -> JStar:
    """J*(U_t) from the three runs that bear on it."""
    return JStar(
        carrier=carrier_term(carrier_report),
        structure=structure_term(content_report),
        lineage=lineage_term(lineage),
    )

"""core/connectome/types.py — the vocabulary a connectome needs.

Connectomics has a settled set of nouns and this module borrows them exactly,
because borrowing them loosely is how a biological name stops being a claim.

* A **unit** is one cell. In Aura a cell is a function or method: the smallest
  thing that has an identity, receives input, and either fires or does not.
* A **compartment** is a part of a cell. H01's SegCLR classifier separates six
  of them — soma, dendrite, axon, axon initial segment, glia, cilium — and four
  of those have exact analogues here. The axon initial segment matters most:
  it is where the decision to fire is taken, and a gate whose precondition its
  own failure keeps true is an initial segment that never reaches threshold.
* A **contact** is one presynaptic site touching one postsynaptic cell. Two
  cells can be joined by many contacts, and H01 found that how many is not
  incidental: 96.5% of connected pairs touch once, and the vanishing fraction
  that touch four or more times behave like a different kind of connection.
* A **neuropil** is the region a cell's arbour sits in. Modules here, packages
  above them, in the way FlyEM groups neurons into named compartments.

The class of a cell — excitatory, inhibitory, modulatory, glial — is measured
from what its exits do, not from its name. A function whose exits mostly refuse
is inhibitory however it is spelled.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "CellClass",
    "Compartment",
    "EdgeKind",
    "ContactSite",
    "Neuropil",
    "Unit",
    "Connection",
    "ConnectomeSnapshot",
    "H01_REFERENCE",
    "FLY_MALE_CNS_REFERENCE",
    "CORTICAL_EI_RATIO",
    "stable_id",
]


def stable_id(*parts: Any, length: int = 16) -> str:
    """A content-addressed identity that survives a rerun."""
    raw = "\x1f".join(str(p) for p in parts).encode("utf-8", "replace")
    return hashlib.blake2b(raw, digest_size=length // 2).hexdigest()


class CellClass(StrEnum):
    """What a cell does to the cells downstream of it.

    Cortex runs at roughly four excitatory cells to one inhibitory cell. The
    ratio is not a curiosity — inhibition is what keeps a recurrent excitatory
    network from saturating, and a network with too much of it cannot get a
    signal across at all.
    """

    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"
    MODULATORY = "modulatory"
    GLIAL = "glial"


class EdgeKind(StrEnum):
    """Which half of a call an edge represents.

    A call carries information twice: the caller drives the callee, and when
    the return value is read the callee drives the caller back. Keeping the two
    apart is what makes reciprocity a measurement instead of an artefact — a
    graph that pairs every drive edge with a return edge is reciprocal by
    construction and says nothing about the circuit.
    """

    DRIVE = "drive"
    RETURN = "return"


class Compartment(StrEnum):
    """The six subcompartments H01 classifies, less the two with no analogue."""

    SOMA = "soma"
    DENDRITE = "dendrite"
    AXON = "axon"
    AXON_INITIAL_SEGMENT = "axon_initial_segment"
    GLIA = "glia"


#: Cortical excitatory:inhibitory cell ratio, the standard 80/20 split.
#: Source: Potjans & Diesmann 2014 population sizes, which sum to 66,070
#: excitatory and 15,326 inhibitory neurons across the four layers.
CORTICAL_EI_RATIO: float = 66070.0 / 15326.0


@dataclass(frozen=True)
class _Reference:
    """A published measurement Aura's own numbers get compared against."""

    name: str
    citation: str
    values: Mapping[str, float]

    def get(self, key: str) -> float:
        return float(self.values[key])


#: Shapson-Coe et al., "A petavoxel fragment of human cerebral cortex
#: reconstructed at nanoscale resolution", Science 384, adk4858 (2024).
H01_REFERENCE = _Reference(
    name="H01",
    citation="Shapson-Coe et al., Science 384:adk4858 (2024)",
    values={
        # Tissue and reconstruction.
        "cubic_mm": 1.0,
        "petabytes": 1.4,
        "cells": 57_000.0,
        "neurons": 16_000.0,
        "glia": 32_000.0,
        "vessel_cells": 8_000.0,
        "synapses": 150_000_000.0,
        "sections": 5_000.0,
        "section_nm": 30.0,
        "imaging_days": 326.0,
        "vessel_mm": 230.0,
        # The connection-multiplicity law. These three are the numbers this
        # repository measures itself against.
        "single_contact_fraction": 0.965,
        "four_or_more_contact_fraction": 0.00092,
        "max_observed_contacts": 50.0,
        # Layer 6 triangular neurons falling into two mirror-symmetric
        # orientation classes rather than being distributed at random.
        "layer6_mirror_fraction": 0.77,
    },
)

#: Janelia FlyEM male CNS connectome v1.0 and its companion paper.
FLY_MALE_CNS_REFERENCE = _Reference(
    name="male-CNS",
    citation="Janelia FlyEM male CNS connectome v1.0 (2026); Cell S0092-8674(26)00942-6",
    values={
        "neurons": 166_000.0,
        "synapses": 125_000_000.0,
        "sex_specific_types": 262.0,
        "dimorphic_types": 114.0,
        # Sex-specific and dimorphic cells as a fraction of the central brain.
        "dimorphic_fraction_of_central_brain": 0.048,
    },
)


@dataclass(frozen=True)
class ContactSite:
    """One presynaptic site. The unit of connection strength.

    ``locus`` is where the contact is made, so two contacts from the same pair
    of cells are distinguishable and countable. ``compartment`` is which part
    of the postsynaptic cell is touched, which is what separates a contact onto
    a gate from a contact onto a body.
    """

    pre: str
    post: str
    locus: str
    compartment: Compartment = Compartment.DENDRITE
    sign: int = 1
    weight: float = 1.0
    kind: EdgeKind = EdgeKind.DRIVE

    def key(self) -> tuple[str, str, str]:
        return (self.pre, self.post, self.locus)


@dataclass(frozen=True)
class Neuropil:
    """A named region an arbour sits in."""

    name: str
    parent: str | None = None

    def path(self) -> str:
        return f"{self.parent}/{self.name}" if self.parent else self.name


@dataclass
class Unit:
    """One cell.

    ``exits`` records what the cell's return paths do, because that is the
    measurement the class assignment rests on: ``suppressive`` exits refuse,
    return nothing, or raise; ``productive`` exits hand a constructed value
    downstream.
    """

    uid: str
    name: str
    neuropil: str
    region: str
    cell_class: CellClass = CellClass.EXCITATORY
    line: int = 0
    size: int = 0
    guards: int = 0
    exits_suppressive: int = 0
    exits_productive: int = 0
    is_async: bool = False
    layer: str = ""
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def exit_count(self) -> int:
        return self.exits_suppressive + self.exits_productive

    @property
    def suppression(self) -> float:
        """Fraction of this cell's exits that refuse rather than produce."""
        total = self.exit_count
        return (self.exits_suppressive / total) if total else 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "name": self.name,
            "neuropil": self.neuropil,
            "region": self.region,
            "cell_class": str(self.cell_class),
            "line": self.line,
            "size": self.size,
            "guards": self.guards,
            "exits_suppressive": self.exits_suppressive,
            "exits_productive": self.exits_productive,
            "is_async": self.is_async,
            "layer": self.layer,
        }


@dataclass(frozen=True)
class Connection:
    """An aggregated pair: every contact from one cell onto another."""

    pre: str
    post: str
    contacts: int
    sign: int
    compartments: tuple[Compartment, ...] = ()
    kind: EdgeKind = EdgeKind.DRIVE

    @property
    def is_strong(self) -> bool:
        """Four or more contacts, H01's threshold for the rare heavy pairs."""
        return self.contacts >= 4


@dataclass
class ConnectomeSnapshot:
    """A whole reconstruction, versioned and deterministically serialisable."""

    version: int
    units: dict[str, Unit]
    connections: dict[tuple[str, str, str], Connection]
    neuropils: dict[str, Neuropil]
    built_at: float = 0.0
    source: str = ""
    attrs: dict[str, Any] = field(default_factory=dict)

    def cell_count(self) -> int:
        return len(self.units)

    def contact_count(self) -> int:
        return sum(c.contacts for c in self.connections.values())

    def class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {str(c): 0 for c in CellClass}
        for unit in self.units.values():
            counts[str(unit.cell_class)] += 1
        return counts

    def ei_ratio(self) -> float:
        counts = self.class_counts()
        inhibitory = counts[str(CellClass.INHIBITORY)]
        if inhibitory <= 0:
            return float("inf")
        return counts[str(CellClass.EXCITATORY)] / inhibitory

    def edges(self, kind: EdgeKind | None = EdgeKind.DRIVE) -> list[Connection]:
        """Every connection, or only those of one kind."""
        if kind is None:
            return list(self.connections.values())
        return [c for c in self.connections.values() if c.kind is kind]

    def out_edges(self, uid: str, kind: EdgeKind | None = EdgeKind.DRIVE) -> list[Connection]:
        return [c for c in self.connections.values() if c.pre == uid and (kind is None or c.kind is kind)]

    def adjacency(
        self,
        uids: Sequence[str] | None = None,
        kind: EdgeKind | None = EdgeKind.DRIVE,
    ) -> dict[str, dict[str, int]]:
        """Contact-weighted adjacency, restricted to ``uids`` when given."""
        keep = set(uids) if uids is not None else None
        out: dict[str, dict[str, int]] = {}
        for conn in self.connections.values():
            if kind is not None and conn.kind is not kind:
                continue
            if keep is not None and (conn.pre not in keep or conn.post not in keep):
                continue
            out.setdefault(conn.pre, {})[conn.post] = conn.contacts
        return out

    def digest(self) -> str:
        payload = {
            "units": sorted(self.units),
            "edges": sorted(
                f"{conn.pre}>{conn.post}:{conn.kind}:{conn.contacts}"
                for conn in self.connections.values()
            ),
        }
        return hashlib.blake2b(
            json.dumps(payload, sort_keys=True).encode("utf-8"), digest_size=16
        ).hexdigest()

    def summary(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source": self.source,
            "cells": self.cell_count(),
            "connections": len(self.connections),
            "drive_connections": len(self.edges(EdgeKind.DRIVE)),
            "return_connections": len(self.edges(EdgeKind.RETURN)),
            "contacts": self.contact_count(),
            "neuropils": len(self.neuropils),
            "class_counts": self.class_counts(),
            "ei_ratio": round(self.ei_ratio(), 4),
            "digest": self.digest(),
        }


def iter_strong(connections: Iterable[Connection]) -> Iterable[Connection]:
    """The rare heavy pairs H01 singles out."""
    return (c for c in connections if c.is_strong)

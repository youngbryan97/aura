"""core/connectome/synaptology.py — how strong a connection is, and what that means.

H01 measured something in human cortex that nobody could measure before, because
nobody had a cubic millimetre reconstructed at synapse resolution: how many
synapses a connected pair of cells actually shares. The answer is that 96.5% of
connected pairs share exactly one, and that the vanishing fraction sharing four
or more — 0.092%, with rare pairs reaching about fifty — are not the tail of the
same process. They look like a different kind of connection, strong enough that
one cell firing is likely to make the other fire.

That gives a measurement Aura can be held to. A call site is a synapse and the
count is exact, so the same distribution can be computed on her and compared
against a human cortical measurement rather than against an intuition.

The comparison found something worth acting on. 81.8% of Aura's connected pairs
touch once and 4.2% touch four or more times, against cortex at 96.5% and
0.092%. Her heavy pairs are forty-five times more common than cortex's, and her
heaviest carries 113 contacts where the human maximum was about fifty. H01's
reading of a heavy pair is that it is special. At 4.2% of everything, hers
cannot be.

The compartment split is the second measurement, and it is the one with a defect
class attached. A contact that lands on a cell's initial segment is a contact
onto the decision to fire, not onto the body of the computation. Aura's
recurring failure is a guard whose own failure keeps its precondition true, and
that failure has a shape here: a cell whose initial segment carries inhibitory
contacts and whose body carries almost none.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .types import (
    CORTICAL_EI_RATIO,
    H01_REFERENCE,
    CellClass,
    Compartment,
    ConnectomeSnapshot,
    EdgeKind,
)

logger = logging.getLogger("Aura.Connectome.Synaptology")

__all__ = [
    "MultiplicityLaw",
    "measure_multiplicity",
    "StrongConnection",
    "strong_connections",
    "CompartmentProfile",
    "compartment_profile",
    "gate_dominated_cells",
    "ei_report",
]


@dataclass(frozen=True)
class MultiplicityLaw:
    """The contact-count distribution, next to the human cortical one."""

    pairs: int
    contacts: int
    histogram: dict[int, int]
    single_fraction: float
    four_or_more_fraction: float
    maximum: int
    mean: float
    reference_single: float
    reference_four_or_more: float
    reference_max: float
    reference_name: str

    @property
    def heavy_excess(self) -> float:
        """How many times more common heavy pairs are here than in cortex."""
        if self.reference_four_or_more <= 0:
            return 0.0
        return self.four_or_more_fraction / self.reference_four_or_more

    def as_json(self) -> dict[str, Any]:
        return {
            "pairs": self.pairs,
            "contacts": self.contacts,
            "single_fraction": round(self.single_fraction, 5),
            "four_or_more_fraction": round(self.four_or_more_fraction, 6),
            "maximum": self.maximum,
            "mean": round(self.mean, 4),
            "reference": self.reference_name,
            "reference_single_fraction": self.reference_single,
            "reference_four_or_more_fraction": self.reference_four_or_more,
            "reference_maximum": self.reference_max,
            "heavy_excess_over_reference": round(self.heavy_excess, 2),
            "histogram": {str(k): v for k, v in sorted(self.histogram.items())[:32]},
        }


def measure_multiplicity(
    snapshot: ConnectomeSnapshot,
    *,
    kind: EdgeKind | None = EdgeKind.DRIVE,
) -> MultiplicityLaw:
    """Count how many contacts join each connected pair."""
    counts = Counter(
        conn.contacts
        for conn in snapshot.connections.values()
        if kind is None or conn.kind is kind
    )
    pairs = sum(counts.values())
    contacts = sum(k * v for k, v in counts.items())
    if pairs == 0:
        return MultiplicityLaw(
            pairs=0,
            contacts=0,
            histogram={},
            single_fraction=0.0,
            four_or_more_fraction=0.0,
            maximum=0,
            mean=0.0,
            reference_single=H01_REFERENCE.get("single_contact_fraction"),
            reference_four_or_more=H01_REFERENCE.get("four_or_more_contact_fraction"),
            reference_max=H01_REFERENCE.get("max_observed_contacts"),
            reference_name=H01_REFERENCE.name,
        )
    heavy = sum(v for k, v in counts.items() if k >= 4)
    return MultiplicityLaw(
        pairs=pairs,
        contacts=contacts,
        histogram=dict(counts),
        single_fraction=counts.get(1, 0) / pairs,
        four_or_more_fraction=heavy / pairs,
        maximum=max(counts),
        mean=contacts / pairs,
        reference_single=H01_REFERENCE.get("single_contact_fraction"),
        reference_four_or_more=H01_REFERENCE.get("four_or_more_contact_fraction"),
        reference_max=H01_REFERENCE.get("max_observed_contacts"),
        reference_name=H01_REFERENCE.name,
    )


@dataclass(frozen=True)
class StrongConnection:
    """A pair carrying enough contacts that H01 would call it powerful."""

    pre: str
    post: str
    pre_name: str
    post_name: str
    contacts: int
    same_module: bool
    same_region: bool

    def as_json(self) -> dict[str, Any]:
        return {
            "pre": self.pre_name,
            "post": self.post_name,
            "contacts": self.contacts,
            "same_module": self.same_module,
            "same_region": self.same_region,
        }


def strong_connections(
    snapshot: ConnectomeSnapshot,
    *,
    threshold: int = 4,
    limit: int = 200,
    kind: EdgeKind | None = EdgeKind.DRIVE,
) -> list[StrongConnection]:
    """The heavy pairs, heaviest first.

    ``same_module`` matters for reading them. A heavy pair inside one module is
    ordinary cohesion. A heavy pair that crosses a package boundary is an
    interface that has been used as though it were an internal one, and it is
    the kind of coupling that makes two packages one.
    """
    out: list[StrongConnection] = []
    for conn in snapshot.connections.values():
        if kind is not None and conn.kind is not kind:
            continue
        if conn.contacts < threshold:
            continue
        pre = snapshot.units.get(conn.pre)
        post = snapshot.units.get(conn.post)
        if pre is None or post is None:
            continue
        out.append(
            StrongConnection(
                pre=conn.pre,
                post=conn.post,
                pre_name=pre.name,
                post_name=post.name,
                contacts=conn.contacts,
                same_module=pre.neuropil == post.neuropil,
                same_region=pre.region == post.region,
            )
        )
    out.sort(key=lambda c: (-c.contacts, c.pre_name, c.post_name))
    return out[:limit]


@dataclass(frozen=True)
class CompartmentProfile:
    """Where contacts land, split by the part of the cell that receives them."""

    total: int
    by_compartment: dict[str, int]
    inhibitory_on_initial_segment: int
    inhibitory_on_dendrite: int
    excitatory_on_initial_segment: int

    def as_json(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "by_compartment": self.by_compartment,
            "inhibitory_on_initial_segment": self.inhibitory_on_initial_segment,
            "inhibitory_on_dendrite": self.inhibitory_on_dendrite,
            "excitatory_on_initial_segment": self.excitatory_on_initial_segment,
        }


def compartment_profile(snapshot: ConnectomeSnapshot) -> CompartmentProfile:
    """Split return contacts by compartment and by the sign they carry.

    Cortex does this deliberately. Chandelier cells put every one of their
    synapses on the axon initial segments of pyramidal cells and nowhere else,
    which makes them able to veto a cell rather than argue with it. A contact
    on the initial segment is a veto; a contact on a dendrite is a vote.
    """
    by_compartment: Counter[str] = Counter()
    inhibitory_ais = 0
    inhibitory_dend = 0
    excitatory_ais = 0
    for conn in snapshot.connections.values():
        if conn.kind is not EdgeKind.RETURN:
            continue
        for compartment in conn.compartments or (Compartment.DENDRITE,):
            by_compartment[str(compartment)] += conn.contacts
            if compartment is Compartment.AXON_INITIAL_SEGMENT:
                if conn.sign < 0:
                    inhibitory_ais += conn.contacts
                else:
                    excitatory_ais += conn.contacts
            elif compartment is Compartment.DENDRITE and conn.sign < 0:
                inhibitory_dend += conn.contacts
    return CompartmentProfile(
        total=sum(by_compartment.values()),
        by_compartment=dict(by_compartment),
        inhibitory_on_initial_segment=inhibitory_ais,
        inhibitory_on_dendrite=inhibitory_dend,
        excitatory_on_initial_segment=excitatory_ais,
    )


def gate_dominated_cells(
    snapshot: ConnectomeSnapshot,
    *,
    min_contacts: int = 4,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Cells whose inputs land almost entirely on the decision to fire.

    A cell like this cannot be argued with by its inputs, only vetoed by them.
    When every one of those vetoes is a call that can itself fail, the cell has
    the structure behind Aura's most repeated failure: a gate whose precondition
    stays true precisely because the gate could not evaluate it.
    """
    per_cell: dict[str, dict[str, int]] = {}
    for conn in snapshot.connections.values():
        if conn.kind is not EdgeKind.RETURN:
            continue
        entry = per_cell.setdefault(conn.post, {"ais": 0, "body": 0, "ais_inhibitory": 0})
        if Compartment.AXON_INITIAL_SEGMENT in conn.compartments:
            entry["ais"] += conn.contacts
            if conn.sign < 0:
                entry["ais_inhibitory"] += conn.contacts
        else:
            entry["body"] += conn.contacts
    rows: list[dict[str, Any]] = []
    for uid, entry in per_cell.items():
        total = entry["ais"] + entry["body"]
        if total < min_contacts or entry["ais"] == 0:
            continue
        share = entry["ais"] / total
        if share < 0.8:
            continue
        unit = snapshot.units.get(uid)
        rows.append(
            {
                "cell": unit.name if unit else uid,
                "initial_segment_contacts": entry["ais"],
                "body_contacts": entry["body"],
                "initial_segment_share": round(share, 4),
                "inhibitory_on_initial_segment": entry["ais_inhibitory"],
                "guards": unit.guards if unit else 0,
            }
        )
    rows.sort(key=lambda r: (-r["initial_segment_contacts"], r["cell"]))
    return rows[:limit]


def ei_report(snapshot: ConnectomeSnapshot) -> dict[str, Any]:
    """The excitation-to-inhibition ratio, against the cortical 80/20.

    Cortex runs about four excitatory cells to one inhibitory. Too little
    inhibition and a recurrent network saturates; too much and nothing crosses
    it. The ratio is reported per region as well as overall, because an average
    of four can hide a package that is nearly all gate.
    """
    counts = snapshot.class_counts()
    per_region: dict[str, Counter[str]] = {}
    for unit in snapshot.units.values():
        per_region.setdefault(unit.region, Counter())[str(unit.cell_class)] += 1
    regions = []
    for region, tally in sorted(per_region.items()):
        excitatory = tally.get(str(CellClass.EXCITATORY), 0)
        inhibitory = tally.get(str(CellClass.INHIBITORY), 0)
        if excitatory + inhibitory < 20:
            continue
        regions.append(
            {
                "region": region,
                "excitatory": excitatory,
                "inhibitory": inhibitory,
                "ratio": round(excitatory / inhibitory, 3) if inhibitory else None,
                "cells": sum(tally.values()),
            }
        )
    regions.sort(key=lambda r: (r["ratio"] if r["ratio"] is not None else 1e9))
    return {
        "class_counts": counts,
        "ei_ratio": round(snapshot.ei_ratio(), 4),
        "cortical_ei_ratio": round(CORTICAL_EI_RATIO, 4),
        "inhibition_excess": round(CORTICAL_EI_RATIO / snapshot.ei_ratio(), 3)
        if snapshot.ei_ratio() > 0
        else 0.0,
        "most_inhibited_regions": regions[:12],
        "least_inhibited_regions": regions[-6:],
    }

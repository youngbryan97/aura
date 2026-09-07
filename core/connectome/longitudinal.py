"""core/connectome/longitudinal.py — the same individual over time, and how to line two up.

Two pieces of the Human Connectome Project's pipeline transfer here and neither
is about imaging.

**A subject is measured against their own template, not against nothing.** The
HCP pipelines added longitudinal processing for exactly this: a within-subject
template built from every timepoint, so a change at one timepoint is a change
relative to that person rather than relative to a population they may not
resemble. Aura's timepoints are commits. A template built across them separates
the part of her that holds still from the part that is moving, and a drift
measured against that template means something a single comparison cannot.

**Two individuals are aligned by what their features do, not by where they
sit.** MSMAll registers cortical surfaces by areal features rather than by
folding, because two brains with the same anatomy in different places are still
the same brains. The identifier-matching used elsewhere in this package has the
opposite failure: rename a function and it becomes a new cell that lost every
edge, and the comparison reports a rewiring that never happened.
:func:`align_by_connectivity` matches cells whose connectivity profiles agree,
so a rename survives it and a genuine rewiring does not.

The third piece is the Allen mouse connectivity atlas's level of description.
Its projection matrix is region to region rather than cell to cell, because
that is the level at which an architecture is decided. :func:`projection_matrix`
produces the same thing for Aura: package to package, normalised by the size of
the source, so a large package does not look influential merely for being large.
"""

from __future__ import annotations

import logging
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .types import ConnectomeSnapshot, EdgeKind

logger = logging.getLogger("Aura.Connectome.Longitudinal")

__all__ = [
    "Template",
    "build_template",
    "drift_against",
    "align_by_connectivity",
    "projection_matrix",
]


@dataclass
class Template:
    """What holds still across timepoints of one individual."""

    timepoints: int
    cell_presence: dict[str, int]
    edge_presence: dict[tuple[str, str], int]
    edge_contacts: dict[tuple[str, str], float]

    def stable_cells(self, threshold: float = 1.0) -> set[str]:
        """Cells present in at least this share of the timepoints."""
        needed = threshold * self.timepoints
        return {uid for uid, count in self.cell_presence.items() if count >= needed}

    def stable_edges(self, threshold: float = 1.0) -> set[tuple[str, str]]:
        needed = threshold * self.timepoints
        return {pair for pair, count in self.edge_presence.items() if count >= needed}

    def summary(self) -> dict[str, Any]:
        core_cells = self.stable_cells()
        core_edges = self.stable_edges()
        return {
            "timepoints": self.timepoints,
            "cells_seen": len(self.cell_presence),
            "cells_in_every_timepoint": len(core_cells),
            "cell_core_share": round(len(core_cells) / max(1, len(self.cell_presence)), 4),
            "edges_seen": len(self.edge_presence),
            "edges_in_every_timepoint": len(core_edges),
            "edge_core_share": round(len(core_edges) / max(1, len(self.edge_presence)), 4),
        }


def build_template(snapshots: Sequence[ConnectomeSnapshot]) -> Template:
    """Count how often each cell and each connection appears across timepoints.

    One timepoint is not a template and the count says so: with a single
    snapshot everything is present in every timepoint and the core share is one,
    which is true and useless. The number of timepoints is carried so a reader
    can tell the difference.
    """
    cell_presence: dict[str, int] = {}
    edge_presence: dict[tuple[str, str], int] = {}
    edge_contacts: dict[tuple[str, str], float] = {}
    for snapshot in snapshots:
        for uid in snapshot.units:
            cell_presence[uid] = cell_presence.get(uid, 0) + 1
        for connection in snapshot.connections.values():
            if connection.kind is not EdgeKind.DRIVE:
                continue
            pair = (connection.pre, connection.post)
            edge_presence[pair] = edge_presence.get(pair, 0) + 1
            edge_contacts[pair] = edge_contacts.get(pair, 0.0) + connection.contacts
    for pair, total in edge_contacts.items():
        edge_contacts[pair] = total / edge_presence[pair]
    return Template(
        timepoints=len(snapshots),
        cell_presence=cell_presence,
        edge_presence=edge_presence,
        edge_contacts=edge_contacts,
    )


def drift_against(
    template: Template,
    snapshot: ConnectomeSnapshot,
    *,
    core_threshold: float = 1.0,
) -> dict[str, Any]:
    """Measure one timepoint against the individual's own template.

    The number worth reading is what happened to the core: cells and edges that
    were present at every previous timepoint and are not present now. A change
    to something that was already coming and going is development; a change to
    something that had never changed is a different event.
    """
    core_cells = template.stable_cells(core_threshold)
    core_edges = template.stable_edges(core_threshold)
    present_cells = set(snapshot.units)
    present_edges = {
        (connection.pre, connection.post)
        for connection in snapshot.connections.values()
        if connection.kind is EdgeKind.DRIVE
    }
    lost_cells = core_cells - present_cells
    lost_edges = core_edges - present_edges
    new_cells = present_cells - set(template.cell_presence)
    new_edges = present_edges - set(template.edge_presence)

    contact_shift: list[float] = []
    for connection in snapshot.connections.values():
        if connection.kind is not EdgeKind.DRIVE:
            continue
        expected = template.edge_contacts.get((connection.pre, connection.post))
        if expected:
            contact_shift.append(abs(connection.contacts - expected) / expected)

    return {
        "timepoints_in_template": template.timepoints,
        "core_cells": len(core_cells),
        "core_cells_lost": len(lost_cells),
        "core_edges": len(core_edges),
        "core_edges_lost": len(lost_edges),
        "cells_new_to_the_individual": len(new_cells),
        "edges_new_to_the_individual": len(new_edges),
        "mean_contact_shift": round(statistics.fmean(contact_shift), 5)
        if contact_shift
        else 0.0,
        "core_loss_share": round(len(lost_cells) / max(1, len(core_cells)), 5),
        "verdict": (
            "the core is intact; every change is to something that was already moving"
            if not lost_cells and not lost_edges
            else f"{len(lost_cells)} cells and {len(lost_edges)} edges that had never "
            "changed are gone"
        ),
    }


def _profile(snapshot: ConnectomeSnapshot) -> dict[str, tuple[frozenset[str], frozenset[str]]]:
    """Each cell's neighbours, by the region they sit in rather than by identity.

    Regions survive a rename where a cell identifier does not, which is the
    whole reason to describe a cell by what its neighbourhood looks like instead
    of by which exact cells are in it.
    """
    profiles: dict[str, tuple[set[str], set[str]]] = {
        uid: (set(), set()) for uid in snapshot.units
    }
    for connection in snapshot.connections.values():
        if connection.kind is not EdgeKind.DRIVE:
            continue
        pre = snapshot.units.get(connection.pre)
        post = snapshot.units.get(connection.post)
        if pre is None or post is None:
            continue
        profiles[connection.pre][1].add(f"{post.region}:{post.cell_class}")
        profiles[connection.post][0].add(f"{pre.region}:{pre.cell_class}")
    return {uid: (frozenset(a), frozenset(b)) for uid, (a, b) in profiles.items()}


def align_by_connectivity(
    left: ConnectomeSnapshot,
    right: ConnectomeSnapshot,
    *,
    minimum_overlap: float = 0.6,
    limit: int = 5_000,
) -> dict[str, Any]:
    """Match cells across two individuals by what their connectivity looks like.

    Cells with the same identifier are matched first and cheaply. What is left
    on each side is matched by Jaccard overlap of the incoming and outgoing
    neighbourhood descriptions, and only above a floor — a weak best match is
    worse than no match, because it turns a rename into a rewiring and a
    rewiring into a rename.

    A match inside one module and a match across two are not the same evidence.
    The neighbourhood description is coarse enough that two unrelated functions
    in different packages can share one, so a cross-module match has to clear a
    higher bar and is labelled either way.
    """
    shared = set(left.units) & set(right.units)
    only_left = sorted(set(left.units) - shared)[:limit]
    only_right = sorted(set(right.units) - shared)[:limit]
    if not only_left or not only_right:
        return {
            "matched_by_identifier": len(shared),
            "matched_by_connectivity": 0,
            "unmatched_left": len(only_left),
            "unmatched_right": len(only_right),
            "pairs": [],
        }

    left_profiles = _profile(left)
    right_profiles = _profile(right)
    by_module: dict[str, list[str]] = {}
    for uid in only_right:
        by_module.setdefault(right.units[uid].neuropil, []).append(uid)

    def _overlap(a: frozenset[str], b: frozenset[str]) -> float:
        if not a and not b:
            return 0.0
        return len(a & b) / len(a | b)

    taken: set[str] = set()
    pairs: list[dict[str, Any]] = []
    cross_module_floor = min(0.95, minimum_overlap + 0.2)
    for uid in only_left:
        unit = left.units[uid]
        candidates = by_module.get(unit.neuropil) or only_right
        incoming, outgoing = left_profiles.get(uid, (frozenset(), frozenset()))
        best_uid = ""
        best_score = 0.0
        for other in candidates:
            if other in taken:
                continue
            other_in, other_out = right_profiles.get(other, (frozenset(), frozenset()))
            score = 0.5 * _overlap(incoming, other_in) + 0.5 * _overlap(outgoing, other_out)
            if score > best_score:
                best_score = score
                best_uid = other
        if not best_uid:
            continue
        same_module = right.units[best_uid].neuropil == unit.neuropil
        floor = minimum_overlap if same_module else cross_module_floor
        if best_score >= floor:
            taken.add(best_uid)
            pairs.append(
                {
                    "left": left.units[uid].name,
                    "right": right.units[best_uid].name,
                    "overlap": round(best_score, 4),
                    "same_module": same_module,
                }
            )
    pairs.sort(key=lambda row: -row["overlap"])
    return {
        "matched_by_identifier": len(shared),
        "matched_by_connectivity": len(pairs),
        "unmatched_left": len(only_left) - len(pairs),
        "unmatched_right": len(only_right) - len(pairs),
        "minimum_overlap": minimum_overlap,
        "cross_module_overlap_floor": cross_module_floor,
        "matched_across_modules": sum(1 for row in pairs if not row["same_module"]),
        "pairs": pairs[:40],
    }


def projection_matrix(
    snapshot: ConnectomeSnapshot,
    *,
    normalise: bool = True,
    limit: int = 40,
) -> dict[str, Any]:
    """Region to region, the level at which an architecture is decided.

    The Allen mouse atlas reports projection strength between structures rather
    than between cells, normalised so that a large source does not look
    influential for being large. The same normalisation applies here: a package
    with four thousand cells reaching another is a different claim from a
    package with forty doing it.
    """
    sizes: dict[str, int] = {}
    for unit in snapshot.units.values():
        sizes[unit.region] = sizes.get(unit.region, 0) + 1
    flows: dict[tuple[str, str], int] = {}
    for connection in snapshot.connections.values():
        if connection.kind is not EdgeKind.DRIVE:
            continue
        pre = snapshot.units.get(connection.pre)
        post = snapshot.units.get(connection.post)
        if pre is None or post is None or pre.region == post.region:
            continue
        key = (pre.region, post.region)
        flows[key] = flows.get(key, 0) + connection.contacts

    rows: list[dict[str, Any]] = []
    for (source, target), contacts in flows.items():
        weight = contacts / sizes[source] if normalise and sizes.get(source) else float(contacts)
        rows.append(
            {
                "source": source,
                "target": target,
                "contacts": contacts,
                "per_source_cell": round(contacts / max(1, sizes.get(source, 1)), 5),
                "weight": round(weight, 5),
            }
        )
    rows.sort(key=lambda row: -row["weight"])
    outward: dict[str, float] = {}
    inward: dict[str, float] = {}
    for row in rows:
        outward[row["source"]] = outward.get(row["source"], 0.0) + row["weight"]
        inward[row["target"]] = inward.get(row["target"], 0.0) + row["weight"]
    return {
        "regions": len(sizes),
        "region_pairs": len(rows),
        "normalised": normalise,
        "strongest": rows[:limit],
        "most_outward": sorted(outward.items(), key=lambda item: -item[1])[:12],
        "most_inward": sorted(inward.items(), key=lambda item: -item[1])[:12],
    }

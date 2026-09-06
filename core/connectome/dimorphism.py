"""core/connectome/dimorphism.py — two individuals, and where they differ.

The male CNS release is the first time two nearly identical nervous systems
could be compared cell by cell. What came out is a shape worth borrowing. The
sex-specific and dimorphic cell types are 4.8% of the central brain, and they
are not spread evenly: sensory and motor regions are close to identical between
the sexes, and the differences pile up in the higher-order centres where signals
are integrated and behaviour is chosen. A small, localised, structural
difference in the right place produces a large behavioural one.

That is a claim about where variation belongs, and it is testable on Aura,
because she has individuals too. Two commits of the same system are two
nervous systems built from the same plan with localised differences, and the
question is whether her differences land where the fly's land.

The test is stated before the numbers arrive. If the changed cells are enriched
in the deep integrative bands relative to the afferent and efferent ones, her
variation follows the biological pattern. If change is spread evenly, or
concentrated at the sensory and motor edges, it does not, and that is worth
knowing: it would mean her development touches the parts of herself that
biology holds still.

Enrichment is tested by permutation rather than by eye. The same number of cells
is drawn at random from the same population many times, and the observed
enrichment is placed in that distribution.
"""

from __future__ import annotations

import logging
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .celltypes import Typing, compare_typings, refine_types
from .microcircuit import LaminarAssignment
from .types import FLY_MALE_CNS_REFERENCE, ConnectomeSnapshot, EdgeKind

logger = logging.getLogger("Aura.Connectome.Dimorphism")

__all__ = [
    "Divergence",
    "compare_individuals",
    "EnrichmentTest",
    "concentration_test",
]


@dataclass
class Divergence:
    """What separates two individuals of the same system."""

    shared_cells: int
    only_in_a: int
    only_in_b: int
    rewired_pairs: int
    shared_pairs: int
    changed_cells: tuple[str, ...]
    typing: dict[str, Any] = field(default_factory=dict)

    @property
    def total_cells(self) -> int:
        return self.shared_cells + self.only_in_a + self.only_in_b

    @property
    def divergent_fraction(self) -> float:
        return len(self.changed_cells) / self.total_cells if self.total_cells else 0.0

    def as_json(self) -> dict[str, Any]:
        return {
            "shared_cells": self.shared_cells,
            "only_in_a": self.only_in_a,
            "only_in_b": self.only_in_b,
            "shared_pairs": self.shared_pairs,
            "rewired_pairs": self.rewired_pairs,
            "changed_cells": len(self.changed_cells),
            "divergent_fraction": round(self.divergent_fraction, 5),
            "fly_dimorphic_fraction": FLY_MALE_CNS_REFERENCE.get(
                "dimorphic_fraction_of_central_brain"
            ),
            "typing": self.typing,
        }


def compare_individuals(
    left: ConnectomeSnapshot,
    right: ConnectomeSnapshot,
    *,
    include_typing: bool = True,
) -> Divergence:
    """Line two reconstructions up and record every place they disagree.

    A cell counts as changed when it exists in both and its wiring is not the
    same, or when it exists in only one of them. Contact counts are compared as
    well as presence, so a pair that went from one call site to nine is a
    change even though the connection was there before and after.
    """
    left_cells = set(left.units)
    right_cells = set(right.units)
    shared = left_cells & right_cells

    def _drive(snapshot: ConnectomeSnapshot) -> dict[tuple[str, str], int]:
        return {
            (conn.pre, conn.post): conn.contacts
            for conn in snapshot.connections.values()
            if conn.kind is EdgeKind.DRIVE
        }

    left_edges = _drive(left)
    right_edges = _drive(right)
    pairs = set(left_edges) | set(right_edges)
    rewired: set[str] = set()
    shared_pairs = 0
    for pair in pairs:
        in_left = left_edges.get(pair)
        in_right = right_edges.get(pair)
        if in_left == in_right:
            shared_pairs += 1
            continue
        if pair[0] in shared:
            rewired.add(pair[0])
        if pair[1] in shared:
            rewired.add(pair[1])
    changed = sorted(rewired | (left_cells ^ right_cells))

    typing: dict[str, Any] = {}
    if include_typing:
        left_typing = refine_types(left, rounds=1)
        right_typing = refine_types(right, rounds=1)
        typing = compare_typings(left_typing, right_typing)

    return Divergence(
        shared_cells=len(shared),
        only_in_a=len(left_cells - right_cells),
        only_in_b=len(right_cells - left_cells),
        rewired_pairs=len(pairs) - shared_pairs,
        shared_pairs=shared_pairs,
        changed_cells=tuple(changed),
        typing=typing,
    )


@dataclass
class EnrichmentTest:
    """Whether change concentrates where the fly's does."""

    band_counts: dict[str, int]
    band_totals: dict[str, int]
    observed_ratio: float
    null_mean: float
    null_sd: float
    z_score: float
    permutations: int
    verdict: str

    def as_json(self) -> dict[str, Any]:
        return {
            "band_counts": self.band_counts,
            "band_totals": self.band_totals,
            "deep_over_edge_ratio": round(self.observed_ratio, 4),
            "null_mean": round(self.null_mean, 4),
            "null_sd": round(self.null_sd, 4),
            "z": round(self.z_score, 3),
            "permutations": self.permutations,
            "verdict": self.verdict,
        }


def concentration_test(
    snapshot: ConnectomeSnapshot,
    assignment: LaminarAssignment,
    changed: Sequence[str],
    *,
    permutations: int = 400,
    seed: int = 0,
) -> EnrichmentTest:
    """Test whether changed cells favour the integrative bands over the edges.

    The two edge bands are the input and output layers, which is where the fly's
    sexes are alike. The two middle bands are where its differences sit. The
    statistic is the ratio of changed cells in the middle to changed cells at
    the edges, and the null draws the same number of cells at random from the
    same assigned population.
    """
    deep_bands = {"L23", "L5"}
    edge_bands = {"L4", "L6"}
    assigned = [uid for uid in snapshot.units if uid in assignment.layer]
    if not assigned:
        return EnrichmentTest({}, {}, 0.0, 0.0, 0.0, 0.0, 0, "no laminar assignment")

    band_totals: dict[str, int] = {}
    for uid in assigned:
        band = assignment.layer[uid]
        band_totals[band] = band_totals.get(band, 0) + 1

    changed_assigned = [uid for uid in changed if uid in assignment.layer]
    if not changed_assigned:
        return EnrichmentTest(
            {}, band_totals, 0.0, 0.0, 0.0, 0.0, 0, "no changed cell was placed in a layer"
        )

    def _ratio(cells: Sequence[str]) -> float:
        deep = sum(1 for uid in cells if assignment.layer.get(uid) in deep_bands)
        edge = sum(1 for uid in cells if assignment.layer.get(uid) in edge_bands)
        return deep / edge if edge else float("inf")

    band_counts: dict[str, int] = {}
    for uid in changed_assigned:
        band = assignment.layer[uid]
        band_counts[band] = band_counts.get(band, 0) + 1

    observed = _ratio(changed_assigned)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(permutations):
        sample = rng.sample(assigned, min(len(changed_assigned), len(assigned)))
        value = _ratio(sample)
        if value != float("inf"):
            draws.append(value)
    null_mean = statistics.fmean(draws) if draws else 0.0
    null_sd = statistics.pstdev(draws) if len(draws) > 1 else 0.0
    z = (observed - null_mean) / null_sd if null_sd > 0 and observed != float("inf") else 0.0

    if z >= 2.0:
        verdict = "change concentrates in the integrative bands, as in the fly"
    elif z <= -2.0:
        verdict = "change concentrates at the sensory and motor edges, unlike the fly"
    else:
        verdict = "change is spread no differently from chance"

    return EnrichmentTest(
        band_counts=band_counts,
        band_totals=band_totals,
        observed_ratio=observed,
        null_mean=null_mean,
        null_sd=null_sd,
        z_score=z,
        permutations=len(draws),
        verdict=verdict,
    )

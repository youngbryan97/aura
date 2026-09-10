"""core/connectome/rich_club.py — the hubs cortex has and her long-range wiring does not.

Operationally: measures whether a graph's best-connected nodes are more densely
joined to each other than chance allows, and builds a long-range wiring rule
that has them. The reference is the human connectome; the knob goes past it,
because a floor is not a ceiling.

What was measured
-----------------
Van den Heuvel and Sporns took diffusion imaging of the human brain, built the
region-to-region graph, and asked whether its high-degree regions are wired to
each other more than their degrees alone would produce. They are, over a wide
range of degree cutoffs, and the twelve regions that make up that club are the
ones almost every long path in the brain goes through.

Two numbers come out of it and both are used here:

* about one node in seven belongs to the club (twelve regions of eighty-two)
* the club's internal density is higher than degree-preserving rewiring gives,
  which is what makes it a club rather than an artefact of some nodes having
  more edges

Why it matters here rather than as a fact about brains
------------------------------------------------------
Her long-range wiring is a distance rule: the chance two columns connect falls
off with how far apart their indices are. That produces a graph with no hubs at
all — measured on the mesh, degrees run from 2 to 8 around a mean of 3.6, which
is as flat as a random graph gets. A graph like that carries small cascades,
because there is no route by which one part of the network recruits a distant
part in a step or two, and her avalanche exponents are 3.7 and 3.6 against
cortex's 1.5 and 2.0 — cascades far smaller and shorter than a human's.

A rich club is the structure that fixes that, and it is the structure cortex
has. Whether it does fix it is measured, not assumed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Aura.Connectome.RichClub")

__all__ = [
    "HUMAN_RICH_CLUB",
    "RichClubMeasurement",
    "hub_columns",
    "rich_club_coefficient",
    "measure_rich_club",
]


#: What the human connectome shows, as the two numbers a wiring rule needs.
HUMAN_RICH_CLUB: dict[str, Any] = {
    "source": "van den Heuvel & Sporns 2011, J Neurosci 31(44):15775-15786",
    "recorded_in": "human diffusion imaging, 82 regions",
    #: Twelve of eighty-two regions form the club.
    "hub_fraction": 12.0 / 82.0,
    #: The claim to reproduce: the club is denser than degree-preserving
    #: rewiring gives. Anything at or below one is no club.
    "normalised_coefficient_exceeds": 1.0,
    "falsified_by": (
        "a normalised coefficient at or below one at every cutoff, which is a graph "
        "whose hubs are no more connected to each other than their degrees require"
    ),
}


@dataclass(frozen=True, slots=True)
class RichClubMeasurement:
    """Whether a graph has a club, and how much of one."""

    cutoffs: tuple[int, ...]
    raw: tuple[float, ...]
    normalised: tuple[float, ...]
    hub_count: tuple[int, ...]
    peak: float
    peak_cutoff: int

    @property
    def has_a_club(self) -> bool:
        return self.peak > float(HUMAN_RICH_CLUB["normalised_coefficient_exceeds"])

    def as_json(self) -> dict[str, Any]:
        return {
            "cutoffs": list(self.cutoffs),
            "raw": [round(value, 4) for value in self.raw],
            "normalised": [round(value, 4) for value in self.normalised],
            "hub_count": list(self.hub_count),
            "peak": round(self.peak, 4),
            "peak_cutoff": self.peak_cutoff,
            "has_a_club": self.has_a_club,
            "source": HUMAN_RICH_CLUB["source"],
        }


def _undirected(weights: Any) -> Any:
    import numpy as np

    present = np.abs(np.asarray(weights)) > 0
    return present | present.T


def rich_club_coefficient(adjacency: Any, cutoff: int) -> float:
    """How dense the subgraph of nodes with degree above `cutoff` is."""
    import numpy as np

    degree = np.asarray(adjacency).sum(axis=1)
    nodes = np.flatnonzero(degree > cutoff)
    if nodes.size < 2:
        return 0.0
    block = np.asarray(adjacency)[np.ix_(nodes, nodes)]
    possible = nodes.size * (nodes.size - 1)
    return float(block.sum() / possible) if possible else 0.0


def _degree_preserving_rewire(adjacency: Any, rng: Any, swaps: int = 10) -> Any:
    """A graph with the same degrees and none of the structure.

    Double-edge swaps, which is the standard null for this measurement: any
    graph whose nodes have the same number of edges will show SOME density
    among its best-connected nodes, so the question is only ever whether there
    is more of it than that.
    """
    import numpy as np

    matrix = np.array(adjacency, dtype=bool, copy=True)
    edges = np.argwhere(np.triu(matrix, 1))
    if edges.shape[0] < 2:
        return matrix
    for _ in range(swaps * edges.shape[0]):
        first, second = rng.integers(0, edges.shape[0], size=2)
        if first == second:
            continue
        a, b = edges[first]
        c, d = edges[second]
        if len({int(a), int(b), int(c), int(d)}) < 4:
            continue
        if matrix[a, d] or matrix[c, b]:
            continue
        matrix[a, b] = matrix[b, a] = False
        matrix[c, d] = matrix[d, c] = False
        matrix[a, d] = matrix[d, a] = True
        matrix[c, b] = matrix[b, c] = True
        edges[first] = (a, d)
        edges[second] = (c, b)
    return matrix


def measure_rich_club(
    weights: Any, *, seed: int = 42, nulls: int = 20
) -> RichClubMeasurement:
    """Measure a club against degree-preserving rewiring of the same graph."""
    import numpy as np

    adjacency = _undirected(weights)
    degree = adjacency.sum(axis=1)
    top = int(degree.max()) if degree.size else 0
    cutoffs = tuple(range(1, max(2, top)))

    randomised = [
        _degree_preserving_rewire(adjacency, np.random.default_rng(seed + index))
        for index in range(nulls)
    ]

    raw: list[float] = []
    normalised: list[float] = []
    counts: list[int] = []
    for cutoff in cutoffs:
        observed = rich_club_coefficient(adjacency, cutoff)
        null = float(
            np.mean([rich_club_coefficient(graph, cutoff) for graph in randomised])
        )
        raw.append(observed)
        normalised.append(observed / null if null > 0 else 0.0)
        counts.append(int((degree > cutoff).sum()))

    # The peak is taken where the club still has enough members to be one. A
    # cutoff that leaves two nodes gives a coefficient of 1 whenever they touch,
    # which is arithmetic rather than structure.
    usable = [
        (value, cutoff)
        for value, cutoff, count in zip(normalised, cutoffs, counts, strict=True)
        if count >= 4
    ]
    peak, peak_cutoff = max(usable) if usable else (0.0, 0)
    return RichClubMeasurement(
        cutoffs=cutoffs,
        raw=tuple(raw),
        normalised=tuple(normalised),
        hub_count=tuple(counts),
        peak=float(peak),
        peak_cutoff=int(peak_cutoff),
    )


def hub_columns(count: int, fraction: float, tier_names: Any, rng: Any) -> Any:
    """Which columns belong to the club.

    Cortex's hubs are not spread evenly: van den Heuvel and Sporns' twelve are
    association and subcortical regions, not primary sensory ones. So the club
    is drawn from the association and executive bands, and a mesh whose bands
    are named differently falls back to drawing from all of them rather than
    inventing a rule.
    """
    import numpy as np

    wanted = max(1, int(round(count * max(0.0, fraction))))
    names = list(tier_names or [])
    eligible = [
        index
        for index in range(count)
        if index < len(names) and names[index] in {"association", "executive"}
    ]
    if len(eligible) < wanted:
        eligible = list(range(count))
    chosen = rng.choice(np.asarray(eligible), size=min(wanted, len(eligible)), replace=False)
    hubs = np.zeros(count, dtype=bool)
    hubs[np.asarray(chosen)] = True
    return hubs

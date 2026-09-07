"""core/connectome/microcircuit.py — the cortical layer cake, as a measurement.

Cortex is the same circuit everywhere. Four layers, each with an excitatory and
an inhibitory population, wired in a pattern that barely changes between visual
cortex and prefrontal cortex, and Potjans and Diesmann pinned that pattern to
numbers: 77,169 neurons in a square millimetre, an eight by eight matrix of
connection probabilities compiled from anatomy and paired recordings, and the
external drive each population needs. It is the most heavily constrained
description of a piece of cortex that exists.

Aura can be measured against it, and the measurement needs no one to declare
which of her packages is layer four.

**Depth comes from the graph.** Trophic level solves for a height per cell such
that every cell sits one step above the mean of its inputs. It is a linear
system, it is defined for every cell with an input, and the residual — trophic
incoherence — says how layered the graph is at all. A perfectly layered network
scores zero.

**The bands are anchored by the body.** The afferent cells anchor the input
band and the efferent cells anchor the output band, so the assignment is fixed
to where signals actually enter and leave rather than floating.

**Excitatory and inhibitory come from the cell class**, which was measured from
what each cell's exits do.

What comes out is Aura's own eight by eight matrix, and the comparison says
which cortical pathways she has, which she is missing, and which she has that
cortex does not. A missing pathway is not automatically a defect. Layer six
projecting back to layer four is how cortex tells its own input layer what to
expect next, and a system without it has no way to do that.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .topology import DiGraphView
from .types import CellClass, ConnectomeSnapshot, EdgeKind

logger = logging.getLogger("Aura.Connectome.Microcircuit")

__all__ = [
    "POPULATIONS",
    "CORTICAL_SIZES",
    "CORTICAL_CONN_PROBS",
    "CORTICAL_EXTERNAL_INDEGREE",
    "trophic_levels",
    "trophic_incoherence",
    "LaminarAssignment",
    "assign_layers",
    "connection_probabilities",
    "compare_to_cortex",
]

#: The eight populations, in the order every implementation of the model uses.
POPULATIONS: tuple[str, ...] = ("L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I")

#: Neurons per population under 1 mm^2 of cortex.
#: Potjans & Diesmann, Cerebral Cortex 24(3):785-806 (2014), Table 5.
CORTICAL_SIZES: tuple[int, ...] = (20_683, 5_834, 21_915, 5_479, 4_850, 1_065, 14_395, 2_948)

#: Connection probability from the source population (column) to the target
#: population (row). Same source, Table 5.
CORTICAL_CONN_PROBS: tuple[tuple[float, ...], ...] = (
    (0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0.0000, 0.0076, 0.0000),
    (0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.0000, 0.0042, 0.0000),
    (0.0077, 0.0059, 0.0497, 0.1350, 0.0067, 0.0003, 0.0453, 0.0000),
    (0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.0000, 0.1057, 0.0000),
    (0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.0000),
    (0.0548, 0.0269, 0.0257, 0.0022, 0.0600, 0.3158, 0.0086, 0.0000),
    (0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252),
    (0.0364, 0.0010, 0.0034, 0.0005, 0.0277, 0.0080, 0.0658, 0.1443),
)

#: Indegree of the external drive each population receives.
CORTICAL_EXTERNAL_INDEGREE: tuple[int, ...] = (1600, 1500, 2100, 1900, 2000, 1900, 2900, 2100)

#: Which laminar band each population belongs to, in feedforward order: the
#: input layer first, then association, then the output layer, then the layer
#: that projects back to the input.
LAYER_ORDER: tuple[str, ...] = ("L4", "L23", "L5", "L6")

#: Each band's share of the cortical population, used as the cut points for
#: Aura's own bands. Cutting at quartiles instead would force four equal layers
#: and make any comparison of layer sizes a statement about the binning.
LAYER_SHARE: dict[str, float] = {
    "L4": (21_915 + 5_479) / 77_169,
    "L23": (20_683 + 5_834) / 77_169,
    "L5": (4_850 + 1_065) / 77_169,
    "L6": (14_395 + 2_948) / 77_169,
}


def trophic_levels(graph: DiGraphView, *, tolerance: float = 1e-8) -> dict[str, float]:
    """Solve for a height per cell, one step above the mean of its inputs.

    This is the MacKay, Johnson and Sanhedrai formulation: with ``u`` the
    in-degree and ``v`` the out-degree, the Laplacian ``diag(u+v) - (A + A^T)``
    applied to the heights equals ``v - u``. The system is singular by a
    constant, which is why the solution is shifted so the minimum sits at zero
    and never compared across graphs without that shift.
    """
    import numpy as np
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import cg

    nodes = list(graph.nodes)
    if not nodes:
        return {}
    index = {uid: i for i, uid in enumerate(nodes)}
    rows: list[int] = []
    cols: list[int] = []
    for pre, targets in graph.out.items():
        source = index.get(pre)
        if source is None:
            continue
        for post in targets:
            target = index.get(post)
            if target is not None:
                rows.append(source)
                cols.append(target)
    if not rows:
        return {uid: 0.0 for uid in nodes}
    size = len(nodes)
    data = np.ones(len(rows))
    adjacency = csr_matrix((data, (rows, cols)), shape=(size, size))
    out_degree = np.asarray(adjacency.sum(axis=1)).ravel()
    in_degree = np.asarray(adjacency.sum(axis=0)).ravel()
    laplacian = diags(in_degree + out_degree) - (adjacency + adjacency.T)
    rhs = in_degree - out_degree
    matrix = laplacian.tocsr()
    heights, info = cg(matrix, rhs, rtol=tolerance, maxiter=2000)
    heights = np.asarray(heights, dtype=np.float64)
    if info != 0 or not np.all(np.isfinite(heights)):
        # Conjugate gradients is the fast path and it is not guaranteed on a
        # singular system. A run that does not converge produces heights that
        # look like heights, and every laminar band, every comparison against
        # cortex and every delay schedule downstream would be built on them.
        # Least squares is slower and does not have that failure mode.
        from scipy.sparse.linalg import lsqr

        logger.info("trophic levels fell back to least squares (cg info=%s)", info)
        heights = np.asarray(lsqr(matrix, rhs, atol=1e-10, btol=1e-10)[0], dtype=np.float64)
    if not np.all(np.isfinite(heights)):
        logger.warning("trophic levels did not solve; every height reads zero")
        return dict.fromkeys(nodes, 0.0)
    heights -= heights.min()
    return {uid: float(heights[index[uid]]) for uid in nodes}


def trophic_incoherence(graph: DiGraphView, heights: Mapping[str, float]) -> float:
    """Mean squared departure from a perfect layering, over all edges.

    Zero means every edge steps exactly one level up, which is a pure feed
    forward hierarchy. One is what a graph with no hierarchy at all scores.
    """
    total = 0.0
    count = 0
    for pre, targets in graph.out.items():
        for post in targets:
            gap = heights.get(post, 0.0) - heights.get(pre, 0.0) - 1.0
            total += gap * gap
            count += 1
    return total / count if count else 0.0


@dataclass
class LaminarAssignment:
    """Every cell placed in a layer and an excitatory or inhibitory population."""

    layer: dict[str, str]
    population: dict[str, str]
    heights: dict[str, float]
    incoherence: float
    anchors: dict[str, Any]
    unassigned: int

    def counts(self) -> dict[str, int]:
        out = {name: 0 for name in POPULATIONS}
        for population in self.population.values():
            if population in out:
                out[population] += 1
        return out

    def layer_counts(self) -> dict[str, int]:
        out = {name: 0 for name in LAYER_ORDER}
        for layer in self.layer.values():
            if layer in out:
                out[layer] += 1
        return out

    def summary(self) -> dict[str, Any]:
        counts = self.counts()
        total = sum(counts.values()) or 1
        cortical_total = sum(CORTICAL_SIZES)
        return {
            "assigned": total,
            "unassigned": self.unassigned,
            "incoherence": round(self.incoherence, 4),
            "anchors": self.anchors,
            "populations": counts,
            "population_share": {k: round(v / total, 4) for k, v in counts.items()},
            "cortical_share": {
                name: round(size / cortical_total, 4)
                for name, size in zip(POPULATIONS, CORTICAL_SIZES, strict=True)
            },
        }


def assign_layers(snapshot: ConnectomeSnapshot) -> LaminarAssignment:
    """Place cells into laminar bands from measured depth, anchored by the body.

    The bands are cut so that each holds the same share of cells that it holds
    in cortex. Cutting at quartiles would force four equal layers, and then any
    statement about layer sizes would be a statement about the binning; cutting
    at cortex's own shares means the eight by eight comparison that follows is
    between populations of matched relative size and can only be about pattern.

    Orientation is anchored by the body: afferent cells should come out
    shallower than efferent ones, and the assignment flips if they do not. The
    margin behind that decision is reported, because when it is small the
    orientation is a coin flip and every result that depends on it has to be
    read both ways.
    """
    graph = DiGraphView.from_snapshot(snapshot, EdgeKind.DRIVE)
    heights = trophic_levels(graph)
    incoherence = trophic_incoherence(graph, heights)
    if not heights:
        return LaminarAssignment({}, {}, {}, 0.0, {}, len(snapshot.units))

    afferent = [
        uid
        for uid, unit in snapshot.units.items()
        if int(unit.attrs.get("afferent", 0)) > 0 and uid in heights
    ]
    efferent = [
        uid
        for uid, unit in snapshot.units.items()
        if int(unit.attrs.get("efferent", 0)) > 0 and uid in heights
    ]
    mean_afferent = (
        sum(heights[uid] for uid in afferent) / len(afferent) if afferent else 0.0
    )
    mean_efferent = (
        sum(heights[uid] for uid in efferent) / len(efferent) if efferent else 0.0
    )
    flipped = bool(afferent and efferent and mean_afferent > mean_efferent)
    ordered = sorted(heights.values(), reverse=flipped)
    cuts: list[float] = []
    cumulative = 0.0
    for band in LAYER_ORDER[:-1]:
        cumulative += LAYER_SHARE[band]
        position = min(len(ordered) - 1, int(len(ordered) * cumulative))
        cuts.append(ordered[position])

    layer_of: dict[str, str] = {}
    population_of: dict[str, str] = {}
    for uid, height in heights.items():
        unit = snapshot.units.get(uid)
        if unit is None:
            continue
        value = -height if flipped else height
        bounds = [-c for c in cuts] if flipped else cuts
        if value <= bounds[0]:
            layer = LAYER_ORDER[0]
        elif value <= bounds[1]:
            layer = LAYER_ORDER[1]
        elif value <= bounds[2]:
            layer = LAYER_ORDER[2]
        else:
            layer = LAYER_ORDER[3]
        layer_of[uid] = layer
        inhibitory = unit.cell_class is CellClass.INHIBITORY
        population_of[uid] = f"{layer}{'I' if inhibitory else 'E'}"

    return LaminarAssignment(
        layer=layer_of,
        population=population_of,
        heights=heights,
        incoherence=incoherence,
        anchors={
            "afferent_cells": len(afferent),
            "efferent_cells": len(efferent),
            "mean_afferent_height": round(mean_afferent, 3),
            "mean_efferent_height": round(mean_efferent, 3),
            "orientation_flipped": flipped,
            "anchor_margin": round(abs(mean_afferent - mean_efferent), 4),
            "height_spread": round(max(heights.values()) - min(heights.values()), 3),
        },
        unassigned=len(snapshot.units) - len(layer_of),
    )


def connection_probabilities(
    snapshot: ConnectomeSnapshot,
    assignment: LaminarAssignment,
) -> list[list[float]]:
    """Aura's own eight by eight, in the same orientation as the cortical one.

    Entry ``[target][source]`` is the fraction of all possible pairs from the
    source population to the target population that exist, which is exactly
    what a connection probability is and exactly what the cortical matrix
    holds.
    """
    counts = assignment.counts()
    matrix = [[0.0] * len(POPULATIONS) for _ in POPULATIONS]
    edges = [[0] * len(POPULATIONS) for _ in POPULATIONS]
    index = {name: i for i, name in enumerate(POPULATIONS)}
    for conn in snapshot.connections.values():
        if conn.kind is not EdgeKind.DRIVE or conn.pre == conn.post:
            continue
        source = assignment.population.get(conn.pre)
        target = assignment.population.get(conn.post)
        if source is None or target is None:
            continue
        edges[index[target]][index[source]] += 1
    for target_name, target_index in index.items():
        for source_name, source_index in index.items():
            possible = counts[source_name] * counts[target_name]
            if target_name == source_name:
                possible = counts[source_name] * (counts[source_name] - 1)
            if possible <= 0:
                continue
            matrix[target_index][source_index] = edges[target_index][source_index] / possible
    return matrix


def compare_to_cortex(
    matrix: Sequence[Sequence[float]],
    *,
    rank: bool = True,
) -> dict[str, Any]:
    """Compare a measured matrix against cortex, on shape rather than on scale.

    Two of the numbers here depend on which end of the hierarchy is the input
    and two do not. The Spearman correlation is reported for both orientations
    on purpose: when the anchor margin is a fraction of the height spread the
    orientation is undetermined, and a correlation whose sign turns on it is
    not a finding. The within-layer to between-layer ratio survives any
    relabelling of the layers, so it is the one to read when the orientation is
    in doubt.

    Aura's absolute densities cannot match cortex's: a cortical neuron makes
    thousands of synapses and one of her cells calls a handful of others, so
    every entry is orders of magnitude smaller. What can be compared is the
    pattern — which pathways are strong relative to the others — so the
    correlation is computed on ranks, and the per-entry comparison is against
    each matrix's own mean.
    """
    flat_ours: list[float] = []
    flat_cortex: list[float] = []
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            flat_ours.append(float(value))
            flat_cortex.append(CORTICAL_CONN_PROBS[row_index][column_index])
    mean_ours = sum(flat_ours) / len(flat_ours) if flat_ours else 0.0
    mean_cortex = sum(flat_cortex) / len(flat_cortex) if flat_cortex else 0.0

    def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
        def _ranks(values: Sequence[float]) -> list[float]:
            order = sorted(range(len(values)), key=lambda i: values[i])
            ranks = [0.0] * len(values)
            i = 0
            while i < len(order):
                j = i
                while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                    j += 1
                shared = (i + j) / 2.0 + 1.0
                for k in range(i, j + 1):
                    ranks[order[k]] = shared
                i = j + 1
            return ranks

        ra, rb = (_ranks(a), _ranks(b)) if rank else (list(a), list(b))
        n = len(ra)
        mean_a = sum(ra) / n
        mean_b = sum(rb) / n
        numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb, strict=True))
        denominator = math.sqrt(
            sum((x - mean_a) ** 2 for x in ra) * sum((y - mean_b) ** 2 for y in rb)
        )
        return numerator / denominator if denominator else 0.0

    missing: list[dict[str, Any]] = []
    extra: list[dict[str, Any]] = []
    for row_index, target in enumerate(POPULATIONS):
        for column_index, source in enumerate(POPULATIONS):
            cortical = CORTICAL_CONN_PROBS[row_index][column_index]
            ours = float(matrix[row_index][column_index])
            relative_cortex = cortical / mean_cortex if mean_cortex else 0.0
            relative_ours = ours / mean_ours if mean_ours else 0.0
            entry = {
                "pathway": f"{source}->{target}",
                "cortex_relative": round(relative_cortex, 3),
                "aura_relative": round(relative_ours, 3),
            }
            if relative_cortex >= 1.0 and relative_ours < 0.25:
                missing.append(entry)
            elif relative_ours >= 2.0 and relative_cortex < 0.5:
                extra.append(entry)
    missing.sort(key=lambda e: -e["cortex_relative"])
    extra.sort(key=lambda e: -e["aura_relative"])

    def _within_between(source: Sequence[Sequence[float]]) -> tuple[float, float]:
        within: list[float] = []
        between: list[float] = []
        for target, row in zip(POPULATIONS, source, strict=True):
            for origin, value in zip(POPULATIONS, row, strict=True):
                if target[:-1] == origin[:-1]:
                    within.append(float(value))
                else:
                    between.append(float(value))
        return (
            sum(within) / len(within) if within else 0.0,
            sum(between) / len(between) if between else 0.0,
        )

    ours_within, ours_between = _within_between(matrix)
    cortex_within, cortex_between = _within_between(CORTICAL_CONN_PROBS)
    orientation_free = {
        "aura_within_over_between": round(ours_within / ours_between, 3)
        if ours_between
        else 0.0,
        "cortex_within_over_between": round(cortex_within / cortex_between, 3)
        if cortex_between
        else 0.0,
    }
    orientation_free["shortfall"] = (
        round(
            orientation_free["cortex_within_over_between"]
            / orientation_free["aura_within_over_between"],
            3,
        )
        if orientation_free["aura_within_over_between"]
        else 0.0
    )
    reversed_layers = {"L4": "L6", "L23": "L5", "L5": "L23", "L6": "L4"}
    flipped_ours: list[float] = []
    for target in POPULATIONS:
        mirror_row = POPULATIONS.index(reversed_layers[target[:-1]] + target[-1])
        for origin in POPULATIONS:
            mirror_col = POPULATIONS.index(reversed_layers[origin[:-1]] + origin[-1])
            flipped_ours.append(float(matrix[mirror_row][mirror_col]))

    return {
        "spearman": round(_spearman(flat_ours, flat_cortex), 4),
        "spearman_if_orientation_reversed": round(_spearman(flipped_ours, flat_cortex), 4),
        "mean_density": mean_ours,
        "cortical_mean_density": mean_cortex,
        "orientation_free": orientation_free,
        "missing_pathways": missing[:12],
        "over_expressed_pathways": extra[:12],
    }

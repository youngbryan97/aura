"""core/connectome/likewise.py — do cells that do the same thing wire together?

MICrONS recorded 75,909 neurons in a mouse's visual cortex and then reconstructed
the same tissue at synapse resolution — 200,000 cells, half a billion synapses,
co-registered to within 3.8 micrometres. What that combination bought was the
first direct test of a rule that had been assumed for decades: neurons tuned to
similar things preferentially connect to each other, and the rule holds across
cortical layers and across visual areas.

The test needs structure and function measured in the same animal, which is why
it took a decade. Aura has both by construction: the cell that fires is the cell
in the graph.

So the question transfers exactly. Do cells whose activity moves together
connect more often than cells that do not?

The null is the one that makes it a test. Connected cells share callers and
share workloads, and a busy cell correlates with everything, so comparing
connected pairs against all pairs would measure how busy the connected ones are.
A degree-preserving rewiring keeps every cell exactly as busy and as connected
as it was and destroys only who is wired to whom. If the correlation among
connected pairs survives that, it is about the wiring.

That null still leaves one thing uncontrolled, and it is the obvious one. Two
connected cells tend to be busy during the same workload, and a rewired partner
may be busy during a different one, so part of any effect is the workload rather
than the wiring. Removing each cell's mean *within each condition* takes the
workload out entirely: what is left is whether two cells move together while the
same thing is being done. ``within_condition`` is on by default because the
uncontrolled version answers an easier question than the one being asked.
"""

from __future__ import annotations

import logging
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .topology import DiGraphView, degree_preserving_rewire
from .types import ConnectomeSnapshot, EdgeKind

logger = logging.getLogger("Aura.Connectome.Likewise")

__all__ = ["LikeToLike", "test_like_to_like"]


@dataclass
class LikeToLike:
    """Correlation among connected pairs, against a degree-preserving null."""

    connected_pairs: int
    connected_mean: float
    null_mean: float
    null_spread: float
    z: float
    nulls: int
    active_cells: int
    within_condition: bool
    verdict: str

    def as_json(self) -> dict[str, Any]:
        return {
            "connected_pairs": self.connected_pairs,
            "connected_mean_correlation": round(self.connected_mean, 5),
            "null_mean_correlation": round(self.null_mean, 5),
            "null_spread": round(self.null_spread, 5),
            "z": round(self.z, 3),
            "nulls": self.nulls,
            "active_cells": self.active_cells,
            "within_condition": self.within_condition,
            "verdict": self.verdict,
        }


def _mean_pair_correlation(
    scaled: Any,
    index: dict[str, int],
    pairs: Sequence[tuple[str, str]],
    *,
    sample: int,
    seed: int,
) -> tuple[float, int]:
    """Mean Pearson correlation over a sample of pairs, on pre-scaled traces."""
    import numpy as np

    rng = np.random.default_rng(seed)
    usable = [
        (index[pre], index[post])
        for pre, post in pairs
        if pre in index and post in index and pre != post
    ]
    if not usable:
        return 0.0, 0
    if len(usable) > sample:
        picks = rng.choice(len(usable), size=sample, replace=False)
        usable = [usable[i] for i in picks]
    left = np.array([a for a, _ in usable])
    right = np.array([b for _, b in usable])
    products = (scaled[:, left] * scaled[:, right]).mean(axis=0)
    return float(products.mean()), len(usable)


def test_like_to_like(
    trace: Any,
    snapshot: ConnectomeSnapshot,
    *,
    nulls: int = 8,
    sample: int = 20_000,
    seed: int = 0,
    min_frames_active: int = 4,
    within_condition: bool = True,
) -> LikeToLike:
    """Measure whether connected cells move together more than chance allows.

    Cells that barely fire are dropped: a trace with three non-zero frames has a
    correlation that is an artefact of which three, and thousands of those swamp
    the measurement.

    With ``within_condition`` set, each cell is centred inside every condition
    before anything is correlated, so two cells that are simply busy during the
    same workload contribute nothing.
    """
    import numpy as np

    matrix = trace.matrix()
    if matrix.size == 0:
        return LikeToLike(0, 0.0, 0.0, 0.0, 0.0, 0, 0, within_condition, "no recording")
    uids = list(trace.uids)
    active_count = (matrix != 0).sum(axis=0)
    alive = active_count >= min_frames_active
    if int(alive.sum()) < 32:
        return LikeToLike(
            0, 0.0, 0.0, 0.0, 0.0, 0, int(alive.sum()), within_condition, "too few active cells"
        )
    kept = matrix[:, alive]
    kept_uids = [uid for uid, keep in zip(uids, alive, strict=False) if keep]
    index = {uid: i for i, uid in enumerate(kept_uids)}

    if within_condition and getattr(trace, "conditions", None):
        centred = np.array(kept, dtype=np.float64)
        conditions = list(trace.conditions)[: kept.shape[0]]
        for name in sorted(set(conditions)):
            rows = [i for i, c in enumerate(conditions) if c == name]
            if len(rows) < 2:
                continue
            centred[rows] -= centred[rows].mean(axis=0)
    else:
        centred = kept - kept.mean(axis=0)
    spread = centred.std(axis=0)
    spread[spread <= 0] = 1.0
    scaled = centred / spread

    graph = DiGraphView.from_snapshot(snapshot, EdgeKind.DRIVE, drop_isolated=False)
    induced = DiGraphView(
        nodes=tuple(kept_uids),
        out={uid: {p for p in graph.out.get(uid, ()) if p in index} for uid in kept_uids},
        inbound={uid: {p for p in graph.inbound.get(uid, ()) if p in index} for uid in kept_uids},
        weights={},
    )
    observed_pairs = induced.edges()
    if len(observed_pairs) < 32:
        return LikeToLike(
            len(observed_pairs), 0.0, 0.0, 0.0, 0.0, 0, len(kept_uids), within_condition,
            "too few connected pairs among the cells that fired",
        )

    connected_mean, counted = _mean_pair_correlation(
        scaled, index, observed_pairs, sample=sample, seed=seed
    )
    null_means: list[float] = []
    for i in range(nulls):
        rewired = degree_preserving_rewire(induced, swaps_per_edge=6, seed=seed + 31 + i)
        value, _ = _mean_pair_correlation(
            scaled, index, rewired.edges(), sample=sample, seed=seed
        )
        null_means.append(value)
    null_mean = statistics.fmean(null_means) if null_means else 0.0
    null_spread = statistics.pstdev(null_means) if len(null_means) > 1 else 0.0
    z = (connected_mean - null_mean) / null_spread if null_spread > 0 else 0.0

    if z >= 3.0:
        verdict = (
            "connected cells move together more than a degree-preserving rewiring "
            "allows, which is MICrONS's like-to-like rule"
        )
    elif z <= -3.0:
        verdict = "connected cells move together less than chance, which needs explaining"
    else:
        verdict = "no detectable like-to-like effect against a degree-preserving null"

    return LikeToLike(
        connected_pairs=counted,
        connected_mean=connected_mean,
        null_mean=null_mean,
        null_spread=null_spread,
        z=z,
        nulls=nulls,
        active_cells=len(kept_uids),
        within_condition=within_condition,
        verdict=verdict,
    )

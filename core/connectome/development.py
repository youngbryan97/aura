"""core/connectome/development.py — a brain is not built, it is grown and then cut back.

Human cortex does not assemble the adult wiring. It overshoots. Synapse density
peaks in the first years of life and then roughly forty percent of those
synapses are removed, and which ones go is decided by whether they carried
anything. Huttenlocher counted this in human tissue; the same overproduce-and-
prune pattern shows up everywhere it has been looked for.

Two properties of that process are worth having, and neither is obvious from the
finished wiring diagram.

**Pruning by use beats designing the wiring.** The developing brain does not
work out which connections it needs. It makes too many, runs, and keeps the ones
that carried traffic. That is cheap, it needs no global view, and it adapts to
the environment the organism actually landed in.

**The window closes.** Plasticity is not constant. A critical period opens,
experience during it sets the structure, and then it shuts — in cortex when
inhibitory maturation and perineuronal nets lock it down. A system that stays
plastic forever cannot commit to anything; a system that never was cannot learn
from where it is.

Applied here, pruning by measured traffic has an obvious null to beat: pruning
the same number of edges at random. If the traffic-pruned graph is no better
than the randomly-pruned one at anything that matters, then the traffic was not
telling us what to keep, and this module reports that rather than assuming the
biology transferred.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .activity import ObservedEdges
from .topology import DiGraphView
from .types import ConnectomeSnapshot, EdgeKind

logger = logging.getLogger("Aura.Connectome.Development")

__all__ = [
    "CriticalPeriod",
    "PruningResult",
    "prune_by_use",
    "prune_at_random",
    "compare_pruning",
    "HUMAN_SYNAPTIC_PRUNING_FRACTION",
]

#: The fraction of peak synapses human cortex removes between the density peak
#: in early childhood and adult levels. Huttenlocher & Dabholkar,
#: J Comp Neurol 387:167 (1997), across frontal and auditory cortex.
HUMAN_SYNAPTIC_PRUNING_FRACTION: float = 0.40


@dataclass(frozen=True)
class CriticalPeriod:
    """A window during which structure can change, and what shuts it.

    ``closes_on`` is required for the same reason a gate needs a closing
    condition: a window with no closing condition is not a critical period, it
    is permanent plasticity wearing the name.
    """

    name: str
    opens_at: float
    closes_at: float
    closes_on: str
    plasticity: float = 1.0

    def open_at(self, when: float) -> bool:
        return self.opens_at <= when < self.closes_at

    def rate(self, when: float) -> float:
        """Plasticity at a moment: full inside the window, zero outside it."""
        return self.plasticity if self.open_at(when) else 0.0


@dataclass
class PruningResult:
    """A pruned connectome and what the pruning cost."""

    snapshot: ConnectomeSnapshot
    kept: int
    removed: int
    method: str
    surviving_traffic: int
    total_traffic: int

    @property
    def traffic_retained(self) -> float:
        return self.surviving_traffic / self.total_traffic if self.total_traffic else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "kept": self.kept,
            "removed": self.removed,
            "removed_fraction": round(self.removed / max(1, self.kept + self.removed), 4),
            "traffic_retained": round(self.traffic_retained, 5),
        }


def _pruned_snapshot(
    snapshot: ConnectomeSnapshot,
    drop: set[tuple[str, str]],
    method: str,
) -> ConnectomeSnapshot:
    connections = {
        key: conn
        for key, conn in snapshot.connections.items()
        if conn.kind is not EdgeKind.DRIVE or (conn.pre, conn.post) not in drop
    }
    grown = ConnectomeSnapshot(
        version=snapshot.version + 1,
        units=snapshot.units,
        connections=connections,
        neuropils=snapshot.neuropils,
        built_at=snapshot.built_at,
        source=snapshot.source,
        attrs=dict(snapshot.attrs),
    )
    grown.attrs.update({"pruned_by": method, "pruned_edges": len(drop)})
    return grown


def prune_by_use(
    snapshot: ConnectomeSnapshot,
    observed: ObservedEdges,
    *,
    fraction: float = HUMAN_SYNAPTIC_PRUNING_FRACTION,
) -> PruningResult:
    """Remove the least-used connections, keeping the ones that carried traffic.

    Ties are broken by contact count and then by cell name, so two runs on the
    same recording prune the same edges. An edge that was never observed is not
    proof of an unused edge — the branch may not have run — so this is a
    development step to be applied to a recording that covered the behaviour
    being developed for, and nothing else.
    """
    drive = [c for c in snapshot.edges(EdgeKind.DRIVE)]
    ranked = sorted(
        drive,
        key=lambda c: (
            observed.counts.get((c.pre, c.post), 0),
            c.contacts,
            c.pre,
            c.post,
        ),
    )
    target = int(len(ranked) * max(0.0, min(1.0, fraction)))
    drop = {(c.pre, c.post) for c in ranked[:target]}
    surviving = sum(
        count for (pre, post), count in observed.counts.items() if (pre, post) not in drop
    )
    return PruningResult(
        snapshot=_pruned_snapshot(snapshot, drop, "use"),
        kept=len(ranked) - len(drop),
        removed=len(drop),
        method="use",
        surviving_traffic=surviving,
        total_traffic=sum(observed.counts.values()),
    )


def prune_at_random(
    snapshot: ConnectomeSnapshot,
    observed: ObservedEdges,
    *,
    fraction: float = HUMAN_SYNAPTIC_PRUNING_FRACTION,
    seed: int = 0,
) -> PruningResult:
    """The null: remove the same number of edges without looking at traffic."""
    drive = [c for c in snapshot.edges(EdgeKind.DRIVE)]
    rng = random.Random(seed)
    ordered = sorted(drive, key=lambda c: (c.pre, c.post))
    rng.shuffle(ordered)
    target = int(len(ordered) * max(0.0, min(1.0, fraction)))
    drop = {(c.pre, c.post) for c in ordered[:target]}
    surviving = sum(
        count for (pre, post), count in observed.counts.items() if (pre, post) not in drop
    )
    return PruningResult(
        snapshot=_pruned_snapshot(snapshot, drop, "random"),
        kept=len(ordered) - len(drop),
        removed=len(drop),
        method="random",
        surviving_traffic=surviving,
        total_traffic=sum(observed.counts.values()),
    )


def compare_pruning(
    snapshot: ConnectomeSnapshot,
    observed: ObservedEdges,
    *,
    fraction: float = HUMAN_SYNAPTIC_PRUNING_FRACTION,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> dict[str, Any]:
    """Prune by use, prune at random, and report whether use was worth using.

    Traffic retained is the primary comparison and it is nearly rigged in
    favour of use-based pruning, which is why the second comparison matters
    more: how much of the graph's reach survives. A pruning that keeps every
    busy edge and disconnects the system has kept the traffic and lost the
    circuit.
    """
    used = prune_by_use(snapshot, observed, fraction=fraction)
    randoms = [
        prune_at_random(snapshot, observed, fraction=fraction, seed=seed) for seed in seeds
    ]

    def _reach(result: PruningResult) -> tuple[int, float]:
        graph = DiGraphView.from_snapshot(result.snapshot, EdgeKind.DRIVE)
        edges = graph.m
        nodes = graph.n
        return nodes, (edges / nodes if nodes else 0.0)

    used_nodes, used_degree = _reach(used)
    random_nodes = [_reach(r)[0] for r in randoms]
    random_degree = [_reach(r)[1] for r in randoms]
    random_traffic = [r.traffic_retained for r in randoms]
    mean_random_traffic = sum(random_traffic) / len(random_traffic)
    return {
        "fraction_removed": fraction,
        "use": used.summary(),
        "random_mean_traffic_retained": round(mean_random_traffic, 5),
        "traffic_advantage": round(used.traffic_retained - mean_random_traffic, 5),
        "connected_cells_use": used_nodes,
        "connected_cells_random_mean": round(sum(random_nodes) / len(random_nodes), 1),
        "mean_degree_use": round(used_degree, 4),
        "mean_degree_random": round(sum(random_degree) / len(random_degree), 4),
        "verdict": (
            "use-based pruning keeps the traffic and the reach"
            if used.traffic_retained > mean_random_traffic
            and used_nodes >= sum(random_nodes) / len(random_nodes)
            else "use-based pruning keeps the traffic and costs reach"
            if used.traffic_retained > mean_random_traffic
            else "traffic did not say what to keep"
        ),
    }

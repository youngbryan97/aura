"""core/connectome/celltypes.py — a type is a group with the same wiring, or it is a label.

FlyEM's male CNS release names more than ten thousand cell types across 166,000
neurons, and the criterion behind the names is worth copying exactly. A cell
type is not a shape someone recognised. It is a group of cells whose connections
to *other types* match, defined by a fixed point: refine the grouping by
connectivity, use the new grouping to describe connectivity, refine again, stop
when nothing moves. Two cells are the same type when the circuit cannot tell
them apart.

That procedure is colour refinement, and it transfers to Aura without any
biological hand-waving, because it needs only a graph. What comes out is a
partition of her cells into groups that occupy the same position in the wiring
— a factory that feeds three validators and a store is the same type as another
factory that does, whatever the two are called.

Types are only worth having if they survive a rerun, so two validations ship
with the typing rather than after it:

**Stability.** Drop a tenth of the edges, retype, and compare partitions with
the adjusted Rand index. A typing that moves when a tenth of the wiring changes
was describing noise.

**Reproducibility across individuals.** FlyEM checks a type by finding it in
another fly. Aura's other individual is another commit of herself, and
:func:`compare_typings` reports how much of the typing survived the code
changing underneath it.
"""

from __future__ import annotations

import hashlib
import logging
import random
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .topology import DiGraphView
from .types import ConnectomeSnapshot, EdgeKind

logger = logging.getLogger("Aura.Connectome.CellTypes")

__all__ = [
    "Typing",
    "refine_types",
    "adjusted_rand_index",
    "stability",
    "compare_typings",
]


def _bucket(contacts: int) -> int:
    """Contact counts, coarsened so a pair with 7 and one with 8 agree.

    Connectivity matching has to tolerate a synapse either way or nothing ever
    matches. Powers of two are the standard coarsening and they keep the H01
    threshold of four intact as its own boundary.
    """
    if contacts <= 1:
        return 1
    if contacts <= 3:
        return 2
    if contacts <= 7:
        return 4
    if contacts <= 15:
        return 8
    return 16


@dataclass
class Typing:
    """A partition of cells into connectivity-defined types."""

    labels: dict[str, str]
    rounds: int
    seed_labels: str

    @property
    def type_count(self) -> int:
        return len(set(self.labels.values()))

    def sizes(self) -> Counter[str]:
        return Counter(self.labels.values())

    def summary(self) -> dict[str, Any]:
        sizes = self.sizes()
        multi = [size for size in sizes.values() if size > 1]
        cells = sum(sizes.values())
        return {
            "cells": cells,
            "types": len(sizes),
            "singleton_types": sum(1 for size in sizes.values() if size == 1),
            "cells_in_multi_member_types": sum(multi),
            "multi_member_share": round(sum(multi) / cells, 4) if cells else 0.0,
            "largest_type": max(sizes.values()) if sizes else 0,
            "mean_type_size": round(cells / len(sizes), 3) if sizes else 0.0,
            "rounds": self.rounds,
            "seed_labels": self.seed_labels,
        }

    def largest(self, limit: int = 10) -> list[tuple[str, int]]:
        return self.sizes().most_common(limit)


def refine_types(
    snapshot: ConnectomeSnapshot,
    *,
    rounds: int = 3,
    seed_labels: str = "cell_class",
    kind: EdgeKind | None = EdgeKind.DRIVE,
    drop_edges: float = 0.0,
    seed: int = 0,
) -> Typing:
    """Refine an initial labelling by connectivity until it stops moving.

    ``rounds`` is small on purpose. Each round widens the neighbourhood a cell
    is described by, and past three rounds almost every cell has a description
    no other cell shares, which is a partition into singletons wearing the word
    type. Stopping early is what keeps a type a group.
    """
    graph = DiGraphView.from_snapshot(snapshot, kind, drop_isolated=False)
    if drop_edges > 0:
        rng = random.Random(seed)
        for pre in list(graph.out):
            keep = {post for post in graph.out[pre] if rng.random() >= drop_edges}
            dropped = graph.out[pre] - keep
            graph.out[pre] = keep
            for post in dropped:
                graph.inbound[post].discard(pre)

    if seed_labels == "cell_class":
        labels = {uid: str(unit.cell_class) for uid, unit in snapshot.units.items()}
    elif seed_labels == "region":
        labels = {uid: unit.region for uid, unit in snapshot.units.items()}
    elif seed_labels == "uniform":
        labels = {uid: "cell" for uid in snapshot.units}
    else:
        raise ValueError(f"unknown seed labelling: {seed_labels}")

    for _ in range(max(0, rounds)):
        nxt: dict[str, str] = {}
        for uid in snapshot.units:
            outgoing = sorted(
                f"o:{labels.get(post, '?')}:{_bucket(graph.weights.get((uid, post), 1))}"
                for post in graph.out.get(uid, ())
            )
            incoming = sorted(
                f"i:{labels.get(pre, '?')}:{_bucket(graph.weights.get((pre, uid), 1))}"
                for pre in graph.inbound.get(uid, ())
            )
            payload = "|".join((labels.get(uid, "?"), *outgoing, *incoming))
            nxt[uid] = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()
        if len(set(nxt.values())) == len(set(labels.values())):
            labels = nxt
            break
        labels = nxt
    return Typing(labels=labels, rounds=rounds, seed_labels=seed_labels)


def adjusted_rand_index(a: Mapping[str, str], b: Mapping[str, str]) -> float:
    """Agreement between two partitions, corrected for chance.

    Zero is what two unrelated partitions of the same cells score on average,
    and one is identical. The correction is the point: two partitions with
    thousands of tiny groups agree on almost every pair by accident, and the
    uncorrected index would call that a match.
    """
    shared = [uid for uid in a if uid in b]
    if len(shared) < 2:
        return 0.0
    contingency: dict[tuple[str, str], int] = defaultdict(int)
    rows: Counter[str] = Counter()
    cols: Counter[str] = Counter()
    for uid in shared:
        contingency[(a[uid], b[uid])] += 1
        rows[a[uid]] += 1
        cols[b[uid]] += 1
    n = len(shared)

    def choose2(value: int) -> float:
        return value * (value - 1) / 2.0

    index = sum(choose2(v) for v in contingency.values())
    row_sum = sum(choose2(v) for v in rows.values())
    col_sum = sum(choose2(v) for v in cols.values())
    total = choose2(n)
    expected = row_sum * col_sum / total if total else 0.0
    maximum = (row_sum + col_sum) / 2.0
    if maximum == expected:
        return 0.0
    return (index - expected) / (maximum - expected)


def stability(
    snapshot: ConnectomeSnapshot,
    *,
    rounds: int = 3,
    drop_edges: float = 0.1,
    repeats: int = 3,
    seed_labels: str = "cell_class",
) -> dict[str, Any]:
    """Retype with a tenth of the wiring missing and see what survives."""
    full = refine_types(snapshot, rounds=rounds, seed_labels=seed_labels)
    scores: list[float] = []
    partial_counts: list[int] = []
    for i in range(repeats):
        partial = refine_types(
            snapshot,
            rounds=rounds,
            seed_labels=seed_labels,
            drop_edges=drop_edges,
            seed=i + 1,
        )
        scores.append(adjusted_rand_index(full.labels, partial.labels))
        partial_counts.append(partial.type_count)
    return {
        "types": full.type_count,
        "drop_edges": drop_edges,
        "repeats": repeats,
        "adjusted_rand_mean": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "adjusted_rand_min": round(min(scores), 4) if scores else 0.0,
        "types_under_dropout_mean": (
            round(sum(partial_counts) / len(partial_counts), 1) if partial_counts else 0.0
        ),
        **full.summary(),
    }


def compare_typings(
    left: Typing,
    right: Typing,
    *,
    left_units: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare a typing against the same procedure run on another individual.

    Cells present in only one of the two are counted rather than dropped
    silently, because a type that vanished because its cells did is a different
    result from a type that vanished because the wiring around it moved.
    """
    shared = set(left.labels) & set(right.labels)
    only_left = set(left.labels) - shared
    only_right = set(right.labels) - shared
    ari = adjusted_rand_index(
        {uid: left.labels[uid] for uid in shared},
        {uid: right.labels[uid] for uid in shared},
    )
    left_groups: dict[str, set[str]] = defaultdict(set)
    right_groups: dict[str, set[str]] = defaultdict(set)
    for uid in shared:
        left_groups[left.labels[uid]].add(uid)
        right_groups[right.labels[uid]].add(uid)
    right_by_cell = {uid: label for label, group in right_groups.items() for uid in group}
    preserved = 0
    for group in left_groups.values():
        if len(group) < 2:
            continue
        targets = {right_by_cell.get(uid) for uid in group}
        if len(targets) == 1:
            preserved += 1
    multi_left = sum(1 for group in left_groups.values() if len(group) >= 2)
    return {
        "shared_cells": len(shared),
        "only_in_left": len(only_left),
        "only_in_right": len(only_right),
        "adjusted_rand": round(ari, 4),
        "multi_member_types_left": multi_left,
        "types_preserved_intact": preserved,
        "preserved_share": round(preserved / multi_left, 4) if multi_left else 0.0,
    }

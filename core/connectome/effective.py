"""core/connectome/effective.py — what a connection does, in the state she is in.

The structural connectome says what can happen. It is the same graph whether she
is holding a conversation, planning, reading code or refusing something, and
that cannot be the whole story, because those are not the same computation.

The object worth having is the effective connectome:

    w_ij^eff(c) = Effect[ do(i), j | c ]

the influence cell *i* has on cell *j* under cognitive condition *c*. Same
anatomy, different active circuits, which is how biological brains work and is
what the recorder's ``condition`` field was for.

Three ways to measure it, and they are not interchangeable. Each carries its
grade and nothing lets a weaker one be reported as a stronger one:

``predictive``
    Does knowing *i*'s recent past improve the prediction of *j*'s next value
    beyond *j*'s own past? This is Granger's question and it is a statement
    about prediction, not about causation: a third cell driving both produces it
    exactly. Scored against a null that keeps *i*'s distribution and destroys its
    timing, so an edge that scores here is at least doing better than a cell of
    the same shape and no relationship.
``model``
    Remove *i* from the graph and propagate again. The number is exact and it is
    exact about a model of propagation, not about Aura. Useful for ranking which
    removals would matter if the model were right, and worth nothing as evidence
    that it is.
``interventional``
    Actually disable *i* while she works, and measure *j*. This is the only one
    that answers the question the notation asks, and it is the expensive one.

The condition is not decoration. An effective connectome averaged over every
condition is the average of circuits that are never active together, and it
describes none of them.
"""

from __future__ import annotations

import logging
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .types import ConnectomeSnapshot, EdgeKind

logger = logging.getLogger("Aura.Connectome.Effective")

__all__ = [
    "Grade",
    "EffectiveEdge",
    "EffectiveConnectome",
    "predictive_influence",
    "model_influence",
    "compare_conditions",
    "MIN_FRAMES_PER_CONDITION",
]

#: Below this many frames a condition cannot support a lagged regression with a
#: null beside it, and the report says the condition was skipped rather than
#: publishing a number from thirty samples.
MIN_FRAMES_PER_CONDITION: int = 120


class Grade(StrEnum):
    """What kind of claim a weight supports."""

    PREDICTIVE = "predictive"
    MODEL = "model"
    INTERVENTIONAL = "interventional"

    @property
    def licenses(self) -> str:
        return {
            Grade.PREDICTIVE: "that one cell's past helps predict another's future",
            Grade.MODEL: "that a model of propagation says removing one changes the other",
            Grade.INTERVENTIONAL: "that disabling one changed the other",
        }[self]


@dataclass(frozen=True)
class EffectiveEdge:
    """One influence, in one condition, at one grade."""

    pre: str
    post: str
    condition: str
    weight: float
    null_mean: float
    null_spread: float
    z: float
    samples: int
    grade: Grade

    @property
    def survives_null(self) -> bool:
        return self.z >= 3.0

    def as_json(self) -> dict[str, Any]:
        return {
            "pre": self.pre,
            "post": self.post,
            "condition": self.condition,
            "weight": round(self.weight, 6),
            "z": round(self.z, 3),
            "samples": self.samples,
            "grade": str(self.grade),
            "survives_null": self.survives_null,
        }


@dataclass
class EffectiveConnectome:
    """The effective graph for one condition."""

    condition: str
    grade: Grade
    edges: dict[tuple[str, str], EffectiveEdge] = field(default_factory=dict)
    frames: int = 0
    skipped: str = ""

    def surviving(self) -> dict[tuple[str, str], EffectiveEdge]:
        return {pair: edge for pair, edge in self.edges.items() if edge.survives_null}

    def summary(self) -> dict[str, Any]:
        surviving = self.surviving()
        weights = [edge.weight for edge in surviving.values()]
        return {
            "condition": self.condition,
            "grade": str(self.grade),
            "licenses": self.grade.licenses,
            "frames": self.frames,
            "edges_measured": len(self.edges),
            "edges_surviving_null": len(surviving),
            "surviving_share": round(len(surviving) / len(self.edges), 4)
            if self.edges
            else 0.0,
            "mean_weight_surviving": round(statistics.fmean(weights), 5) if weights else 0.0,
            "skipped": self.skipped,
        }

    def strongest(self, snapshot: ConnectomeSnapshot, limit: int = 15) -> list[dict[str, Any]]:
        rows = sorted(self.surviving().values(), key=lambda edge: -edge.weight)[:limit]
        return [
            {
                **edge.as_json(),
                "pre_name": snapshot.units[edge.pre].name
                if edge.pre in snapshot.units
                else edge.pre,
                "post_name": snapshot.units[edge.post].name
                if edge.post in snapshot.units
                else edge.post,
            }
            for edge in rows
        ]


def _lagged(series: Any, lags: int) -> Any:
    """Stack a series' recent past as columns, one row per predictable step."""
    import numpy as np

    length = series.shape[0]
    if length <= lags:
        return np.zeros((0, lags), dtype=np.float64)
    return np.stack([series[lag : length - lags + lag] for lag in range(lags)], axis=1)


def _residual(design: Any, target: Any, ridge: float) -> float:
    import numpy as np

    if design.shape[0] == 0:
        return float("inf")
    gram = design.T @ design
    gram[np.diag_indices_from(gram)] += ridge
    try:
        weights = np.linalg.solve(gram, design.T @ target)
    except np.linalg.LinAlgError:
        return float("inf")
    residual = target - design @ weights
    return float(residual @ residual)


def predictive_influence(
    trace: Any,
    snapshot: ConnectomeSnapshot,
    condition: str,
    *,
    lags: int = 3,
    ridge: float = 1e-3,
    nulls: int = 8,
    seed: int = 0,
    max_edges: int = 40_000,
    only_cells: Sequence[str] | None = None,
) -> EffectiveConnectome:
    """Does knowing the source's past help predict the target's next value?

    Granger's test, per edge, inside one condition. The null keeps the source's
    values and destroys its timing by rotating the series, which is the right
    shuffle here: a plain permutation would also destroy the source's own
    autocorrelation and make almost anything look significant against it.

    Only edges that exist in the structural graph are tested. Testing every pair
    would be a different and much larger experiment, and it would need a
    correction for forty million comparisons that would leave nothing standing.

    ``only_cells`` narrows it further to edges with both ends in a named set,
    which is what a question about a particular circuit wants: measuring sixty
    thousand edges to ask about seven stations spends nine regressions each on
    the fifty-nine thousand that cannot answer.
    """
    import numpy as np

    matrix = trace.matrix()
    conditions = list(trace.conditions)
    rows = [i for i, name in enumerate(conditions) if name == condition]
    if len(rows) < MIN_FRAMES_PER_CONDITION:
        return EffectiveConnectome(
            condition=condition,
            grade=Grade.PREDICTIVE,
            frames=len(rows),
            skipped=(
                f"{len(rows)} frames is below the {MIN_FRAMES_PER_CONDITION} a lagged "
                "regression needs beside a null"
            ),
        )
    activity = np.asarray(matrix[rows], dtype=np.float64)
    index = {uid: i for i, uid in enumerate(trace.uids)}
    rng = np.random.default_rng(seed)

    result = EffectiveConnectome(
        condition=condition, grade=Grade.PREDICTIVE, frames=len(rows)
    )
    keep = set(only_cells) if only_cells is not None else None
    tested = 0
    for connection in snapshot.edges(EdgeKind.DRIVE):
        if tested >= max_edges:
            break
        if keep is not None and (connection.pre not in keep or connection.post not in keep):
            continue
        source = index.get(connection.pre)
        target = index.get(connection.post)
        if source is None or target is None or source == target:
            continue
        y = activity[lags:, target]
        if y.std() <= 0:
            continue
        x_source = activity[:, source]
        if x_source.std() <= 0:
            continue
        own = _lagged(activity[:, target], lags)
        other = _lagged(x_source, lags)
        if own.shape[0] != y.shape[0] or other.shape[0] != y.shape[0]:
            continue
        bias = np.ones((y.shape[0], 1))
        restricted = _residual(np.hstack([own, bias]), y, ridge)
        full = _residual(np.hstack([own, other, bias]), y, ridge)
        if not np.isfinite(restricted) or restricted <= 0:
            continue
        weight = max(0.0, (restricted - full) / restricted)

        null_weights = []
        for _ in range(nulls):
            shift = int(rng.integers(lags + 1, max(lags + 2, len(x_source) - lags)))
            rotated = np.roll(x_source, shift)
            rotated_lagged = _lagged(rotated, lags)
            null_full = _residual(np.hstack([own, rotated_lagged, bias]), y, ridge)
            null_weights.append(max(0.0, (restricted - null_full) / restricted))
        null_mean = float(np.mean(null_weights)) if null_weights else 0.0
        null_spread = float(np.std(null_weights)) if len(null_weights) > 1 else 0.0
        z = (weight - null_mean) / null_spread if null_spread > 0 else 0.0
        tested += 1
        result.edges[(connection.pre, connection.post)] = EffectiveEdge(
            pre=connection.pre,
            post=connection.post,
            condition=condition,
            weight=weight,
            null_mean=null_mean,
            null_spread=null_spread,
            z=z,
            samples=int(y.shape[0]),
            grade=Grade.PREDICTIVE,
        )
    return result


def model_influence(
    snapshot: ConnectomeSnapshot,
    sources: Sequence[str],
    *,
    steps: int = 4,
    decay: float = 0.6,
    limit: int = 400,
) -> EffectiveConnectome:
    """Remove a cell from the graph, propagate again, and take the difference.

    Exact, and exact about a model. Activation spreads along contact-weighted
    edges with a decay per step, so what comes out ranks which removals would
    change the most if propagation worked that way. It is not evidence that it
    does, and the grade says so.
    """
    import numpy as np
    from scipy import sparse

    nodes = sorted(snapshot.units)
    index = {uid: i for i, uid in enumerate(nodes)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for connection in snapshot.edges(EdgeKind.DRIVE):
        pre = index.get(connection.pre)
        post = index.get(connection.post)
        if pre is None or post is None or pre == post:
            continue
        rows.append(post)
        cols.append(pre)
        data.append(float(connection.contacts))
    size = len(nodes)
    if not rows:
        return EffectiveConnectome(condition="model", grade=Grade.MODEL, skipped="no edges")
    adjacency = sparse.csr_matrix((data, (rows, cols)), shape=(size, size))
    column_sums = np.asarray(adjacency.sum(axis=0)).ravel()
    column_sums[column_sums == 0] = 1.0
    normalised = adjacency @ sparse.diags(1.0 / column_sums)

    def _propagate(seed_vector: Any, silenced: int | None) -> Any:
        state = seed_vector.copy()
        total = seed_vector.copy()
        for _ in range(steps):
            if silenced is not None:
                state[silenced] = 0.0
            state = decay * (normalised @ state)
            total = total + state
        return total

    result = EffectiveConnectome(condition="model", grade=Grade.MODEL)
    seeds = [uid for uid in sources if uid in index][:limit]
    for uid in seeds:
        seed_vector = np.zeros(size)
        seed_vector[index[uid]] = 1.0
        intact = _propagate(seed_vector, None)
        for target_uid in nodes:
            target = index[target_uid]
            if target == index[uid] or intact[target] <= 0:
                continue
            damaged = _propagate(seed_vector, index[uid])
            effect = float(intact[target] - damaged[target])
            if effect <= 0:
                continue
            result.edges[(uid, target_uid)] = EffectiveEdge(
                pre=uid,
                post=target_uid,
                condition="model",
                weight=effect,
                null_mean=0.0,
                null_spread=0.0,
                z=float("inf"),
                samples=steps,
                grade=Grade.MODEL,
            )
            break
    return result


def compare_conditions(
    left: EffectiveConnectome,
    right: EffectiveConnectome,
    *,
    snapshot: ConnectomeSnapshot | None = None,
    limit: int = 12,
) -> dict[str, Any]:
    """What one cognitive state recruits that another does not.

    Two effective graphs over the same anatomy are comparable edge by edge. The
    correlation says how much of the circuitry is shared; the edges that survive
    the null in one and not the other are the ones that state recruits.
    """
    shared = sorted(set(left.edges) & set(right.edges))
    if len(shared) < 8:
        return {
            "left": left.condition,
            "right": right.condition,
            "shared_edges": len(shared),
            "verdict": "too few edges measured in both conditions to compare",
        }
    a = [left.edges[pair].weight for pair in shared]
    b = [right.edges[pair].weight for pair in shared]
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b, strict=True))
    denominator = (
        sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b)
    ) ** 0.5
    correlation = numerator / denominator if denominator else 0.0
    only_left = sorted(
        (pair for pair in shared if left.edges[pair].survives_null
         and not right.edges[pair].survives_null),
        key=lambda pair: -left.edges[pair].weight,
    )
    only_right = sorted(
        (pair for pair in shared if right.edges[pair].survives_null
         and not left.edges[pair].survives_null),
        key=lambda pair: -right.edges[pair].weight,
    )
    def _name(uid: str) -> str:
        if snapshot is not None and uid in snapshot.units:
            return snapshot.units[uid].name
        return uid

    return {
        "left": left.condition,
        "right": right.condition,
        "shared_edges": len(shared),
        "correlation": round(correlation, 4),
        "surviving_in_left_only": len(only_left),
        "surviving_in_right_only": len(only_right),
        "recruited_by_left": [
            f"{_name(pair[0])} -> {_name(pair[1])}" for pair in only_left[:limit]
        ],
        "recruited_by_right": [
            f"{_name(pair[0])} -> {_name(pair[1])}" for pair in only_right[:limit]
        ],
        "verdict": (
            "the two states run the same circuits"
            if correlation >= 0.9 and not only_left and not only_right
            else f"{len(only_left) + len(only_right)} edges are active in one state and not "
            "the other"
        ),
    }

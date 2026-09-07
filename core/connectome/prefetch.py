"""core/connectome/prefetch.py — knowing which cells are about to run, and warming them.

The forecasting benchmark asks what value a cell will take next and finds, on
this recording, that the recent past barely says. That is a real answer to a
real question and it is the wrong question for doing anything useful, because
nothing here needs to know a cell's activation to a decimal place. It needs to
know *which cells are about to run*, which is a set, and a set is a much easier
thing to predict than a number.

The connectome answers it directly. A cell fires when something calls it, and
the things that can call it are its presynaptic partners. So the cells likely to
run in the next moment are the ones downstream of the cells running now.

Whether that beats knowing nothing is the whole question, and it has two nulls
that both have to lose:

``frequent``
    Predict the cells that run most often. This is what any cache does by
    default and it is a hard baseline, because a small set of cells really does
    run almost all the time.
``persistent``
    Predict the cells that just ran. On a smooth signal this is very hard to
    beat and it needs no connectome at all.

If neither is beaten, the connectome adds nothing to knowing what is hot, and
that is worth finding out before anything is wired to it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .types import ConnectomeSnapshot, EdgeKind

logger = logging.getLogger("Aura.Connectome.Prefetch")

__all__ = [
    "PrefetchPlan",
    "weighted_next_active",
    "SetPrediction",
    "downstream_of",
    "predict_next_active",
    "evaluate_prefetch",
    "warm",
]


@dataclass(frozen=True)
class SetPrediction:
    """How well one rule predicted the set of cells that ran next."""

    rule: str
    precision: float
    recall: float
    f1: float
    predicted: float
    actual: float
    frames: int

    def as_json(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "precision": round(self.precision, 5),
            "recall": round(self.recall, 5),
            "f1": round(self.f1, 5),
            "mean_predicted": round(self.predicted, 2),
            "mean_actual": round(self.actual, 2),
            "frames": self.frames,
        }


def downstream_of(
    snapshot: ConnectomeSnapshot,
    cells: Iterable[str],
    *,
    hops: int = 1,
    limit: int = 4_000,
) -> set[str]:
    """Everything reachable from a set of cells within a few hops.

    One hop is what a single frame can carry. More hops widen the prediction and
    make it easier to be right and less useful, so the number is a parameter and
    the evaluation reports what it was set to.
    """
    adjacency: dict[str, set[str]] = {}
    for connection in snapshot.connections.values():
        if connection.kind is EdgeKind.DRIVE:
            adjacency.setdefault(connection.pre, set()).add(connection.post)
    frontier = set(cells)
    reached: set[str] = set()
    for _ in range(max(1, hops)):
        nxt: set[str] = set()
        for cell in frontier:
            nxt |= adjacency.get(cell, set())
        nxt -= reached
        reached |= nxt
        frontier = nxt
        if len(reached) >= limit:
            break
    return reached


def predict_next_active(
    snapshot: ConnectomeSnapshot,
    active_now: Sequence[str],
    *,
    hops: int = 1,
    also_persist: bool = True,
    limit: int = 4_000,
) -> set[str]:
    """The cells to warm: what the active ones can reach, and themselves.

    Keeping the currently active cells in the prediction is not padding. A cell
    that ran this frame is very likely to run again, and a rule that drops them
    to look more selective is trading recall for an appearance.
    """
    predicted = downstream_of(snapshot, active_now, hops=hops, limit=limit)
    if also_persist:
        predicted |= set(active_now)
    return predicted


def weighted_next_active(
    snapshot: ConnectomeSnapshot,
    active_now: Sequence[str],
    *,
    budget: int,
    hops: int = 1,
) -> set[str]:
    """The strongest downstream neighbours, ranked by how many call sites join them.

    The unweighted rule treats a pair joined by one call site and a pair joined
    by twenty as the same prediction, and H01's whole finding is that they are
    not the same connection. Ranking by contact count and stopping at a budget
    lets the rule be compared against a baseline of the same size, which is the
    only comparison that means anything when precision is the metric.
    """
    weights: dict[str, int] = {}
    adjacency: dict[str, list[tuple[str, int]]] = {}
    for connection in snapshot.connections.values():
        if connection.kind is EdgeKind.DRIVE:
            adjacency.setdefault(connection.pre, []).append(
                (connection.post, connection.contacts)
            )
    frontier = list(active_now)
    for _ in range(max(1, hops)):
        nxt: list[str] = []
        for cell in frontier:
            for post, contacts in adjacency.get(cell, ()):
                weights[post] = weights.get(post, 0) + contacts
                nxt.append(post)
        frontier = nxt
        if len(weights) > budget * 8:
            break
    ranked = sorted(weights.items(), key=lambda item: (-item[1], item[0]))
    chosen = {uid for uid, _ in ranked[: max(0, budget - len(set(active_now)))]}
    return chosen | set(active_now)


def _score(rule: str, predictions: Sequence[set[str]], truths: Sequence[set[str]]) -> SetPrediction:
    hits = 0
    predicted_total = 0
    actual_total = 0
    for predicted, actual in zip(predictions, truths, strict=True):
        hits += len(predicted & actual)
        predicted_total += len(predicted)
        actual_total += len(actual)
    precision = hits / predicted_total if predicted_total else 0.0
    recall = hits / actual_total if actual_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    frames = len(predictions)
    return SetPrediction(
        rule=rule,
        precision=precision,
        recall=recall,
        f1=f1,
        predicted=predicted_total / frames if frames else 0.0,
        actual=actual_total / frames if frames else 0.0,
        frames=frames,
    )


@dataclass
class PrefetchPlan:
    """Every rule scored on the same frames, with the verdict between them."""

    rules: list[SetPrediction] = field(default_factory=list)
    hops: int = 1

    def best(self) -> SetPrediction | None:
        return max(self.rules, key=lambda r: r.f1) if self.rules else None

    def as_json(self) -> dict[str, Any]:
        best = self.best()
        structural = [r for r in self.rules if r.rule.startswith("connectome")]
        nulls = [r for r in self.rules if not r.rule.startswith("connectome")]
        best_structural = max(structural, key=lambda r: r.f1) if structural else None
        beaten = (
            best_structural is not None
            and nulls
            and all(best_structural.f1 > null.f1 for null in nulls)
        )
        return {
            "hops": self.hops,
            "rules": [r.as_json() for r in self.rules],
            "best": best.rule if best else None,
            "verdict": (
                "the connectome predicts what runs next better than knowing what is hot"
                if beaten
                else f"the connectome does not beat its nulls; {best.rule if best else 'none'} wins"
            ),
        }


def evaluate_prefetch(
    trace: Any,
    snapshot: ConnectomeSnapshot,
    *,
    hops: int = 1,
    frequent_size: int | None = None,
    max_frames: int = 600,
) -> PrefetchPlan:
    """Score the connectome rule against knowing what is hot and what just ran.

    ``frequent_size`` defaults to the mean number of cells the connectome rule
    predicts, so the baseline is allowed to name as many cells as the rule it is
    competing with. A precision comparison between a rule naming forty cells and
    one naming four thousand is not a comparison.
    """
    import numpy as np

    matrix = trace.matrix()
    if matrix.size == 0 or matrix.shape[0] < 3:
        return PrefetchPlan(hops=hops)
    uids = list(trace.uids)
    frames = min(max_frames, matrix.shape[0] - 1)
    step = max(1, (matrix.shape[0] - 1) // frames)
    indices = list(range(0, matrix.shape[0] - 1, step))[:frames]

    totals = matrix.sum(axis=0)
    ranking = [uids[i] for i in np.argsort(-totals) if totals[i] > 0]

    active_sets: dict[int, set[str]] = {}

    def _active(row: int) -> set[str]:
        cached = active_sets.get(row)
        if cached is None:
            cached = {uids[i] for i in np.nonzero(matrix[row])[0]}
            active_sets[row] = cached
        return cached

    connectome_predictions: list[set[str]] = []
    weighted_predictions: list[set[str]] = []
    persistent_predictions: list[set[str]] = []
    truths: list[set[str]] = []
    for row in indices:
        now = _active(row)
        truths.append(_active(row + 1))
        connectome_predictions.append(
            predict_next_active(snapshot, sorted(now), hops=hops)
        )
        weighted_predictions.append(
            weighted_next_active(
                snapshot, sorted(now), budget=max(1, int(len(now) * 1.25)), hops=hops
            )
        )
        persistent_predictions.append(set(now))

    size = frequent_size or int(
        sum(len(p) for p in connectome_predictions) / max(1, len(connectome_predictions))
    )
    frequent = set(ranking[: max(1, size)])
    frequent_predictions = [set(frequent) for _ in indices]

    plan = PrefetchPlan(hops=hops)
    plan.rules.append(_score("connectome", connectome_predictions, truths))
    plan.rules.append(_score("connectome_weighted", weighted_predictions, truths))
    plan.rules.append(_score("frequent", frequent_predictions, truths))
    plan.rules.append(_score("persistent", persistent_predictions, truths))
    return plan


def warm(
    snapshot: ConnectomeSnapshot,
    active_now: Sequence[str],
    warmer: Callable[[str], None],
    *,
    hops: int = 1,
    budget: int = 64,
) -> list[str]:
    """Warm what is about to run, most-connected first, within a budget.

    A warmer that raises takes itself out rather than stopping the warm-up: this
    runs ahead of work that has not been asked for yet, and it must never be the
    reason the work that was asked for fails.
    """
    predicted = predict_next_active(snapshot, active_now, hops=hops)
    in_degree: dict[str, int] = {}
    for connection in snapshot.connections.values():
        if connection.kind is EdgeKind.DRIVE and connection.post in predicted:
            in_degree[connection.post] = in_degree.get(connection.post, 0) + 1
    ordered = sorted(predicted, key=lambda uid: (-in_degree.get(uid, 0), uid))[:budget]
    warmed: list[str] = []
    for uid in ordered:
        try:
            warmer(uid)
        except (RuntimeError, OSError, ValueError, KeyError, AttributeError) as exc:
            logger.debug("warming %s was refused: %s", uid, exc)
            continue
        warmed.append(uid)
    return warmed

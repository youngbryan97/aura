"""A quick judgement learned from slow ones: what her first look misses, fitted to what her longer looks found.

Anything she judges two ways, at a glance and after thinking it through, gives her pairs to learn from: the thing, what
the glance said, and what the thinking found. Where the two differ, the glance missed something, and what it missed can
be learned, so that next time the glance says what thinking would have found without the thinking. A player's sense
of a position grows this way, and so does a craftsman's eye.

The glance here is a weighted sum (each part of the thing counted on its own); what it misses is how the parts go
together, so what is learned is one hidden layer over which parts are present. A thing is the set of its parts present,
as indices; nothing here knows what the parts are. The fit is to the chance of the outcome each judgement implies, on
the scale her own outcomes showed, so a large difference between two judgements of a thing already lost counts for
little. How wide the layer is, and how long it is fitted, are chosen by how well it foretells what it was not fitted
on: things from separate occasions are held back, and the narrowest layer that foretells them best is kept.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["LearnedJudgement", "learned_from_slow_ones"]

#: Of the occasions the things came from, the share held back to judge a fit by.
HELD_BACK_SHARE = 0.1
#: Widths of the hidden layer tried, narrowest first: each twice the last.
WIDTHS = (8, 16, 32, 64)
#: Passes over what it is fitted on without the held-back loss improving, before a fit stops.
PATIENCE = 2
#: Passes at most.
PASSES_AT_MOST = 40
#: The fit's learning rate and its pull toward nothing (Adam, with its usual moments).
RATE = 1e-3
PULL = 1e-5
#: Things fitted together in one step.
BATCH = 1024
#: The unit the layer's output is learned in, in the judgement's own units: the layer is fitted on numbers near one.
UNIT = 100.0


@dataclass
class LearnedJudgement:
    """One hidden layer: a row of weights into it for each part, its biases, and its output's weights and bias; the
    output is in the judgement's own units."""

    first: np.ndarray
    first_bias: np.ndarray
    out: np.ndarray
    out_bias: float
    held_out_before: float
    held_out_after: float

    def of(self, parts: list[np.ndarray]) -> np.ndarray:
        """What the layer adds to the first look, for each thing given as the indices of its parts."""
        hidden = np.stack([self.first_bias + self.first[p[p >= 0]].sum(axis=0) for p in parts])
        return np.maximum(hidden, 0.0) @ self.out + self.out_bias


def _dense(parts: np.ndarray, width: int) -> np.ndarray:
    present = np.zeros((len(parts), width), dtype=np.float32)
    rows, columns = np.nonzero(parts >= 0)
    present[rows, parts[rows, columns]] = 1.0
    return present


def _fit(parts: np.ndarray, glance: np.ndarray, target: np.ndarray, scale: float, fitted_on: np.ndarray,
         held: np.ndarray, inputs: int, hidden: int, rng: np.random.Generator) -> tuple[float, list[np.ndarray]]:
    weights = [rng.normal(0, 0.05, (inputs, hidden)).astype(np.float32), np.zeros(hidden, dtype=np.float32),
               rng.normal(0, 0.05, hidden).astype(np.float32), np.zeros(1, dtype=np.float32)]
    first_moment = [np.zeros_like(w) for w in weights]
    second_moment = [np.zeros_like(w) for w in weights]

    def chance(index: np.ndarray, w: list[np.ndarray]) -> tuple[np.ndarray, ...]:
        present = _dense(parts[index], inputs)
        inner = present @ w[0] + w[1]
        hidden_out = np.maximum(inner, 0)
        added = (hidden_out @ w[2] + w[3][0]) * UNIT
        return present, inner, hidden_out, 1 / (1 + np.exp(-(glance[index] + added) / scale))

    def loss(index: np.ndarray, w: list[np.ndarray]) -> float:
        total = 0.0
        for start in range(0, len(index), 8192):
            part = index[start:start + 8192]
            total += float(np.sum((chance(part, w)[3] - target[part]) ** 2))
        return total / max(1, len(index))

    best, best_weights, waited, step = loss(held, weights), [w.copy() for w in weights], 0, 0
    for _ in range(PASSES_AT_MOST):
        order = rng.permutation(fitted_on)
        for start in range(0, len(order), BATCH):
            index = order[start:start + BATCH]
            present, inner, hidden_out, p = chance(index, weights)
            g = (2 * (p - target[index]) * p * (1 - p) / scale * UNIT / len(index)).astype(np.float32)
            into_hidden = (g[:, None] * weights[2][None, :]) * (inner > 0)
            grads = [present.T @ into_hidden, into_hidden.sum(axis=0), hidden_out.T @ g, np.array([g.sum()])]
            step += 1
            for k, (w, grad) in enumerate(zip(weights, grads)):
                grad = grad + PULL * w
                first_moment[k] = 0.9 * first_moment[k] + 0.1 * grad
                second_moment[k] = 0.999 * second_moment[k] + 0.001 * grad * grad
                w -= (RATE * (first_moment[k] / (1 - 0.9 ** step))
                      / (np.sqrt(second_moment[k] / (1 - 0.999 ** step)) + 1e-8)).astype(np.float32)
        now = loss(held, weights)
        if now < best:
            best, best_weights, waited = now, [w.copy() for w in weights], 0
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    return best, best_weights


def learned_from_slow_ones(parts: np.ndarray, glance: np.ndarray, slow: np.ndarray, occasions: np.ndarray, *,
                           inputs: int, scale: float, seed: int = 0) -> LearnedJudgement | None:
    """What a glance misses, learned: ``parts`` holds each thing's present parts as indices below ``inputs`` (padded
    with -1), ``glance`` and ``slow`` its two judgements in one unit, ``occasions`` which occasion it came from, and
    ``scale`` the judgement that reads as about three chances in four. None where too few occasions to hold any back,
    or where no layer foretells what it was not fitted on better than the glance alone."""
    occasions = np.asarray(occasions)
    distinct = np.unique(occasions)
    if len(distinct) < 2 or len(parts) < 2:
        return None
    rng = np.random.default_rng(seed)
    held_occasions = set(rng.choice(distinct, size=max(1, int(len(distinct) * HELD_BACK_SHARE)), replace=False).tolist())
    is_held = np.array([o in held_occasions for o in occasions.tolist()])
    held, fitted_on = np.where(is_held)[0], np.where(~is_held)[0]
    if not len(fitted_on):
        return None
    glance = np.asarray(glance, dtype=np.float64)
    target = 1 / (1 + np.exp(-np.clip(np.asarray(slow, dtype=np.float64), -30 * scale, 30 * scale) / scale))
    before = float(np.mean((1 / (1 + np.exp(-glance[held] / scale)) - target[held]) ** 2))
    best: tuple[float, list[np.ndarray]] | None = None
    for hidden in WIDTHS:
        tried = _fit(parts, glance, target, scale, fitted_on, held, inputs, hidden, rng)
        if best is not None and tried[0] >= best[0]:
            # A wider layer that foretells no better is not kept, and no wider one is tried.
            break
        best = tried
    if best is None or not best[0] < before:
        return None
    w = best[1]
    # Kept in the judgement's units: the first layer and its bias scaled up, so the layer's output needs no scaling.
    return LearnedJudgement(first=w[0] * UNIT, first_bias=w[1] * UNIT, out=w[2].copy(), out_bias=float(w[3][0] * UNIT),
                            held_out_before=before, held_out_after=float(best[0]))

"""What changes slowly and decides how things come out, found in what was seen and how each time it came out.

A judgement made at one moment sees as far as it looks. What decides how things come out long after is often what
changes little from one moment to the next: in a game, the shape of the pieces rather than the next take; in anything
that unfolds over steps, its slow quantities rather than its quick ones. Here no one says what those are. Given
episodes, each a sequence of what was seen (one row of numbers a step) and how it came out, the slow directions are
found in the rows themselves: the combinations of what is seen that change least from one step to the next for how
much they vary across steps (slow feature analysis). Only those slower than noise are kept: one that changes as much
between neighbouring steps as across the whole of an episode is no slower than chance.

Which of them decides, and how much, is learned from how the episodes came out, beyond what the judgement already held
saw: its estimate at each step, in whatever units it is given, is first read against how things came out (a scale and
a level), then held fixed, and only what the slow directions add is fitted. What is learned is given back in the held
judgement's own units, to be added to it. How an episode came out
is any number the world gave back, a win or a loss, a score, a reward, a standing; it is read as its rank among the
outcomes seen, so nothing is assumed of its scale and no goal need be a win. What is learned is tried on episodes held
back from it, and nothing is learned where it does not foretell them better than the judgement held alone, by more
than chance: each episode's gain, and their mean above its standard error.

Nothing here knows a game, a world or a goal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["SlowAndDeciding", "learned", "slow_directions"]

#: How far each fit is held back toward nothing, as rehearsal's fits are.
HELD_BACK = 1.0
#: One episode in this many is held back from the fit, to judge what it foretells.
HELD_OUT_EVERY = 5


@dataclass
class SlowAndDeciding:
    """The slow directions found, how slow each is (its change between steps over its spread; under one is slower than
    noise), what each is worth to how things come out (log-odds per unit), and how episodes held back were foretold
    without and with it."""

    directions: np.ndarray
    slowness: np.ndarray
    worth: np.ndarray
    held_out: tuple[float, float]
    #: How many log-odds one unit of the held judgement was read as: zero where none was given.
    per_unit_held: float = 0.0

    def weights(self) -> np.ndarray:
        """The whole of what was learned as one weight on each number seen: log-odds per unit."""
        return self.worth @ self.directions

    def in_units_held(self) -> np.ndarray | None:
        """What was learned in the held judgement's own units, to be added to it; None where it gave none."""
        return self.weights() / self.per_unit_held if self.per_unit_held > 0 else None

    def value(self, row: np.ndarray) -> float:
        return float(np.asarray(row, dtype=np.float64) @ self.weights())


def _as_ranks(ends: np.ndarray) -> np.ndarray:
    """How each episode came out, as its rank among them between nought and one: ties share their middle rank."""
    order = np.argsort(ends, kind="mergesort")
    ranks = np.empty(len(ends), dtype=np.float64)
    ranks[order] = np.arange(len(ends), dtype=np.float64)
    for value in np.unique(ends):
        alike = ends == value
        ranks[alike] = ranks[alike].mean()
    return ranks / max(1, len(ends) - 1)


def slow_directions(episodes: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """The directions in which the rows change least from one step to the next for how much they vary, slowest first,
    with how slow each is; only those slower than noise. A direction's slowness is the spread of its changes between
    neighbouring steps over twice its spread across them: about one where the steps are as alike as strangers."""
    rows = np.concatenate([np.asarray(e, dtype=np.float64) for e in episodes if len(e)])
    steps = [np.diff(np.asarray(e, dtype=np.float64), axis=0) for e in episodes if len(e) > 1]
    if not steps or rows.shape[0] < 3:
        return np.zeros((0, rows.shape[1] if rows.ndim == 2 else 0)), np.zeros(0)
    changes = np.concatenate(steps)
    spread = np.cov(rows, rowvar=False).reshape(rows.shape[1], rows.shape[1])
    change = np.cov(changes, rowvar=False).reshape(rows.shape[1], rows.shape[1]) / 2.0
    # Held back toward nothing by the spread's own scale, so a number that never varies is no direction.
    floor = 1e-6 * max(float(np.trace(spread)) / max(1, spread.shape[0]), 1e-12)
    values, vectors = np.linalg.eigh(spread + floor * np.eye(spread.shape[0]))
    kept = values > 10 * floor
    if not kept.any():
        return np.zeros((0, rows.shape[1])), np.zeros(0)
    whitening = vectors[:, kept] / np.sqrt(values[kept])
    slowness, turned = np.linalg.eigh(whitening.T @ change @ whitening)
    directions = (whitening @ turned).T
    slower = slowness < 1.0
    return directions[slower], slowness[slower]


def _fitted(z: np.ndarray, offset: np.ndarray, y: np.ndarray, *, held_back: float = HELD_BACK) -> np.ndarray | None:
    theta = np.zeros(z.shape[1])
    penalty = held_back * np.eye(z.shape[1]) + 1e-9 * np.eye(z.shape[1])
    for _ in range(30):
        p = 1.0 / (1.0 + np.exp(-np.clip(offset + z @ theta, -30, 30)))
        try:
            step = np.linalg.solve((z * (p * (1 - p))[:, None]).T @ z + penalty, z.T @ (p - y) + penalty @ theta)
        # not a failure: rows too alike to fit say nothing yet.
        except np.linalg.LinAlgError:
            return None
        theta -= step
        if float(np.abs(step).max()) < 1e-7:
            break
    return theta if np.all(np.isfinite(theta)) else None


def _read_against(held: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """The held judgement read as log-odds of how things came out: a scale and a level."""
    spread = float(held.std())
    x = np.column_stack([np.ones_like(held), held / spread if spread > 0 else np.zeros_like(held)])
    fit = _fitted(x, np.zeros(len(y)), y, held_back=0.0)
    if fit is None:
        return 0.0, 0.0
    return (float(fit[1] / spread) if spread > 0 else 0.0), float(fit[0])


def _loss(offset: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(1.0 / (1.0 + np.exp(-np.clip(offset, -30, 30))), 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def learned(episodes: list[np.ndarray], ends: list[float], offsets: list[np.ndarray] | None = None, *,
            seed: int = 0) -> SlowAndDeciding | None:
    """What changes slowly and decides how ``episodes`` came out: ``ends`` is how each came out, any number; ``offsets``
    the judgement already held at each step, in log-odds, held fixed. None where too little is seen, or where nothing
    slow foretells episodes held back better than that judgement alone."""
    if len(episodes) < 2 * HELD_OUT_EVERY or len(episodes) != len(ends):
        return None
    y_episode = _as_ranks(np.asarray(ends, dtype=np.float64))
    order = np.random.default_rng(seed).permutation(len(episodes))
    held = set(order[:max(1, len(episodes) // HELD_OUT_EVERY)].tolist())
    fit = [i for i in range(len(episodes)) if i not in held]

    def stacked(which: list[int], directions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = np.concatenate([np.asarray(episodes[i], dtype=np.float64) @ directions.T for i in which])
        y = np.concatenate([np.full(len(episodes[i]), y_episode[i]) for i in which])
        o = np.concatenate([np.asarray(offsets[i], dtype=np.float64) if offsets is not None
                            else np.zeros(len(episodes[i])) for i in which])
        return z, o, y

    def trained(which: list[int], directions: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float] | None:
        z, o, y = stacked(which, directions)
        per_unit, level = _read_against(o, y)
        # Each direction scaled to its own spread, so holding back treats them alike.
        scale = z.std(axis=0)
        scale[scale == 0] = 1.0
        theta = _fitted(z / scale, per_unit * o + level, y)
        return None if theta is None else (theta, scale, per_unit, level)

    def foretold(which: list[int], directions: np.ndarray, model) -> tuple[float, float]:
        theta, scale, per_unit, level = model
        z, o, y = stacked(which, directions)
        o = per_unit * o + level
        return _loss(o, y), _loss(o + (z / scale) @ theta, y)

    # How many of the slowest directions to fit is chosen on part of the episodes fitted, never on those held back:
    # a board's pieces give hundreds of directions slower than noise, and a few dozen games cannot fit them all.
    inner_held = fit[::HELD_OUT_EVERY]
    inner = [i for i in fit if i not in inner_held]
    found, _slowness = slow_directions([episodes[i] for i in inner])
    if not len(found) or not inner_held:
        return None
    sizes, best = [], None
    k = 1
    while k <= len(found):
        sizes.append(k)
        k *= 2
    for k in sizes:
        model = trained(inner, found[:k])
        if model is None:
            continue
        _before, after = foretold(inner_held, found[:k], model)
        if best is None or after < best[0]:
            best = (after, k)
    if best is None:
        return None
    directions, slowness = slow_directions([episodes[i] for i in fit])
    directions, slowness = directions[:best[1]], slowness[:best[1]]
    if not len(directions):
        return None
    model = trained(fit, directions)
    if model is None:
        return None
    theta, scale, per_unit, _level = model
    before, after = foretold(sorted(held), directions, model)
    # Better by more than chance: each held-back episode's own gain, and their mean above its standard error.
    gains = [np.subtract(*foretold([i], directions, model)) for i in sorted(held)]
    if len(gains) < 2 or not np.mean(gains) > np.std(gains, ddof=1) / np.sqrt(len(gains)):
        return None
    return SlowAndDeciding(directions=directions, slowness=slowness, worth=theta / scale,
                           held_out=(round(before, 6), round(after, 6)),
                           per_unit_held=per_unit if offsets is not None else 0.0)

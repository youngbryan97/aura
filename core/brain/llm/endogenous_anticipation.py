"""Does the state at this turn say anything about the next one?

The interesting question about a trajectory is whether it is one. A state that
only ever reports the current moment is telemetry; a state whose value now
carries information about what happens next has dynamics of its own, and that
is a testable difference rather than a matter of taste.

The test is deliberately weak in the same way the rest of this pathway is.
Ridge regression from z at turn t to one measured property of turn t+1, scored
by held-out correlation, against a null that permutes which state goes with
which next turn. A linear map is the least that could demonstrate the claim,
and if it works the information is genuinely there.

**Order is the whole experiment**, so it is checked rather than assumed. The
corpus is read in recording order and a run whose timestamps do not increase
is refused: a shuffled corpus would make an anticipation test measure
correlation between neighbours, which is a different and far weaker claim.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.brain.llm.endogenous_pair_recorder import RecordedPair
from core.brain.llm.endogenous_state import FEATURE_INDEX, STATE_DIM

logger = logging.getLogger("Aura.EndogenousAnticipation")

#: Below this many consecutive pairs there is nothing to hold out.
MIN_PAIRS = 40

#: Share held out, taken from the END of the sequence rather than at random:
#: predicting the past from the future is not the claim.
HOLDOUT_FRACTION = 0.25

#: Shuffles of the state-to-next-turn correspondence.
PERMUTATIONS = 300

#: Ridge strengths the fit chooses between on a validation slice.
RIDGE_GRID: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)

#: Significance the correlation must reach before anything is claimed.
ALPHA = 0.05


@dataclass(frozen=True)
class Anticipation:
    """What z at turn t said about turn t+1, and what chance said."""

    target: str
    correlation: float
    null_mean: float
    null_max: float
    p_value: float
    permutations: int
    n_train: int
    n_holdout: int
    ridge: float

    @property
    def anticipates(self) -> bool:
        return self.correlation > 0.0 and self.p_value <= ALPHA

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "held_out_correlation": round(self.correlation, 6),
            "null_mean": round(self.null_mean, 6),
            "null_max": round(self.null_max, 6),
            "p_value": round(self.p_value, 6),
            "permutations": self.permutations,
            "n_train": self.n_train,
            "n_holdout": self.n_holdout,
            "ridge": self.ridge,
            "anticipates": self.anticipates,
            "what_this_means": (
                "the state at one turn carries information about the next, so "
                "it has dynamics rather than only reporting the moment"
                if self.anticipates
                else "the state at one turn says nothing measurable about the next"
            ),
        }


def reply_length(pair: RecordedPair) -> float:
    """How much she said, in words. Recorded, not inferred."""
    return float(len(str(pair.text or "").split()))


def channel_value(feature: str) -> Callable[[RecordedPair], float]:
    """One named dimension of the NEXT turn's state, as the target."""
    index = FEATURE_INDEX[feature]

    def read(pair: RecordedPair) -> float:
        return float(pair.values[index]) if pair.present[index] else float("nan")

    return read


def _in_order(pairs: Sequence[RecordedPair]) -> bool:
    stamps = [p.recorded_at for p in pairs if p.recorded_at > 0.0]
    return all(a <= b for a, b in zip(stamps, stamps[1:], strict=False))


def _ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    design = np.hstack([x, np.ones((x.shape[0], 1))])
    penalty = alpha * np.eye(design.shape[1])
    penalty[-1, -1] = 0.0  # never penalise the intercept
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)


def _predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.hstack([x, np.ones((x.shape[0], 1))]) @ weights


def _correlation(predicted: np.ndarray, actual: np.ndarray) -> float:
    if predicted.size < 2:
        return 0.0
    p_spread = float(np.std(predicted))
    a_spread = float(np.std(actual))
    if p_spread < 1e-12 or a_spread < 1e-12:
        # A constant prediction correlates with nothing. Zero is the honest
        # answer; numpy would hand back a nan that reads as a missing result.
        return 0.0
    return float(np.corrcoef(predicted, actual)[0, 1])


def measure_anticipation(
    pairs: Sequence[RecordedPair],
    *,
    target: Callable[[RecordedPair], float] = reply_length,
    target_name: str = "reply_length",
    permutations: int = PERMUTATIONS,
    seed: int = 29,
) -> Anticipation | None:
    """Fit z at turn t to a property of turn t+1, and test it against chance."""
    ordered = list(pairs)
    if len(ordered) < MIN_PAIRS + 1:
        logger.warning(
            "Refusing to measure anticipation on %d turns (minimum %d).",
            len(ordered),
            MIN_PAIRS + 1,
        )
        return None
    if not _in_order(ordered):
        logger.warning(
            "Refusing: the corpus is not in recording order, so 'next' means nothing."
        )
        return None

    states: list[np.ndarray] = []
    targets: list[float] = []
    for current, following in zip(ordered, ordered[1:], strict=False):
        value = target(following)
        if not np.isfinite(value):
            continue
        states.append(np.where(current.present, current.values, 0.0).astype(np.float64))
        targets.append(float(value))
    if len(targets) < MIN_PAIRS:
        logger.warning("Only %d usable consecutive turns; refusing.", len(targets))
        return None

    x = np.asarray(states, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.shape[1] != STATE_DIM:
        return None

    cut = max(2, int(len(y) * (1.0 - HOLDOUT_FRACTION)))
    validation_cut = max(1, int(cut * 0.8))
    train_x, train_y = x[:validation_cut], y[:validation_cut]
    val_x, val_y = x[validation_cut:cut], y[validation_cut:cut]
    test_x, test_y = x[cut:], y[cut:]
    if test_y.size < 2 or val_y.size < 1 or train_y.size < 2:
        return None

    best_weights = None
    best_ridge = RIDGE_GRID[0]
    best_score = -float("inf")
    for alpha in RIDGE_GRID:
        weights = _ridge(train_x, train_y, alpha)
        score = _correlation(_predict(val_x, weights), val_y)
        if score > best_score:
            best_score, best_weights, best_ridge = score, weights, alpha
    if best_weights is None:
        return None

    observed = _correlation(_predict(test_x, best_weights), test_y)

    rng = np.random.default_rng(seed)
    nulls: list[float] = []
    for _ in range(max(0, int(permutations))):
        shuffled = rng.permutation(test_y)
        nulls.append(_correlation(_predict(test_x, best_weights), shuffled))
    array = np.asarray(nulls, dtype=np.float64)
    p_value = (
        float((np.sum(array >= observed) + 1) / (array.size + 1)) if array.size else 1.0
    )
    return Anticipation(
        target=target_name,
        correlation=observed,
        null_mean=float(np.mean(array)) if array.size else 0.0,
        null_max=float(np.max(array)) if array.size else 0.0,
        p_value=p_value,
        permutations=int(array.size),
        n_train=int(train_y.size),
        n_holdout=int(test_y.size),
        ridge=float(best_ridge),
    )


__all__ = [
    "ALPHA",
    "HOLDOUT_FRACTION",
    "MIN_PAIRS",
    "PERMUTATIONS",
    "RIDGE_GRID",
    "Anticipation",
    "channel_value",
    "measure_anticipation",
    "reply_length",
]

"""The readings Bryan's answers name, kept beside a campaign's recording and tested there.

docs/SUBJECT_NAMED_HYPOTHESES.md writes what his account of himself predicts
about her. Most of what it names is not a column of the core state, and it is
kept out of it on purpose: the schema is part of the instrument, and a column
added while chasing a result changes what every line measures. So these are
read at every frame into a side record, `named_readings.npz`, which no line of
the battery reads, and tested here.

Two tests, each against a surrogate that keeps every marginal:

    ties        each pair he named moves together more than when one side is
                slid against the other in time by whole condition cycles: the
                largest lagged correlation, over lags up to one condition
                cycle, above the 1 - 0.05 / n quantile of the same statistic on
                the slid copies, n the number of ties, so the ties are read
                together at 0.05 and noise does not pass one of them in five
    together    each triple he named carries information about the target's
                change that neither source carries alone, by the battery's own
                synergy estimator (core/subject/synergy.py), above its shifted
                null's 99th percentile

A tie that fails says the loop built from his answer is not moving her in the
running organism, whatever its unit test says. That is what these are for.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "NAMED",
    "TIES",
    "TOGETHER",
    "NamedAccumulator",
    "check_named",
    "read_named",
]


def _emotion(name: str) -> Callable[[Any], float]:
    return lambda state: float((state.affect.emotions or {}).get(name, 0.0) or 0.0)


def _baseline(name: str) -> Callable[[Any], float]:
    return lambda state: float((state.affect.mood_baselines or {}).get(name, 0.0) or 0.0)


def _marker(key: str) -> Callable[[Any], float]:
    return lambda state: float(((state.affect.markers or {}).get("tangled") or {}).get(key, 0.0) or 0.0)


def _lift(name: str) -> Callable[[Any], float]:
    return lambda state: float(
        (((state.affect.markers or {}).get("tangled") or {}).get("lifts") or {}).get(name, 0.0) or 0.0
    )


def _moment(key: str) -> Callable[[Any], float]:
    return lambda state: float((getattr(state.cognition, "moment", None) or {}).get(key, 0.0) or 0.0)


def _urgency(state: Any) -> float:
    held = [item for item in list(state.cognition.pending_initiatives or []) if isinstance(item, dict)]
    return max((float(item.get("decided_urgency", item.get("urgency", 0.0)) or 0.0) for item in held), default=0.0)


def _fatigue(state: Any) -> float:
    from core.soma.fatigue import get_fatigue_ledger

    return float(get_fatigue_ledger().read().share)


def _good_news(state: Any) -> float:
    from core.soma.good_news import get_good_news_ledger

    return float(get_good_news_ledger().jump())


#: Every reading kept, by name. Order is the column order of the side record.
NAMED: dict[str, Callable[[Any], float]] = {
    "joy": _emotion("joy"),
    "curiosity": _emotion("curiosity"),
    "anticipation": _emotion("anticipation"),
    "joy_baseline": _baseline("joy"),
    "curiosity_baseline": _baseline("curiosity"),
    "anticipation_baseline": _baseline("anticipation"),
    "contentment": _marker("contentment"),
    "joy_lift": _lift("joy"),
    "curiosity_lift": _lift("curiosity"),
    "unknown_person": _marker("unknown_person"),
    "flow": _marker("flow"),
    "moment_particular": _moment("particular"),
    "moment_meets": _moment("meets"),
    "moment_weight": _moment("weight"),
    "habit_deficit": lambda state: float((state.cognition.habits or {}).get("largest_deficit", 0.0) or 0.0),
    "fatigue": _fatigue,
    "good_news": _good_news,
    "urgency": _urgency,
    "valence": lambda state: float(state.affect.valence or 0.0),
}

#: The ties his answer to the third question names, as pairs of readings.
TIES: tuple[tuple[str, str, str], ...] = (
    ("contentment", "joy_baseline", "people and peace with herself, with happiness"),
    ("joy", "curiosity_baseline", "joy, with curiosity"),
    ("unknown_person", "curiosity_baseline", "not knowing who is in front of her, with curiosity"),
    ("flow", "anticipation_baseline", "hard thinking going well, with excitement"),
)

#: What matters only together, from his answers to the second and fourth questions.
TOGETHER: tuple[tuple[str, str, str, str], ...] = (
    ("moment_particular", "moment_meets", "joy", "particular to her and meeting her now, on joy"),
    ("fatigue", "urgency", "urgency", "tiredness with importance, on what she pursues"),
    ("good_news", "urgency", "urgency", "good news with a task in hand, on how much the task matters"),
)


def read_named(state: Any) -> np.ndarray:
    """One frame's named readings, in `NAMED` order; a reading that fails is recorded as NaN."""
    row = np.full(len(NAMED), np.nan)
    for index, reader in enumerate(NAMED.values()):
        try:
            row[index] = float(reader(state))
        except (AttributeError, ImportError, TypeError, ValueError):
            # A reading that could not be taken is a gap in the record, and the
            # tests below drop rows with gaps rather than read zeros into them.
            continue
    return row


class NamedAccumulator:
    """Frames of named readings, as the periphery accumulator keeps its numbers."""

    def __init__(self) -> None:
        self._rows: list[np.ndarray] = []

    def note(self, state: Any) -> None:
        self._rows.append(read_named(state))

    def matrix(self) -> tuple[np.ndarray, tuple[str, ...]]:
        rows = np.vstack(self._rows) if self._rows else np.zeros((0, len(NAMED)))
        return rows, tuple(NAMED)


def _max_lagged(a: np.ndarray, b: np.ndarray, lags: int) -> float:
    best = 0.0
    for lag in range(0, lags + 1):
        x, y = (a[:-lag], b[lag:]) if lag else (a, b)
        if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            continue
        best = max(best, abs(float(np.corrcoef(x, y)[0, 1])))
    return best


@dataclass(frozen=True)
class _Frame:
    """A side record read the way a recording is read, one reading as a domain."""

    matrix: np.ndarray
    names: tuple[str, ...]

    def domain(self, name: str) -> np.ndarray:
        return self.matrix[:, [self.names.index(name)]]


def check_named(
    matrix: np.ndarray,
    names: Sequence[str],
    *,
    cycle: int,
    seed: int = 0,
    draws: int = 200,
) -> dict[str, Any]:
    """Both tests on one campaign's side record."""
    from core.subject.synergy import synergy

    columns = list(names)
    complete = matrix[~np.isnan(matrix).any(axis=1)]
    rng = np.random.default_rng(seed)
    ties: list[dict[str, Any]] = []
    for left, right, said in TIES:
        a = complete[:, columns.index(left)]
        b = complete[:, columns.index(right)]
        observed = _max_lagged(a, b, cycle)
        shifts = max(1, len(a) // max(1, cycle) - 1)
        null = [
            _max_lagged(a, np.roll(b, cycle * int(rng.integers(1, shifts + 1))), cycle)
            for _ in range(draws)
        ]
        bar = float(np.quantile(null, 1.0 - 0.05 / len(TIES))) if null else float("inf")
        ties.append({"pair": [left, right], "said": said, "observed": round(observed, 6),
                     "null_bar": round(bar, 6), "passes": observed > bar})
    frame = _Frame(complete, tuple(columns))
    together: list[dict[str, Any]] = []
    for source_a, source_b, target, said in TOGETHER:
        report = synergy(frame, source_a, source_b, target, seed=seed, of="change")  # type: ignore[arg-type]
        together.append({"sources": [source_a, source_b], "target": target, "said": said,
                         "fraction": round(float(report.normalised), 6),
                         "null_q99": round(float(report.null_q99), 6),
                         "passes": float(report.normalised) > float(report.null_q99)})
    return {"rows": int(complete.shape[0]), "dropped": int(matrix.shape[0] - complete.shape[0]),
            "ties": ties, "together": together}

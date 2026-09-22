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


def _never_moved(matrix: np.ndarray, columns: list[str], wanted: tuple[str, ...]) -> str:
    """The first of these readings that held one value throughout, or ''."""
    for name in wanted:
        if float(np.std(matrix[:, columns.index(name)])) < 1e-12:
            return name
    return ""


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
        still = _never_moved(complete, columns, (left, right))
        if still:
            ties.append({"pair": [left, right], "said": said, "measured": False, "passes": None,
                         "why": f"{still} never moved over {complete.shape[0]} frames"})
            continue
        # Read by what changed, not by level. Her readings drift slowly, and two
        # drifting series correlate whatever they are: on seed 7 a shifted copy
        # of the joy baseline correlated with contentment at 0.9953, and the
        # tie "passed" at 0.9956. Changes carry the coupling and not the drift.
        a = np.diff(complete[:, columns.index(left)])
        b = np.diff(complete[:, columns.index(right)])
        observed = _max_lagged(a, b, cycle)
        shifts = max(1, len(a) // max(1, cycle) - 1)
        null = [
            _max_lagged(a, np.roll(b, cycle * int(rng.integers(1, shifts + 1))), cycle)
            for _ in range(draws)
        ]
        bar = float(np.quantile(null, 1.0 - 0.05 / len(TIES))) if null else float("inf")
        ties.append({"pair": [left, right], "said": said, "measured": True, "observed": round(observed, 6),
                     "null_bar": round(bar, 6), "passes": observed > bar})
    frame = _Frame(complete, tuple(columns))
    together: list[dict[str, Any]] = []
    for source_a, source_b, target, said in TOGETHER:
        still = _never_moved(complete, columns, (source_a, source_b, target))
        if still:
            together.append({"sources": [source_a, source_b], "target": target, "said": said,
                             "measured": False, "passes": None,
                             "why": f"{still} never moved over {complete.shape[0]} frames"})
            continue
        report = synergy(frame, source_a, source_b, target, seed=seed, of="change")  # type: ignore[arg-type]
        # The battery's own synergy line (ISC-v3): the raw synergy against the
        # bootstrap null and the shifted null, and the held-out interaction.
        # The fraction against its shifted null, which this read first, is the
        # bar v3 retired: a ratio of two small numbers whose null stays wide.
        together.append({"sources": [source_a, source_b], "target": target, "said": said,
                         "measured": True,
                         "synergy": round(float(report.synergy), 6),
                         "bootstrap_q99": round(float(report.bootstrap_q99), 6),
                         "raw_null_q99": round(float(report.raw_null_q99), 6),
                         "interaction_lower_bound": round(float(report.interaction_lower_bound), 6),
                         "passes": bool(report.passes_v3)})
    return {"rows": int(complete.shape[0]), "dropped": int(matrix.shape[0] - complete.shape[0]),
            "ties": ties, "together": together}


# ── H1: the families of feeling, read off a content run ────────────────────

#: His families, as the percept kinds the content run presents. Anger has no
#: kind (in her it comes from somebody objecting, not from something seen),
#: so the frustration kinds nearest it stand in. docs/SUBJECT_NAMED_HYPOTHESES.md.
FAMILIES: dict[str, tuple[str, ...]] = {
    "warmth": ("cared_for", "positive_interaction", "interaction", "extended_dialogue"),
    "loss": ("disconnection",),
    "frustration": ("error", "internal_error", "self_correction", "inner_conflict"),
}


def check_families(report: dict[str, Any]) -> dict[str, Any]:
    """H1 as registered: warmth kinds nearer each other than to loss and to frustration.

    Both inequalities on the content run's internal geometry, the Fisher-Rao
    distance between the futures two classes induce from one fork. Refuted if
    either fails. Whether each margin clears the run's own internal floor is
    reported beside the verdict and does not change it.
    """
    classes = [c for c in report.get("classes") or () if isinstance(c, dict)]
    internal = report.get("internal") or {}
    kinds = [str(c.get("kind") or "") for c in classes]
    if not classes or not internal:
        return {"measured": False, "passes": None, "why": "the content run has no internal geometry"}

    def members(family: str) -> list[int]:
        return [i for i, kind in enumerate(kinds) if kind in FAMILIES[family]]

    warmth, loss, frustration = members("warmth"), members("loss"), members("frustration")
    missing = [name for name, found in (("warmth", len(warmth) >= 2), ("loss", loss), ("frustration", frustration)) if not found]
    if missing:
        return {"measured": False, "passes": None,
                "why": f"no class of the {', '.join(missing)} family among {sorted(set(kinds))}"}

    def distance(i: int, j: int) -> float | None:
        value = internal.get(f"{min(i, j)}-{max(i, j)}")
        return None if value is None else float(value)

    def mean(pairs: list[tuple[int, int]]) -> float | None:
        values = [d for d in (distance(i, j) for i, j in pairs) if d is not None]
        return float(np.mean(values)) if values else None

    within = mean([(i, j) for n, i in enumerate(warmth) for j in warmth[n + 1:]])
    to_loss = mean([(i, j) for i in warmth for j in loss])
    to_frustration = mean([(i, j) for i in warmth for j in frustration])
    if within is None or to_loss is None or to_frustration is None:
        return {"measured": False, "passes": None, "why": "a family pair has no measured distance"}
    floor = max((float(v) for v in (report.get("internal_floor") or {}).values()), default=0.0)
    return {
        "measured": True,
        "within_warmth": round(within, 6),
        "warmth_to_loss": round(to_loss, 6),
        "warmth_to_frustration": round(to_frustration, 6),
        "internal_floor": round(floor, 6),
        "margins_clear_the_floor": bool(to_loss - within > floor and to_frustration - within > floor),
        "passes": bool(within < to_loss and within < to_frustration),
    }


# ── H4: a being of drives alone, read off a campaign's nulls ───────────────


def check_drives_only(nulls_detail: dict[str, Any] | None) -> dict[str, Any]:
    """H4 as registered: the drives-only null fails the conjunction, on one component.

    Failing one component is failing the conjunction, since every line is
    required, so the verdict is that one field of the architecture's row.
    Refuted if the row reads one component, which is also the only way it
    could pass. A campaign whose nulls stage did not run it has not measured it.
    """
    row = (nulls_detail or {}).get("drives_only")
    if not isinstance(row, dict) or "one_component" not in row:
        return {"measured": False, "passes": None, "why": "the campaign's nulls stage has no drives_only row"}
    return {
        "measured": True,
        "one_component": bool(row["one_component"]),
        "passes": not bool(row["one_component"]),
    }


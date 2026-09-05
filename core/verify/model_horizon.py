"""core/verify/model_horizon.py — where a model has been checked and was right.

A simulator is trustworthy over the region where it has been tested and found
accurate, and nowhere else. That boundary is not a property anyone can read
off the model; it has to be measured, and until it is, every prediction
carries the same confident tone whether it is interpolating between a hundred
checked cases or extrapolating past all of them.

The consequence is what matters. A prediction outside the region must not be
what decides an irreversible action. Not because it is wrong — it may well be
right — but because nothing has established that it is, and an irreversible
act is the one case where "probably" is not a good enough reason.

The horizon here is model-agnostic on purpose. It needs no gradients, no
ensemble, and no access to the model's internals: only a feature vector
describing each query, the prediction made, and eventually the outcome. From
those it answers two separate questions:

**Support.** Are there enough resolved cases near this one? A query with no
neighbours is an extrapolation whatever the model says about it.

**Local calibration.** Where there are neighbours, how wrong was the model
there? A well-supported region the model gets consistently wrong is outside
the horizon too, which a support-only check would miss.

Both can come back UNKNOWN, and that is the important verdict. "Nothing near
this has been checked" is a different finding from "this region is bad", and
collapsing them is how a model with no track record ends up trusted by
default. The accuracy bar is a sealed criterion from
:mod:`core.verify.epistemic_independence`, so it cannot be lowered after a
model has been seen to miss it.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.Verify.Horizon")


def _checked_lock(name: str, *, reentrant: bool = False):
    """The repo's instrumented lock, so lockdep can see this one too.

    A raw threading lock is invisible to the ABBA detector, and a detector
    that sees only some of the locks reports clean while the deadlock it
    exists to find is assembled out of the others.
    """

    from core.runtime.lockdep import checked_lock

    return checked_lock(name, reentrant=reentrant)


#: Resolved cases needed near a query before support can be claimed. Below
#: this the neighbourhood is one or two points and their accuracy is anecdote.
MIN_NEIGHBOURS = 5

#: How close a case has to be to count as a neighbour, as a fraction of the
#: typical distance between resolved cases.
#:
#: A fraction rather than a distance, because whether 0.35 is "near" depends
#: entirely on how the features happen to be scaled — which is a per-model
#: fact this module cannot know. With an absolute radius, a query sitting
#: exactly between two clusters counted every point in both as a neighbour
#: and was reported as well supported by cases that were nowhere near it.
#:
#: This is derived from the record, not from the query's outcome. The bar
#: being judged — local error — stays fixed, so nothing here sets its own
#: success criterion.
NEIGHBOUR_FRACTION = 0.35

#: Pairs sampled to estimate the typical distance. The estimate does not need
#: to be exact and the record can hold a couple of thousand cases.
_SPREAD_SAMPLE = 400

#: Resolved cases needed anywhere before the horizon can say anything at all.
MIN_RECORD = 12

#: Local error above which a well-supported region is still outside the
#: horizon. Sealed as a criterion, so it cannot be raised after a model has
#: been seen to miss it.
MAX_LOCAL_ERROR = 0.25

#: How long a resolved case stays evidence about the region. A model's world
#: changes; a check from last month is weaker evidence than one from today,
#: and one from a year ago is a historical note.
EVIDENCE_HALF_LIFE_S = 7.0 * 86400.0


class Standing(StrEnum):
    """Whether a prediction is inside the region the model has earned."""

    #: Supported by enough nearby resolved cases, and accurate in them.
    INSIDE = "inside"
    #: Supported, and the model is wrong around here.
    UNRELIABLE = "unreliable"
    #: Nothing near this has been resolved. Not a verdict about the model.
    UNSUPPORTED = "unsupported"
    #: The record is too small to say anything at all.
    UNMEASURED = "unmeasured"

    @property
    def may_drive_irreversible(self) -> bool:
        """Only a checked and accurate region may settle an irreversible act."""
        return self is Standing.INSIDE


@dataclass(frozen=True)
class Case:
    """One prediction, its query, and what actually happened."""

    features: tuple[float, ...]
    predicted: float
    actual: float | None = None
    at: float = field(default_factory=time.time)
    label: str = ""

    @property
    def resolved(self) -> bool:
        return self.actual is not None

    @property
    def error(self) -> float:
        return 0.0 if self.actual is None else abs(self.predicted - self.actual)

    def weight_at(self, now: float) -> float:
        age = max(0.0, now - self.at)
        return 0.5 ** (age / EVIDENCE_HALF_LIFE_S)


@dataclass(frozen=True)
class Verdict:
    """Where this query sits relative to what the model has earned."""

    standing: Standing
    neighbours: int
    local_error: float | None
    #: Distance to the nearest resolved case, or None when there are none.
    nearest: float | None
    because: str

    @property
    def may_drive_irreversible(self) -> bool:
        return self.standing.may_drive_irreversible

    def ceiling(self) -> float:
        """How irreversible an act this prediction may support, in [0, 1].

        Not a hard switch. A prediction the model has earned may settle
        anything; one in a region it gets wrong may still inform a cheap,
        undoable step, because being outside the horizon is a reason to be
        careful rather than a reason to stop.
        """
        return _CEILINGS[self.standing]

    def to_dict(self) -> dict[str, Any]:
        return {
            "standing": str(self.standing),
            "neighbours": self.neighbours,
            "local_error": None if self.local_error is None else round(self.local_error, 4),
            "nearest": None if self.nearest is None else round(self.nearest, 4),
            "may_drive_irreversible": self.may_drive_irreversible,
            "ceiling": self.ceiling(),
            "because": self.because,
        }


#: What each standing may support. UNSUPPORTED sits above UNRELIABLE: not
#: knowing whether the model works here is a better position than knowing it
#: does not.
_CEILINGS: Mapping[Standing, float] = {
    Standing.INSIDE: 1.0,
    Standing.UNSUPPORTED: 0.4,
    Standing.UNMEASURED: 0.4,
    Standing.UNRELIABLE: 0.15,
}


def _typical_distance(cases: Sequence[Case], seed: int = 0x5EED) -> float:
    """The median distance between resolved cases, as the scale of "near".

    Sampled and deterministic: an estimate that changed between processes
    would make the same query supported on one boot and not on the next.
    """
    import random as _random

    if len(cases) < 2:
        return 0.0
    rng = _random.Random(seed)
    span = len(cases)
    pairs = min(_SPREAD_SAMPLE, span * (span - 1) // 2)
    seen: list[float] = []
    for _ in range(pairs):
        left = rng.randrange(span)
        right = rng.randrange(span)
        if left == right:
            continue
        distance = _distance(cases[left].features, cases[right].features)
        if math.isfinite(distance):
            seen.append(distance)
    if not seen:
        return 0.0
    seen.sort()
    middle = len(seen) // 2
    if len(seen) % 2:
        return seen[middle]
    return (seen[middle - 1] + seen[middle]) / 2.0


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    """Normalised Euclidean distance over the shared prefix."""
    span = min(len(left), len(right))
    if span == 0:
        return math.inf
    total = sum((float(left[i]) - float(right[i])) ** 2 for i in range(span))
    return math.sqrt(total / span)


class ModelHorizon:
    """The region one model has been checked in, and was right."""

    def __init__(self, model: str, *, capacity: int = 2048) -> None:
        self.model = str(model)
        self._cases: list[Case] = []
        self._capacity = max(MIN_RECORD, int(capacity))
        self._lock = _checked_lock("model_horizon", reentrant=True)

    # ── recording ────────────────────────────────────────────────────────

    def predicted(
        self, features: Sequence[float], prediction: float, *, label: str = ""
    ) -> int:
        """Record a prediction. Returns its index, for resolving it later."""
        case = Case(
            features=tuple(float(f) for f in features),
            predicted=float(prediction),
            label=str(label)[:120],
        )
        with self._lock:
            self._cases.append(case)
            if len(self._cases) > self._capacity:
                # Drop the oldest resolved case first: an unresolved one is
                # still waiting to become evidence, and dropping it loses a
                # check that was already paid for.
                for index, existing in enumerate(self._cases):
                    if existing.resolved:
                        del self._cases[index]
                        break
                else:
                    del self._cases[0]
            return len(self._cases) - 1

    def resolved(self, index: int, actual: float) -> bool:
        """Record what actually happened. True if the case was still held."""
        with self._lock:
            if not 0 <= index < len(self._cases):
                return False
            case = self._cases[index]
            self._cases[index] = Case(
                features=case.features,
                predicted=case.predicted,
                actual=float(actual),
                at=case.at,
                label=case.label,
            )
            return True

    def observe(
        self, features: Sequence[float], prediction: float, actual: float, *, label: str = ""
    ) -> None:
        """Record a prediction and its outcome together."""
        self.resolved(self.predicted(features, prediction, label=label), actual)

    # ── judging ──────────────────────────────────────────────────────────

    def standing(self, features: Sequence[float]) -> Verdict:
        """Where a query sits relative to what this model has earned."""
        query = tuple(float(f) for f in features)
        now = time.time()
        with self._lock:
            resolved = [c for c in self._cases if c.resolved]

        if len(resolved) < MIN_RECORD:
            return Verdict(
                standing=Standing.UNMEASURED,
                neighbours=0,
                local_error=None,
                nearest=None,
                because=(
                    f"{len(resolved)} resolved cases; {MIN_RECORD} are needed "
                    "before anything can be said about any region"
                ),
            )

        # Sort on the distance alone. Two cases at the same point make the
        # tuple comparison fall through to the Case, which is not orderable,
        # and identical points are exactly what a repeated check produces.
        scored = sorted(
            ((_distance(query, c.features), c) for c in resolved),
            key=lambda pair: pair[0],
        )
        nearest = scored[0][0] if scored else None
        typical = _typical_distance(resolved)
        radius = typical * NEIGHBOUR_FRACTION
        if radius <= 0.0:
            # Every resolved case is at the same point. Only a query at that
            # point is near anything, and "near" has no other meaning here.
            radius = 1e-9
        near = [(d, c) for d, c in scored if d <= radius]

        if len(near) < MIN_NEIGHBOURS:
            return Verdict(
                standing=Standing.UNSUPPORTED,
                neighbours=len(near),
                local_error=None,
                nearest=nearest,
                because=(
                    f"{len(near)} resolved cases within {radius:.3f} of this one "
                    f"(a third of the typical {typical:.3f} between checks); "
                    f"nearest is {nearest:.3f} away. Nothing like this has been "
                    "checked"
                ),
            )

        weights = [c.weight_at(now) for _d, c in near]
        total = sum(weights)
        if total <= 0.0:
            return Verdict(
                standing=Standing.UNSUPPORTED,
                neighbours=len(near),
                local_error=None,
                nearest=nearest,
                because="every nearby check has aged out of being evidence",
            )
        error = sum(c.error * w for (_d, c), w in zip(near, weights, strict=True)) / total

        if _error_within_bar(error):
            return Verdict(
                standing=Standing.INSIDE,
                neighbours=len(near),
                local_error=error,
                nearest=nearest,
                because=(
                    f"{len(near)} checked cases nearby, mean error {error:.3f} "
                    f"within {MAX_LOCAL_ERROR}"
                ),
            )
        return Verdict(
            standing=Standing.UNRELIABLE,
            neighbours=len(near),
            local_error=error,
            nearest=nearest,
            because=(
                f"{len(near)} checked cases nearby and the model is wrong here: "
                f"mean error {error:.3f} over {MAX_LOCAL_ERROR}"
            ),
        )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            cases = list(self._cases)
        resolved = [c for c in cases if c.resolved]
        return {
            "model": self.model,
            "cases": len(cases),
            "resolved": len(resolved),
            "mean_error": (
                round(sum(c.error for c in resolved) / len(resolved), 4)
                if resolved
                else None
            ),
        }

    def clear(self) -> None:
        with self._lock:
            self._cases.clear()


def _error_within_bar(error: float) -> bool:
    """Judge local error against a bar sealed before any model was measured."""
    try:
        from core.verify.epistemic_independence import declare

        criterion = declare(
            "model_horizon.local_error",
            threshold=MAX_LOCAL_ERROR,
            direction="below",
            rationale=(
                "the mean absolute error, in the units of the thing predicted, "
                "above which a region is not one the model has earned; fixed "
                "before any model was measured against it"
            ),
        )
        return criterion.judge(error).passed
    except (ImportError, RuntimeError, ValueError):
        return error < MAX_LOCAL_ERROR


_HORIZONS: dict[str, ModelHorizon] = {}
_HORIZONS_LOCK = _checked_lock("model_horizon")


def horizon(model: str) -> ModelHorizon:
    """The horizon for one named model, created on first use."""
    with _HORIZONS_LOCK:
        found = _HORIZONS.get(model)
        if found is None:
            found = ModelHorizon(model)
            _HORIZONS[model] = found
        return found


def all_horizons() -> dict[str, dict[str, Any]]:
    with _HORIZONS_LOCK:
        return {name: h.snapshot() for name, h in _HORIZONS.items()}


def reset_horizons() -> None:
    with _HORIZONS_LOCK:
        _HORIZONS.clear()


__all__ = [
    "EVIDENCE_HALF_LIFE_S",
    "MAX_LOCAL_ERROR",
    "MIN_NEIGHBOURS",
    "MIN_RECORD",
    "NEIGHBOUR_FRACTION",
    "Case",
    "ModelHorizon",
    "Standing",
    "Verdict",
    "all_horizons",
    "horizon",
    "reset_horizons",
]

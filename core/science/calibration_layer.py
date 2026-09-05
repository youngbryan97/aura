"""core/science/calibration_layer.py — every confidence on one scale, checked against outcomes.

Aura reports confidence from at least six places on at least four scales. The
cortex reports a probability from logits. The world model reports one from
posterior variance through a function of surprise. Rules report a Wilson lower
bound. The AtomSpace reports count/(count+1). Grounding reports a Brier-scored
reliability. Self-report produces a number from language. Adding them, comparing
them or thresholding them all at 0.7 assumes they mean the same thing, and none
of them was ever checked against whether the thing happened.

``core/ontogeny/calibration.py`` already does the checking properly for one
source. This is the layer every source passes through, so that:

* each source gets its OWN reliability curve, because a cortex probability of
  0.9 and a rule confidence of 0.9 do not have the same hit rate and never
  will;
* a raw reading becomes a **calibrated** reading through that source's own
  measured curve, so downstream code can compare them;
* a source with too little history is reported as UNCALIBRATED and its raw
  value passes through untouched rather than being silently trusted.

The last point is the one that matters. A calibration layer that invents a
mapping from three observations is worse than none, because it launders a
guess into a corrected number. Below the floor, this says so.

Calibration is done by binning: within a bin, the calibrated value is the
observed hit rate. That is the same estimator a reliability diagram plots, and
it is deliberately the simplest thing that cannot overfit — a fitted sigmoid on
sixty observations would look better and mean less.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from core.runtime.lockdep import checked_lock

__all__ = [
    "Reading",
    "SourceCalibration",
    "CalibrationLayer",
    "get_calibration_layer",
    "reset_calibration_layer_for_test",
]

#: Observations a source needs before its curve is used. Below this the raw
#: value passes through and the source reports UNCALIBRATED.
MIN_OBSERVATIONS = 30

#: Number of probability bins. Ten is what a reliability diagram uses and what
#: a reader can check by eye.
BINS = 10


def _bin_of(probability: float) -> int:
    return min(BINS - 1, max(0, int(probability * BINS)))


@dataclass
class SourceCalibration:
    """One source's history: what it said, and what happened."""

    source: str
    counts: list[int] = field(default_factory=lambda: [0] * BINS)
    hits: list[int] = field(default_factory=lambda: [0] * BINS)
    raw_sum: float = 0.0
    outcome_sum: float = 0.0
    squared_error: float = 0.0
    n: int = 0

    def observe(self, raw: float, outcome: bool) -> None:
        index = _bin_of(raw)
        self.counts[index] += 1
        self.hits[index] += 1 if outcome else 0
        self.raw_sum += raw
        self.outcome_sum += 1.0 if outcome else 0.0
        self.squared_error += (raw - (1.0 if outcome else 0.0)) ** 2
        self.n += 1

    @property
    def calibrated(self) -> bool:
        return self.n >= MIN_OBSERVATIONS

    @property
    def brier(self) -> float | None:
        return self.squared_error / self.n if self.n else None

    @property
    def bias(self) -> float | None:
        """Mean confidence minus mean outcome. Positive is overconfidence."""
        if not self.n:
            return None
        return (self.raw_sum - self.outcome_sum) / self.n

    def calibrate(self, raw: float) -> float:
        """Map a raw reading onto the observed hit rate for its bin.

        An empty bin falls back to the nearest populated one rather than to the
        raw value, because a source with a gap in its range has still been
        measured either side of the gap.
        """
        if not self.calibrated:
            return raw
        index = _bin_of(raw)
        for offset in range(BINS):
            for candidate in (index - offset, index + offset):
                if 0 <= candidate < BINS and self.counts[candidate] > 0:
                    return self.hits[candidate] / self.counts[candidate]
        return raw

    def reliability_diagram(self) -> list[dict[str, Any]]:
        return [
            {
                "bin": i,
                "range": [i / BINS, (i + 1) / BINS],
                "n": self.counts[i],
                "observed": (self.hits[i] / self.counts[i]) if self.counts[i] else None,
            }
            for i in range(BINS)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "n": self.n,
            "calibrated": self.calibrated,
            "brier": self.brier,
            "bias": self.bias,
            "diagram": self.reliability_diagram(),
        }


@dataclass(frozen=True, slots=True)
class Reading:
    """One confidence, before and after its source's own correction."""

    source: str
    raw: float
    calibrated: float
    status: str  # calibrated | uncalibrated
    n: int

    @property
    def usable_for_comparison(self) -> bool:
        return self.status == "calibrated"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "raw": self.raw,
            "calibrated": self.calibrated,
            "status": self.status,
            "n": self.n,
        }


class CalibrationLayer:
    """The one place a confidence becomes comparable to another confidence."""

    def __init__(self) -> None:
        self._lock = checked_lock("core.science.calibration_layer.CalibrationLayer", reentrant=True)
        self._sources: dict[str, SourceCalibration] = {}

    def observe(self, source: str, raw: float, outcome: bool) -> None:
        """Record what a source said and what actually happened."""
        with self._lock:
            self._sources.setdefault(source, SourceCalibration(source)).observe(
                min(1.0, max(0.0, float(raw))), bool(outcome)
            )

    def read(self, source: str, raw: float) -> Reading:
        """Turn a raw confidence into a comparable one, or say why not."""
        raw = min(1.0, max(0.0, float(raw)))
        with self._lock:
            calibration = self._sources.get(source)
        if calibration is None:
            return Reading(source, raw, raw, "uncalibrated", 0)
        return Reading(
            source,
            raw,
            calibration.calibrate(raw),
            "calibrated" if calibration.calibrated else "uncalibrated",
            calibration.n,
        )

    def combine(self, readings: Sequence[Reading]) -> dict[str, Any]:
        """Aggregate several sources, using only the ones that mean something.

        Uncalibrated readings are reported and excluded. Averaging a measured
        0.9 with an unmeasured 0.9 produces a number whose meaning is half
        known, which is the arithmetic this layer exists to stop.
        """
        usable = [r for r in readings if r.usable_for_comparison]
        excluded = [r.source for r in readings if not r.usable_for_comparison]
        if not usable:
            return {
                "combined": None,
                "usable_sources": [],
                "excluded_sources": excluded,
                "reason": "no calibrated source",
            }
        combined = sum(r.calibrated for r in usable) / len(usable)
        return {
            "combined": combined,
            "usable_sources": [r.source for r in usable],
            "excluded_sources": excluded,
            "spread": max(r.calibrated for r in usable) - min(r.calibrated for r in usable),
        }

    def report(self) -> dict[str, Any]:
        with self._lock:
            sources = list(self._sources.values())
        return {
            "sources": len(sources),
            "calibrated_sources": sum(1 for s in sources if s.calibrated),
            "by_source": {s.source: s.to_dict() for s in sources},
            "worst_bias": max(
                ((s.source, s.bias) for s in sources if s.bias is not None),
                key=lambda pair: abs(pair[1]),
                default=None,
            ),
        }


_lock = checked_lock("core.science.calibration_layer.singleton")
_layer: CalibrationLayer | None = None


def get_calibration_layer() -> CalibrationLayer:
    global _layer
    with _lock:
        if _layer is None:
            _layer = CalibrationLayer()
        return _layer


def reset_calibration_layer_for_test() -> CalibrationLayer:
    global _layer
    with _lock:
        _layer = CalibrationLayer()
        return _layer

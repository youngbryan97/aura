"""core/connectome/criticality.py — measuring the distance to the critical point without lying about it.

Cortex sits close to a phase transition. Below it activity dies out and nothing
propagates; above it activity explodes and nothing is distinguishable; at it,
cascades follow a power law, correlations reach furthest, and the dynamic range
is widest. The order parameter is the branching ratio ``m``: the mean number of
units one active unit activates on the next step. Critical is ``m = 1``.

There is a trap in measuring it, and it is the reason this module exists.

The obvious estimator — count how many units each active unit lit up, take the
mean — is badly biased when only part of the system is observed. Wilting and
Priesemann showed that under subsampling it collapses towards zero, so a system
sitting exactly at the critical point reads as strongly subcritical, and any
controller wired to that reading pushes gain up until the real system is
supercritical. Aura's criticality regulator watches 64 columns of a far larger
mesh, which is precisely the regime where that happens.

The multistep regression estimator avoids it. For a branching process the
regression slope of activity ``k`` steps apart decays as ``b·m^k``. Subsampling
changes ``b`` and leaves ``m`` alone, so fitting the decay recovers the branching
ratio whatever fraction of the system is visible.

The second measurement is the avalanche distribution, and it comes with its own
honesty check. Power laws in size and duration are necessary for criticality and
nowhere near sufficient — plenty of uninteresting processes produce them. The
crackling-noise relation is the discriminator: the exponents must satisfy
``(alpha - 1)/(tau - 1) = 1/(sigma·nu·z)``, and the right-hand side can be
measured independently from how mean avalanche size grows with duration. When
the two disagree, the power laws are not evidence of criticality, and this
module reports the disagreement rather than the two exponents on their own.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .topology import power_law_fit

logger = logging.getLogger("Aura.Connectome.Criticality")

__all__ = [
    "BranchingEstimate",
    "branching_ratio_mr",
    "naive_branching_ratio",
    "Avalanches",
    "extract_avalanches",
    "CriticalityReport",
    "assess",
]


@dataclass(frozen=True)
class BranchingEstimate:
    """A branching ratio with the fit it came from."""

    m: float
    offset: float
    r_squared: float
    kmax: int
    samples: int
    autocorrelation_time: float
    method: str

    @property
    def regime(self) -> str:
        if self.m < 0.9:
            return "subcritical"
        if self.m > 1.02:
            return "supercritical"
        return "critical"

    def as_json(self) -> dict[str, Any]:
        return {
            "m": round(self.m, 5),
            "offset": round(self.offset, 5),
            "r_squared": round(self.r_squared, 4),
            "kmax": self.kmax,
            "samples": self.samples,
            "autocorrelation_time_steps": round(self.autocorrelation_time, 3),
            "method": self.method,
            "regime": self.regime,
        }


def naive_branching_ratio(series: Sequence[float]) -> float:
    """The biased estimator, kept so the bias can be shown rather than argued.

    Mean of ``a(t+1)/a(t)`` over steps where the system was active. This is what
    a controller reads when nobody has thought about subsampling.
    """
    ratios = [
        float(series[t + 1]) / float(series[t])
        for t in range(len(series) - 1)
        if float(series[t]) > 0
    ]
    return sum(ratios) / len(ratios) if ratios else 0.0


def branching_ratio_mr(
    series: Sequence[float],
    *,
    kmax: int = 40,
    with_offset: bool = True,
) -> BranchingEstimate:
    """Multistep regression estimator, unbiased under subsampling.

    ``r_k`` is the slope of a linear regression of activity at ``t+k`` on
    activity at ``t``. For a branching process it decays geometrically in ``k``
    and the ratio is the base of that decay. The offset variant absorbs a
    constant external drive, which otherwise flattens the tail of the decay and
    drags the estimate towards one.
    """
    import numpy as np

    values = np.asarray(list(series), dtype=np.float64)
    n = values.size
    if n < 8:
        return BranchingEstimate(0.0, 0.0, 0.0, 0, int(n), 0.0, "insufficient-data")
    kmax = int(max(2, min(kmax, n // 4)))
    variance = float(values[: n - kmax].var())
    if variance <= 0:
        return BranchingEstimate(0.0, 0.0, 0.0, kmax, int(n), 0.0, "no-variance")

    slopes: list[float] = []
    for k in range(1, kmax + 1):
        base = values[: n - k]
        shifted = values[k:]
        base_centered = base - base.mean()
        shifted_centered = shifted - shifted.mean()
        denominator = float((base_centered**2).sum())
        if denominator <= 0:
            slopes.append(0.0)
            continue
        slopes.append(float((base_centered * shifted_centered).sum() / denominator))

    slope_array = np.asarray(slopes, dtype=np.float64)
    # Lags past the point where the correlation has decayed into the noise
    # contribute nothing but scatter, and including them drags a fast-decaying
    # process towards one. The usable range ends at the first lag that drops
    # below a fiftieth of the first, which is a few autocorrelation times for
    # any ratio worth estimating.
    if slope_array.size and slope_array[0] > 0:
        floor = slope_array[0] / 50.0
        usable = kmax
        for position, value in enumerate(slope_array):
            if value <= floor:
                usable = max(3, position)
                break
        if usable < kmax:
            kmax = usable
            slope_array = slope_array[:kmax]
    ks = np.arange(1, kmax + 1, dtype=np.float64)

    offset = 0.0
    if with_offset:
        # A constant drive shows up as a floor the decay never crosses. Taking
        # the mean of the last fifth of the lags as that floor is crude and it
        # is checked: if removing it makes the fit worse, it is not applied.
        tail = slope_array[max(1, int(kmax * 0.8)) :]
        offset = float(tail.mean()) if tail.size else 0.0

    def _fit(target: Any) -> tuple[float, float]:
        positive = target > 1e-12
        if positive.sum() < 3:
            return 0.0, 0.0
        x = ks[positive]
        y = np.log(target[positive])
        slope, intercept = np.polyfit(x, y, 1)
        predicted = slope * x + intercept
        residual = float(((y - predicted) ** 2).sum())
        total = float(((y - y.mean()) ** 2).sum())
        r_squared = 1.0 - residual / total if total > 0 else 0.0
        return float(np.exp(slope)), r_squared

    plain_m, plain_r2 = _fit(slope_array)
    if with_offset and offset > 0:
        shifted_m, shifted_r2 = _fit(slope_array - offset)
        if shifted_r2 > plain_r2:
            m, r_squared, method = shifted_m, shifted_r2, "multistep-regression+offset"
        else:
            m, r_squared, method, offset = plain_m, plain_r2, "multistep-regression", 0.0
    else:
        m, r_squared, method, offset = plain_m, plain_r2, "multistep-regression", 0.0

    tau = -1.0 / math.log(m) if 0 < m < 1 else float("inf")
    return BranchingEstimate(
        m=m,
        offset=offset,
        r_squared=r_squared,
        kmax=kmax,
        samples=int(n),
        autocorrelation_time=tau,
        method=method,
    )


@dataclass
class Avalanches:
    """Cascades pulled out of an activity trace."""

    sizes: list[int]
    durations: list[int]
    threshold: float
    active_fraction: float

    def summary(self) -> dict[str, Any]:
        return {
            "count": len(self.sizes),
            "threshold": round(self.threshold, 4),
            "active_fraction": round(self.active_fraction, 4),
            "largest": max(self.sizes) if self.sizes else 0,
            "longest": max(self.durations) if self.durations else 0,
            "mean_size": round(sum(self.sizes) / len(self.sizes), 3) if self.sizes else 0.0,
        }


def extract_avalanches(
    activity: Sequence[float],
    *,
    percentile: float = 25.0,
) -> Avalanches:
    """Contiguous runs above a quiet threshold, with the area under each.

    Beggs and Plenz define an avalanche between two silent bins. A recording of
    a machine is never silent, so the threshold is a low percentile of the trace
    and the size is the area above it. That choice changes the exponents, which
    is why the percentile is reported alongside them.
    """
    import numpy as np

    values = np.asarray(list(activity), dtype=np.float64)
    if values.size == 0:
        return Avalanches([], [], 0.0, 0.0)
    threshold = float(np.percentile(values, percentile))
    above = values > threshold
    sizes: list[int] = []
    durations: list[int] = []
    run_size = 0.0
    run_length = 0
    for value, is_active in zip(values, above, strict=True):
        if is_active:
            run_size += float(value) - threshold
            run_length += 1
        elif run_length:
            sizes.append(max(1, int(round(run_size))))
            durations.append(run_length)
            run_size = 0.0
            run_length = 0
    if run_length:
        sizes.append(max(1, int(round(run_size))))
        durations.append(run_length)
    return Avalanches(
        sizes=sizes,
        durations=durations,
        threshold=threshold,
        active_fraction=float(above.mean()),
    )


@dataclass
class CriticalityReport:
    """Where the system sits, and whether the evidence supports saying so."""

    branching: BranchingEstimate
    naive: float
    avalanches: dict[str, Any]
    size_exponent: dict[str, float]
    duration_exponent: dict[str, float]
    measured_gamma: float
    predicted_gamma: float
    crackling_error: float
    verdict: str

    def as_json(self) -> dict[str, Any]:
        return {
            "branching": self.branching.as_json(),
            "naive_branching_ratio": round(self.naive, 5),
            "subsampling_bias": round(self.branching.m - self.naive, 5),
            "avalanches": self.avalanches,
            "size_exponent_tau": {k: round(v, 4) for k, v in self.size_exponent.items()},
            "duration_exponent_alpha": {
                k: round(v, 4) for k, v in self.duration_exponent.items()
            },
            "measured_gamma": round(self.measured_gamma, 4),
            "predicted_gamma": round(self.predicted_gamma, 4),
            "crackling_error": round(self.crackling_error, 4),
            "verdict": self.verdict,
        }


def assess(activity: Sequence[float], *, percentile: float = 25.0) -> CriticalityReport:
    """Estimate the branching ratio and test whether the avalanches agree with it."""
    import numpy as np

    branching = branching_ratio_mr(activity)
    naive = naive_branching_ratio(activity)
    avalanches = extract_avalanches(activity, percentile=percentile)
    size_fit = power_law_fit(avalanches.sizes)
    duration_fit = power_law_fit(avalanches.durations)

    measured_gamma = 0.0
    if avalanches.sizes and avalanches.durations:
        by_duration: dict[int, list[int]] = {}
        for size, duration in zip(avalanches.sizes, avalanches.durations, strict=True):
            by_duration.setdefault(duration, []).append(size)
        points = [
            (math.log(duration), math.log(sum(sizes) / len(sizes)))
            for duration, sizes in sorted(by_duration.items())
            if duration >= 2 and sizes
        ]
        if len(points) >= 3:
            xs = np.asarray([p[0] for p in points])
            ys = np.asarray([p[1] for p in points])
            measured_gamma = float(np.polyfit(xs, ys, 1)[0])

    tau = size_fit.get("alpha", 0.0)
    alpha = duration_fit.get("alpha", 0.0)
    predicted_gamma = (alpha - 1.0) / (tau - 1.0) if tau > 1.0 else 0.0
    error = abs(predicted_gamma - measured_gamma) if measured_gamma else float("inf")

    fits_are_usable = (
        size_fit.get("ks", 1.0) < 0.2
        and duration_fit.get("ks", 1.0) < 0.2
        and size_fit.get("tail_n", 0) >= 32
    )
    if not fits_are_usable:
        verdict = "avalanche exponents are not reliable enough to test criticality"
    elif error < 0.2:
        verdict = f"consistent with criticality: exponents satisfy the scaling relation, m={branching.m:.3f}"
    else:
        verdict = (
            "power laws present but the scaling relation fails, so they are not "
            "evidence of criticality"
        )

    return CriticalityReport(
        branching=branching,
        naive=naive,
        avalanches=avalanches.summary(),
        size_exponent=size_fit,
        duration_exponent=duration_fit,
        measured_gamma=measured_gamma,
        predicted_gamma=predicted_gamma,
        crackling_error=error,
        verdict=verdict,
    )

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
from dataclasses import dataclass
from typing import Any

from .topology import power_law_fit

logger = logging.getLogger("Aura.Connectome.Criticality")

__all__ = [
    "Avalanches",
    "BranchingEstimate",
    "CriticalityReport",
    "MINIMUM_DECADES",
    "assess",
    "branching_ratio_mr",
    "cascades_beyond_rate",
    "exponent_against_recording_width",
    "extract_avalanches",
    "naive_branching_ratio",
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
        # Taking logs turns an exponential into a line and turns the noise on a
        # small slope into a large error on its logarithm. Weighting each lag by
        # the size of its own correlation is the standard correction, and
        # without it a fast-decaying process reads as slower than it is.
        weights = target[positive]
        slope, intercept = np.polyfit(x, y, 1, w=weights)
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

    The area is measured in units of the threshold, not in whatever the trace
    is measured in. It was measured in the trace's own units and rounded to an
    integer, so a mesh whose activity runs at a ten-thousandth gave every
    cascade a size of 1 — 136 of them, all identical, and no distribution to
    fit. Dividing by the threshold makes the size a ratio and the exponents
    stop depending on the amplitude of the signal.

    For the definition Beggs and Plenz actually used, where a bin is silent
    only when no UNIT is active and a cascade's size is how many unit-bins it
    contains, pass the per-unit trace to `extract_avalanches_per_unit`. A
    scalar trace cannot express it: the mean over units is above its own
    quartile three quarters of the time, so the runs are not cascades.
    """
    import numpy as np

    values = np.asarray(list(activity), dtype=np.float64)
    if values.size == 0:
        return Avalanches([], [], 0.0, 0.0)
    threshold = float(np.percentile(values, percentile))
    above = values > threshold
    scale = threshold if threshold > 0 else 1.0
    sizes: list[int] = []
    durations: list[int] = []
    run_size = 0.0
    run_length = 0
    for value, is_active in zip(values, above, strict=True):
        if is_active:
            run_size += (float(value) - threshold) / scale
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


def extract_avalanches_per_unit(
    activity: Any,
    *,
    percentile: float = 75.0,
) -> Avalanches:
    """Cascades the way Beggs and Plenz counted them.

    `activity` is (time, units). A unit is active in a bin when it is above its
    OWN quiet level, a bin is silent when no unit is active in it, and a cascade
    is a run of non-silent bins whose size is how many unit-bins it contains.
    That is an integer by construction and does not move when the whole
    recording is scaled, which is what makes the exponents comparable to a
    recording of tissue.

    The threshold is per unit because units are not on the same scale. A mesh
    column that barely moves would never cross a threshold set from the busiest
    one, and would be silent for the whole recording.
    """
    import numpy as np

    values = np.asarray(activity, dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        return Avalanches([], [], 0.0, 0.0)
    magnitude = np.abs(values)
    thresholds = np.percentile(magnitude, percentile, axis=0)
    active = magnitude > thresholds[None, :]
    per_bin = active.sum(axis=1)
    sizes: list[int] = []
    durations: list[int] = []
    run_size = 0
    run_length = 0
    for count in per_bin:
        if count > 0:
            run_size += int(count)
            run_length += 1
        elif run_length:
            sizes.append(run_size)
            durations.append(run_length)
            run_size = 0
            run_length = 0
    if run_length:
        sizes.append(run_size)
        durations.append(run_length)
    return Avalanches(
        sizes=sizes,
        durations=durations,
        threshold=float(np.mean(thresholds)),
        active_fraction=float((per_bin > 0).mean()),
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


def cascades_beyond_rate(
    spikes: Any,
    *,
    shuffles: int = 8,
    seed: int = 11,
    percentile: float = 0.0,
) -> dict[str, Any]:
    """Are these cascades, or is this the firing rate arriving in bins?

    Shift each unit's own spike train by its own random amount. Every firing
    rate survives untouched and every coincidence between units is destroyed,
    so what separates the real recording from the shifted one is exactly the
    coincidence. Beggs and Plenz ran this control against their own arrays.

    What gets compared is how much the number of units active in a bin varies.
    Independent units make that variance the sum of their own, which is what
    the shuffled ensemble measures, and units that recruit each other push it
    above that. Comparing the avalanche exponent instead does not work and was
    tried first: at high occupancy the avalanches are separated by whichever
    bins happen to be silent, so rolling the units moves the largest cascade
    around for reasons that have nothing to do with recruitment, and the test
    called independent Poisson structured and real recruitment flat.
    """
    import numpy as np

    values = np.asarray(spikes, dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        return {"verdict": "no recording"}
    magnitude = np.abs(values)
    thresholds = np.percentile(magnitude, percentile, axis=0)
    active = (magnitude > thresholds[None, :]).astype(np.float64)
    real_counts = active.sum(axis=1)
    real_spread = float(real_counts.var())

    rng = np.random.default_rng(seed)
    ticks = active.shape[0]
    spreads: list[float] = []
    for _ in range(max(2, shuffles)):
        rolled = np.empty_like(active)
        for unit in range(active.shape[1]):
            rolled[:, unit] = np.roll(active[:, unit], int(rng.integers(0, ticks)))
        spreads.append(float(rolled.sum(axis=1).var()))
    null_mean = sum(spreads) / len(spreads)
    null_sd = (sum((value - null_mean) ** 2 for value in spreads) / len(spreads)) ** 0.5
    # Three sigma, the same threshold this module already uses to call an
    # excursion a spike.
    excess = (real_spread - null_mean) / null_sd if null_sd > 0 else 0.0
    coincident = excess >= 3.0

    cascades = extract_avalanches_per_unit(values, percentile=percentile)
    fit = power_law_fit(cascades.sizes)
    return {
        "active_fraction": round(cascades.active_fraction, 4),
        "exponent": round(float(fit.get("alpha", 0.0)), 4),
        "population_variance": round(real_spread, 4),
        "shuffled_variance": round(null_mean, 4),
        "shuffled_spread": round(null_sd, 6),
        "sigmas_above_independence": round(float(excess), 3),
        "shuffles": max(2, shuffles),
        "verdict": (
            f"units fire together {excess:.1f} sigma more than their own rates "
            "explain, so these runs are cascades"
            if coincident
            else (
                "shifting every unit against every other one leaves the same "
                f"spread ({excess:.1f} sigma), so this is the firing rate "
                "arriving in bins and not a cascade measurement"
            )
        ),
    }


def exponent_against_recording_width(
    spikes: Any,
    *,
    widths: Sequence[int] = (30, 60, 120, 240, 480, 960, 1920, 0),
    seed: int = 1000,
) -> dict[str, Any]:
    """Refit the avalanche exponents while reading more and more of the system.

    An exponent is a claim about scaling and does not care how much of a system
    a recording reads. The slope of a finite window's cutoff does, and it gets
    steeper as the window narrows, so a curve that moves is the answer to
    whether a number was ever a measurement.

    Each width takes its own subsample, sets the analysis bin to that
    subsample's mean inter-event interval the way Beggs and Plenz set theirs,
    and refits. A width of 0 means every unit.
    """
    import numpy as np

    values = np.asarray(spikes, dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        return {"rows": [], "verdict": "no recording"}
    ticks, units = values.shape
    rows: list[dict[str, Any]] = []
    for width in widths:
        keep = units if width <= 0 else min(int(width), units)
        chosen = np.sort(
            np.random.default_rng(seed + keep).choice(units, size=keep, replace=False)
        )
        sub = values[:, chosen]
        events = int((sub.sum(axis=1) > 0).sum())
        bin_ticks = max(1, round(ticks / max(1, events)))
        usable = (ticks // bin_ticks) * bin_ticks
        folded = (
            sub[:usable].reshape(usable // bin_ticks, bin_ticks, keep).sum(axis=1)
            if bin_ticks > 1 and usable
            else sub
        )
        # A unit is active in a wide bin if it fired at all in it. Summing
        # leaves a count, and the extractor's per-unit threshold is that
        # unit's own minimum, so a unit that fired in every wide bin would
        # have its quietest bins counted as silence -- which at these bin
        # widths is most of them.
        cascades = extract_avalanches_per_unit(
            (folded > 0).astype(float) if bin_ticks > 1 else folded, percentile=0.0
        )
        fit = power_law_fit(cascades.sizes)
        duration = power_law_fit(cascades.durations)
        rows.append(
            {
                "units_recorded": keep,
                "bin_ticks": bin_ticks,
                "avalanches": len(cascades.sizes),
                "active_fraction": round(cascades.active_fraction, 4),
                "largest": max(cascades.sizes) if cascades.sizes else 0,
                "size_exponent": round(float(fit.get("alpha", 0.0)), 4),
                "size_decades": round(float(fit.get("decades", 0.0)), 4),
                "size_ks": round(float(fit.get("ks", 1.0)), 4),
                "duration_exponent": round(float(duration.get("alpha", 0.0)), 4),
                "duration_decades": round(float(duration.get("decades", 0.0)), 4),
            }
        )
    fitted = [row for row in rows if row["size_exponent"] > 0.0]
    moved = (
        max(row["size_exponent"] for row in fitted) - min(row["size_exponent"] for row in fitted)
        if fitted
        else 0.0
    )
    return {
        "rows": rows,
        "units": units,
        "exponent_range": round(moved, 4),
        "verdict": (
            f"the size exponent moves {moved:.3f} across recording widths, so it is "
            "reading the window and not only the dynamics"
            if moved > 0.5
            else f"the size exponent holds within {moved:.3f} across recording widths"
        ),
    }


#: How far a fitted tail has to run before an exponent means anything.
#:
#: Clauset, Shalizi and Newman put the floor above a decade: under that a power
#: law is not separable from a lognormal, an exponential, or the shoulder of a
#: finite system's cutoff, whatever the KS distance says. Beggs and Plenz fitted
#: theirs over nearly two decades of avalanche size. Fitting through a cutoff
#: returns an exponent STEEPER than the real one, so a system measured over half
#: a decade will look like its cascades are smaller than they are.
MINIMUM_DECADES: float = 1.0


def assess(
    activity: Sequence[float],
    *,
    percentile: float = 25.0,
    per_unit: Any = None,
    per_unit_percentile: float = 75.0,
) -> CriticalityReport:
    """Estimate the branching ratio and test whether the avalanches agree with it.

    The branching ratio comes from the summed trace. The cascades come from
    `per_unit` when it is given — (time, units), the shape Beggs and Plenz
    counted in — and from the summed trace otherwise, which is weaker and says
    so through its own fit quality.
    """
    import numpy as np

    branching = branching_ratio_mr(activity)
    naive = naive_branching_ratio(activity)
    avalanches = (
        extract_avalanches_per_unit(per_unit, percentile=per_unit_percentile)
        if per_unit is not None
        else extract_avalanches(activity, percentile=percentile)
    )
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
        and size_fit.get("decades", 0.0) >= MINIMUM_DECADES
        and duration_fit.get("decades", 0.0) >= MINIMUM_DECADES
    )
    if not fits_are_usable:
        verdict = (
            "avalanche exponents are not reliable enough to test criticality: "
            f"{size_fit.get('decades', 0.0):.2f} decades of size and "
            f"{duration_fit.get('decades', 0.0):.2f} of duration, "
            f"KS {size_fit.get('ks', 1.0):.3f} and {duration_fit.get('ks', 1.0):.3f}, "
            f"{int(size_fit.get('tail_n', 0))} in the fitted tail"
        )
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

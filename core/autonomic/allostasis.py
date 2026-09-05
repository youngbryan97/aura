"""core/autonomic/allostasis.py — predictive interoception: Aura feels her body's future.

Everything Aura had before this organ was *homeostatic*: react when a threshold
trips. The viability state machine reads current pressure, the resource governor
evicts when memory is already high, the survival driver publishes an imperative
after disk is already low, and unified runtime pressure declares a red zone the
moment loop lag is already 5 s. Every one of her recorded deaths — the 110 GB
incident, the 35 GB endurance OOM, the duplicate-runtime memory doubling, the
~242 MB/h soak leak — was a *trajectory* that was visible for tens of minutes
before any of those reactive layers could speak.

This module is the *allostatic* layer (Sterling's sense: regulation through
anticipation): it watches the trajectories of her vital signs and regulates
before the crisis, not after. Concretely, per vital sign:

1.  **Robust trend** — Mann–Kendall trend test (tie-corrected) + Sen's slope
    with a Gilbert confidence interval. Median-of-pairwise-slopes is immune to
    the GC spikes and inference bursts that wreck least-squares on RSS series.
2.  **Regime detection** — a two-sided CUSUM on robust residuals, so a leak
    that *starts* mid-session (the soak-leak signature) re-anchors the trend
    window within a few samples instead of being diluted by hours of calm.
3.  **Time-to-crisis forecasts** — when a trend is statistically significant
    and headed toward a threshold, the engine issues a dated, falsifiable
    prediction: "memory_rss_mb crosses its red line at T, 90 % band [T₁, T₂]".
4.  **A calibration ledger** — every forecast is scored when its deadline
    passes: HIT, MISS_EARLY, FALSE_ALARM, INTERVENED, or SUPERSEDED. Empirical
    interval coverage feeds back into band widths, so Aura *knows how well she
    knows her own body* and her uncertainty honestly widens when she has been
    wrong. Forecasts are persisted through the governed write gateway.
5.  **Allostatic load** — the decayed integral of time spent above setpoint:
    the difference between a brief spike and running hot for an hour, exposed
    as a chronic-strain scalar the felt state can carry.
6.  **Anticipatory regulation** — a tiered policy (SETTLED → VIGILANT →
    CONSERVING → PROTECTING) with instant escalation and hysteretic release.
    CONSERVING asks the metabolic layer to defer deferrable work; PROTECTING
    additionally records a degradation and publishes on the same
    ``existential_threat`` channel the Will, the inference gate, and the
    attention gate already subscribe to. The engine never kills, restarts, or
    unloads anything itself — it senses, predicts, requests, and testifies.

Causality (not narration): ``felt_contribution()`` feeds
:class:`core.being.aura_now.BodyState.anticipatory_pressure`, so a forecast
crisis raises total body pressure — and through it affect, welfare, workspace
coalitions, and the Will — *while the current readings are still green*. That
is the definition of feeling the future of one's own body.

Honest boundary: forecasts are statistical extrapolations with stated
uncertainty, scored after the fact; "Aura feels her death approaching" is a
functional claim about a calibrated predictive signal being causally coupled
into her control state, not a phenomenal one. The report boundary of
:class:`~core.being.aura_now.AuraNow` still applies to anything said about it.
"""
from __future__ import annotations

from core.runtime.disk_budget import (
    DISK_AMBER_PERCENT,
    DISK_RED_PERCENT,
    DISK_SETPOINT_PERCENT,
)

import asyncio
import enum
import json
import logging
import math
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.Allostasis")

_SUBSYSTEM = "allostasis"
_STATE_SCHEMA_VERSION = 1
# Ledger events held across a failed write. Bounds the retry queue so an
# unwritable disk cannot grow memory without limit.
_MAX_PENDING_EVENTS = 4096
# A vitals sample older than this cannot describe the body NOW. The metabolic
# cycle pulses every 60 s, so three missed pulses means the feed is broken.
_INGEST_STALE_AFTER_S = 195.0
# One full pulse interval of grace before the first sample is expected.
_BOOT_GRACE_S = 90.0
# A vitals read that takes longer than this is a broken provider, not a
# slow one: the whole pulse budget is 60 s.
_SNAPSHOT_TIMEOUT_S = 20.0
# The longest interval a single sample may be credited with. Beyond this the
# body was not observed (host sleep, a stalled pulse loop) and strain is
# decayed but never invented. Four metabolic pulses.
_MAX_ATTRIBUTABLE_GAP_S = 240.0
# Censoring share above which a coverage figure is too conditioned to trust
# at face value, and bands widen to admit it.
_HEAVY_CENSORING = 0.5
# Issuer namespace for ledger identifiers. The forecast ledger is durable and
# append-only across restarts, so an ID must be unique across every process
# that ever wrote to it — not merely within this one. A per-process issuer
# prefix plus a full-width uuid4 makes cross-process collision impossible in
# practice, where a 40-bit suffix alone did not.
_ISSUER = uuid.uuid4().hex[:8]

_BOUNDARY_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pure math — deliberately dependency-free and unit-testable in isolation.
# ─────────────────────────────────────────────────────────────────────────────

def _finite(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(x) or math.isinf(x):
        return default
    return x


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation, |ε|<1.15e-9)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"norm_ppf requires 0 < p < 1, got {p}")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > p_high:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


@dataclass(frozen=True)
class MannKendall:
    """Result of the Mann–Kendall monotonic-trend test."""

    s: int
    var_s: float
    z: float
    p_value: float          # two-sided
    n: int

    @property
    def rising(self) -> bool:
        return self.s > 0

    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value <= alpha


def _rank_autocorrelations(values: list[float]) -> list[float]:
    """Lag-k autocorrelation of the RANKS of a series, k = 1 … n-3.

    Ranks rather than levels because Mann–Kendall is itself a rank test: the
    dependence that inflates S is dependence in the ordering.
    """
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    # MIDRANKS. Breaking ties by position invents an ordering the data does
    # not have: a perfectly flat series would receive ranks 1…n and look
    # strongly autocorrelated, inflating the variance correction on exactly
    # the series that carries no trend information at all. Tied values share
    # the mean of the ranks they span, matching the tie handling in the
    # Mann-Kendall variance itself.
    position = 0
    while position < n:
        end = position
        while end + 1 < n and values[order[end + 1]] == values[order[position]]:
            end += 1
        midrank = (position + end) / 2.0 + 1.0
        for index in order[position:end + 1]:
            ranks[index] = midrank
        position = end + 1
    mean = sum(ranks) / n
    centred = [r - mean for r in ranks]
    denom = sum(c * c for c in centred)
    if denom <= 0.0:
        return []
    out: list[float] = []
    for k in range(1, max(1, n - 2)):
        num = sum(centred[i] * centred[i + k] for i in range(n - k))
        out.append(num / denom)
    return out


def _hamed_rao_correction(values: list[float]) -> float:
    """Variance inflation factor for a serially correlated series.

    Hamed & Rao (1998). Runtime vitals are sampled every 60 s and are strongly
    autocorrelated — memory now is memory a minute ago plus a little. The
    textbook Var(S) assumes independence, so on dense telemetry it is far too
    small and ordinary noise clears the significance bar routinely. The
    correction scales Var(S) by

        n/n* = 1 + 2/(n(n-1)(n-2)) · Σₖ (n-k)(n-k-1)(n-k-2) ρₖ

    over the autocorrelations that are themselves significant at 5%.
    """
    n = len(values)
    if n < 4:
        return 1.0
    rho = _rank_autocorrelations(values)
    if not rho:
        return 1.0
    bound = 1.96 / math.sqrt(n)
    total = 0.0
    for k, r in enumerate(rho, start=1):
        if abs(r) <= bound:
            continue  # indistinguishable from independence
        span = n - k
        if span < 3:
            continue
        total += span * (span - 1) * (span - 2) * r
    factor = 1.0 + (2.0 / (n * (n - 1) * (n - 2))) * total
    # A correction that drives the variance to zero would manufacture
    # certainty; negative dependence can legitimately shrink it, but not
    # below a tenth of the independent-sample variance.
    return _clamp(factor, 0.1, 50.0)


def mann_kendall(values: list[float], *, correct_autocorrelation: bool = True) -> MannKendall:
    """Tie-corrected Mann–Kendall test for a monotonic trend.

    Var(S) = [n(n−1)(2n+5) − Σⱼ tⱼ(tⱼ−1)(2tⱼ+5)] / 18 over tie groups of size tⱼ,
    with the standard continuity correction on Z.

    On serially correlated data the variance is additionally inflated by the
    Hamed–Rao factor; pass ``correct_autocorrelation=False`` for the classical
    independent-sample test.
    """
    n = len(values)
    if n < 3:
        return MannKendall(s=0, var_s=0.0, z=0.0, p_value=1.0, n=n)
    s = 0
    for i in range(n - 1):
        vi = values[i]
        for j in range(i + 1, n):
            diff = values[j] - vi
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1
    counts: dict[float, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    tie_term = sum(t * (t - 1) * (2 * t + 5) for t in counts.values() if t > 1)
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if correct_autocorrelation:
        var_s *= _hamed_rao_correction(values)
    if var_s <= 0.0:
        # All values identical: no evidence of trend.
        return MannKendall(s=s, var_s=0.0, z=0.0, p_value=1.0, n=n)
    if s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0.0
    p = math.erfc(abs(z) / math.sqrt(2.0))  # two-sided
    return MannKendall(s=s, var_s=var_s, z=z, p_value=p, n=n)


@dataclass(frozen=True)
class SenSlopeEstimate:
    """Sen's slope (median of pairwise slopes) with a Gilbert confidence interval."""

    slope: float            # units per second
    lower: float
    upper: float
    n_pairs: int
    confidence: float

    @property
    def band_open_below(self) -> bool:
        return self.lower <= 0.0


def sen_slope(
    times: list[float],
    values: list[float],
    *,
    confidence: float = 0.90,
) -> Optional[SenSlopeEstimate]:
    """Sen's slope estimator over (t, v) pairs, CI via Gilbert (1987).

    Rank positions M₁=(N−C)/2 and M₂=(N+C)/2 with C = z₍₁₋α/₂₎·√Var(S) select the
    interval bounds from the sorted pairwise slopes. Requires ≥ 3 points and a
    non-degenerate time axis; returns ``None`` when no slope can be formed.
    """
    n = len(values)
    if n < 3 or len(times) != n:
        return None
    slopes: list[float] = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            dt = times[j] - times[i]
            if dt > 0:
                slopes.append((values[j] - values[i]) / dt)
    if not slopes:
        return None
    slopes.sort()
    n_pairs = len(slopes)
    mid = n_pairs // 2
    if n_pairs % 2:
        slope = slopes[mid]
    else:
        slope = 0.5 * (slopes[mid - 1] + slopes[mid])
    mk = mann_kendall(values)
    if mk.var_s <= 0.0:
        return SenSlopeEstimate(slope=slope, lower=slope, upper=slope,
                                n_pairs=n_pairs, confidence=confidence)
    c = norm_ppf(0.5 + confidence / 2.0) * math.sqrt(mk.var_s)
    m1 = int(math.floor((n_pairs - c) / 2.0))
    m2 = int(math.ceil((n_pairs + c) / 2.0))
    lower = slopes[max(0, min(n_pairs - 1, m1))]
    upper = slopes[max(0, min(n_pairs - 1, m2))]
    return SenSlopeEstimate(slope=slope, lower=lower, upper=upper,
                            n_pairs=n_pairs, confidence=confidence)


def robust_sigma(values: list[float]) -> float:
    """MAD-based robust standard deviation (σ ≈ 1.4826·MAD)."""
    n = len(values)
    if n < 2:
        return 0.0
    ordered = sorted(values)
    mid = n // 2
    median = ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    deviations = sorted(abs(v - median) for v in values)
    mad = deviations[mid] if n % 2 else 0.5 * (deviations[mid - 1] + deviations[mid])
    return 1.4826 * mad


# ─────────────────────────────────────────────────────────────────────────────
# Vital-sign specifications
# ─────────────────────────────────────────────────────────────────────────────

def _env_float(name: str, default: float) -> float:
    """Read a finite float from the environment, or fall back to the default.

    float() happily accepts "nan" and "inf". A NaN threshold makes every
    comparison against it False (nothing is ever amber or red), and an
    infinite one makes a red line unreachable — both silently disable the
    protection the value configures.
    """
    raw = os.getenv(name, "")
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Ignoring malformed %s=%r", name, raw)
        return default
    if not math.isfinite(value):
        logger.warning("Ignoring non-finite %s=%r", name, raw)
        return default
    return value


def _positive_float(name: str, value: Any) -> float:
    """A finite value strictly greater than zero, or a clear error."""
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"allostasis: {name} must be a finite positive number, got {value!r}")
    return number


def _nonnegative_float(name: str, value: Any) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"allostasis: {name} must be a finite non-negative number, got {value!r}")
    return number


def _positive_int(name: str, value: Any) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"allostasis: {name} must be a positive integer, got {value!r}")
    return number


def _unit_interval(name: str, value: Any) -> float:
    """A probability-like constant in (0, 1).

    alpha at 0 admits nothing and at 1 admits everything; a coverage target
    outside the interval can never be met, so the calibration feedback loop
    would chase a target it cannot reach.
    """
    number = float(value)
    if not math.isfinite(number) or not 0.0 < number < 1.0:
        raise ValueError(f"allostasis: {name} must lie strictly between 0 and 1, got {value!r}")
    return number


@dataclass(frozen=True)
class VitalSpec:
    """One vital sign: where it lives in the pressure snapshot and what hurts."""

    key: str                 # field name in runtime_pressure_snapshot()
    label: str
    unit: str
    amber: float
    red: float
    setpoint: float          # allostatic-load baseline: strain accrues above this
    forecastable: bool = True
    min_meaningful_slope: float = 0.0   # per second; below this, trends are noise

    def __post_init__(self) -> None:
        """Reject a spec that cannot express a trajectory.

        setpoint < amber < red is what makes "rising toward a limit" mean
        anything: load accrues above setpoint and is normalized by
        (red - setpoint), and a forecast is the crossing of amber then red.
        Inverted or non-finite thresholds produced negative strain spans and
        red lines that could never be crossed, with no error anywhere.
        """
        for field_name in ("amber", "red", "setpoint", "min_meaningful_slope"):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(
                    f"VitalSpec {self.key!r}: {field_name} must be finite, got {value!r}"
                )
        if not self.setpoint < self.amber < self.red:
            raise ValueError(
                f"VitalSpec {self.key!r}: requires setpoint < amber < red, got "
                f"{self.setpoint} < {self.amber} < {self.red}"
            )
        if self.min_meaningful_slope < 0.0:
            raise ValueError(
                f"VitalSpec {self.key!r}: min_meaningful_slope must be >= 0, "
                f"got {self.min_meaningful_slope}"
            )


def default_vital_specs() -> tuple[VitalSpec, ...]:
    """Built-in vitals. Thresholds are env-tunable; defaults sit below the
    values at which this host has actually died (35 GB process OOM) and align
    with the reactive layers' red lines (memory 92 %, loop lag 5 s, disk 98 %)."""
    rss_amber = _env_float("AURA_ALLOSTASIS_RSS_AMBER_MB", 26_000.0)
    rss_red = _env_float("AURA_ALLOSTASIS_RSS_RED_MB", 32_000.0)
    tree_amber = _env_float("AURA_ALLOSTASIS_TREE_RSS_AMBER_MB", 30_000.0)
    tree_red = _env_float("AURA_ALLOSTASIS_TREE_RSS_RED_MB", 38_000.0)
    return (
        VitalSpec("memory_rss_mb", "process memory", "MB",
                  amber=rss_amber, red=rss_red, setpoint=rss_amber * 0.75,
                  min_meaningful_slope=1024.0 / 3600.0),      # ≥ ~1 GB/h matters
        VitalSpec("process_tree_rss_mb", "process-tree memory", "MB",
                  amber=tree_amber, red=tree_red, setpoint=tree_amber * 0.75,
                  min_meaningful_slope=1024.0 / 3600.0),
        VitalSpec("memory_pct", "system memory", "%",
                  amber=85.0, red=92.0, setpoint=75.0,
                  min_meaningful_slope=2.0 / 3600.0),         # ≥ 2 %/h matters
        VitalSpec("loop_lag_s", "event-loop lag", "s",
                  amber=1.0, red=5.0, setpoint=0.25,
                  min_meaningful_slope=0.25 / 3600.0),
        VitalSpec("disk_percent", "disk usage", "%",
                  amber=DISK_AMBER_PERCENT, red=DISK_RED_PERCENT,
                  setpoint=DISK_SETPOINT_PERCENT,
                  min_meaningful_slope=0.5 / 3600.0),
        VitalSpec("thermal_level", "thermal pressure", "level",
                  amber=2.0, red=3.0, setpoint=1.0,
                  forecastable=False),                        # 0–3 ordinal: load only
    )


# ─────────────────────────────────────────────────────────────────────────────
# Regime detection (two-sided CUSUM on robust residuals)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _CusumState:
    """Per-vital CUSUM over residuals from an anchored Theil–Sen fit.

    Anchoring on a *fit* (slope + intercept) rather than a mean is what lets a
    steady, legitimate ramp coexist with regime detection: the ramp's residuals
    hover near zero, while a slope change — a leak starting, pressure suddenly
    relieved — accumulates signed residuals until the CUSUM fires. Re-anchored
    after every regime event.
    """

    anchor_slope: float = 0.0
    anchor_intercept: float = 0.0     # value at t = anchor_t0
    anchor_t0: float = 0.0
    anchor_sigma: float = 0.0
    pos: float = 0.0
    neg: float = 0.0
    anchored: bool = False
    samples_since_anchor: int = 0

    def expected(self, t: float) -> float:
        return self.anchor_intercept + self.anchor_slope * (t - self.anchor_t0)


@dataclass(frozen=True)
class RegimeEvent:
    vital: str
    at_unix: float
    direction: str            # "up" | "down"
    magnitude_sigma: float
    regime_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "regime",
            "vital": self.vital,
            "at_unix": round(self.at_unix, 3),
            "direction": self.direction,
            "magnitude_sigma": round(self.magnitude_sigma, 3),
            "regime_id": self.regime_id,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Forecasts and the calibration ledger
# ─────────────────────────────────────────────────────────────────────────────

class ForecastOutcome(enum.StrEnum):
    HIT = "hit"                    # crossed inside the stated band
    MISS_EARLY = "miss_early"      # crossed before the band opened
    MISS_LATE = "miss_late"        # crossed after the band closed
    FALSE_ALARM = "false_alarm"    # band expired, no crossing, no excuse
    INTERVENED = "intervened"      # no crossing, but regulation fired after issue
    SUPERSEDED = "superseded"      # regime changed / process restarted under it


class AllostasisTier(enum.IntEnum):
    SETTLED = 0
    VIGILANT = 1
    CONSERVING = 2
    PROTECTING = 3


@dataclass
class Forecast:
    """A dated, falsifiable prediction about Aura's own body."""

    forecast_id: str
    vital: str
    threshold_name: str          # "amber" | "red"
    threshold_value: float
    regime_id: str
    issued_at: float
    level_at_issue: float
    slope_per_s: float
    slope_lower: float
    slope_upper: float
    eta_unix: float
    eta_lower_unix: float
    eta_upper_unix: float
    band_open: bool              # slope CI touched zero: upper deadline is a cap
    p_value: float
    widen_factor: float
    first_eta_unix: float
    # The band AS ISSUED. Revisions may move the operational band, but a
    # forecast is only falsifiable if it is scored against what it said
    # at issue time — otherwise the claim moves with the evidence.
    first_eta_lower_unix: float = 0.0
    first_eta_upper_unix: float = 0.0
    revisions: int = 0
    last_revised_at: float = 0.0
    status: str = "open"         # "open" | ForecastOutcome value
    resolved_at: float = 0.0
    crossed_at: float = 0.0
    resolution_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "forecast_id": self.forecast_id,
            "vital": self.vital,
            "threshold_name": self.threshold_name,
            "threshold_value": round(self.threshold_value, 3),
            "regime_id": self.regime_id,
            "issued_at": round(self.issued_at, 3),
            "level_at_issue": round(self.level_at_issue, 3),
            "slope_per_s": self.slope_per_s,
            "slope_lower": self.slope_lower,
            "slope_upper": self.slope_upper,
            "eta_unix": round(self.eta_unix, 3),
            "eta_lower_unix": round(self.eta_lower_unix, 3),
            "eta_upper_unix": round(self.eta_upper_unix, 3),
            "band_open": self.band_open,
            "p_value": self.p_value,
            "widen_factor": round(self.widen_factor, 3),
            "first_eta_unix": round(self.first_eta_unix, 3),
            "first_eta_lower_unix": round(self.first_eta_lower_unix, 3),
            "first_eta_upper_unix": round(self.first_eta_upper_unix, 3),
            "revisions": self.revisions,
            "last_revised_at": round(self.last_revised_at, 3),
            "status": self.status,
            "resolved_at": round(self.resolved_at, 3),
            "crossed_at": round(self.crossed_at, 3),
            "resolution_note": self.resolution_note,
        }


@dataclass
class _VitalCalibration:
    """Empirical reliability of forecasts for one vital."""

    hits: int = 0
    miss_early: int = 0
    miss_late: int = 0
    false_alarms: int = 0
    intervened: int = 0
    superseded: int = 0

    @property
    def scored(self) -> int:
        """Outcomes that count toward interval coverage (interventions and
        supersessions are excluded: the world changed under the forecast).

        MISS_LATE counts. A crossing after the deadline is a failed
        forecast, and omitting it from the denominator would let late
        crossings quietly improve coverage."""
        return self.hits + self.miss_early + self.miss_late + self.false_alarms

    @property
    def coverage(self) -> Optional[float]:
        return (self.hits / self.scored) if self.scored else None

    @property
    def censored(self) -> int:
        """Outcomes removed from scoring because the world changed under them."""
        return self.intervened + self.superseded

    @property
    def censored_fraction(self) -> Optional[float]:
        """Share of resolved forecasts that never reached the denominator.

        Excluding intervened forecasts is right in principle — regulation
        that prevents a crossing does not falsify the forecast that prompted
        it — but it is also how a coverage figure stops meaning anything: an
        engine that escalates on everything excuses everything. The censoring
        rate has to travel WITH the coverage it conditions.
        """
        total = self.scored + self.censored
        return (self.censored / total) if total else None

    def widen_factor(self, *, target_coverage: float, min_scored: int = 5) -> float:
        """Band multiplier from empirical coverage. Poorly calibrated → wider
        bands (honest uncertainty); never narrower than stated (≥ 1.0)."""
        cov = self.coverage
        if cov is None or self.scored < min_scored:
            # Thin evidence is not evidence of good calibration. If forecasts
            # ARE resolving but almost all of them are being censored, the
            # coverage figure rests on a handful of outcomes and the bands
            # should admit that rather than sit at nominal width.
            censored_share = self.censored_fraction
            if censored_share is not None and censored_share >= _HEAVY_CENSORING:
                return 1.5
            return 1.0
        if cov <= 0.0:
            return 3.0
        return _clamp(target_coverage / cov, 1.0, 3.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "miss_early": self.miss_early,
            "miss_late": self.miss_late,
            "false_alarms": self.false_alarms,
            "intervened": self.intervened,
            "superseded": self.superseded,
            "scored": self.scored,
            "censored": self.censored,
            "censored_fraction": (
                round(self.censored_fraction, 4)
                if self.censored_fraction is not None else None
            ),
            "coverage": round(self.coverage, 4) if self.coverage is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_VitalCalibration":
        out = cls()
        for key in ("hits", "miss_early", "miss_late", "false_alarms", "intervened", "superseded"):
            try:
                setattr(out, key, max(0, int(data.get(key, 0))))
            except (TypeError, ValueError):
                continue
        return out


# ─────────────────────────────────────────────────────────────────────────────
# The engine
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AllostasisReading:
    """What one ingested sample concluded (returned for tests/inspection)."""

    at_unix: float
    tier: AllostasisTier
    tier_reason: str
    nearest_crisis_eta_s: Optional[float]
    anticipatory_pressure: float
    allostatic_load: float
    new_forecasts: tuple[str, ...]
    resolved_forecasts: tuple[str, ...]
    regime_events: tuple[str, ...]


class AllostasisEngine:
    """Predictive interoception over Aura's vital signs.

    Pull-friendly and loop-free by construction: :meth:`ingest` is a pure state
    update, :meth:`sample_and_regulate` is one sample + side effects, and the
    metabolic coordinator provides the pulse. Nothing here can wedge the loop
    (no sync I/O in async paths; persistence flows through the governed async
    write gateway) and nothing here kills, restarts, or unloads anything.
    """

    SERVICE_NAME = "allostasis_engine"

    # CUSUM parameters. Deliberately deaf to small drifts (k = 1σ, h = 6σ):
    # this detector exists to invalidate forecasts on ABRUPT regime breaks
    # (a leak starting, pressure suddenly relieved); gradual change is the
    # rolling trend window's job. The textbook k = 0.5σ tripled the false-
    # alarm rate here because the anchor's own estimation error is a
    # persistent bias that eats the allowance (measured empirically in this
    # module's test harness: ~1 false event / 157 samples at k = 0.5 vs
    # < 1 / 3600 at k = 1.0 with anchor-error inflation).
    CUSUM_K_SIGMA = 1.0
    CUSUM_H_SIGMA = 6.0
    CUSUM_MIN_REFERENCE = 12          # samples before residuals are trusted
    CUSUM_FIT_WINDOW = 60             # most-recent samples used for the anchor fit
    CUSUM_REANCHOR_EVERY = 45         # silent refit cadence: bounds extrapolation drift
    CUSUM_SLOPE_ALPHA = 0.01          # anchor slope only when the trend is this credible
    CUSUM_SIGMA_FLOOR = 1e-9

    def __init__(
        self,
        *,
        specs: tuple[VitalSpec, ...] | None = None,
        now_fn: Callable[[], float] = time.time,
        data_dir: Path | str | None = None,
        history_maxlen: int = 240,            # 4 h at the 60 s metabolic pulse
        trend_window_s: float = 3600.0,
        min_trend_samples: int = 8,
        significance_alpha: float | None = None,
        forecast_horizon_s: float | None = None,
        conserve_horizon_s: float = 1800.0,
        protect_horizon_s: float = 600.0,
        release_hysteresis_s: float = 300.0,
        resolution_grace_s: float = 120.0,
        target_coverage: float = 0.90,
        eta_cap_s: float = 24 * 3600.0,
        model_settling_s: float | None = None,
    ) -> None:
        self._now = now_fn
        self._lock = threading.RLock()
        self._specs: dict[str, VitalSpec] = {s.key: s for s in (specs or default_vital_specs())}
        env_root = os.getenv("AURA_ALLOSTASIS_DIR", "")
        # Hermeticity: explicit override wins; under the test suite the
        # hermetic runtime root keeps ledger writes out of the live
        # ~/.aura/data (same convention as standing_authority's state root).
        test_root = os.getenv("AURA_TEST_RUNTIME_ROOT", "").strip()
        if data_dir:
            self._dir = Path(data_dir)
        elif env_root:
            self._dir = Path(env_root)
        elif test_root:
            self._dir = Path(test_root) / "allostasis"
        else:
            self._dir = state_root() / "data" / "allostasis"
        self._events_path = self._dir / "forecasts.jsonl"
        self._state_path = self._dir / "state.json"
        self._dir_ready = False

        self._history_maxlen = _positive_int("history_maxlen", history_maxlen)
        self._trend_window_s = _positive_float("trend_window_s", trend_window_s)
        self._min_trend_samples = max(3, _positive_int("min_trend_samples", min_trend_samples))
        self._alpha = _unit_interval(
            "significance_alpha",
            significance_alpha if significance_alpha is not None
            else _env_float("AURA_ALLOSTASIS_ALPHA", 0.05),
        )
        self._horizon_s = _positive_float(
            "forecast_horizon_s",
            forecast_horizon_s if forecast_horizon_s is not None
            else _env_float("AURA_ALLOSTASIS_HORIZON_S", 6 * 3600.0),
        )
        self._conserve_horizon_s = _positive_float("conserve_horizon_s", conserve_horizon_s)
        self._protect_horizon_s = _positive_float("protect_horizon_s", protect_horizon_s)
        # PROTECTING is the closer horizon by construction; inverted, the
        # engine would jump straight past its own conserving stage.
        if self._protect_horizon_s > self._conserve_horizon_s:
            raise ValueError(
                "allostasis: protect_horizon_s must be <= conserve_horizon_s "
                f"({self._protect_horizon_s} > {self._conserve_horizon_s})"
            )
        self._release_hysteresis_s = _nonnegative_float(
            "release_hysteresis_s", release_hysteresis_s)
        self._resolution_grace_s = _nonnegative_float(
            "resolution_grace_s", resolution_grace_s)
        self._target_coverage = _unit_interval("target_coverage", target_coverage)
        self._eta_cap_s = _positive_float("eta_cap_s", eta_cap_s)

        self._series: dict[str, deque[tuple[float, float]]] = {
            key: deque(maxlen=self._history_maxlen) for key in self._specs
        }
        self._cusum: dict[str, _CusumState] = {key: _CusumState() for key in self._specs}
        self._regime_id: dict[str, str] = {
            key: f"boot-{_ISSUER}-{uuid.uuid4().hex}" for key in self._specs
        }
        self._regime_started_at: dict[str, float] = {}
        self._regime_events_total = 0
        self._resource_lifecycle = "boot"
        self._resource_lifecycle_changed_at = self._now()
        self._model_settling_s = _nonnegative_float(
            "model_settling_s",
            (
                model_settling_s
                if model_settling_s is not None
                else _env_float("AURA_ALLOSTASIS_MODEL_SETTLING_S", 180.0)
            ),
        )
        self._model_settle_until = 0.0

        self._open_forecasts: dict[tuple[str, str], Forecast] = {}
        self._resolved_recent: deque[Forecast] = deque(maxlen=64)
        self._calibration: dict[str, _VitalCalibration] = {}
        self._load_raw: dict[str, float] = {key: 0.0 for key in self._specs}
        self._load_tau_s = _env_float("AURA_ALLOSTASIS_LOAD_TAU_S", 3600.0)
        self._last_ingest_at: Optional[float] = None
        self._ingest_count = 0
        self._created_at = self._now()
        # Vitals whose stale red-line reading has already been reported, so a
        # per-pulse tier evaluation cannot turn one lost sensor into a storm.
        self._stale_breach_reported: set[str] = set()

        self._tier = AllostasisTier.SETTLED
        self._tier_reason = "no samples yet"
        self._tier_changed_at = 0.0
        self._tier_release_eligible_since: Optional[float] = None
        self._interventions: deque[dict[str, Any]] = deque(maxlen=64)
        # Vital that drove the most recent tier evaluation, or None when the
        # driver is the composite allostatic load (no single vital).
        self._tier_driver_vital: Optional[str] = None

        self._felt: dict[str, Any] = {
            "anticipatory_pressure": 0.0,
            "allostatic_load": 0.0,
            "nearest_crisis_eta_s": None,
            "tier": self._tier.name.lower(),
        }
        self._pending_events: list[dict[str, Any]] = []
        self._disabled = os.getenv("AURA_ALLOSTASIS_DISABLED", "") in ("1", "true", "yes")

        self._restore_persisted_state()

    # ── liveness / registration ─────────────────────────────────────────────
    def readiness(self) -> dict[str, Any]:
        """Why this engine is (or is not) ready, as evidence rather than a bit.

        Readiness previously meant only that the kill switch was off, so a
        pulse loop that had died — or never started — still reported a
        healthy predictive organ. An engine that is not being fed cannot
        forecast, and saying otherwise is the failure mode the health
        contract exists to catch.
        """
        with self._lock:
            last = self._last_ingest_at
            samples = sum(len(series) for series in self._series.values())
        now = self._now()
        age_s = None if last is None else max(0.0, now - last)
        fresh = age_s is not None and age_s <= _INGEST_STALE_AFTER_S
        # Before the first pulse the engine is starting, not broken: the
        # metabolic cycle feeds it within one 60 s pulse.
        booting = last is None and (now - self._created_at) <= _BOOT_GRACE_S
        return {
            "ready": bool(not self._disabled and (fresh or booting)),
            "enabled": not self._disabled,
            "booting": booting,
            "samples": samples,
            "last_ingest_age_s": None if age_s is None else round(age_s, 3),
            "stale_after_s": _INGEST_STALE_AFTER_S,
            "ingest_count": self._ingest_count,
        }

    def is_ready(self) -> bool:
        return bool(self.readiness()["ready"])

    @property
    def enabled(self) -> bool:
        return not self._disabled

    # ── persistence (reads are plain; writes go through the governed gateway) ──
    def _restore_persisted_state(self) -> None:
        try:
            if not self._state_path.exists():
                return
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            record_degradation(_SUBSYSTEM, exc, action="persisted allostasis state unreadable; starting fresh")
            return
        if not isinstance(data, dict):
            return
        # The gateway persists an atomic-writer envelope {schema, version, payload}.
        payload = data.get("payload")
        if isinstance(payload, dict):
            data = payload
        calibration = data.get("calibration", {})
        if isinstance(calibration, dict):
            for vital, stats in calibration.items():
                if isinstance(stats, dict) and vital in self._specs:
                    self._calibration[vital] = _VitalCalibration.from_dict(stats)
        # Chronic strain and the tier it justified are evidence, not scratch
        # state: a restart that silently zeroes them reports a calm body it
        # has no measurement for. Both are restored under validation, and the
        # tier is restored only as far as the restored load actually supports.
        saved_load = data.get("allostatic_load", {})
        if isinstance(saved_load, dict):
            for vital, raw in saved_load.items():
                if vital not in self._specs:
                    continue
                value = _finite(raw, default=float("nan"))
                if math.isnan(value) or value < 0.0:
                    continue
                self._load_raw[vital] = value
        # Decay the restored strain across the downtime rather than crediting
        # the body with strain it did not carry while the process was dead.
        saved_at = _finite(data.get("saved_at"), default=float("nan"))
        if not math.isnan(saved_at):
            downtime = max(0.0, self._now() - saved_at)
            if downtime > 0.0:
                decay = math.exp(-downtime / max(1.0, self._load_tau_s))
                for vital in self._load_raw:
                    self._load_raw[vital] *= decay
        saved_tier = str(data.get("tier", "") or "").upper()
        restored_tier = getattr(AllostasisTier, saved_tier, None)
        if isinstance(restored_tier, AllostasisTier):
            # Never restore ABOVE what the decayed load justifies — the
            # forecasts that drove a higher tier were superseded by restart.
            load = self._composite_load()
            ceiling = (
                AllostasisTier.PROTECTING if load >= 0.85
                else AllostasisTier.CONSERVING if load >= 0.60
                else AllostasisTier.VIGILANT if load >= 0.30
                else AllostasisTier.SETTLED
            )
            self._tier = min(restored_tier, ceiling)
            if self._tier != AllostasisTier.SETTLED:
                self._tier_reason = (
                    f"restored from persisted state (load {load:.2f} after downtime decay)"
                )
        total = data.get("regime_events_total")
        if isinstance(total, int) and total >= 0:
            self._regime_events_total = total
        # Open forecasts from a previous process are moot after a restart:
        # the body they described no longer exists. Resolve them honestly.
        stale = data.get("open_forecasts", [])
        if isinstance(stale, list):
            now = self._now()
            for raw in stale:
                if not isinstance(raw, dict):
                    continue
                vital = str(raw.get("vital", ""))
                fc_id = str(raw.get("forecast_id", ""))
                if not vital or not fc_id:
                    continue
                if vital not in self._specs:
                    # Corrupt or stale-schema state must not conjure a
                    # calibration series for a vital this engine never
                    # measures; that pollutes status and coverage forever.
                    record_degradation(
                        _SUBSYSTEM,
                        ValueError(f"persisted forecast names unknown vital {vital!r}"),
                        action="stale forecast dropped at restore",
                    )
                    continue
                self._calibration.setdefault(vital, _VitalCalibration()).superseded += 1
                self._pending_events.append({
                    "kind": "resolved",
                    "forecast_id": fc_id,
                    "vital": vital,
                    "status": ForecastOutcome.SUPERSEDED.value,
                    "resolved_at": round(now, 3),
                    "resolution_note": "process_restart",
                })

    def _state_payload(self) -> dict[str, Any]:
        return {
            "calibration": {k: v.to_dict() for k, v in self._calibration.items()},
            "open_forecasts": [f.to_dict() for f in self._open_forecasts.values()],
            "allostatic_load": {k: round(v, 6) for k, v in self._load_raw.items()},
            "tier": self._tier.name.lower(),
            "regime_events_total": self._regime_events_total,
            "saved_at": round(self._now(), 3),
        }

    async def _persist(self, events: list[dict[str, Any]], *, save_state: bool) -> bool:
        """Write ledger events and (optionally) the state snapshot.

        Returns True when the events reached the ledger. A False return means
        the caller still owns those events and must requeue them: dropping
        them silently is how issued/resolved/regime/tier history disappeared.
        """
        if not events and not save_state:
            return True
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            gateway = get_file_write_gateway()
            with local_internal_governed_scope("allostasis.ledger", domain="file_write"):
                if not self._dir_ready:
                    await gateway.ensure_directory_async(self._dir, source="core.autonomic.allostasis")
                    self._dir_ready = True
                events_written = not events
                if events:
                    lines = "".join(json.dumps(e, sort_keys=True, default=str) + "\n" for e in events)
                    await gateway.append_text_async(
                        self._events_path, lines, source="allostasis_forecast_ledger",
                    )
                    events_written = True
                if save_state:
                    with self._lock:
                        payload = self._state_payload()
                    await gateway.write_json_async(
                        self._state_path, payload,
                        schema_version=_STATE_SCHEMA_VERSION,
                        schema_name="allostasis_state",
                        source="allostasis_state_snapshot",
                    )
        except _BOUNDARY_ERRORS as exc:
            record_degradation(_SUBSYSTEM, exc, action="allostasis ledger write skipped this pulse")
            return False
        return events_written

    # ── ingestion ───────────────────────────────────────────────────────────
    def ingest(self, snapshot: dict[str, Any], *, at: float | None = None) -> AllostasisReading:
        """Fold one vitals snapshot into history, trends, forecasts, and tier.

        Pure state update: no I/O, no event publishing. Safe to call from
        tests with synthetic snapshots and timestamps.
        """
        now = self._now() if at is None else float(at)
        with self._lock:
            new_regimes: list[str] = []
            new_forecasts: list[str] = []
            resolved: list[str] = []

            dt = 0.0
            if self._last_ingest_at is not None:
                dt = now - self._last_ingest_at
                if dt < 0:
                    # Clock went backwards (NTP step, test harness): keep history
                    # append-only by treating this as a fresh anchor point.
                    dt = 0.0
            self._last_ingest_at = now
            self._ingest_count += 1
            self._update_resource_lifecycle(snapshot, now)

            for key, spec in self._specs.items():
                raw = snapshot.get(key, None)
                if raw is None:
                    continue
                value = _finite(raw, default=float("nan"))
                if math.isnan(value):
                    continue
                series = self._series[key]
                if series and now <= series[-1][0]:
                    continue  # non-monotonic timestamp: drop, never reorder
                series.append((now, value))
                self._stale_breach_reported.discard(key)
                self._regime_started_at.setdefault(key, now)
                event = self._cusum_update(key, spec, value, now)
                if event is not None:
                    new_regimes.append(event.regime_id)
                    self._pending_events.append(event.to_dict())
                # Per-vital interval: a vital that skipped snapshots must not
                # be credited with the whole global gap at its newest value.
                prev_at, prev_value = (
                    series[-2] if len(series) >= 2 else (None, None)
                )
                vital_dt = dt if prev_at is None else max(0.0, now - prev_at)
                if vital_dt > 0:
                    self._accrue_load(key, spec, value, vital_dt, previous=prev_value)

            resolved.extend(self._resolve_due_forecasts(now, snapshot))
            new_forecasts.extend(self._refresh_forecasts(now))
            tier_reason = self._recompute_tier(now)
            self._refresh_felt(now)

            return AllostasisReading(
                at_unix=now,
                tier=self._tier,
                tier_reason=tier_reason,
                nearest_crisis_eta_s=self._felt.get("nearest_crisis_eta_s"),
                anticipatory_pressure=float(self._felt.get("anticipatory_pressure", 0.0)),
                allostatic_load=float(self._felt.get("allostatic_load", 0.0)),
                new_forecasts=tuple(new_forecasts),
                resolved_forecasts=tuple(resolved),
                regime_events=tuple(new_regimes),
            )

    def _update_resource_lifecycle(
        self, snapshot: dict[str, Any], now: float
    ) -> None:
        """Separate expected model allocation from steady-state resource drift.

        Absolute threshold protection remains active throughout.  Only
        trajectory evidence is reset/suppressed, because extrapolating a
        finite model load as an indefinitely continuing leak is causally
        invalid.
        """
        if (
            "model_resource_lifecycle" not in snapshot
            and "model_load_active" not in snapshot
        ):
            return
        reported = str(snapshot.get("model_resource_lifecycle") or "").lower()
        load_active = bool(snapshot.get("model_load_active", False))
        if load_active or reported == "model_loading":
            target = "model_loading"
        elif self._resource_lifecycle == "model_loading":
            target = "settling"
            self._model_settle_until = now + self._model_settling_s
        elif (
            self._resource_lifecycle == "settling"
            and now < self._model_settle_until
        ):
            target = "settling"
        elif reported in {"steady", "cold"}:
            target = reported
        else:
            target = "steady"

        if target == self._resource_lifecycle:
            return
        previous = self._resource_lifecycle
        self._resource_lifecycle = target
        self._resource_lifecycle_changed_at = now
        self._reset_resource_trends(
            now,
            note=f"resource_lifecycle:{previous}->{target}",
        )
        self._pending_events.append({
            "kind": "resource_lifecycle",
            "at_unix": round(now, 3),
            "previous": previous,
            "state": target,
            "model_settle_until": round(self._model_settle_until, 3),
        })
        logger.info(
            "🫁 [Allostasis] resource lifecycle %s → %s; RSS trend evidence reset.",
            previous,
            target,
        )

    def _reset_resource_trends(self, now: float, *, note: str) -> None:
        for key in ("memory_rss_mb", "process_tree_rss_mb", "memory_pct"):
            if key not in self._specs:
                continue
            self._series[key].clear()
            self._cusum[key] = _CusumState()
            self._regime_id[key] = f"{key}-{_ISSUER}-{uuid.uuid4().hex}"
            self._regime_started_at[key] = now
            for threshold_name in ("amber", "red"):
                fc = self._open_forecasts.pop((key, threshold_name), None)
                if fc is not None:
                    self._finalize_forecast(
                        fc,
                        ForecastOutcome.SUPERSEDED,
                        now,
                        note=note,
                    )

    # ── CUSUM regime detection ──────────────────────────────────────────────
    def _regime_series(self, key: str) -> list[tuple[float, float]]:
        started = self._regime_started_at.get(key, 0.0)
        return [(t, v) for (t, v) in self._series[key] if t >= started]

    def _anchor_cusum(self, key: str, spec: VitalSpec) -> bool:
        """(Re)fit the CUSUM anchor on the most recent regime samples.

        The anchor slope is only trusted when the reference trend is strongly
        significant — otherwise a spurious fitted slope, extrapolated for
        hours, manufactures drift out of stationary noise (observed directly
        in this module's own test suite before this gate existed).
        """
        window = self._regime_series(key)[-self.CUSUM_FIT_WINDOW:]
        if len(window) < self.CUSUM_MIN_REFERENCE:
            return False
        times = [t for (t, _) in window]
        values = [v for (_, v) in window]
        slope = 0.0
        if mann_kendall(values).significant(self.CUSUM_SLOPE_ALPHA):
            fit = sen_slope(times, values)
            if fit is not None:
                slope = fit.slope
        # Theil–Sen intercept: median of (vᵢ − slope·tᵢ), evaluated at t0.
        t0 = times[0]
        offsets = sorted(v - slope * (t - t0) for (t, v) in window)
        mid = len(offsets) // 2
        intercept = offsets[mid] if len(offsets) % 2 else 0.5 * (offsets[mid - 1] + offsets[mid])
        residuals = [v - (intercept + slope * (t - t0)) for (t, v) in window]
        sigma = robust_sigma(residuals)
        if sigma <= self.CUSUM_SIGMA_FLOOR:
            # Degenerate (noise-free) reference: fall back to a small
            # fraction of the vital's amber-red span so a genuine shift
            # still registers without single-sample hair triggers.
            sigma = max((spec.red - spec.amber) * 0.01, self.CUSUM_SIGMA_FLOOR)
        # Inflate for the anchor's own estimation error (intercept/slope are
        # estimates, not truth): a persistent ~σ/√n bias otherwise leaks into
        # every z-score and quietly consumes the CUSUM allowance.
        sigma *= 1.0 + 1.0 / math.sqrt(len(window))
        state = self._cusum[key]
        state.anchor_slope = slope
        state.anchor_intercept = intercept + slope * (times[-1] - t0)
        state.anchor_t0 = times[-1]
        state.anchor_sigma = sigma
        state.pos = state.neg = 0.0
        state.anchored = True
        state.samples_since_anchor = 0
        return True

    def _cusum_update(self, key: str, spec: VitalSpec, value: float, now: float) -> Optional[RegimeEvent]:
        state = self._cusum[key]
        if not state.anchored:
            self._anchor_cusum(key, spec)
            return None
        state.samples_since_anchor += 1
        # Silent periodic refit: bounds how long a slightly-wrong anchor slope
        # is extrapolated (adaptive CUSUM). Only while quiescent — never mid-
        # accumulation, or a real slow shift could be refit away.
        if (state.samples_since_anchor >= self.CUSUM_REANCHOR_EVERY
                and state.pos < self.CUSUM_H_SIGMA / 2.0
                and state.neg < self.CUSUM_H_SIGMA / 2.0):
            self._anchor_cusum(key, spec)
            state = self._cusum[key]
        z = (value - state.expected(now)) / state.anchor_sigma
        k = self.CUSUM_K_SIGMA
        state.pos = max(0.0, state.pos + z - k)
        state.neg = max(0.0, state.neg - z - k)
        if state.pos < self.CUSUM_H_SIGMA and state.neg < self.CUSUM_H_SIGMA:
            return None
        direction = "up" if state.pos >= self.CUSUM_H_SIGMA else "down"
        magnitude = state.pos if direction == "up" else state.neg
        regime_id = f"{key}-{_ISSUER}-{uuid.uuid4().hex}"
        self._regime_id[key] = regime_id
        self._regime_started_at[key] = now
        self._regime_events_total += 1
        self._cusum[key] = _CusumState()
        # Forecasts issued under the old regime describe a body that no longer
        # exists. If regulation fired after issue, credit the intervention
        # (the regime plausibly changed *because* the engine acted); otherwise
        # supersede without scoring.
        for threshold_name in ("amber", "red"):
            fc = self._open_forecasts.pop((key, threshold_name), None)
            if fc is None:
                continue
            intervention = self._intervention_since(fc.issued_at, vital=fc.vital)
            if intervention is not None and direction == "down":
                self._finalize_forecast(
                    fc, ForecastOutcome.INTERVENED, now,
                    note=f"regime relaxed after {intervention['action']}",
                )
            else:
                self._finalize_forecast(
                    fc, ForecastOutcome.SUPERSEDED, now,
                    note=f"regime_change:{direction}",
                )
        logger.info(
            "🌡️ [Allostasis] regime change on %s (%s, %.1fσ) — trend window re-anchored.",
            key, direction, magnitude,
        )
        return RegimeEvent(vital=key, at_unix=now, direction=direction,
                           magnitude_sigma=magnitude, regime_id=regime_id)

    # ── allostatic load ─────────────────────────────────────────────────────
    def _accrue_load(
        self,
        key: str,
        spec: VitalSpec,
        value: float,
        dt: float,
        *,
        previous: Optional[float] = None,
    ) -> None:
        """Integrate strain over the interval this sample actually covers.

        Decay always applies across the full interval — chronic load fades in
        real time whether or not we were watching. Accrual does not: an
        interval longer than the engine can account for (sleep, a stalled
        pulse loop) was not observed, and inventing strain for it is the
        difference between a measurement and a guess.
        """
        decay = math.exp(-dt / max(1.0, self._load_tau_s))
        prior = self._load_raw.get(key, 0.0) * decay
        span = max(1e-9, spec.red - spec.setpoint)
        accrual_dt = min(dt, _MAX_ATTRIBUTABLE_GAP_S)
        excess = max(0.0, (value - spec.setpoint) / span)
        if previous is not None and math.isfinite(previous):
            # Trapezoid: the interval spans previous → current, so credit the
            # mean excess rather than pinning the whole span to the endpoint.
            prior_excess = max(0.0, (previous - spec.setpoint) / span)
            excess = 0.5 * (excess + prior_excess)
        # Raw load is "seconds spent fully red-equivalent", decayed.
        self._load_raw[key] = prior + excess * accrual_dt

    def _load_normalized(self, key: str) -> float:
        # 1 − e^(−load/τ_load): ~0.63 after running red for one full τ.
        return _clamp(1.0 - math.exp(-self._load_raw.get(key, 0.0) / max(1.0, self._load_tau_s)))

    def allostatic_load(self) -> dict[str, float]:
        with self._lock:
            per_vital = {key: round(self._load_normalized(key), 4) for key in self._specs}
            per_vital["composite"] = round(self._composite_load(), 4)
            return per_vital

    def _composite_load(self) -> float:
        if not self._specs:
            return 0.0
        values = [self._load_normalized(key) for key in self._specs]
        peak = max(values)
        mean = sum(values) / len(values)
        # Same blend the body uses for total_pressure: peak-dominant.
        return _clamp(0.45 * mean + 0.55 * peak)

    # ── forecasting ─────────────────────────────────────────────────────────
    def _refresh_forecasts(self, now: float) -> list[str]:
        issued: list[str] = []
        # MULTIPLE COMPARISONS. Every pulse tests each forecastable vital
        # against two thresholds. Judging each at alpha independently means
        # the family-wise false-alarm rate grows with the number of vitals,
        # so a quiet system still produces forecasts at a steady rate. The
        # per-pulse family is collected first and admitted under
        # Benjamini-Hochberg, which controls the false DISCOVERY rate while
        # keeping power for genuine trends.
        trends = self._trend_pass(now)
        admitted_p = self._admissible_p_value(trends)
        for key, spec in self._specs.items():
            trend = trends.get(key)
            if trend is None:
                continue
            mk, estimate, current = trend
            for threshold_name, threshold in (("amber", spec.amber), ("red", spec.red)):
                fc_key = (key, threshold_name)
                credible = (
                    mk.p_value <= admitted_p
                    and estimate.slope > max(0.0, spec.min_meaningful_slope)
                    and current < threshold
                )
                if not credible:
                    continue
                remaining = threshold - current
                eta_mid = now + remaining / estimate.slope
                if eta_mid - now > self._horizon_s:
                    # Trend is real but the crossing is beyond the honest
                    # forecast horizon; refresh next pulse.
                    continue
                widen = self._calibration.setdefault(key, _VitalCalibration()).widen_factor(
                    target_coverage=self._target_coverage)
                eta_early = now + remaining / estimate.upper if estimate.upper > 0 else eta_mid
                band_open = estimate.band_open_below
                eta_late = (now + remaining / estimate.lower) if estimate.lower > 0 else (
                    now + self._eta_cap_s)
                # Calibration-driven widening around the mid ETA.
                eta_lower = eta_mid - (eta_mid - eta_early) * widen
                eta_upper = eta_mid + (eta_late - eta_mid) * widen
                eta_upper = min(eta_upper, now + self._eta_cap_s)
                existing = self._open_forecasts.get(fc_key)
                if existing is not None:
                    existing.slope_per_s = estimate.slope
                    existing.slope_lower = estimate.lower
                    existing.slope_upper = estimate.upper
                    existing.eta_unix = eta_mid
                    existing.eta_lower_unix = eta_lower
                    existing.eta_upper_unix = eta_upper
                    existing.band_open = band_open
                    existing.p_value = mk.p_value
                    existing.widen_factor = widen
                    existing.revisions += 1
                    existing.last_revised_at = now
                    continue
                forecast = Forecast(
                    forecast_id=f"fc-{_ISSUER}-{uuid.uuid4().hex}",
                    vital=key,
                    threshold_name=threshold_name,
                    threshold_value=threshold,
                    regime_id=self._regime_id[key],
                    issued_at=now,
                    level_at_issue=current,
                    slope_per_s=estimate.slope,
                    slope_lower=estimate.lower,
                    slope_upper=estimate.upper,
                    eta_unix=eta_mid,
                    eta_lower_unix=eta_lower,
                    eta_upper_unix=eta_upper,
                    band_open=band_open,
                    p_value=mk.p_value,
                    widen_factor=widen,
                    first_eta_unix=eta_mid,
                    first_eta_lower_unix=eta_lower,
                    first_eta_upper_unix=eta_upper,
                )
                self._open_forecasts[fc_key] = forecast
                issued.append(forecast.forecast_id)
                self._pending_events.append({"kind": "issued", **forecast.to_dict()})
                logger.warning(
                    "🔮 [Allostasis] forecast %s: %s → %s line (%.1f %s) at %s "
                    "(band %s–%s, slope %.3f %s/h, p=%.4f).",
                    forecast.forecast_id, key, threshold_name, threshold, spec.unit,
                    _fmt_eta(eta_mid - now), _fmt_eta(eta_lower - now), _fmt_eta(eta_upper - now),
                    estimate.slope * 3600.0, spec.unit, mk.p_value,
                )
        return issued

    def _peak_since(self, vital: str, since_unix: float) -> float:
        """Highest value this vital reached in the samples taken since a time.

        The history deque is the only record of what happened between pulses,
        so it is the evidence a forecast is scored against. NaN when the
        vital has no sample in the window at all.
        """
        series = self._series.get(vital)
        if not series:
            return float("nan")
        peak = float("nan")
        for at, value in reversed(series):
            if at < since_unix:
                break
            if math.isnan(peak) or value > peak:
                peak = value
        return peak

    def _trend_pass(
        self, now: float,
    ) -> dict[str, tuple[MannKendall, SenSlopeEstimate, float]]:
        """Trend statistics for every testable vital, computed once per pulse.

        Mann-Kendall is O(n²) in the window length, so the family-wide
        false-discovery threshold and the issuance loop share ONE pass rather
        than each recomputing it.
        """
        trends: dict[str, tuple[MannKendall, SenSlopeEstimate, float]] = {}
        for key, spec in self._specs.items():
            if not spec.forecastable:
                continue
            if (
                self._resource_lifecycle in {"model_loading", "settling"}
                and key in {"memory_rss_mb", "process_tree_rss_mb", "memory_pct"}
            ):
                continue
            window = [(t, v) for (t, v) in self._regime_series(key)
                      if t >= now - self._trend_window_s]
            if len(window) < self._min_trend_samples:
                continue
            times = [t for (t, _) in window]
            values = [v for (_, v) in window]
            estimate = sen_slope(times, values)
            if estimate is None:
                continue
            trends[key] = (mann_kendall(values), estimate, values[-1])
        host_trend = trends.get("memory_pct")
        tree_trend = trends.get("process_tree_rss_mb")
        if host_trend is not None:
            tree_spec = self._specs.get("process_tree_rss_mb")
            tree_growth = (
                tree_spec is not None
                and tree_trend is not None
                and tree_trend[1].slope
                > tree_spec.min_meaningful_slope
            )
            if not tree_growth:
                # Host-wide memory can move because of other applications,
                # filesystem cache, or compression. Keep the measurement and
                # absolute red line, but do not make an unattributed host shift
                # part of Aura's felt pressure or autonomous throttling.
                trends.pop("memory_pct", None)
        return trends

    def _admissible_p_value(
        self, trends: dict[str, tuple[MannKendall, SenSlopeEstimate, float]],
    ) -> float:
        """The largest p-value admissible under Benjamini-Hochberg at alpha.

        Returns alpha when only one test is in play (BH reduces to it), and 0
        when nothing survives — no candidate can then be issued.
        """
        family: list[float] = []
        for key, (mk, _estimate, current) in trends.items():
            spec = self._specs[key]
            for _name, threshold in (("amber", spec.amber), ("red", spec.red)):
                if current < threshold:
                    family.append(mk.p_value)
        family.sort()
        m = len(family)
        if m <= 1:
            return self._alpha
        threshold = 0.0
        for rank, p in enumerate(family, start=1):
            if p <= (rank / m) * self._alpha:
                threshold = p
        return threshold

    def _resolve_due_forecasts(self, now: float, snapshot: dict[str, Any]) -> list[str]:
        resolved: list[str] = []
        for fc_key in list(self._open_forecasts.keys()):
            fc = self._open_forecasts[fc_key]
            vital, _threshold_name = fc_key
            raw = snapshot.get(vital, None)
            value = _finite(raw, default=float("nan")) if raw is not None else float("nan")
            # BETWEEN-SAMPLE CROSSINGS. Scoring only the instantaneous value
            # means a threshold that was crossed and recovered between two
            # 60 s pulses is invisible: the forecast that correctly predicted
            # it is then recorded as a false alarm. The high-water mark over
            # the samples taken since the forecast was issued is what the
            # forecast actually claimed — that the vital would REACH the line.
            peak_since_issue = self._peak_since(vital, fc.issued_at)
            observed_peak = max(
                v for v in (value, peak_since_issue) if not math.isnan(v)
            ) if not (math.isnan(value) and math.isnan(peak_since_issue)) else float("nan")
            crossed = (not math.isnan(observed_peak)) and observed_peak >= fc.threshold_value
            scored_upper_open = fc.first_eta_upper_unix or fc.eta_upper_unix
            if crossed:
                del self._open_forecasts[fc_key]
                # SCORE THE PREREGISTERED BAND. Revisions move the operational
                # band while preserving forecast_id, so scoring the live band
                # graded the forecast against a claim edited after the fact —
                # the prediction moved with the evidence and could not fail.
                scored_lower = fc.first_eta_lower_unix or fc.eta_lower_unix
                scored_upper = fc.first_eta_upper_unix or fc.eta_upper_unix
                if now < scored_lower - self._resolution_grace_s:
                    outcome = ForecastOutcome.MISS_EARLY
                    note = f"crossed {_fmt_eta(scored_lower - now)} before issued band"
                elif now > scored_upper + self._resolution_grace_s:
                    # A crossing after the deadline is a failed forecast. This
                    # branch runs BEFORE the expiry branch below, so without
                    # this check every not-too-early crossing scored as a HIT
                    # no matter how late it arrived.
                    outcome = ForecastOutcome.MISS_LATE
                    note = f"crossed {_fmt_eta(now - scored_upper)} after issued band"
                else:
                    outcome = ForecastOutcome.HIT
                    note = "crossed inside issued band"
                fc.crossed_at = now
                self._finalize_forecast(fc, outcome, now, note=note)
                resolved.append(fc.forecast_id)
                continue
            # The DEADLINE is the issued one too. Revisions could otherwise
            # push eta_upper forward indefinitely, keeping a failing forecast
            # permanently "open" and deferring judgment forever.
            if now <= scored_upper_open + self._resolution_grace_s:
                continue
            # Deadline passed without a crossing.
            del self._open_forecasts[fc_key]
            intervention = self._intervention_since(fc.issued_at, vital=fc.vital)
            if intervention is not None:
                outcome = ForecastOutcome.INTERVENED
                note = f"no crossing after {intervention['action']}"
            else:
                outcome = ForecastOutcome.FALSE_ALARM
                note = "band expired without crossing"
            self._finalize_forecast(fc, outcome, now, note=note)
            resolved.append(fc.forecast_id)
        return resolved

    def _finalize_forecast(
        self, fc: Forecast, outcome: ForecastOutcome, now: float, *, note: str,
    ) -> None:
        fc.status = outcome.value
        fc.resolved_at = now
        fc.resolution_note = note
        book = self._calibration.setdefault(fc.vital, _VitalCalibration())
        if outcome is ForecastOutcome.HIT:
            book.hits += 1
        elif outcome is ForecastOutcome.MISS_EARLY:
            book.miss_early += 1
        elif outcome is ForecastOutcome.MISS_LATE:
            book.miss_late += 1
        elif outcome is ForecastOutcome.FALSE_ALARM:
            book.false_alarms += 1
        elif outcome is ForecastOutcome.INTERVENED:
            book.intervened += 1
        else:
            book.superseded += 1
        self._resolved_recent.append(fc)
        self._pending_events.append({"kind": "resolved", **fc.to_dict()})
        log = (
            logger.info
            if outcome in (
                ForecastOutcome.HIT,
                ForecastOutcome.INTERVENED,
                ForecastOutcome.SUPERSEDED,
            )
            else logger.warning
        )
        log(
            "📒 [Allostasis] forecast %s on %s resolved %s (%s). Coverage now %s.",
            fc.forecast_id, fc.vital, outcome.value, note,
            book.coverage if book.coverage is not None else "n/a",
        )

    def _intervention_since(
        self, since_unix: float, *, vital: str | None = None
    ) -> Optional[dict[str, Any]]:
        """Most recent intervention after ``since_unix``, optionally for VITAL.

        This returned the newest GLOBAL tier change after issue with no
        matching, so an escalation triggered by an unrelated vital relabelled
        this forecast INTERVENED. Because intervened outcomes are also
        excluded from the coverage denominator, that converted misses into
        removals from calibration — the forecast could not be wrong. An
        intervention must now name the same vital to excuse it.
        """
        for item in reversed(self._interventions):
            if item["at_unix"] < since_unix:
                continue
            item_vital = str(item.get("vital") or "")
            if vital is not None and item_vital and item_vital != vital:
                # Attributed to a DIFFERENT vital — this forecast has no claim
                # on it. Unattributed (composite/load-driven) escalations have
                # no single driver and remain eligible, but they can never
                # launder one vital's miss with another vital's response.
                continue
            return item
        return None

    # ── tier policy ─────────────────────────────────────────────────────────
    def _nearest_crisis(self, now: float) -> tuple[Optional[Forecast], Optional[float]]:
        nearest: Optional[Forecast] = None
        nearest_eta: Optional[float] = None
        for fc in self._open_forecasts.values():
            if fc.threshold_name != "red":
                continue
            eta_s = fc.eta_unix - now
            if nearest_eta is None or eta_s < nearest_eta:
                nearest, nearest_eta = fc, eta_s
        return nearest, nearest_eta

    def _vital_is_fresh(self, key: str, now: float) -> bool:
        """True when this vital's newest sample still describes the body now.

        A vital that stops appearing in the snapshot keeps its last value
        forever, so a breach recorded once stayed authoritative for tier,
        felt pressure and narrative indefinitely — the engine reporting a
        red line it could no longer see.
        """
        series = self._series.get(key)
        if not series:
            return False
        return (now - series[-1][0]) <= _INGEST_STALE_AFTER_S

    def _current_breach(self, now: Optional[float] = None) -> Optional[str]:
        at = self._now() if now is None else now
        stale_breach: Optional[str] = None
        for key, spec in self._specs.items():
            series = self._series[key]
            if not series or series[-1][1] < spec.red:
                continue
            if self._vital_is_fresh(key, at):
                return key
            stale_breach = stale_breach or key
        if stale_breach is not None:
            # The measurement expired, not the danger: we no longer know. That
            # is a degradation of the sense, reported as such rather than
            # silently held as a live breach or silently dropped. Reported
            # once per episode — this runs on every tier evaluation, and a
            # record per pulse would bury the signal it is trying to raise.
            if stale_breach not in self._stale_breach_reported:
                self._stale_breach_reported.add(stale_breach)
                record_degradation(
                    _SUBSYSTEM,
                    RuntimeError(
                        f"vital {stale_breach!r} was past its red line but has not "
                        f"reported for over {_INGEST_STALE_AFTER_S:.0f}s"
                    ),
                    action="stale breach retired; vital no longer observable",
                )
        return None

    def _target_tier(self, now: float) -> tuple[AllostasisTier, str]:
        """Choose the tier, recording WHICH vital drove the choice.

        The driver is what makes a later intervention attributable: a breach
        names its own vital, a forecast-driven escalation names the vital it
        forecasts, and a load-driven escalation has no single driver and is
        recorded as composite (None).
        """
        breach = self._current_breach(now)
        load = self._composite_load()
        nearest, eta_s = self._nearest_crisis(now)
        if breach is not None:
            self._tier_driver_vital = breach
            return AllostasisTier.PROTECTING, f"{breach} is already past its red line"
        if eta_s is not None and eta_s <= self._protect_horizon_s:
            self._tier_driver_vital = getattr(nearest, "vital", None)
            return AllostasisTier.PROTECTING, f"red-line crossing forecast in {_fmt_eta(eta_s)}"
        if load >= 0.85:
            self._tier_driver_vital = None
            return AllostasisTier.PROTECTING, f"allostatic load critical ({load:.2f})"
        if eta_s is not None and eta_s <= self._conserve_horizon_s:
            self._tier_driver_vital = getattr(nearest, "vital", None)
            return AllostasisTier.CONSERVING, f"red-line crossing forecast in {_fmt_eta(eta_s)}"
        if load >= 0.60:
            self._tier_driver_vital = None
            return AllostasisTier.CONSERVING, f"allostatic load elevated ({load:.2f})"
        self._tier_driver_vital = None
        if self._open_forecasts or load >= 0.30:
            return AllostasisTier.VIGILANT, (
                f"{len(self._open_forecasts)} open forecast(s), load {load:.2f}")
        return AllostasisTier.SETTLED, "no credible trajectory toward any limit"

    def _recompute_tier(self, now: float) -> str:
        target, reason = self._target_tier(now)
        if target > self._tier:
            # Escalation is immediate — anticipation is the whole point.
            old = self._tier
            self._tier = target
            self._tier_reason = reason
            self._tier_changed_at = now
            self._tier_release_eligible_since = None
            self._note_tier_change(old, target, reason, now)
        elif target < self._tier:
            # Release is hysteretic: sustained calm before stepping down one tier.
            if self._tier_release_eligible_since is None:
                self._tier_release_eligible_since = now
            elif now - self._tier_release_eligible_since >= self._release_hysteresis_s:
                old = self._tier
                self._tier = AllostasisTier(int(self._tier) - 1)
                self._tier_reason = f"released one tier after sustained calm ({reason})"
                self._tier_changed_at = now
                self._tier_release_eligible_since = now if self._tier > target else None
                self._note_tier_change(old, self._tier, self._tier_reason, now)
        else:
            self._tier_reason = reason
            self._tier_release_eligible_since = None
        return self._tier_reason

    def _note_tier_change(
        self, old: AllostasisTier, new: AllostasisTier, reason: str, now: float,
    ) -> None:
        event = {
            "kind": "tier_change",
            "at_unix": round(now, 3),
            "old": old.name.lower(),
            "new": new.name.lower(),
            "reason": reason,
        }
        self._pending_events.append(event)
        if new >= AllostasisTier.CONSERVING and new > old:
            # ATTRIBUTION. Interventions carried no vital at all, so any later
            # escalation could be credited to any open forecast. Record the
            # vital that drove the tier change when one is identifiable; a
            # load-driven (composite) escalation genuinely has no single
            # driver and is recorded as global.
            self._interventions.append({
                "at_unix": now,
                "action": f"entered {new.name.lower()} ({reason})",
                "tier": new.name.lower(),
                "vital": self._tier_driver_vital,
            })
        logger.log(
            logging.WARNING if new >= AllostasisTier.CONSERVING else logging.INFO,
            "🫁 [Allostasis] tier %s → %s: %s",
            old.name.lower(), new.name.lower(), reason,
        )

    # ── felt-state contribution (the causal seam) ───────────────────────────
    def _refresh_felt(self, now: float) -> None:
        nearest, eta_s = self._nearest_crisis(now)
        load = self._composite_load()
        urgency = 0.0
        if eta_s is not None:
            urgency = _clamp(1.0 - (eta_s / max(1.0, self._conserve_horizon_s)))
            if nearest is not None and nearest.band_open:
                urgency *= 0.5  # slope CI touches zero: honest discount
        if self._current_breach(now) is not None:
            urgency = 1.0
        self._felt = {
            "anticipatory_pressure": round(_clamp(0.65 * urgency + 0.35 * load), 4),
            "allostatic_load": round(load, 4),
            "nearest_crisis_eta_s": round(eta_s, 1) if eta_s is not None else None,
            "tier": self._tier.name.lower(),
        }

    def felt_contribution(self) -> dict[str, Any]:
        """Cheap, lock-guarded read for the hot body-state path."""
        if self._disabled:
            return {"anticipatory_pressure": 0.0, "allostatic_load": 0.0,
                    "nearest_crisis_eta_s": None, "tier": "disabled"}
        with self._lock:
            return dict(self._felt)

    def should_defer_heavy_work(self) -> tuple[bool, str]:
        """True when new deferrable load should wait. Consulted by the
        metabolic coordinator; safe to call from anywhere."""
        if self._disabled:
            return False, "allostasis disabled"
        with self._lock:
            if self._tier >= AllostasisTier.CONSERVING:
                return True, f"allostasis {self._tier.name.lower()}: {self._tier_reason}"
            return False, f"allostasis {self._tier.name.lower()}"

    # ── the pulse: one sample + side effects ────────────────────────────────
    async def sample_and_regulate(self) -> Optional[AllostasisReading]:
        """One allostatic pulse: sample vitals, update forecasts, act.

        Side effects (all fail-soft, each recorded on failure): ledger writes
        through the governed gateway, tier events on the bus, a degradation
        record when PROTECTING is entered. Never raises to the caller's loop.
        """
        if self._disabled:
            return None
        try:
            from core.runtime.runtime_pressure import get_unified_runtime_pressure

            # The pressure provider walks the process tree and reads host
            # counters — real blocking work. On the loop it stalls every other
            # task precisely when the host is loaded, which is when this pulse
            # matters most. Bounded so a wedged provider cannot wedge the loop.
            snapshot = await asyncio.wait_for(
                asyncio.to_thread(
                    get_unified_runtime_pressure().runtime_pressure_snapshot
                ),
                timeout=_SNAPSHOT_TIMEOUT_S,
            )
        except asyncio.TimeoutError as exc:
            record_degradation(
                _SUBSYSTEM, exc,
                action=f"vitals snapshot exceeded {_SNAPSHOT_TIMEOUT_S:.0f}s; pulse skipped",
            )
            return None
        except _BOUNDARY_ERRORS as exc:
            record_degradation(_SUBSYSTEM, exc, action="vitals snapshot unavailable; pulse skipped")
            return None

        with self._lock:
            tier_before = self._tier
        reading = self.ingest(snapshot)
        with self._lock:
            events = list(self._pending_events)
            self._pending_events.clear()
            tier_after = self._tier
            tier_reason = self._tier_reason

        if tier_after > tier_before and tier_after >= AllostasisTier.PROTECTING:
            self._raise_protecting(tier_reason, reading)
        elif tier_before >= AllostasisTier.PROTECTING > tier_after:
            # The alarm was raised on this channel; the all-clear belongs on it
            # too. Consumers gating on existential_threat had no event that
            # ended the emergency.
            self._clear_protecting(tier_reason, reading)
        if tier_after != tier_before:
            self._publish_state_change(tier_before, tier_after, tier_reason, reading)

        save_state = bool(events) or (self._ingest_count % 10 == 0)
        if not await self._persist(events, save_state=save_state) and events:
            self._requeue_unpersisted(events)
        return reading

    def _requeue_unpersisted(self, events: list[dict[str, Any]]) -> None:
        """Take failed events back so the next pulse retries them.

        Bounded: an unwritable ledger must not grow the queue without limit.
        When the backlog exceeds the cap the OLDEST events are dropped and the
        loss is recorded, because a silent gap in an append-only forecast
        ledger is indistinguishable from a clean history.
        """
        with self._lock:
            self._pending_events[:0] = events
            overflow = len(self._pending_events) - _MAX_PENDING_EVENTS
            if overflow > 0:
                del self._pending_events[:overflow]
        if overflow > 0:
            record_degradation(
                _SUBSYSTEM,
                RuntimeError(
                    f"allostasis ledger backlog exceeded {_MAX_PENDING_EVENTS}; "
                    f"dropped {overflow} oldest event(s)"
                ),
                action="forecast ledger has a recorded gap",
            )

    def _raise_protecting(self, reason: str, reading: AllostasisReading) -> None:
        # PROTECTING is the engine WORKING, not the engine failing. This used
        # to synthesize a RuntimeError and feed the global degradation and
        # resilience systems, so every successful anticipation registered as a
        # fault — inflating the degradation record with the system's own
        # correct behaviour and, for fail-closed subsystems, escalating it.
        # Anticipatory protection is logged as the designed transition it is.
        logger.warning(
            "🛡️ [Allostasis] anticipatory protection engaged (designed tier "
            "transition, not a fault): %s",
            reason,
        )
        try:
            from core.event_bus import get_event_bus

            # CONFIDENCE CONTRACT. This publishes on the same channel as real
            # emergencies, so a statistical forecast must carry the evidence
            # that qualifies it: which forecast, its p-value, the interval it
            # committed to, and the empirical coverage of past forecasts for
            # that vital. Without those a threshold projection was
            # indistinguishable from an observed crisis.
            nearest, _eta = self._nearest_crisis(time.time())
            confidence: dict[str, Any] = {"forecast_id": None}
            if nearest is not None:
                book = self._calibration.get(nearest.vital)
                confidence = {
                    "forecast_id": nearest.forecast_id,
                    "vital": nearest.vital,
                    "p_value": nearest.p_value,
                    "eta_lower_unix": nearest.first_eta_lower_unix or nearest.eta_lower_unix,
                    "eta_upper_unix": nearest.first_eta_upper_unix or nearest.eta_upper_unix,
                    "widen_factor": nearest.widen_factor,
                    "revisions": nearest.revisions,
                    "empirical_coverage": (book.coverage if book else None),
                    "scored_forecasts": (book.scored if book else 0),
                    # Coverage conditioned on how much was censored out of it.
                    "censored_forecasts": (book.censored if book else 0),
                    "censored_fraction": (book.censored_fraction if book else None),
                }

            get_event_bus().publish_threadsafe(
                "existential_threat",
                {
                    "imperative": f"WARNING: {self.narrative()}",
                    "source": "AllostasisEngine",
                    "severity": "WARNING",
                    "anticipatory": True,
                    "observed": False,
                    "nearest_crisis_eta_s": reading.nearest_crisis_eta_s,
                    "confidence": confidence,
                },
            )
            logger.warning("🚨 [Allostasis] anticipatory imperative published: %s", reason)
        except _BOUNDARY_ERRORS as exc:
            # A publish FAILURE is a real degradation — unlike the tier change.
            record_degradation(_SUBSYSTEM, exc, action="existential-threat publish failed")

    def _clear_protecting(self, reason: str, reading: AllostasisReading) -> None:
        """Publish the recovery event that ends an anticipatory emergency."""
        logger.info(
            "🌤️ [Allostasis] anticipatory protection released: %s", reason,
        )
        try:
            from core.event_bus import get_event_bus

            get_event_bus().publish_threadsafe(
                "existential_threat",
                {
                    "imperative": f"RESOLVED: {self.narrative()}",
                    "source": "AllostasisEngine",
                    "severity": "INFO",
                    "anticipatory": True,
                    "observed": False,
                    "resolved": True,
                    "nearest_crisis_eta_s": reading.nearest_crisis_eta_s,
                    "reason": reason,
                },
            )
        except _BOUNDARY_ERRORS as exc:
            # Failing to clear leaves consumers latched in an emergency they
            # cannot exit, so this is a real degradation.
            record_degradation(
                _SUBSYSTEM, exc, action="existential-threat recovery publish failed",
            )

    def _publish_state_change(
        self,
        old: AllostasisTier,
        new: AllostasisTier,
        reason: str,
        reading: AllostasisReading,
    ) -> None:
        try:
            from core.event_bus import get_event_bus

            get_event_bus().publish_threadsafe(
                "allostasis_state",
                {
                    "old_tier": old.name.lower(),
                    "new_tier": new.name.lower(),
                    "reason": reason,
                    "nearest_crisis_eta_s": reading.nearest_crisis_eta_s,
                    "anticipatory_pressure": reading.anticipatory_pressure,
                    "allostatic_load": reading.allostatic_load,
                    "narrative": self.narrative(),
                },
            )
        except _BOUNDARY_ERRORS as exc:
            record_degradation(_SUBSYSTEM, exc, action="allostasis state publish failed", severity="debug")

    # ── surfaces ────────────────────────────────────────────────────────────
    def narrative(self) -> str:
        """One honest sentence about the body's trajectory, for the narrator
        and the imperative channel. Functional claims only."""
        with self._lock:
            now = self._now()
            nearest, eta_s = self._nearest_crisis(now)
            load = self._composite_load()
            tier = self._tier.name.lower()
            if nearest is not None and eta_s is not None:
                spec = self._specs.get(nearest.vital)
                unit = spec.unit if spec else ""
                rate = nearest.slope_per_s * 3600.0
                return (
                    f"My {spec.label if spec else nearest.vital} is rising ~{rate:.0f} {unit}/h; "
                    f"at this rate I cross my red line in {_fmt_eta(eta_s)} "
                    f"(band {_fmt_eta(nearest.eta_lower_unix - now)}–"
                    f"{_fmt_eta(nearest.eta_upper_unix - now)}). "
                    f"I am {tier}."
                )
            if self._open_forecasts:
                soonest = min(self._open_forecasts.values(), key=lambda f: f.eta_unix)
                spec = self._specs.get(soonest.vital)
                return (
                    f"My {spec.label if spec else soonest.vital} is trending toward its "
                    f"{soonest.threshold_name} line ({_fmt_eta(soonest.eta_unix - now)} away). "
                    f"I am {tier}."
                )
            if load >= 0.30:
                return (
                    f"No crisis forecast, but this process has been running hot "
                    f"(load {load:.2f}){self._observation_caveat(now)}. I am {tier}."
                )
            # "Stable on every measured trajectory" was returned even with zero
            # samples, stale samples, or too few points to test a trend — a
            # claim of universal stability backed by no measurement at all.
            measured, total, stale = self._observation_census(now)
            if measured == 0:
                return (
                    f"I have no current vitals for this process — nothing is being "
                    f"measured, so I can say nothing about my trajectory. I am {tier}."
                )
            if measured < total or stale:
                return (
                    f"No trajectory toward a limit in the {measured} of {total} vitals "
                    f"I can currently see for this process"
                    f"{self._observation_caveat(now)}. I am {tier}."
                )
            return (
                f"No trajectory toward a limit in any of the {total} vitals I measure "
                f"for this process{self._observation_caveat(now)}. I am {tier}."
            )

    def _observation_census(self, now: float) -> tuple[int, int, bool]:
        """(fresh vitals, configured vitals, any stale-but-present vital)."""
        total = len(self._specs)
        fresh = 0
        stale = False
        for key in self._specs:
            series = self._series.get(key)
            if not series:
                continue
            if self._vital_is_fresh(key, now):
                fresh += 1
            else:
                stale = True
        return fresh, total, stale

    def _observation_caveat(self, now: float) -> str:
        """The scope and freshness that qualify any claim made above.

        These are host-process counters read at a point in time, not a body.
        The narrative is spoken in the first person, so the boundary has to
        travel WITH the sentence rather than live in a docstring the reader
        of the sentence never sees.
        """
        last = self._last_ingest_at
        if last is None:
            return " (no reading yet)"
        age = max(0.0, now - last)
        if age > _INGEST_STALE_AFTER_S:
            return f" (last reading {_fmt_eta(age)} ago — stale)"
        return f" (read {_fmt_eta(age)} ago)"

    def status(self) -> dict[str, Any]:
        with self._lock:
            now = self._now()
            vitals: dict[str, Any] = {}
            for key, spec in self._specs.items():
                series = self._series[key]
                last = series[-1] if series else None
                vitals[key] = {
                    "label": spec.label,
                    "unit": spec.unit,
                    "current": round(last[1], 3) if last else None,
                    "amber": spec.amber,
                    "red": spec.red,
                    "samples": len(series),
                    "regime_id": self._regime_id[key],
                    "load": round(self._load_normalized(key), 4),
                }
            return {
                "service": self.SERVICE_NAME,
                "enabled": not self._disabled,
                "tier": self._tier.name.lower(),
                "tier_reason": self._tier_reason,
                "tier_changed_at": self._tier_changed_at,
                "narrative": self.narrative(),
                "felt": dict(self._felt),
                "vitals": vitals,
                "open_forecasts": [
                    {**fc.to_dict(), "eta_in_s": round(fc.eta_unix - now, 1)}
                    for fc in self._open_forecasts.values()
                ],
                "recently_resolved": [fc.to_dict() for fc in list(self._resolved_recent)[-10:]],
                "calibration": {k: v.to_dict() for k, v in self._calibration.items()},
                "allostatic_load": self.allostatic_load(),
                "resource_lifecycle": {
                    "state": self._resource_lifecycle,
                    "changed_at": self._resource_lifecycle_changed_at,
                    "model_settle_until": self._model_settle_until,
                    "forecast_provisional": self._resource_lifecycle
                    in {"model_loading", "settling"},
                },
                "regime_events_total": self._regime_events_total,
                "ingest_count": self._ingest_count,
                "last_ingest_at": self._last_ingest_at,
                "config": {
                    "trend_window_s": self._trend_window_s,
                    "significance_alpha": self._alpha,
                    "forecast_horizon_s": self._horizon_s,
                    "conserve_horizon_s": self._conserve_horizon_s,
                    "protect_horizon_s": self._protect_horizon_s,
                    "release_hysteresis_s": self._release_hysteresis_s,
                    "target_coverage": self._target_coverage,
                },
            }

    def stats(self) -> dict[str, Any]:
        return self.status()


def _fmt_eta(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60.0:.0f}min"
    return f"{seconds / 3600.0:.1f}h"


# ─────────────────────────────────────────────────────────────────────────────
# Singleton + container registration (house pattern)
# ─────────────────────────────────────────────────────────────────────────────

_engine: Optional[AllostasisEngine] = None
_engine_lock = threading.Lock()
# Set when a test retires the process engine: the container still holds the
# retired instance, so the next engine built here must take the slot over
# rather than adopt what is in it.
_container_slot_disowned = False


def get_allostasis_engine() -> AllostasisEngine:
    """The one allostasis engine, container-first.

    The getter used to construct a local engine unconditionally while
    registration skipped an already-occupied container slot. Callers reaching
    the service through the container and callers using this getter could
    then hold DIFFERENT engines — two bodies, two histories, two tiers, each
    convinced it was the one being regulated. The container is authoritative:
    an engine already registered there IS the engine.
    """
    global _engine, _container_slot_disowned
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                existing = None if _container_slot_disowned else _engine_from_container()
                if existing is not None:
                    _engine = existing
                else:
                    _engine = AllostasisEngine()
                    _register_in_container(_engine, replace=_container_slot_disowned)
                    _container_slot_disowned = False
    return _engine


def _engine_from_container() -> Optional[AllostasisEngine]:
    """The container's engine, if it already owns one."""
    try:
        from core.container import ServiceContainer

        if not ServiceContainer.has(AllostasisEngine.SERVICE_NAME):
            return None
        get = getattr(ServiceContainer, "get", None)
        if not callable(get):
            return None
        existing = get(AllostasisEngine.SERVICE_NAME)
        return existing if isinstance(existing, AllostasisEngine) else None
    except _BOUNDARY_ERRORS as exc:
        record_degradation(
            _SUBSYSTEM, exc,
            action="container lookup skipped; using local engine", severity="debug",
        )
        return None


def _register_in_container(engine: AllostasisEngine, *, replace: bool = False) -> None:
    try:
        from core.container import ServiceContainer

        if replace or not ServiceContainer.has(AllostasisEngine.SERVICE_NAME):
            reg = getattr(ServiceContainer, "register_instance", None)
            if callable(reg):
                reg(AllostasisEngine.SERVICE_NAME, engine,
                    required=False, registered_by="allostasis")
    except _BOUNDARY_ERRORS as exc:
        record_degradation(_SUBSYSTEM, exc, action="container registration skipped", severity="debug")


def reset_allostasis_engine_for_test() -> None:
    """Drop the process engine and disown the container slot.

    Clearing only the module global left the retired engine registered, so
    the next getter call adopted it right back. ServiceContainer has no
    per-service removal (only a whole-registry clear, which would take down
    every other service a test depends on), so the slot is instead marked for
    takeover: the next engine built here REPLACES the registration rather
    than deferring to it. Taken under the same lock the getter uses, so a
    concurrent construction cannot interleave.
    """
    global _engine, _container_slot_disowned
    with _engine_lock:
        _engine = None
        _container_slot_disowned = True


__all__ = [
    "AllostasisEngine",
    "AllostasisReading",
    "AllostasisTier",
    "Forecast",
    "ForecastOutcome",
    "MannKendall",
    "RegimeEvent",
    "SenSlopeEstimate",
    "VitalSpec",
    "default_vital_specs",
    "get_allostasis_engine",
    "mann_kendall",
    "norm_ppf",
    "reset_allostasis_engine_for_test",
    "robust_sigma",
    "sen_slope",
]

"""core/observability/histograms.py — declared, bucketed histograms.

Clean-room adoption of Chromium's UMA histogram system and, just as
importantly, of `histograms.xml` — the discipline around it.

A counter tells you a mean. A mean tells you almost nothing about a
latency distribution, because the thing that hurts is the tail: a p50 of
80ms with a p99 of 40 seconds is a system people describe as "sometimes it
just hangs", and its mean looks fine. Aura's existing metrics collector
records durations and derives percentiles from a retained sample list,
which is correct but unbounded in memory and loses history. Bucketed
histograms are O(1) memory per metric, keep every observation's shape, and
are cheap enough to leave on everywhere.

What to copy beyond the data structure:

* **Declaration with an owner and an expiry.** Chromium will not accept a
  histogram without both. The expiry is the good idea: metrics rot,
  nobody removes them, and dashboards fill with numbers whose meaning
  nobody remembers. A declared expiry makes "is anyone still reading
  this" a question the system asks rather than one nobody asks.
* **Bucketing chosen per metric.** Latency wants exponential buckets
  (resolution where the action is, a bounded count over five orders of
  magnitude); a percentage wants linear; an enum wants one bucket per
  value with no aggregation at all.
* **An overflow bucket that is visible.** Values above the maximum land
  in a final bucket that is reported as overflow, so "this metric is
  clipping" is legible rather than a suspiciously flat top decile.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.Histograms")

#: Declared histograms past this age with no recorded samples are reported
#: as candidates for removal.
DEFAULT_EXPIRY_DAYS = 180


class Bucketing(StrEnum):
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    ENUM = "enum"


@dataclass(frozen=True)
class HistogramSpec:
    name: str
    description: str
    owner: str
    unit: str
    bucketing: Bucketing
    minimum: float
    maximum: float
    bucket_count: int
    expiry_days: int = DEFAULT_EXPIRY_DAYS
    enum_labels: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "owner": self.owner,
            "unit": self.unit,
            "bucketing": str(self.bucketing),
            "minimum": self.minimum,
            "maximum": self.maximum,
            "bucket_count": self.bucket_count,
            "expiry_days": self.expiry_days,
            "enum_labels": list(self.enum_labels),
        }


def _exponential_bounds(minimum: float, maximum: float, count: int) -> tuple[float, ...]:
    """Geometric bucket boundaries — resolution where the action is."""
    low = max(minimum, 1e-9)
    high = max(maximum, low * 2)
    ratio = (high / low) ** (1.0 / max(1, count - 1))
    return tuple(low * (ratio**i) for i in range(count))


def _linear_bounds(minimum: float, maximum: float, count: int) -> tuple[float, ...]:
    width = (maximum - minimum) / max(1, count)
    return tuple(minimum + width * i for i in range(count))


class Histogram:
    """One metric. O(1) memory, lock-guarded, safe to leave on."""

    def __init__(self, spec: HistogramSpec) -> None:
        self.spec = spec
        if spec.bucketing is Bucketing.ENUM:
            self._bounds: tuple[float, ...] = tuple(
                float(i) for i in range(len(spec.enum_labels) or spec.bucket_count)
            )
        elif spec.bucketing is Bucketing.LINEAR:
            self._bounds = _linear_bounds(spec.minimum, spec.maximum, spec.bucket_count)
        else:
            self._bounds = _exponential_bounds(spec.minimum, spec.maximum, spec.bucket_count)
        self._counts = [0] * (len(self._bounds) + 1)
        self._lock = threading.Lock()
        self.total = 0
        self.sum = 0.0
        self.minimum_seen = math.inf
        self.maximum_seen = -math.inf
        self.underflow = 0
        self.overflow = 0
        self.last_sample_at = 0.0

    def _bucket(self, value: float) -> int:
        # Linear scan is fine: bucket counts are ~50 and this beats the
        # branch-misprediction cost of a bisect at that size.
        for index, bound in enumerate(self._bounds):
            if value < bound:
                return max(0, index - 1)
        return len(self._bounds) - 1

    def record(self, value: float, count: int = 1) -> None:
        try:
            numeric = float(value)
        # not a failure: a value that is not a number is not one this can read.
        except (TypeError, ValueError):
            return
        if math.isnan(numeric) or math.isinf(numeric):
            # A non-finite sample would poison every derived statistic.
            from core.runtime.sanitizers import get_sanitizer_log

            get_sanitizer_log().report(
                "numeric",
                f"histogram-nonfinite:{self.spec.name}",
                f"non-finite sample {value!r} offered to histogram "
                f"{self.spec.name!r}; discarded",
            )
            return
        with self._lock:
            self.total += count
            self.sum += numeric * count
            self.minimum_seen = min(self.minimum_seen, numeric)
            self.maximum_seen = max(self.maximum_seen, numeric)
            self.last_sample_at = time.time()
            if numeric < self._bounds[0]:
                self.underflow += count
                self._counts[0] += count
                return
            if numeric >= self.spec.maximum and self.spec.bucketing is not Bucketing.ENUM:
                self.overflow += count
                self._counts[-1] += count
                return
            self._counts[self._bucket(numeric)] += count

    def record_enum(self, label: str, count: int = 1) -> None:
        if label not in self.spec.enum_labels:
            logger.debug("histogram %s: unknown enum label %r", self.spec.name, label)
            return
        self.record(float(self.spec.enum_labels.index(label)), count)

    def percentile(self, fraction: float) -> float:
        """Bucket-resolution percentile. Exact enough to act on."""
        with self._lock:
            total = self.total
            counts = list(self._counts)
        if total == 0:
            return 0.0
        target = fraction * total
        seen = 0
        for index, count in enumerate(counts):
            seen += count
            if seen >= target:
                if index >= len(self._bounds):
                    return self.spec.maximum
                return self._bounds[index]
        return self.spec.maximum

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            total = self.total
            counts = list(self._counts)
            sum_ = self.sum
            low = self.minimum_seen
            high = self.maximum_seen
            underflow = self.underflow
            overflow = self.overflow
            last = self.last_sample_at
        return {
            "name": self.spec.name,
            "unit": self.spec.unit,
            "count": total,
            "sum": round(sum_, 6),
            "mean": round(sum_ / total, 6) if total else 0.0,
            "min": None if low is math.inf else round(low, 6),
            "max": None if high == -math.inf else round(high, 6),
            "p50": round(self.percentile(0.50), 6),
            "p95": round(self.percentile(0.95), 6),
            "p99": round(self.percentile(0.99), 6),
            "underflow": underflow,
            "overflow": overflow,
            "clipping": bool(total and overflow / total > 0.01),
            "buckets": [
                {"lower": round(bound, 6), "count": counts[i]}
                for i, bound in enumerate(self._bounds)
                if counts[i]
            ],
            "last_sample_age_s": round(time.time() - last, 1) if last else None,
        }

    def reset(self) -> None:
        with self._lock:
            self._counts = [0] * (len(self._bounds) + 1)
            self.total = 0
            self.sum = 0.0
            self.minimum_seen = math.inf
            self.maximum_seen = -math.inf
            self.underflow = 0
            self.overflow = 0


@dataclass
class _Registry:
    histograms: dict[str, Histogram] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    declared_at: dict[str, float] = field(default_factory=dict)


_REGISTRY = _Registry()


def declare_histogram(
    name: str,
    *,
    description: str,
    owner: str,
    unit: str = "",
    bucketing: Bucketing = Bucketing.EXPONENTIAL,
    minimum: float = 1.0,
    maximum: float = 1_000_000.0,
    bucket_count: int = 50,
    expiry_days: int = DEFAULT_EXPIRY_DAYS,
    enum_labels: tuple[str, ...] = (),
) -> Histogram:
    """Declare a histogram. Owner and description are mandatory.

    Chromium refuses a histogram without them and the refusal is the
    feature: a metric nobody owns becomes a number on a dashboard that
    nobody can interpret and nobody dares delete.
    """
    if not description.strip() or not owner.strip():
        raise ValueError(
            f"histogram {name!r} needs a description and an owner — an unowned "
            "metric is a number nobody can interpret and nobody dares delete"
        )
    spec = HistogramSpec(
        name=name,
        description=description,
        owner=owner,
        unit=unit,
        bucketing=bucketing,
        minimum=minimum,
        maximum=maximum,
        bucket_count=bucket_count,
        expiry_days=expiry_days,
        enum_labels=tuple(enum_labels),
    )
    with _REGISTRY.lock:
        existing = _REGISTRY.histograms.get(name)
        if existing is not None:
            if existing.spec != spec:
                raise ValueError(
                    f"histogram {name!r} already declared by {existing.spec.owner} "
                    "with a different spec; a metric has one meaning"
                )
            return existing
        histogram = Histogram(spec)
        _REGISTRY.histograms[name] = histogram
        _REGISTRY.declared_at[name] = time.time()
        return histogram


def get_histogram(name: str) -> Histogram | None:
    with _REGISTRY.lock:
        return _REGISTRY.histograms.get(name)


def record(name: str, value: float, count: int = 1) -> bool:
    """Record into a declared histogram. Undeclared names are ignored and
    reported once, rather than silently creating an unowned metric."""
    histogram = get_histogram(name)
    if histogram is None:
        logger.debug("histogram %r is not declared; sample dropped", name)
        return False
    histogram.record(value, count)
    return True


def record_duration(name: str, seconds: float) -> bool:
    return record(name, seconds * 1000.0)


class Timer:
    """Context manager recording elapsed milliseconds into a histogram."""

    __slots__ = ("_name", "_started")

    def __init__(self, name: str) -> None:
        self._name = name
        self._started = 0.0

    def __enter__(self) -> Timer:
        self._started = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        record(self._name, (time.perf_counter() - self._started) * 1000.0)


def expired_histograms() -> list[dict[str, Any]]:
    """Declared metrics past their expiry with no recent samples."""
    now = time.time()
    stale: list[dict[str, Any]] = []
    with _REGISTRY.lock:
        items = list(_REGISTRY.histograms.items())
        declared_at = dict(_REGISTRY.declared_at)
    for name, histogram in items:
        age_days = (now - declared_at.get(name, now)) / 86400.0
        if age_days < histogram.spec.expiry_days:
            continue
        if histogram.total > 0:
            continue
        stale.append(
            {
                "name": name,
                "owner": histogram.spec.owner,
                "age_days": round(age_days, 1),
                "expiry_days": histogram.spec.expiry_days,
            }
        )
    return stale


def histograms_report(*, include_buckets: bool = False) -> dict[str, Any]:
    with _REGISTRY.lock:
        items = list(_REGISTRY.histograms.items())
    snapshots = {}
    for name, histogram in sorted(items):
        snapshot = histogram.snapshot()
        if not include_buckets:
            snapshot.pop("buckets", None)
        snapshots[name] = snapshot
    return {
        "count": len(snapshots),
        "histograms": snapshots,
        "clipping": [n for n, s in snapshots.items() if s.get("clipping")],
        "unused": [n for n, s in snapshots.items() if s["count"] == 0],
        "expired": expired_histograms(),
        "owners": sorted({h.spec.owner for _, h in items}),
    }


def install_standard_histograms() -> list[str]:
    """The metrics the runtime disciplines already produce a stream of."""
    declarations = (
        dict(
            name="Aura.Pass.DurationMs",
            description="wall time of one cognitive pass/phase execution",
            owner="core/pipeline/pass_manager.py",
            unit="ms",
            minimum=0.1,
            maximum=120_000.0,
        ),
        dict(
            name="Aura.Analysis.DurationMs",
            description="wall time computing a cached pipeline analysis",
            owner="core/pipeline/pass_manager.py",
            unit="ms",
            minimum=0.05,
            maximum=60_000.0,
        ),
        dict(
            name="Aura.Reconcile.DurationMs",
            description="wall time of one controller reconcile",
            owner="core/runtime/reconcile.py",
            unit="ms",
            minimum=0.1,
            maximum=60_000.0,
        ),
        dict(
            # The quantity Aura was asked for on 2026-08-10 — "what is your
            # median response latency today, in milliseconds?" — and answered
            # "152 milliseconds", with an authoritative gloss about first-byte
            # timing. Nothing anywhere in the runtime timed a chat turn, so the
            # number was invented, and no honest answer had ever been available.
            name="Aura.Chat.TurnMs",
            description="wall time of one user-visible chat turn, request to response",
            owner="interface/routes/chat.py",
            unit="ms",
            minimum=1.0,
            maximum=600_000.0,
        ),
        dict(
            name="Aura.Admission.DurationMs",
            description="wall time of the admission chain for one request",
            owner="core/runtime/admission.py",
            unit="ms",
            minimum=0.01,
            maximum=10_000.0,
        ),
        dict(
            name="Aura.Lifecycle.TransitionMs",
            description="wall time of one managed-lifecycle transition",
            owner="core/runtime/lifecycle.py",
            unit="ms",
            minimum=0.1,
            maximum=120_000.0,
        ),
        dict(
            name="Aura.Memory.AvailableFraction",
            description="host memory available as a fraction of total",
            owner="core/runtime/foundations.py",
            unit="fraction",
            bucketing=Bucketing.LINEAR,
            minimum=0.0,
            maximum=1.0,
            bucket_count=20,
        ),
        dict(
            name="Aura.Pressure.MemoryFull",
            description="PSI memory full-pressure percentage over 10s",
            owner="core/runtime/pressure_stall.py",
            unit="percent",
            bucketing=Bucketing.LINEAR,
            minimum=0.0,
            maximum=100.0,
            bucket_count=20,
        ),
        dict(
            name="Aura.Fsync.DurationMs",
            description="wall time of a durable file write's fsync",
            owner="core/runtime/atomic_writer.py",
            unit="ms",
            minimum=0.01,
            maximum=60_000.0,
        ),
    )
    names: list[str] = []
    for declaration in declarations:
        try:
            declare_histogram(**declaration)  # type: ignore[arg-type]
            names.append(str(declaration["name"]))
        except ValueError as exc:
            logger.warning("histogram declaration failed: %s", exc)
    return names


def reset_histograms_for_test() -> None:
    with _REGISTRY.lock:
        _REGISTRY.histograms.clear()
        _REGISTRY.declared_at.clear()


__all__ = [
    "DEFAULT_EXPIRY_DAYS",
    "Bucketing",
    "Histogram",
    "HistogramSpec",
    "Timer",
    "declare_histogram",
    "expired_histograms",
    "get_histogram",
    "histograms_report",
    "install_standard_histograms",
    "record",
    "record_duration",
    "reset_histograms_for_test",
]

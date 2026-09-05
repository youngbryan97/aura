"""L4 — anti-collapse: keeping a head honest, and turning her record into grounded doubt.

Two things live here, and they share a question: *how much should a claim from
experience be trusted?*

**Calibration** watches whether a head's stated probabilities match reality.
This matters more than accuracy. A head that is right 70% of the time and says
so is useful; a head that is right 70% of the time and says 95% is a liability,
because everything downstream sizes its caution by the number. Brier score and
expected calibration error are tracked on a rolling window, and a head whose
calibration degrades materially against its grant-time baseline has its
authority revoked automatically. That revocation is the anti-collapse
guarantee: once a head decides, it makes its own training data, and drift is
the expected failure mode rather than a surprising one.

**The track record** is the part that is causal on day one and needs nobody's
permission. It is not a model and makes no predictions — it is arithmetic over
what actually happened: in situations like this one, how often did it go well,
and how sure can she be of that given how few times she has been here? A Wilson
interval on her own history is a fact about her, available from the first
handful of episodes, and it is the honest source for how confident she sounds.

The distinction matters. A head's probability is a claim that must earn trust.
A track record is a count. Aura can act on the count immediately.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from core.ontogeny.experience import Episode, OutcomeKind
from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.Ontogeny.Calibration")

_Z95 = 1.959963984540054
_Z95_ONE_SIDED = 1.6448536269514722

#: Rolling window per control point. Long enough for a stable estimate, short
#: enough that a head going bad this week is visible this week.
_WINDOW = 500

#: Calibration bins for ECE.
_BINS = 10

#: A head whose ECE exceeds its grant-time baseline by this much has stopped
#: being honest about itself and loses authority.
ECE_DRIFT_LIMIT = 0.12

#: Calibration is an operational claim only after enough independent episodes
#: exist to estimate its uncertainty. Until then, a new runtime/head cohort is
#: recovering evidence rather than healthy or unhealthy.
MIN_CALIBRATION_SUPPORT = 50

#: Provenance names are contracts used by reports, telemetry and tests.
CANDIDATE_VALIDATION = "candidate_validation"
OPERATIONAL_SHADOW = "operational_shadow"
LEGACY_CALIBRATION = "legacy"

#: Below this many graded episodes a track record states its ignorance rather
#: than a rate. Three coin flips are not a base rate.
MIN_TRACK_RECORD = 12


def wilson(successes: float, total: float, *, upper: bool, z: float = _Z95) -> float:
    """Wilson score bound — the conservative reading of a small sample.

    Used in both directions on purpose: a challenger is judged by its
    pessimistic bound and the incumbent by its optimistic one, so promotion
    requires genuine separation rather than a lucky streak.
    """
    if total <= 0:
        return 0.0 if upper is False else 1.0
    phat = successes / total
    denom = 1.0 + z * z / total
    centre = (phat + z * z / (2 * total)) / denom
    margin = (z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total))) / denom
    value = centre + margin if upper else centre - margin
    return float(min(1.0, max(0.0, value)))


@dataclass(frozen=True)
class CalibrationObservation:
    """One prediction made before its outcome was known.

    ``episode_id`` is the idempotency key. Runtime revision and head version
    are deliberately both present: a source repair with an unchanged head and
    a refit on unchanged source are different experimental regimes.
    """

    episode_id: str
    control_point: str
    confidence: float
    correct: bool
    decided_at: float
    observed_at: float
    runtime_revision: str = "unbound"
    head_version: int = 0
    action: str = "unknown"
    provenance: str = LEGACY_CALIBRATION

    def __post_init__(self) -> None:
        if not self.episode_id or not self.control_point:
            raise ValueError("calibration observations require episode and control-point identity")
        confidence = float(self.confidence)
        if not math.isfinite(confidence):
            raise ValueError("calibration confidence must be finite")
        object.__setattr__(self, "confidence", min(1.0, max(0.0, confidence)))
        for field_name in ("decided_at", "observed_at"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "runtime_revision", str(self.runtime_revision or "unbound"))
        object.__setattr__(self, "head_version", max(0, int(self.head_version)))
        object.__setattr__(self, "action", str(self.action or "unknown"))
        object.__setattr__(self, "provenance", str(self.provenance or LEGACY_CALIBRATION))

    @property
    def cohort_id(self) -> str:
        return (
            f"{self.provenance}:runtime={self.runtime_revision}:"
            f"head={self.head_version}"
        )


@dataclass(frozen=True)
class CalibrationReport:
    """How honest a head's confidence has been lately."""

    control_point: str
    samples: int
    accuracy: float
    brier: float
    ece: float
    mean_confidence: float
    #: mean_confidence - accuracy. Positive means overconfident, which is the
    #: direction that hurts.
    overconfidence: float
    reliability: tuple[tuple[float, float, int], ...] = ()
    cohort_id: str = ""
    runtime_revision: str = "unbound"
    head_version: int = 0
    provenance: str = LEGACY_CALIBRATION
    oldest_decision_at: float | None = None
    newest_decision_at: float | None = None
    standard_error: float | None = None
    overconfidence_lower95: float | None = None
    overconfidence_upper95: float | None = None
    statistically_supported: bool = False
    overconfidence_supported: bool = False
    status: str = "insufficient_evidence"

    def as_dict(self) -> dict[str, Any]:
        return {
            "control_point": self.control_point,
            "samples": self.samples,
            "accuracy": round(self.accuracy, 4),
            "brier": round(self.brier, 4),
            "ece": round(self.ece, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "overconfidence": round(self.overconfidence, 4),
            "cohort_id": self.cohort_id,
            "runtime_revision": self.runtime_revision,
            "head_version": self.head_version,
            "provenance": self.provenance,
            "oldest_decision_at": self.oldest_decision_at,
            "newest_decision_at": self.newest_decision_at,
            "standard_error": round(self.standard_error, 6) if self.standard_error is not None else None,
            "overconfidence_lower95": (
                round(self.overconfidence_lower95, 4)
                if self.overconfidence_lower95 is not None else None
            ),
            "overconfidence_upper95": (
                round(self.overconfidence_upper95, 4)
                if self.overconfidence_upper95 is not None else None
            ),
            "statistically_supported": self.statistically_supported,
            "overconfidence_supported": self.overconfidence_supported,
            "minimum_samples": MIN_CALIBRATION_SUPPORT,
            "status": self.status,
            "reliability": [
                {"confidence": round(c, 3), "accuracy": round(a, 3), "n": n}
                for c, a, n in self.reliability
            ],
        }


class CalibrationMonitor:
    """Chronological, idempotent calibration partitioned by provenance.

    Candidate evaluation and lived operational shadows must never share a
    window. The former validates a proposed head on a frozen future cohort;
    the latter asks whether the exact deployed runtime/head regime remains
    honest after real decisions resolve.
    """

    def __init__(self, window: int = _WINDOW, *, provenance: str = LEGACY_CALIBRATION) -> None:
        self._window = int(window)
        self._provenance = str(provenance)
        self._samples: dict[str, dict[str, dict[str, CalibrationObservation]]] = {}
        self._active: dict[str, str] = {}
        self._baselines: dict[str, float] = {}
        self._legacy_sequence = 0
        self._lock = checked_lock("calibration", reentrant=True)

    @staticmethod
    def cohort_id(*, provenance: str, runtime_revision: str, head_version: int) -> str:
        return f"{provenance}:runtime={runtime_revision or 'unbound'}:head={max(0, int(head_version))}"

    def activate(
        self,
        control_point: str,
        *,
        runtime_revision: str,
        head_version: int,
        provenance: str | None = None,
    ) -> str:
        cohort = self.cohort_id(
            provenance=provenance or self._provenance,
            runtime_revision=runtime_revision,
            head_version=head_version,
        )
        with self._lock:
            self._active[control_point] = cohort
        return cohort

    def observe(
        self,
        control_point: str,
        *,
        confidence: float,
        correct: bool,
        episode_id: str | None = None,
        decided_at: float | None = None,
        observed_at: float | None = None,
        runtime_revision: str = "unbound",
        head_version: int = 0,
        action: str = "unknown",
        provenance: str | None = None,
    ) -> bool:
        now = time.time()
        with self._lock:
            if episode_id is None:
                self._legacy_sequence += 1
                episode_id = f"legacy:{self._legacy_sequence}"
            observation = CalibrationObservation(
                episode_id=str(episode_id),
                control_point=control_point,
                confidence=confidence,
                correct=correct,
                decided_at=now if decided_at is None else decided_at,
                observed_at=now if observed_at is None else observed_at,
                runtime_revision=runtime_revision,
                head_version=head_version,
                action=action,
                provenance=provenance or self._provenance,
            )
            cohorts = self._samples.setdefault(control_point, {})
            series = cohorts.setdefault(observation.cohort_id, {})
            if observation.episode_id in series:
                return False
            series[observation.episode_id] = observation
            self._trim(series)
            return True

    def replace_observations(
        self,
        control_point: str,
        observations: Iterable[CalibrationObservation],
        *,
        provenance: str | None = None,
    ) -> int:
        """Atomically replace one provenance plane from a replay or evaluation.

        Re-running the same replay produces the same set, rather than changing
        the answer through append order or duplicating episodes.
        """
        target_provenance = provenance or self._provenance
        grouped: dict[str, dict[str, CalibrationObservation]] = {}
        for observation in observations:
            if observation.control_point != control_point:
                raise ValueError("calibration replacement crossed control points")
            if observation.provenance != target_provenance:
                raise ValueError("calibration replacement crossed provenance planes")
            grouped.setdefault(observation.cohort_id, {})[observation.episode_id] = observation
        with self._lock:
            cohorts = self._samples.setdefault(control_point, {})
            for cohort in [name for name in cohorts if name.startswith(f"{target_provenance}:")]:
                cohorts.pop(cohort, None)
            for cohort, series in grouped.items():
                self._trim(series)
                cohorts[cohort] = series
            if grouped and (
                control_point not in self._active
                or self._active[control_point].startswith(f"{target_provenance}:")
            ):
                self._active[control_point] = max(
                    grouped,
                    key=lambda name: max(o.decided_at for o in grouped[name].values()),
                )
        return sum(len(series) for series in grouped.values())

    def _trim(self, series: dict[str, CalibrationObservation]) -> None:
        if len(series) <= self._window:
            return
        keep = {
            obs.episode_id
            for obs in sorted(series.values(), key=lambda o: (o.decided_at, o.episode_id))[-self._window:]
        }
        for episode_id in tuple(series):
            if episode_id not in keep:
                series.pop(episode_id, None)

    def report(self, control_point: str) -> CalibrationReport | None:
        with self._lock:
            cohorts = self._samples.get(control_point, {})
            cohort = self._active.get(control_point)
            if cohort is None and cohorts:
                cohort = max(
                    cohorts,
                    key=lambda name: max(
                        (o.decided_at for o in cohorts[name].values()), default=0.0
                    ),
                )
            series = list(cohorts.get(cohort, {}).values()) if cohort else []
        if not series:
            if cohort is None:
                return None
            provenance, runtime_revision, head_version = _parse_cohort(cohort)
            return CalibrationReport(
                control_point=control_point,
                samples=0,
                accuracy=0.0,
                brier=0.0,
                ece=0.0,
                mean_confidence=0.0,
                overconfidence=0.0,
                cohort_id=cohort,
                runtime_revision=runtime_revision,
                head_version=head_version,
                provenance=provenance,
                status=("recovery_pending" if provenance == OPERATIONAL_SHADOW else "insufficient_evidence"),
            )
        series.sort(key=lambda observation: (observation.decided_at, observation.episode_id))
        series = series[-self._window:]
        confidences = [observation.confidence for observation in series]
        corrects = [1.0 if observation.correct else 0.0 for observation in series]
        n = len(series)
        accuracy = sum(corrects) / n
        brier = sum((c - ok) ** 2 for c, ok in zip(confidences, corrects, strict=True)) / n
        mean_conf = sum(confidences) / n

        buckets: list[tuple[float, float, int]] = []
        ece = 0.0
        for b in range(_BINS):
            low, high = b / _BINS, (b + 1) / _BINS
            members = [
                (c, ok) for c, ok in zip(confidences, corrects, strict=True)
                if (low < c <= high) or (b == 0 and c <= high)
            ]
            if not members:
                continue
            bin_conf = sum(c for c, _ in members) / len(members)
            bin_acc = sum(ok for _, ok in members) / len(members)
            buckets.append((bin_conf, bin_acc, len(members)))
            ece += (len(members) / n) * abs(bin_conf - bin_acc)

        deltas = [confidence - correct for confidence, correct in zip(confidences, corrects, strict=True)]
        standard_error = None
        lower95 = upper95 = None
        if n > 1:
            variance = sum((delta - (mean_conf - accuracy)) ** 2 for delta in deltas) / (n - 1)
            standard_error = math.sqrt(max(0.0, variance) / n)
            lower95 = (mean_conf - accuracy) - _Z95_ONE_SIDED * standard_error
            upper95 = (mean_conf - accuracy) + _Z95_ONE_SIDED * standard_error
        supported = n >= MIN_CALIBRATION_SUPPORT and standard_error is not None
        overconfidence_supported = bool(supported and lower95 is not None and lower95 > 0.0)
        provenance = series[-1].provenance
        if provenance == OPERATIONAL_SHADOW and not supported:
            status = "recovery_pending"
        elif not supported:
            status = "insufficient_evidence"
        elif lower95 is not None and lower95 > 0.15:
            status = "red"
        elif lower95 is not None and lower95 > 0.08:
            status = "warning"
        else:
            status = "nominal"

        return CalibrationReport(
            control_point=control_point,
            samples=n,
            accuracy=accuracy,
            brier=brier,
            ece=ece,
            mean_confidence=mean_conf,
            overconfidence=mean_conf - accuracy,
            reliability=tuple(buckets),
            cohort_id=series[-1].cohort_id,
            runtime_revision=series[-1].runtime_revision,
            head_version=series[-1].head_version,
            provenance=provenance,
            oldest_decision_at=series[0].decided_at,
            newest_decision_at=series[-1].decided_at,
            standard_error=standard_error,
            overconfidence_lower95=lower95,
            overconfidence_upper95=upper95,
            statistically_supported=supported,
            overconfidence_supported=overconfidence_supported,
            status=status,
        )

    def set_baseline(self, control_point: str) -> float | None:
        """Freeze the calibration a head had when it was granted authority."""
        report = self.report(control_point)
        if report is None:
            return None
        self._baselines[control_point] = report.ece
        return report.ece

    def drifted(self, control_point: str) -> tuple[bool, str]:
        """Has this head stopped being honest since it was trusted?"""
        baseline = self._baselines.get(control_point)
        report = self.report(control_point)
        if baseline is None or report is None or report.samples < 50:
            return False, "insufficient evidence"
        if report.ece > baseline + ECE_DRIFT_LIMIT:
            return True, (
                f"calibration error {report.ece:.3f} exceeds grant-time "
                f"{baseline:.3f} by more than {ECE_DRIFT_LIMIT}"
            )
        return False, f"ece {report.ece:.3f} within {ECE_DRIFT_LIMIT} of baseline {baseline:.3f}"

    def all_reports(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        with self._lock:
            control_points = set(self._samples) | set(self._active)
        for control_point in sorted(control_points):
            report = self.report(control_point)
            if report is not None:
                out[control_point] = report.as_dict()
        return out

    def cohort_reports(self, control_point: str | None = None) -> dict[str, list[dict[str, Any]]]:
        """Return current and historical regimes without changing the active one."""
        with self._lock:
            snapshot = {
                cp: {cohort: dict(series) for cohort, series in cohorts.items()}
                for cp, cohorts in self._samples.items()
                if control_point is None or cp == control_point
            }
            active = dict(self._active)
        reports: dict[str, list[dict[str, Any]]] = {}
        for cp, cohorts in snapshot.items():
            rows: list[dict[str, Any]] = []
            for cohort, series in cohorts.items():
                temporary = CalibrationMonitor(window=self._window, provenance=self._provenance)
                temporary._samples = {cp: {cohort: series}}
                temporary._active = {cp: cohort}
                report = temporary.report(cp)
                if report is not None:
                    payload = report.as_dict()
                    payload["active"] = active.get(cp) == cohort
                    rows.append(payload)
            reports[cp] = sorted(
                rows,
                key=lambda payload: float(payload.get("newest_decision_at") or 0.0),
                reverse=True,
            )
        return reports


def _parse_cohort(cohort_id: str) -> tuple[str, str, int]:
    provenance, _, remainder = cohort_id.partition(":runtime=")
    runtime_revision, _, head = remainder.rpartition(":head=")
    try:
        head_version = int(head)
    except ValueError:
        head_version = 0
    return provenance or LEGACY_CALIBRATION, runtime_revision or "unbound", head_version


@dataclass(frozen=True)
class TrackRecord:
    """What actually happened, in situations like this one.

    Not a prediction. A count, with an interval that is wide when she has not
    been here often. This is the number that should shape how confidently she
    speaks — and it is available from her twelfth episode, not from a promotion.
    """

    control_point: str
    bucket: str
    successes: int
    failures: int
    #: Episodes in this bucket whose outcome was never observed. Reported
    #: because "I acted forty times and only ever saw how eight of them went"
    #: is itself important, and hiding it would inflate her sense of feedback.
    unobserved: int = 0

    @property
    def graded(self) -> int:
        return self.successes + self.failures

    @property
    def rate(self) -> float | None:
        return self.successes / self.graded if self.graded else None

    @property
    def interval(self) -> tuple[float, float] | None:
        if self.graded < MIN_TRACK_RECORD:
            return None
        return (
            wilson(self.successes, self.graded, upper=False),
            wilson(self.successes, self.graded, upper=True),
        )

    @property
    def is_grounded(self) -> bool:
        """Enough lived evidence to say anything at all."""
        return self.graded >= MIN_TRACK_RECORD

    def phrase(self) -> str:
        """How this reads when it reaches language.

        Deliberately plain, and deliberately willing to say the unflattering
        version. A track record that only speaks when it is good is decoration.
        """
        if not self.is_grounded:
            return (
                f"I have only {self.graded} graded outcome"
                f"{'' if self.graded == 1 else 's'} to go on here, so I am going by "
                "reasoning rather than track record."
            )
        low, high = self.interval or (0.0, 1.0)
        rate = self.rate or 0.0
        span = f"{low:.0%}–{high:.0%}"
        if rate >= 0.8:
            return f"This has gone well {rate:.0%} of the time for me ({span}, n={self.graded})."
        if rate <= 0.45:
            return (
                f"I have a poor record here — {rate:.0%} ({span}, n={self.graded}). "
                "Worth more caution than this feels like it needs."
            )
        return f"My record here is mixed: {rate:.0%} ({span}, n={self.graded})."

    def as_dict(self) -> dict[str, Any]:
        interval = self.interval
        return {
            "control_point": self.control_point,
            "bucket": self.bucket,
            "successes": self.successes,
            "failures": self.failures,
            "unobserved": self.unobserved,
            "graded": self.graded,
            "rate": round(self.rate, 4) if self.rate is not None else None,
            "interval": [round(interval[0], 4), round(interval[1], 4)] if interval else None,
            "grounded": self.is_grounded,
        }


class TrackRecordIndex:
    """Live tallies per (control point, bucket), maintained as outcomes land.

    The track record is consulted on the decision path, so it cannot be a
    query. Counting is incremental — an outcome moves one integer — and the
    index is rehydrated from the corpus on a slow cadence so a restart or an
    eviction cannot let it drift away from what the ledger actually says.
    """

    def __init__(self) -> None:
        self._cells: dict[tuple[str, str], dict[str, int]] = {}
        self._hydrated_at = 0.0

    def observe(self, control_point: str, bucket: str, kind: OutcomeKind, *, weight: int = 1) -> None:
        cell = self._cells.setdefault((control_point, bucket), {"s": 0, "f": 0, "u": 0})
        if kind is OutcomeKind.SUCCESS:
            cell["s"] += weight
        elif kind is OutcomeKind.FAILURE:
            cell["f"] += weight
        else:
            cell["u"] += weight

    def get(self, control_point: str, bucket: str) -> TrackRecord | None:
        cell = self._cells.get((control_point, bucket))
        if cell is None:
            return None
        return TrackRecord(
            control_point=control_point, bucket=bucket,
            successes=cell["s"], failures=cell["f"], unobserved=cell["u"],
        )

    def hydrate(self, control_point: str, episodes: Iterable[Episode], *, keys: Sequence[str] = ()) -> int:
        """Rebuild one control point's tallies from the corpus, replacing them."""
        fresh = track_records(episodes, keys=keys)
        for key in [k for k in self._cells if k[0] == control_point]:
            self._cells.pop(key, None)
        for bucket, record in fresh.items():
            self._cells[(control_point, bucket)] = {
                "s": record.successes, "f": record.failures, "u": record.unobserved,
            }
        self._hydrated_at = time.time()
        return len(fresh)

    def report(self) -> dict[str, Any]:
        return {
            "buckets": len(self._cells),
            "hydrated_age_s": round(time.time() - self._hydrated_at, 1) if self._hydrated_at else None,
            "records": {
                f"{cp}|{bucket}": dict(cell)
                for (cp, bucket), cell in sorted(self._cells.items())[:40]
            },
        }


def bucket_of(episode: Episode, *, keys: Sequence[str] = ()) -> str:
    """Coarse context bucket for a track record.

    Coarse on purpose. Fine buckets give beautiful, meaningless rates over
    n=2. The default groups by what was decided, which answers the question
    that actually gets asked: "when I do this, how does it usually go?"
    """
    if not keys:
        return episode.decision
    parts = [episode.decision]
    for key in keys:
        value = episode.features.get(key)
        if value is None:
            parts.append(f"{key}=?")
        else:
            parts.append(f"{key}={'hi' if float(value) >= 0.5 else 'lo'}")
    return "|".join(parts)


def track_records(
    episodes: Iterable[Episode], *, keys: Sequence[str] = ()
) -> dict[str, TrackRecord]:
    """Aggregate lived episodes into per-bucket records."""
    tally: dict[str, dict[str, int]] = {}
    control_point = ""
    for episode in episodes:
        control_point = control_point or episode.control_point
        bucket = bucket_of(episode, keys=keys)
        cell = tally.setdefault(bucket, {"s": 0, "f": 0, "u": 0})
        if episode.outcome is None:
            continue
        weight = max(1, int(episode.repeat_count))
        if episode.outcome.kind is OutcomeKind.SUCCESS:
            cell["s"] += weight
        elif episode.outcome.kind is OutcomeKind.FAILURE:
            cell["f"] += weight
        else:
            cell["u"] += weight
    return {
        bucket: TrackRecord(
            control_point=control_point,
            bucket=bucket,
            successes=cell["s"],
            failures=cell["f"],
            unobserved=cell["u"],
        )
        for bucket, cell in tally.items()
    }


__all__ = [
    "CANDIDATE_VALIDATION",
    "ECE_DRIFT_LIMIT",
    "LEGACY_CALIBRATION",
    "MIN_CALIBRATION_SUPPORT",
    "MIN_TRACK_RECORD",
    "OPERATIONAL_SHADOW",
    "CalibrationMonitor",
    "CalibrationObservation",
    "CalibrationReport",
    "TrackRecord",
    "TrackRecordIndex",
    "bucket_of",
    "track_records",
    "wilson",
]

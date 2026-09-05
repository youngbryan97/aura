"""core/canonical/state.py — one authoritative value, many estimators.

Every subsystem that currently owns a version of "how Aura is" becomes an
estimator here. It contributes an :class:`Estimate` — a value, a confidence,
and who said it — and reads back the fused answer. The substrate keeps its
dynamics, interiority keeps its faculties, sentiment keeps its analysis; what
none of them keeps is a private copy of the truth.

Fusion is precision-weighted, which is the only defensible way to combine
estimators of different quality: an estimator that says it is sure counts
more, and an estimator that says it is guessing counts less, without anyone
choosing weights by hand. Stale estimates decay rather than being deleted, so
a subsystem that has gone quiet stops driving the answer without its last
word vanishing from the record.

**Disagreement is not averaged away.** Three subsystems putting uncertainty
at 0.2, 0.5 and 0.9 is not a system that believes 0.53. It is a system whose
parts disagree, and that is a fact about her worth more than the mean — it
means one of them is wrong, and which one is a question she can ask. Every
fused reading carries its spread, and a spread past the threshold raises a
:class:`Disagreement` that a consumer can pick up as a cognitive event rather
than a rounding error.

The package imports nothing from the rest of ``core`` except the degradation
recorder. It cannot reach the subsystems that estimate into it, which is what
keeps it from growing opinions of its own.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from core.canonical.channels import CHANNELS, Domain, channel
from core.runtime.lockdep import checked_lock

#: How long an estimate keeps full weight, and how long until it has none. An
#: estimator that has gone quiet should stop steering the answer, but a
#: subsystem that updates every thirty seconds is not stale at thirty-one.
FULL_WEIGHT_S = 30.0
NO_WEIGHT_S = 300.0

#: Spread, as a fraction of the channel's range, past which the estimators are
#: taken to be disagreeing rather than scattering. Below it, two producers
#: differing slightly is measurement noise; above it, they are describing
#: different situations and the mean describes neither.
DISAGREEMENT_THRESHOLD = 0.25

#: Estimators needed before disagreement means anything. One estimator cannot
#: disagree, and two disagreeing gives no way to tell which is the outlier.
MIN_ESTIMATORS_FOR_DISAGREEMENT = 3

#: The channels reconcile() writes to, and therefore may not read from.
#: Without the exclusion, disagreeing about coherence would lower coherence,
#: which would then be disagreed about, and the number would drift on its own
#: evidence until it meant nothing.
_RECONCILE_TARGETS = frozenset({"epistemic.uncertainty", "self.coherence"})

#: The spread at which disagreement is taken to be total. Half the channel's
#: range apart is two subsystems describing opposite situations.
_RECONCILE_FULL_SPREAD = 0.5

#: How many channels disagreeing at once counts as everything disagreeing.
#: Two arguments do not make her twice as lost as one.
_RECONCILE_BREADTH_SATURATES = 4.0

#: What a disagreement is worth as an estimator. High, because it is a direct
#: observation of her own parts rather than an inference about the world —
#: scaled by how severe the disagreement actually is.
_RECONCILE_CONFIDENCE = 0.8


@dataclass(frozen=True)
class Estimate:
    """One subsystem's reading of one canonical channel."""

    channel_id: str
    value: float
    #: How much the producer trusts its own reading, in [0, 1]. An estimator
    #: that always says 1.0 is not confident, it is uncalibrated, and it will
    #: dominate every channel it touches.
    confidence: float
    producer: str
    at: float = field(default_factory=time.time)
    #: Free-text note for the receipt. Never read by fusion.
    note: str = ""

    def weight_at(self, now: float) -> float:
        """Confidence, decayed by age."""
        age = max(0.0, now - self.at)
        if age <= FULL_WEIGHT_S:
            decay = 1.0
        elif age >= NO_WEIGHT_S:
            decay = 0.0
        else:
            decay = 1.0 - (age - FULL_WEIGHT_S) / (NO_WEIGHT_S - FULL_WEIGHT_S)
        return max(0.0, min(1.0, self.confidence)) * decay

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel_id,
            "value": self.value,
            "confidence": self.confidence,
            "producer": self.producer,
            "at": self.at,
            "note": self.note,
        }


@dataclass(frozen=True)
class Disagreement:
    """Estimators describing different situations on one channel.

    Raised rather than averaged. The mean of two subsystems that disagree
    describes neither of them, and the fact that they disagree is information
    about her that the mean destroys.
    """

    channel_id: str
    spread: float
    fused: float
    #: Producer and value, worst outlier first.
    positions: tuple[tuple[str, float], ...]
    at: float = field(default_factory=time.time)

    @property
    def extremes(self) -> tuple[tuple[str, float], tuple[str, float]]:
        return self.positions[0], self.positions[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel_id,
            "spread": round(self.spread, 4),
            "fused": round(self.fused, 4),
            "positions": [[name, round(value, 4)] for name, value in self.positions],
            "at": self.at,
        }


@dataclass(frozen=True)
class Reading:
    """The authoritative value of one channel, and what it rests on."""

    channel_id: str
    value: float
    confidence: float
    #: Weighted standard deviation across contributing estimators, as a
    #: fraction of the channel's range.
    spread: float
    contributors: tuple[str, ...]
    #: True when nobody has estimated this and the channel's neutral is being
    #: reported. A default is not a measurement and says so.
    is_default: bool = False

    @property
    def disagreed(self) -> bool:
        return (
            len(self.contributors) >= MIN_ESTIMATORS_FOR_DISAGREEMENT
            and self.spread > DISAGREEMENT_THRESHOLD
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel_id,
            "value": round(self.value, 4),
            "confidence": round(self.confidence, 4),
            "spread": round(self.spread, 4),
            "contributors": list(self.contributors),
            "is_default": self.is_default,
            "disagreed": self.disagreed,
        }


class CanonicalState:
    """The one place each canonical variable lives."""

    def __init__(self, *, now: Any = time.time) -> None:
        self._now = now
        self._lock = checked_lock("core.canonical.state.CanonicalState", reentrant=True)
        #: channel id → producer → that producer's latest estimate. One per
        #: producer, because a subsystem estimating twice has changed its
        #: mind, not gained a second vote.
        self._estimates: dict[str, dict[str, Estimate]] = {}
        self._disagreements: list[Disagreement] = []
        self._max_disagreements = 64

    # ── writing ──────────────────────────────────────────────────────────

    def estimate(
        self,
        channel_id: str,
        value: float,
        *,
        confidence: float,
        producer: str,
        note: str = "",
    ) -> Estimate:
        """Contribute a reading. Raises on an undeclared channel."""
        spec = channel(channel_id)
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{channel_id} estimate is not a number: {value!r}") from exc
        if not math.isfinite(number):
            raise ValueError(f"{channel_id} estimate is not finite")
        record = Estimate(
            channel_id=spec.id,
            value=spec.clamp(number),
            confidence=max(0.0, min(1.0, float(confidence))),
            producer=str(producer)[:64] or "anonymous",
            at=float(self._now()),
            note=str(note)[:200],
        )
        with self._lock:
            self._estimates.setdefault(spec.id, {})[record.producer] = record
        return record

    def retract(self, channel_id: str, producer: str) -> bool:
        """Withdraw a producer's estimate. True if there was one."""
        with self._lock:
            table = self._estimates.get(str(channel_id))
            if not table:
                return False
            return table.pop(str(producer), None) is not None

    # ── reading ──────────────────────────────────────────────────────────

    def get(self, channel_id: str) -> Reading:
        """The authoritative value of one channel."""
        spec = channel(channel_id)
        now = float(self._now())
        with self._lock:
            live = [
                (e, e.weight_at(now))
                for e in self._estimates.get(spec.id, {}).values()
            ]
        contributing = [(e, w) for e, w in live if w > 0.0]
        if not contributing:
            return Reading(
                channel_id=spec.id,
                value=spec.neutral,
                confidence=0.0,
                spread=0.0,
                contributors=(),
                is_default=True,
            )

        total = sum(w for _e, w in contributing)
        fused = sum(e.value * w for e, w in contributing) / total
        variance = sum(w * (e.value - fused) ** 2 for e, w in contributing) / total
        spread = math.sqrt(max(0.0, variance)) / spec.span

        reading = Reading(
            channel_id=spec.id,
            value=spec.clamp(fused),
            # Confidence is the total weight, saturating: five agreeing
            # estimators are more than one, and fifty are not ten times five.
            confidence=min(1.0, total / 2.0),
            spread=spread,
            contributors=tuple(sorted(e.producer for e, _w in contributing)),
        )
        if reading.disagreed:
            self._note_disagreement(reading, contributing)
        return reading

    def value(self, channel_id: str) -> float:
        return self.get(channel_id).value

    def domain(self, domain: Domain) -> dict[str, Reading]:
        return {
            c.id: self.get(c.id) for c in CHANNELS if c.domain is domain
        }

    def snapshot(self) -> dict[str, Any]:
        readings = {c.id: self.get(c.id) for c in CHANNELS}
        return {
            "channels": {cid: r.to_dict() for cid, r in readings.items()},
            "defaulted": sorted(cid for cid, r in readings.items() if r.is_default),
            "disagreed": sorted(cid for cid, r in readings.items() if r.disagreed),
            "producers": sorted(
                {p for table in self._estimates.values() for p in table}
            ),
        }

    # ── disagreement as an event ─────────────────────────────────────────

    def _note_disagreement(
        self, reading: Reading, contributing: list[tuple[Estimate, float]]
    ) -> None:
        positions = tuple(
            (e.producer, e.value)
            for e, _w in sorted(
                contributing, key=lambda pair: abs(pair[0].value - reading.value), reverse=True
            )
        )
        event = Disagreement(
            channel_id=reading.channel_id,
            spread=reading.spread,
            fused=reading.value,
            positions=positions,
            at=float(self._now()),
        )
        with self._lock:
            # One standing event per channel: a channel that keeps disagreeing
            # is one situation, not a hundred, and a queue that grows per read
            # would make reading the state a way to fill memory.
            self._disagreements = [
                d for d in self._disagreements if d.channel_id != event.channel_id
            ][-(self._max_disagreements - 1):]
            self._disagreements.append(event)

    def disagreements(self) -> tuple[Disagreement, ...]:
        """Channels whose estimators are describing different situations."""
        with self._lock:
            return tuple(self._disagreements)

    def take_disagreements(self) -> tuple[Disagreement, ...]:
        """Read and clear, for a consumer that turns these into cognition."""
        with self._lock:
            out = tuple(self._disagreements)
            self._disagreements = []
        return out

    def reconcile(self) -> dict[str, Any]:
        """Turn standing disagreement into evidence about how sure she is.

        A disagreement is only a cognitive event if something happens next.
        Three subsystems putting uncertainty at 0.2, 0.5 and 0.9 is not a
        system that believes 0.53; it is a system whose parts are describing
        different situations, and the fact that they are is itself the most
        informative thing available. She is, in the plainest sense, less sure
        how she is than any one of them thinks.

        So disagreement estimates into ``epistemic.uncertainty`` and against
        ``self.coherence``, with a confidence that rises with how far apart
        the parts are. The two target channels are excluded from their own
        source set: without that, disagreeing about coherence would lower
        coherence, which would be disagreed about, and the loop would run
        until the number meant nothing.
        """
        # Evaluate every channel first. Disagreement is detected when a
        # channel is read, so relying on the standing list would make this
        # depend on whether somebody happened to have looked — a channel
        # nobody reads would never be found to be in conflict, and the answer
        # would change with call order.
        for spec in CHANNELS:
            if spec.id not in _RECONCILE_TARGETS:
                self.get(spec.id)
        standing = [
            d for d in self.disagreements() if d.channel_id not in _RECONCILE_TARGETS
        ]
        if not standing:
            return {"reconciled": 0, "channels": []}

        worst = max(d.spread for d in standing)
        # Several channels disagreeing at once is a worse sign than one, and
        # not a proportionally worse one: two arguments do not make her twice
        # as lost. The count saturates.
        breadth = min(1.0, len(standing) / _RECONCILE_BREADTH_SATURATES)
        severity = min(1.0, worst / max(1e-9, _RECONCILE_FULL_SPREAD))
        strength = max(severity, breadth * 0.5)

        self.estimate(
            "epistemic.uncertainty",
            strength,
            confidence=_RECONCILE_CONFIDENCE * severity,
            producer="disagreement",
            note=f"{len(standing)} channels, worst spread {worst:.3f}",
        )
        self.estimate(
            "self.coherence",
            1.0 - strength,
            confidence=_RECONCILE_CONFIDENCE * severity,
            producer="disagreement",
            note="her parts describe different situations",
        )
        return {
            "reconciled": len(standing),
            "channels": sorted(d.channel_id for d in standing),
            "worst_spread": round(worst, 4),
            "strength": round(strength, 4),
        }

    def estimates_for(self, channel_id: str) -> tuple[Estimate, ...]:
        with self._lock:
            return tuple(self._estimates.get(str(channel_id), {}).values())

    def producers(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted({p for t in self._estimates.values() for p in t}))

    def clear(self) -> None:
        with self._lock:
            self._estimates.clear()
            self._disagreements = []


_STATE: CanonicalState | None = None
_STATE_LOCK = checked_lock("core.canonical.state.singleton")


def get_canonical_state() -> CanonicalState:
    global _STATE
    with _STATE_LOCK:
        if _STATE is None:
            _STATE = CanonicalState()
        return _STATE


def estimate(
    channel_id: str,
    value: float,
    *,
    confidence: float,
    producer: str,
    note: str = "",
) -> Estimate:
    """Contribute to the canonical state from anywhere."""
    return get_canonical_state().estimate(
        channel_id, value, confidence=confidence, producer=producer, note=note
    )


def read(channel_id: str) -> Reading:
    return get_canonical_state().get(channel_id)


def estimate_many(
    values: Mapping[str, float] | Iterable[tuple[str, float]],
    *,
    confidence: float,
    producer: str,
) -> tuple[Estimate, ...]:
    items = values.items() if isinstance(values, Mapping) else values
    return tuple(
        estimate(cid, v, confidence=confidence, producer=producer) for cid, v in items
    )


__all__ = [
    "DISAGREEMENT_THRESHOLD",
    "FULL_WEIGHT_S",
    "MIN_ESTIMATORS_FOR_DISAGREEMENT",
    "NO_WEIGHT_S",
    "CanonicalState",
    "Disagreement",
    "Estimate",
    "Reading",
    "estimate",
    "estimate_many",
    "get_canonical_state",
    "read",
]

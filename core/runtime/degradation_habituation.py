"""core/runtime/degradation_habituation.py — the hundredth time is not the first.

Clean-room adoption of Anima's structural-scar mechanism (non-commercial
licence; mechanism reimplemented from its design, no code taken). In Anima,
sub-threshold pressure accumulates until it breaks through, and each
breakthrough of a given type lays down a scar whose strength saturates as
``1 - exp(-n*k)``. The scar then ATTENUATES future breakthroughs of that
same type. The organism habituates to its own recurring crises.

Aura already de-weights repeats *inside* one window:
``existential_stakes`` divides by ``(1 + repeat)**2`` so a burst of one
failure cannot read as a cascade. What she has never had is memory across
windows. A degradation that has recurred every day for three weeks opens
each new window at full weight, and because degradation weight feeds the
survival term, a known, understood, chronic condition goes on generating
fresh existential threat forever.

That is alarm fatigue with the polarity reversed: instead of the operator
learning to ignore the alarm, the organism never learns, and keeps paying
the full affective cost of a fact it has already absorbed.

Three constraints shape the design:

* **The record never attenuates.** Habituation applies to the *felt* weight
  — survival pressure, threat, alarm — and never to the degradation log,
  the receipt, or the count. An audit trail that fades is not an audit
  trail. This is the whole reason attenuation lives here rather than inside
  ``record_degradation``.
* **It never reaches zero.** Attenuation is capped, leaving a floor of
  :data:`_RESIDUAL`. A chronic problem must stay perceptible; a mechanism
  that could silence one completely would be a way for a persistent fault
  to become invisible by persisting, which is precisely backwards.
* **It re-sensitises.** Scars decay when a signature stops recurring, so a
  fault that was fixed and returns a month later lands close to full
  weight again. Habituation is to the ONGOING, not to the historical.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from core.runtime.lockdep import LockRank, checked_lock

__all__ = [
    "Scar",
    "DegradationHabituation",
    "get_habituation",
    "note_recurrence",
    "attenuation_for",
    "signature_for",
]


def signature_for(subsystem: object, error_type: object) -> str:
    """THE key for habituation. Every writer and reader must call this.

    A counter that accumulates under one key while its reader tests another
    is a gate that never fires, and it fails silently in the direction of
    looking healthy. This function is the single source of truth so that
    cannot happen: ``record_degradation`` writes through it and
    ``existential_stakes`` reads through it.

    The granularity is deliberately the failure CLASS — subsystem plus
    exception type — and not the message. Habituation is about "this kind
    of thing again", and message text carries ids, paths and timings that
    would make every occurrence a new signature and defeat the mechanism
    entirely. Callers that need message-level granularity for a different
    purpose (within-window deduplication, say) should keep their own key
    and not reuse this one.
    """
    sub = str(subsystem or "unknown").strip() or "unknown"
    kind = str(error_type or "unknown").strip() or "unknown"
    return f"{sub}|{kind}"

#: Occurrences that attract NO attenuation at all. Habituation must never
#: be able to muffle a genuine new cascade while it is still unfolding, and
#: a saturating curve alone does not give that: with growth alone, the
#: second occurrence of a brand-new failure already lost 30% of its weight.
#: Below this count the multiplier is exactly 1.0, so the early events of
#: something new always land at full force.
_FREE_OCCURRENCES = 5

#: Growth constant, applied to occurrences BEYOND the free ones. At 0.35,
#: attenuation reaches half its cap about three recurrences after the free
#: allowance and is near saturation ten past it — fast enough that a daily
#: fault stops screaming within a week or so of becoming daily.
_GROWTH = 0.35

#: Maximum share of weight habituation may remove. The remaining
#: :data:`_RESIDUAL` is what a fully-habituated chronic condition still
#: contributes, forever.
_MAX_ATTENUATION = 0.6
_RESIDUAL = 1.0 - _MAX_ATTENUATION

#: A scar loses this fraction of its strength per hour of silence. At
#: 0.02/h a saturated scar is back near full sensitivity in about two
#: days of quiet — long enough to cover a normal working gap, short enough
#: that a fault which returns next week is heard properly.
_DECAY_PER_HOUR = 0.02

#: Bound on distinct tracked signatures, so an unbounded variety of error
#: strings cannot make this a memory surface.
_MAX_SIGNATURES = 2048


@dataclass
class Scar:
    """Accumulated familiarity with one recurring failure signature."""

    signature: str
    count: int = 0
    strength: float = 0.0
    first_seen: float = field(default_factory=lambda: time.time())
    last_seen: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature,
            "count": self.count,
            "strength": round(self.strength, 4),
            "attenuation": round(1.0 - self.multiplier(), 4),
            "multiplier": round(self.multiplier(), 4),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }

    def multiplier(self) -> float:
        """What fraction of the alarm weight still lands. Never below the floor."""
        return max(_RESIDUAL, 1.0 - self.strength * _MAX_ATTENUATION)


class DegradationHabituation:
    """Tracks how familiar each failure signature has become."""

    def __init__(self) -> None:
        self._lock = checked_lock("degradation_habituation", rank=LockRank.LEAF)
        self._scars: dict[str, Scar] = {}

    # ------------------------------------------------------------------ decay

    def _decayed_strength(self, scar: Scar, *, now: float) -> float:
        """Strength after the silence since ``last_seen``."""
        quiet_hours = max(0.0, (now - scar.last_seen) / 3600.0)
        if quiet_hours <= 0.0:
            return scar.strength
        return max(0.0, scar.strength - quiet_hours * _DECAY_PER_HOUR)

    # ------------------------------------------------------------------ write

    def note(self, signature: str, *, now: float | None = None) -> Scar:
        """Record one more occurrence of ``signature`` and grow its scar."""
        key = str(signature or "").strip() or "unknown"
        moment = time.time() if now is None else now
        with self._lock:
            scar = self._scars.get(key)
            if scar is None:
                scar = Scar(signature=key, first_seen=moment, last_seen=moment)
                self._scars[key] = scar
            else:
                scar.strength = self._decayed_strength(scar, now=moment)
            scar.count += 1
            # Saturating, and only past the free allowance: each recurrence
            # adds less than the one before, so a fault cannot be habituated
            # away by sheer volume in a burst, and the first few occurrences
            # of anything new are never discounted at all.
            beyond = max(0, scar.count - _FREE_OCCURRENCES)
            scar.strength = 1.0 - math.exp(-beyond * _GROWTH) if beyond else 0.0
            scar.last_seen = moment
            self._evict_locked()
            return Scar(**{**scar.__dict__})

    def _evict_locked(self) -> None:
        if len(self._scars) <= _MAX_SIGNATURES:
            return
        # Drop the least recently seen: a signature nobody has hit in a long
        # time is also the one whose scar has decayed closest to nothing.
        stale = sorted(self._scars.values(), key=lambda s: s.last_seen)
        for scar in stale[: len(self._scars) - _MAX_SIGNATURES]:
            self._scars.pop(scar.signature, None)

    # ------------------------------------------------------------------- read

    def multiplier(self, signature: str, *, now: float | None = None) -> float:
        """Weight multiplier for ``signature`` — 1.0 when it is unfamiliar.

        Read-only: asking what something would weigh must not itself make
        the system more familiar with it.
        """
        # Fast path, taken on the overwhelming majority of calls: nothing
        # has ever been habituated, so there is nothing to look up. This
        # runs inside the per-record loop that computes survival pressure,
        # where the weight is age-decayed and the saturation margin is
        # measured in milliseconds — a lock acquisition per record was
        # enough to change the answer on a loaded host. Reading a dict for
        # emptiness is atomic under CPython, and the worst case is that a
        # scar added microseconds ago is missed for one call, which cannot
        # matter to an alarm-weighting heuristic.
        if not self._scars:
            return 1.0
        key = str(signature or "").strip() or "unknown"
        moment = time.time() if now is None else now
        with self._lock:
            scar = self._scars.get(key)
            if scar is None:
                return 1.0
            strength = self._decayed_strength(scar, now=moment)
        return max(_RESIDUAL, 1.0 - strength * _MAX_ATTENUATION)

    def attenuation(self, signature: str, *, now: float | None = None) -> float:
        """Fraction of weight removed. The complement of :meth:`multiplier`."""
        return round(1.0 - self.multiplier(signature, now=now), 4)

    def scar(self, signature: str) -> Scar | None:
        with self._lock:
            scar = self._scars.get(str(signature or "").strip() or "unknown")
            return Scar(**{**scar.__dict__}) if scar else None

    def chronic(self, *, minimum_count: int = 5) -> list[dict[str, Any]]:
        """Signatures familiar enough to be called chronic.

        This is the useful output for a human: not "what failed" but "what
        keeps failing, and how long it has been doing so".
        """
        with self._lock:
            scars = [s for s in self._scars.values() if s.count >= minimum_count]
        return [
            {**s.to_dict(), "recurring_for_h": round((s.last_seen - s.first_seen) / 3600.0, 2)}
            for s in sorted(scars, key=lambda s: -s.count)
        ]

    def status(self) -> dict[str, Any]:
        with self._lock:
            total = len(self._scars)
            saturated = sum(1 for s in self._scars.values() if s.strength > 0.9)
        return {
            "signatures_tracked": total,
            "saturated": saturated,
            "max_attenuation": _MAX_ATTENUATION,
            "residual_floor": _RESIDUAL,
            "chronic": self.chronic(),
        }

    def reset_for_test(self) -> None:
        with self._lock:
            self._scars.clear()


_HABITUATION = DegradationHabituation()


def get_habituation() -> DegradationHabituation:
    return _HABITUATION


def note_recurrence(signature: str) -> Scar:
    return _HABITUATION.note(signature)


def attenuation_for(signature: str) -> float:
    return _HABITUATION.multiplier(signature)

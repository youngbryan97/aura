"""core/social/social_stamina.py — wanting company and being able to sustain it.

Being sociable is usually modelled as one number: how much an agent likes
being with people. That number cannot represent the ordinary situation of
wanting to see someone and not having it in you, which is not ambivalence and
not a contradiction. Two quantities are involved and they move independently.

**Belonging** is a need. It relaxes toward met in company and toward unmet in
solitude, first-order, on one time constant — so an agent who spends half its
time with people settles near the middle, and the share of time in company is
the only thing that sets where it settles. Scaling the two directions
differently is how a first draft of this made someone lonelier the more
evenings they spent out.

**Stamina** is a capacity. It falls during interaction at a rate that depends
on who and what kind, and recovers in solitude. It is a budget, not a
preference, and no amount of wanting refills it.

The pair makes the interesting quantity computable. If interaction costs `d`
per unit time and solitude returns `r` per unit time, the largest share of
time that can be spent with people indefinitely is

    duty cycle = r / (d + r)

Nothing is chosen there — it is the balance point, and everything above it is
borrowing. A schedule that exceeds it does not feel different on any single
day, which is exactly why it needs computing rather than sensing:
``sustainable_share`` gives the number, and ``overdrawn_for`` says how long
the current pattern has been above it.

## Empty is not the bottom of a scale

Running to zero and then recovering does not put an agent back where a
smaller deficit would have. Recovery from empty runs slower than recovery from
low, so the same total deficit costs more when it is taken all at once. That
asymmetry is what makes exhaustion a state rather than a low reading, and it
is the reason ``recovery_time`` is not simply the deficit over the rate.

## What it is not

Not empathy, which is in ``core/affect/empathic_coupling.py`` and is about
being moved by someone. Not care, which is in
``core/ethics/care_allocation.py`` and is about spending on their behalf. An
agent can have a full battery and no empathy, or deep empathy and nothing
left. Collapsing the three loses every case where they disagree, and those
are the cases worth being able to see.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.Social.Stamina")

#: How much slower recovery runs from empty than from a partial deficit. The
#: value is the shape of the asymmetry rather than a measurement: what the
#: module asserts is that the two rates differ, and a caller with data should
#: fit it. Set to one and exhaustion becomes an ordinary low reading, which is
#: the ablation.
EMPTY_PENALTY = 2.0

#: Encounters kept per counterpart.
MAX_ENCOUNTERS = 512


@dataclass
class Company:
    """What one counterpart costs and what being with them returns.

    Cost is per unit time and is estimated from what actually happened, not
    declared. Some company is restful and some is not, and an agent that
    cannot tell them apart will schedule the wrong evenings.
    """

    key: str
    drain_per_s: float = 1.0
    encounters: int = 0
    total_seconds: float = 0.0
    observed_drain: list[float] = field(default_factory=list)

    def observe(self, seconds: float, spent: float) -> None:
        """Record an encounter and what it actually cost."""
        if seconds <= 0:
            return
        self.encounters += 1
        self.total_seconds += seconds
        self.observed_drain.append(spent / seconds)
        if len(self.observed_drain) > MAX_ENCOUNTERS:
            del self.observed_drain[: len(self.observed_drain) - MAX_ENCOUNTERS]
        # A running mean rather than the last reading: one bad evening with
        # someone is not a fact about them.
        self.drain_per_s = float(sum(self.observed_drain) / len(self.observed_drain))

    @property
    def restful(self) -> bool:
        """Company that costs less than the average. Not a judgement of them."""
        return self.drain_per_s <= 0.0


@dataclass(frozen=True)
class Reading:
    """Where both quantities stand, kept apart."""

    belonging: float
    """Unmet need for company, in [0, 1]. Rises with time apart."""

    stamina: float
    """Capacity remaining, in [0, 1]. Falls with interaction."""

    sustainable_share: float
    """Largest share of time that can be spent in company indefinitely."""

    recent_share: float
    """Share actually spent in company over the recent window."""

    overdrawn: bool
    exhausted: bool
    at: float

    @property
    def wants_but_cannot(self) -> bool:
        """The case a single sociability number cannot represent."""
        return self.belonging > 0.5 and self.stamina < 0.25

    def as_dict(self) -> dict[str, Any]:
        return {
            "belonging": round(self.belonging, 4),
            "stamina": round(self.stamina, 4),
            "sustainable_share": round(self.sustainable_share, 4),
            "recent_share": round(self.recent_share, 4),
            "overdrawn": self.overdrawn,
            "exhausted": self.exhausted,
            "wants_but_cannot": self.wants_but_cannot,
        }


class SocialStamina:
    """A belonging need and a capacity, tracked separately.

    Time is passed in rather than read from a clock, so a caller can replay a
    week in a test and get the same answer the live path would give.
    """

    def __init__(
        self,
        *,
        drain_per_s: float = 1.0 / 3600.0,
        recovery_per_s: float = 1.0 / 7200.0,
        belonging_per_s: float = 1.0 / (3 * 86400.0),
    ) -> None:
        #: Capacity spent per second of ordinary company, before per-person
        #: correction. Falls out of the caller's units; the ratio against
        #: recovery is what the arithmetic below actually uses.
        self.drain_per_s = float(drain_per_s)
        self.recovery_per_s = float(recovery_per_s)
        #: Rate constant of the belonging relaxation, in reciprocal seconds.
        #: One constant for both directions, so the resting level is set by
        #: the share of time in company and by nothing else.
        self.belonging_per_s = float(belonging_per_s)

        self.stamina = 1.0
        self.belonging = 0.0
        self._company: dict[str, Company] = {}
        self._log: list[tuple[float, float, bool]] = []
        self._emptied_at: float | None = None
        self._overdrawn_since: float | None = None

    def company(self, key: str) -> Company:
        record = self._company.get(key)
        if record is None:
            record = Company(key=key, drain_per_s=self.drain_per_s)
            self._company[key] = record
        return record

    # ------------------------------------------------------------- dynamics

    def sustainable_share(self, *, with_person: str | None = None) -> float:
        """Share of time in company that holds the capacity level.

        ``recovery / (drain + recovery)``. It is the balance point, so
        everything above it is borrowed against a later recovery and
        everything below it accumulates slack.
        """
        drain = (
            self.company(with_person).drain_per_s
            if with_person is not None
            else self.drain_per_s
        )
        if drain <= 0:
            return 1.0
        return self.recovery_per_s / (drain + self.recovery_per_s)

    def spend(self, seconds: float, *, with_person: str = "", at: float | None = None) -> float:
        """Time in company. Returns what it cost."""
        moment = at if at is not None else time.time()
        drain = (
            self.company(with_person).drain_per_s if with_person else self.drain_per_s
        )
        cost = max(0.0, drain) * max(0.0, seconds)
        self.stamina = max(0.0, self.stamina - cost)
        # Being with people meets the need, whatever it costs to be there.
        # Exponential relaxation toward met, on the same constant solitude
        # uses toward unmet — the two directions must match or the resting
        # level stops being a function of the share of time in company.
        self.belonging *= math.exp(-self.belonging_per_s * max(0.0, seconds))
        if with_person:
            self.company(with_person).observe(seconds, cost)
        if self.stamina <= 0.0 and self._emptied_at is None:
            self._emptied_at = moment
        self._record(moment, seconds, True)
        return cost

    def rest(self, seconds: float, *, at: float | None = None) -> float:
        """Time alone. Returns what it recovered.

        Recovery from empty runs slower, so the same deficit costs more when
        it was taken all at once. Without that asymmetry, exhaustion is just a
        low number and the schedule that produced it looks fine in hindsight.
        """
        moment = at if at is not None else time.time()
        rate = self.recovery_per_s
        if self._emptied_at is not None:
            rate /= EMPTY_PENALTY
        gained = min(1.0 - self.stamina, max(0.0, rate) * max(0.0, seconds))
        self.stamina = min(1.0, self.stamina + gained)
        self.belonging = 1.0 - (1.0 - self.belonging) * math.exp(
            -self.belonging_per_s * max(0.0, seconds)
        )
        if self.stamina >= 1.0:
            self._emptied_at = None
        self._record(moment, seconds, False)
        return gained

    def _record(self, at: float, seconds: float, in_company: bool) -> None:
        self._log.append((at, seconds, in_company))
        if len(self._log) > MAX_ENCOUNTERS:
            del self._log[: len(self._log) - MAX_ENCOUNTERS]

    def recent_share(self, *, window_s: float = 7 * 86400.0,
                     at: float | None = None) -> float:
        moment = at if at is not None else time.time()
        recent = [(s, c) for t, s, c in self._log if moment - t <= window_s]
        total = sum(s for s, _ in recent)
        if total <= 0:
            return 0.0
        return sum(s for s, c in recent if c) / total

    def recovery_time(self, *, to: float = 1.0) -> float:
        """Seconds of solitude needed to reach a level.

        Not the deficit over the rate whenever the capacity has been run to
        nothing: the asymmetry above makes recovery from empty slower.
        """
        deficit = max(0.0, min(to, 1.0) - self.stamina)
        if deficit <= 0 or self.recovery_per_s <= 0:
            return 0.0
        rate = self.recovery_per_s
        if self._emptied_at is not None:
            rate /= EMPTY_PENALTY
        return deficit / rate

    def read(self, *, window_s: float = 7 * 86400.0, at: float | None = None) -> Reading:
        moment = at if at is not None else time.time()
        share = self.recent_share(window_s=window_s, at=moment)
        sustainable = self.sustainable_share()
        overdrawn = share > sustainable
        if overdrawn and self._overdrawn_since is None:
            self._overdrawn_since = moment
        elif not overdrawn:
            self._overdrawn_since = None
        return Reading(
            belonging=self.belonging, stamina=self.stamina,
            sustainable_share=sustainable, recent_share=share,
            overdrawn=overdrawn, exhausted=self._emptied_at is not None,
            at=moment,
        )

    def overdrawn_for(self, *, at: float | None = None) -> float:
        if self._overdrawn_since is None:
            return 0.0
        return max(0.0, (at if at is not None else time.time()) - self._overdrawn_since)

    def status(self, *, at: float | None = None) -> dict[str, Any]:
        reading = self.read(at=at)
        return {
            **reading.as_dict(),
            "overdrawn_for_s": round(self.overdrawn_for(at=at), 2),
            "recovery_time_s": round(self.recovery_time(), 2),
            "company": {
                key: {
                    "drain_per_s": round(c.drain_per_s, 8),
                    "encounters": c.encounters,
                    "restful": c.restful,
                }
                for key, c in sorted(self._company.items())
            },
        }


def sustainable_hours_per_week(drain_per_hour: float, recovery_per_hour: float) -> float:
    """The duty cycle in the units a person would actually ask it in."""
    if drain_per_hour <= 0:
        return 168.0
    return 168.0 * recovery_per_hour / (drain_per_hour + recovery_per_hour)


_STAMINA: SocialStamina | None = None


def get_social_stamina() -> SocialStamina:
    global _STAMINA
    if _STAMINA is None:
        _STAMINA = SocialStamina()
    return _STAMINA


def reset_social_stamina_for_test() -> None:
    global _STAMINA
    _STAMINA = None

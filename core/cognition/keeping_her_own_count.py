"""What an instrument reads, kept by her own count, and read again only when her count may have drifted.

A clock she has read, a balance, a gauge, a counter of anything that changes by what she does and by a steady rule:
once read, what it shows next follows from what she has done since, and her own count says it without looking. A look
costs time and a count costs nothing. Each reading tells her how far her count had drifted from it, and the part of
that drift which comes with every step (time a game adds after each move, the moment it takes her to see a change) is
learned and added to every later step. She reads again only when what her count could have missed since the last
reading is more than she can afford to be wrong by.

Nothing here knows what is counted. A step is whatever she did; its change is in the instrument's units.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

__all__ = ["KeptCount"]

#: Readings remembered, for what a step's count usually misses.
REMEMBERED = 32


@dataclass
class KeptCount:
    """One instrument's value as her count keeps it between readings."""

    #: What the instrument would show now by her count: the last reading, carried on by every step since.
    value: float | None = None
    #: Steps counted since the last reading.
    since: int = 0
    #: What each step changed that her count did not know, learned from readings: added to every step.
    per_step: float = 0.0
    #: At each reading after the first: what her count had missed, per step since the reading before.
    missed: list[float] = field(default_factory=list)

    def read(self, reading: float) -> None:
        """The instrument read ``reading``: her count is set to it, and what the count missed is learned."""
        reading = float(reading)
        if self.value is not None and self.since:
            # What a step changed that her count did not know: what the last steps missed, with what was already being
            # added for it put back.
            unknown = (reading - self.value) / self.since + self.per_step
            self.missed = [*self.missed[-(REMEMBERED - 1):], unknown]
            self.per_step = statistics.fmean(self.missed)
        self.value = reading
        self.since = 0

    def step(self, change: float) -> None:
        """She did something that changes what the instrument shows by ``change``, by her own count."""
        if self.value is None:
            return
        self.value += float(change) + self.per_step
        self.since += 1

    def may_be_off_by(self) -> float | None:
        """How far her count may be from what the instrument would show: what a step's count usually misses, for each
        step since the last reading. None before readings have shown both what a step misses and how far that
        varies."""
        if self.value is None or len(self.missed) < 2:
            return None
        usual = statistics.median(abs(m - self.per_step) for m in self.missed)
        return usual * self.since

    def must_read(self, can_be_off_by: float) -> bool:
        """Whether to read the instrument rather than go by her count: there is no count yet, no reading has shown
        what her count misses, or it may now be off by more than ``can_be_off_by``. Nothing has been counted since a
        reading, and the count is the reading."""
        if self.value is not None and self.since == 0:
            return False
        off = self.may_be_off_by()
        return off is None or off > can_be_off_by

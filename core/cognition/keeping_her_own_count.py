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
    #: At each reading after the first: how far her count was from it, and over how many steps.
    missed: list[tuple[float, int]] = field(default_factory=list)
    #: Of the readings remembered, what the steps changed that her count did not know, in all, and how many steps.
    unknown: list[tuple[float, int]] = field(default_factory=list)

    def read(self, reading: float) -> None:
        """The instrument read ``reading``: her count is set to it, and what the count missed is learned. What each
        step adds unknown is everything the steps since the readings began added unknown, over how many steps there
        were: an instrument read in whole units misses by up to half of one each time, so a step's share of one
        reading's miss says little, and a share of many says more."""
        reading = float(reading)
        if self.value is not None and self.since:
            miss = reading - self.value
            self.missed = [*self.missed[-(REMEMBERED - 1):], (miss, self.since)]
            # What the steps added that her count did not know: what they missed, with what was already being added
            # for it put back.
            self.unknown = [*self.unknown[-(REMEMBERED - 1):], (miss + self.per_step * self.since, self.since)]
            self.per_step = sum(u for u, _ in self.unknown) / sum(n for _, n in self.unknown)
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
        usual = statistics.median(abs(miss) / steps for miss, steps in self.missed)
        return usual * self.since

    def must_read(self, can_be_off_by: float) -> bool:
        """Whether to read the instrument rather than go by her count: there is no count yet, no reading has shown
        what her count misses, or it may now be off by more than ``can_be_off_by``. Nothing has been counted since a
        reading, and the count is the reading."""
        if self.value is not None and self.since == 0:
            return False
        off = self.may_be_off_by()
        return off is None or off > can_be_off_by

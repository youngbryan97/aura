"""Bars on a screen that fill and empty: how full each is, and when it went up or down.

Health, energy, fuel, paint left, a boss's strength, a charge, time left, how
much of a field is covered: a great many things say how much is left by a bar,
not a number. A strip of one colour along a track, longer for more. Nothing
reads it aloud; a person sees it shorten as they are hit, and knows.

LIVE 2026-10-09 a game's three heroes shared a health bar, and its rules said
so; she played as if nothing could hurt her. Another drained a bucket of paint
with every patch painted and a fifth of it at every knock from the campers
chasing her; she stood still while they knocked her five times, and lost in
seconds, each time.

So a strip of one colour, long and thin, that stays where it is on the screen
while the rest goes on, is followed: how far its colour runs along it, from
the end that stays put. A strip whose run changes, and does not move, is a bar,
and each change is something going up or down. What it measures is read from
the words beside it where there are any (core/agency/what_meeting_things_does.py
reads a counter's label the same way); a bar with no words is taken to be
something running out, worse lower, until play says otherwise.

Nothing here knows a game. Measured on whatever pictures it is given.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["Bar", "BarsOnTheScreen"]

#: The width pictures are read at.
READ_WIDTH = 320

#: How alike two pixels' colours must be, on their most different channel, to be one colour of a bar.
ONE_COLOUR = 28

#: A strip: at least this share of the picture's width long, at most this share wide, and this many times longer than
#: it is thick; at most this share of the picture's height thick.
LONGEST = 0.6
SHORTEST = 0.06
LONGER_THAN_THICK = 3.5
THICKEST = 0.08

#: How far, in read pixels, a run must change to have gone up or down, and how many changes make a strip a bar.
CHANGED_BY = 2
CHANGES_TO_BE_A_BAR = 2

#: How many reads a bar holds its level before a change counts: a bar stands at a level between changes, where a run
#: of sky or floor changes at every read as things cross it.
STEADY_READS = 3

#: How often the strips are looked for afresh, in seconds, and the most followed at once.
LOOK_FOR_STRIPS_S = 3.0
MOST_STRIPS = 40


@dataclass
class Bar:
    """A strip of one colour followed along its row: where it is, its colour, and how far it ran each time it was read."""

    row: int
    thick: int
    colour: tuple[int, int, int]
    start: int
    end: int
    #: The end that has stayed put, once one has moved: "start" or "end", or "" until then.
    anchored: str = ""
    longest: int = 0
    changes: int = 0
    #: Reads in a row at the same level; and changes that came without the level having held first.
    steady: int = 0
    unsteady: int = 0
    #: (when, how far it ran) at each read.
    levels: list[tuple[float, int]] = field(default_factory=list)
    #: The words beside it, and what a change in them means ("up is good", "down is bad", "neither", "").
    label: str = ""
    meaning: str = ""

    @property
    def is_a_bar(self) -> bool:
        return self.changes >= CHANGES_TO_BE_A_BAR and self.unsteady <= self.changes

    @property
    def full(self) -> float:
        """How full it is, as a share of the longest it has been: 1 at its fullest yet."""
        return (self.end - self.start) / max(1, self.longest)

    def where(self, wide: int, tall: int) -> tuple[float, float, float, float]:
        """Where it is, as shares of the picture: left, top, right, bottom."""
        return self.start / wide, (self.row - self.thick / 2) / tall, self.end / wide, (self.row + self.thick / 2) / tall


def _strips(picture: np.ndarray) -> list[Bar]:
    """Runs of one colour along rows, long and thin, as strips not yet known to be bars."""
    tall, wide = picture.shape[:2]
    rows = picture.astype(np.int16)
    same_as_left = np.zeros((tall, wide), dtype=bool)
    same_as_left[:, 1:] = np.abs(rows[:, 1:] - rows[:, :-1]).max(axis=2) <= ONE_COLOUR // 2
    found: list[Bar] = []
    for y in range(1, tall - 1, 2):
        breaks = np.flatnonzero(~same_as_left[y])
        edges = np.append(breaks, wide)
        for a, b in zip(edges[:-1], edges[1:], strict=True):
            length = int(b - a)
            if not SHORTEST * wide <= length <= LONGEST * wide:
                continue
            colour = rows[y, a:b].mean(axis=0)
            thick = 1
            for step in (-1, 1):
                yy = y + step
                while 0 <= yy < tall and thick <= THICKEST * tall:
                    if np.abs(rows[yy, a:b] - colour).max(axis=1).mean() > ONE_COLOUR:
                        break
                    thick += 1
                    yy += step
            if thick > THICKEST * tall or length < LONGER_THAN_THICK * thick or thick < 2:
                continue
            if a == 0 or b == wide:
                continue  # running off the picture: sky, floor or a wall, not a bar drawn on it
            if any(abs(bar.row - y) <= bar.thick and abs(bar.start - a) <= 2 and abs(bar.end - b) <= 2 for bar in found):
                continue
            found.append(Bar(y, thick, tuple(int(c) for c in colour), int(a), int(b), longest=length))
    return found


def _run_of(picture: np.ndarray, bar: Bar) -> tuple[int, int] | None:
    """Where a strip's colour runs along its row now, through where it was: its start and end, or None where gone."""
    tall, wide = picture.shape[:2]
    if not 0 <= bar.row < tall:
        return None
    line = np.abs(picture[bar.row].astype(np.int16) - np.asarray(bar.colour)).max(axis=1) <= ONE_COLOUR
    middle = (bar.start + bar.end) // 2
    # From the end that has stayed put, else from the middle of where it was.
    seed = bar.start if bar.anchored == "start" else bar.end - 1 if bar.anchored == "end" else middle
    seed = min(wide - 1, max(0, seed))
    if not line[seed]:
        nearby = [x for x in range(max(0, seed - 2), min(wide, seed + 3)) if line[x]]
        if not nearby:
            return None
        seed = nearby[0]
    start = seed
    while start > 0 and line[start - 1]:
        start -= 1
    end = seed + 1
    while end < wide and line[end]:
        end += 1
    return start, end


@dataclass
class BarsOnTheScreen:
    """Strips followed from picture to picture, and the ones that have shown they are bars."""

    strips: list[Bar] = field(default_factory=list)
    looked_at: float = -math.inf
    shape: tuple[int, int] = (0, 0)

    def bars(self) -> list[Bar]:
        return [bar for bar in self.strips if bar.is_a_bar]

    def read(self, picture: Any, at: float) -> list[dict[str, Any]]:
        """Read every strip in one picture; each bar's change, as {"bar", "from", "to", "at", "since"}."""
        from core.perception.picture_arithmetic import shrink

        picture = np.asarray(picture)
        tall, wide = picture.shape[:2]
        if wide > READ_WIDTH:
            picture = shrink(picture, READ_WIDTH, max(1, round(tall * READ_WIDTH / wide)))
        self.shape = picture.shape[:2]
        if at - self.looked_at >= LOOK_FOR_STRIPS_S:
            self.looked_at = at
            self._take_in(_strips(picture))
        changes: list[dict[str, Any]] = []
        for bar in list(self.strips):
            run = _run_of(picture, bar)
            if run is None:
                if not bar.is_a_bar:
                    self.strips.remove(bar)
                continue
            change = self._went(bar, run)
            since = bar.levels[-1][0] if bar.levels else at
            bar.levels.append((at, run[1] - run[0]))
            del bar.levels[:-120]
            if change and bar.is_a_bar:
                changes.append({"bar": bar, "from": change[0], "to": change[1], "at": at, "since": since})
        return changes

    def _take_in(self, found: list[Bar]) -> None:
        for strip in found:
            if any(abs(bar.row - strip.row) <= max(bar.thick, 2) and strip.start < bar.end + 3 and bar.start < strip.end + 3
                   for bar in self.strips):
                continue
            self.strips.append(strip)
        if len(self.strips) > MOST_STRIPS:
            bars = self.bars()
            self.strips = bars + [s for s in self.strips if not s.is_a_bar][-(MOST_STRIPS - len(bars)):]

    @staticmethod
    def _went(bar: Bar, run: tuple[int, int]) -> tuple[float, float] | None:
        """Whether a strip's run changed at one end only, the other staying put: how full it was and is."""
        start, end = run
        moved_start, moved_end = abs(start - bar.start) >= CHANGED_BY, abs(end - bar.end) >= CHANGED_BY
        if moved_start and moved_end:
            # Both ends went: the strip moved, or something crossed it. Not a bar's change.
            bar.start, bar.end, bar.steady = start, end, 0
            bar.unsteady += 1
            return None
        if not moved_start and not moved_end:
            bar.steady += 1
            return None
        held, bar.steady = bar.steady >= STEADY_READS, 0
        if not held:
            bar.unsteady += 1
            bar.start, bar.end = start, end
            return None
        was = bar.full
        anchored = "end" if moved_start else "start"
        if bar.anchored and bar.anchored != anchored:
            bar.start, bar.end = start, end
            return None
        bar.anchored = anchored
        bar.start, bar.end = start, end
        bar.longest = max(bar.longest, end - start)
        bar.changes += 1
        return was, bar.full

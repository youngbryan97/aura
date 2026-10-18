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

from core.verify import invariant

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
CHANGES_TO_BE_A_BAR = 3

#: The widest break, in read pixels, between two runs of a bar's colour that still leaves them one bar: the line between
#: the cells of a bar drawn in cells, as health is drawn in hearts or segments. Offline 2026-10-12 a food fight's two
#: bars of five cells each were never found, each cell too short to be a strip, and she never knew she was being hit.
DIVIDER = 3

#: How long a strip not found where it was is still looked for there: a screen that shakes when she is hit takes every
#: bar with it for a moment.
MISSING_S = 1.0

#: How long a bar holds its level before a change counts, in seconds: a bar stands at a level between changes, where a
#: run of sky, sand or water changes as things cross it or as it ripples. LIVE 2026-10-09 three strips of a beach scene,
#: each held for a tenth of a second, were taken for bars going down.
STEADY_S = 0.4

#: How often the strips are looked for afresh, in seconds, and the most followed at once.
LOOK_FOR_STRIPS_S = 3.0
MOST_STRIPS = 40
DISCOVERY_ROWS_PER_PICTURE = 8


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
    #: Since when it has held its level; and changes that came without the level having held first.
    steady_since: float = 0.0
    unsteady: int = 0
    #: (when, how far it ran) at each read.
    levels: list[tuple[float, int]] = field(default_factory=list)
    #: The words beside it, and what a change in them means ("up is good", "down is bad", "neither", "").
    label: str = ""
    meaning: str = ""
    #: Since when it has not been found where it was (0 while it is).
    missing_since: float = 0.0

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


def _median_of_columns(values: np.ndarray) -> np.ndarray:
    """The median of each column, as ``np.median(values, axis=0)`` gives it, from one partition of the column."""
    n = values.shape[0]
    if n % 2:
        return np.partition(values, n // 2, axis=0)[n // 2].astype(np.float64)
    middle = np.partition(values, (n // 2 - 1, n // 2), axis=0)
    return np.mean(middle[n // 2 - 1:n // 2 + 1].astype(np.float64), axis=0)


def _mostly_unlike(far: np.ndarray, within: float) -> bool:
    """Whether the median of ``far`` is more than ``within``, as ``np.median(far) > within`` says, from a count: half or
    more of the values beyond it, and where exactly half are, the two middle values taken as the median takes them."""
    n = far.size
    beyond = int(np.count_nonzero(far > within))
    if n % 2:
        return beyond >= (n + 1) // 2
    if beyond != n // 2:
        return beyond > n // 2
    return (float(far[far <= within].max()) + float(far[far > within].min())) / 2 > within


def _strip_at(rows: np.ndarray, y: int, a: int, b: int) -> Bar | None:
    """Measure the ordinary strip predicates on this picture's signed pixels."""
    tall, wide = rows.shape[:2]
    length = int(b-a)
    if not SHORTEST*wide <= length <= LONGEST*wide:
        return None
    colour = _median_of_columns(rows[y, a:b])
    thick, reach = 1, {-1: y, 1: y}
    for step in (-1, 1):
        yy = y+step
        while 0 <= yy < tall and thick <= THICKEST*tall:
            if _mostly_unlike(np.abs(rows[yy, a:b]-colour).max(axis=1), ONE_COLOUR):
                break
            thick += 1
            reach[step] = yy
            yy += step
    if (thick > THICKEST*tall or length < LONGER_THAN_THICK*thick or thick < 2
            or a == 0 or b == wide):
        return None
    return Bar((reach[-1]+reach[1])//2, thick, tuple(int(c) for c in colour), int(a), int(b), longest=length)


class _StripDiscovery:
    """One owned capture, with a finite row allowance between incoming pictures."""

    def __init__(self, picture: np.ndarray) -> None:
        self.rows = picture.astype(np.int16)
        tall, wide = self.rows.shape[:2]
        self.same_as_left = np.zeros((tall, wide), dtype=bool)
        self.same_as_left[:, 1:] = np.abs(self.rows[:, 1:]-self.rows[:, :-1]).max(axis=2) <= ONE_COLOUR//2
        self.sums = np.zeros((tall, wide+1, self.rows.shape[2]), dtype=np.float64)
        np.cumsum(self.rows, axis=1, dtype=np.float64, out=self.sums[:, 1:])
        self.found: list[Bar] = []
        self.next_y, self.done = 1, tall <= 2

    def _row(self, y: int) -> None:
        wide = self.rows.shape[1]
        breaks = np.flatnonzero(~self.same_as_left[y])
        edges = np.append(breaks, wide)
        for a, b in _runs_across_dividers(self.rows[y], list(zip(edges[:-1], edges[1:], strict=True)), self.sums[y]):
            strip = _strip_at(self.rows, y, a, b)
            if strip is None:
                continue
            if any(abs(bar.row-y) <= bar.thick and abs(bar.start-a) <= 2 and abs(bar.end-b) <= 2 for bar in self.found):
                continue
            self.found.append(strip)

    def step(self) -> list[Bar]:
        before = len(self.found)
        for _ in range(DISCOVERY_ROWS_PER_PICTURE):
            if self.done:
                break
            self._row(self.next_y)
            self.next_y += 2
            self.done = self.next_y >= self.rows.shape[0]-1
        return self.found[before:]


def _strips(picture: np.ndarray) -> list[Bar]:
    """The complete ordinary scan, for callers that do not hold a live picture loop."""
    discovery = _StripDiscovery(picture)
    for _ in range(math.ceil(picture.shape[0]/(2*DISCOVERY_ROWS_PER_PICTURE))):
        discovery.step()
    return discovery.found


@invariant('perception.bar_discovery_keeps_time_for_the_next_picture', scope='perception',
           owner='core/perception/how_full_a_bar_is.py', observational=False)
def _bar_discovery_budget_invariant() -> tuple:
    picture = np.zeros((160, 240, 3), np.uint8)
    picture[30:38, 20:110] = (80, 140, 230)
    discovery = _StripDiscovery(picture)
    assert not discovery.step() and discovery.next_y == 1+2*DISCOVERY_ROWS_PER_PICTURE
    strip, = discovery.step()
    assert (strip.start, strip.end) == (20, 110)
    reader = BarsOnTheScreen()
    reader.read(picture, 1.)
    reader.read(np.zeros_like(picture), 1.1)
    assert not reader.strips
    return ()


def _runs_across_dividers(row: np.ndarray, runs: list[tuple[int, int]], sums: np.ndarray | None = None) -> list[tuple[int, int]]:
    """Each run of one colour along a row, and with them the runs made by joining runs of a colour across the narrow
    lines between them (DIVIDER): the cells of a bar drawn in cells, read as the bar. ``sums`` is the row's running
    sums, where they are kept, for the mean colour of a run."""

    def mean(a: int, b: int) -> np.ndarray:
        return (sums[b] - sums[a]) / (b - a) if sums is not None else row[a:b].mean(axis=0)

    out = list(runs)
    i = 0
    while i < len(runs):
        a, b = runs[i]
        colour = mean(a, b)
        j, end = i + 1, b
        while j < len(runs):
            gap_end = j
            while gap_end < len(runs) and runs[gap_end][0] - end <= DIVIDER and runs[gap_end][1] - runs[gap_end][0] <= DIVIDER:
                gap_end += 1
            if gap_end >= len(runs) or runs[gap_end][0] - end > DIVIDER:
                break
            na, nb = runs[gap_end]
            if np.abs(mean(na, nb) - colour).max() > ONE_COLOUR:
                break
            end, j = nb, gap_end + 1
        if end != b:
            out.append((a, end))
        i += 1
    return out


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
    return _along(line, seed, -1), _along(line, seed, 1) + 1


def _along(line: np.ndarray, x: int, step: int) -> int:
    """How far a bar's colour runs from ``x`` one way, across the narrow lines between its cells."""
    wide = len(line)
    while True:
        for jump in range(1, DIVIDER + 2):
            nx = x + step * jump
            if 0 <= nx < wide and line[nx]:
                x = nx
                break
        else:
            return x


def _the_picture_shook(runs: list[tuple[Bar, tuple[int, int] | None]]) -> bool:
    """Whether two strips or more went whole, each its own length, the same way in one picture: the picture moved."""
    shifts = [run[0] - bar.start for bar, run in runs
              if run is not None and abs(run[0] - bar.start) >= CHANGED_BY and abs(run[1] - bar.end) >= CHANGED_BY
              and abs((run[1] - run[0]) - (bar.end - bar.start)) < CHANGED_BY]
    return len(shifts) >= 2 and max(shifts) - min(shifts) <= CHANGED_BY


@dataclass
class BarsOnTheScreen:
    """Strips followed from picture to picture, and the ones that have shown they are bars."""

    strips: list[Bar] = field(default_factory=list)
    looked_at: float = -math.inf
    shape: tuple[int, int] = (0, 0)
    #: The one bar with no words that has been said to be what she has left: one is told, not every strip that runs down.
    said_unnamed: str = ""
    _discovery: Any = field(default=None, repr=False)
    _discovery_scope: Any = field(default=None, repr=False)
    _last_read_at: float = -math.inf

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
        if not math.isfinite(at):
            return []
        scope = self.shape, getattr(at, 'surface', None), getattr(at, 'epoch', None)
        if scope != self._discovery_scope or at < self._last_read_at:
            self._discovery, self.looked_at = None, -math.inf
            self._discovery_scope = scope
        self._last_read_at = float(at)
        if self._discovery is None and at-self.looked_at >= LOOK_FOR_STRIPS_S:
            self.looked_at = at
            self._discovery = _StripDiscovery(picture)
        if self._discovery is not None:
            candidates = self._discovery.step()
            if candidates:
                current_rows = picture.astype(np.int16)
                measured = []
                for candidate in candidates:
                    run = _run_of(picture, candidate)
                    current = _strip_at(current_rows, candidate.row, *run) if run is not None else None
                    if current is not None:
                        measured.append(current)
                self._take_in(measured)
            if self._discovery.done:
                self._discovery = None
        changes: list[dict[str, Any]] = []
        runs = [(bar, _run_of(picture, bar)) for bar in list(self.strips)]
        shaken = _the_picture_shook(runs)
        for bar, run in runs:
            if run is None:
                bar.missing_since = bar.missing_since or at
                if not bar.is_a_bar and at - bar.missing_since > MISSING_S:
                    self.strips.remove(bar)
                continue
            bar.missing_since = 0.0
            change = self._went(bar, run, at, shaken)
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
    def _went(bar: Bar, run: tuple[int, int], at: float, shaken: bool = False) -> tuple[float, float] | None:
        """Whether a strip's run changed at one end only, the other staying put, having held its level first: how full
        it was and is."""
        start, end = run
        moved_start, moved_end = abs(start - bar.start) >= CHANGED_BY, abs(end - bar.end) >= CHANGED_BY
        if not bar.steady_since:
            bar.steady_since = at
        if moved_start and moved_end:
            # Both ends went: the strip moved, or something crossed it. Not a bar's change; and where it went whole, the
            # same length, with others going the same way at once, the picture went with it (a screen that shakes as she
            # is hit), which says nothing against it. A strip that goes on its own is scenery: offline 2026-10-12 five
            # strips of a game's scenery, let off for going whole, were each said to be what she had left.
            if abs((end - start) - (bar.end - bar.start)) >= CHANGED_BY or not shaken:
                bar.unsteady += 1
            bar.start, bar.end, bar.steady_since = start, end, at
            return None
        if not moved_start and not moved_end:
            return None
        held, bar.steady_since = at - bar.steady_since >= STEADY_S, at
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

"""Watching a surface as one continuous stream, with her own acts on the same timeline.

Looking at a still picture, acting, and looking at another is how a board is
played, and it misses everything in between: a scene that plays out on its
own, a line that appears and is gone, an answer that came and went before the
second picture. LIVE-like 2026-10-05 an archived game opened on a scene of
speech bubbles appearing one after another; read one still at a time, every
bubble looked like a button, she clicked each in turn, and each click seemed to
work, because the scene went on by itself.

A person watches. What changes while they do nothing is the thing going on by
itself; what changes just after they act, where it was not already changing, is
the answer to what they did; and while a scene is playing they wait for it.
Here the surface's frames are sampled continuously, a few times a second, into
a timeline of how much changed and where; her acts are put on it as she makes
them; and from the two:

- ``answered``: whether an act was answered, beyond what the surface was doing anyway;
- ``playing_on_its_own``: whether something is going on by itself now (a scene, an animation that ends);
- ``keeps_changing_on_its_own``: the parts of the picture that keep changing while she does nothing,
  such as a scene's lines, which are read and not pressed;
- ``watch``: waiting while it plays, as long as it plays, up to a bound.

A part that changes all the time (a spinning coin, a ticking clock) is part of
the picture, not something going on: it is left out of all of these. Nothing
here knows what a game, a scene or a button is.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger("Aura.Watching")

__all__ = ["Watching"]

#: Pictures a second the stream is sampled at.
PER_SECOND = 8.0

#: How long the timeline reaches back, in seconds.
REMEMBERED_S = 60.0

#: The grid the picture is measured on, and the width it is shrunk to first.
CELLS = (16, 12)
LOOKED_AT_WIDE = 96

#: How different a pixel must be (of 255) to have changed, and how much of a
#: cell must change for the cell to have.
PIXEL_CHANGED = 18.0
CELL_CHANGED = 0.04

#: How long after an act a change is its answer.
ANSWERS_WITHIN_S = 2.0

#: A cell that changed in more than this share of the recent pictures changes
#: all the time, and is part of the picture.
ALWAYS = 0.6

#: How long a stretch with no change of its own ends a scene, and the longest
#: she watches one before acting anyway.
SCENE_ENDS_AFTER_S = 4.0
WATCH_AT_MOST_S = 30.0


@dataclass
class _Moment:
    at: float
    cells: np.ndarray  # bool, CELLS[1] x CELLS[0]: which cells changed since the picture before
    share: float  # of the picture that changed


@dataclass
class Watching:
    """The stream of one surface: what changed when and where, and when she acted."""

    take: Callable[[], Awaitable[Any]]
    per_second: float = PER_SECOND
    moments: deque[_Moment] = field(default_factory=lambda: deque(maxlen=int(REMEMBERED_S * PER_SECOND)))
    acts: deque[tuple[float, str]] = field(default_factory=lambda: deque(maxlen=200))
    _last: np.ndarray | None = None
    _task: asyncio.Task | None = None
    _watched_until: float = 0.0

    # ── the stream ───────────────────────────────────────────────────────

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.ensure_future(self._run())

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 - a stream stopped is stopped however it ended
                pass

    async def _run(self) -> None:
        period = 1.0 / max(0.5, self.per_second)
        while True:
            began = time.monotonic()
            try:
                taken = await self.take()
            except Exception as why:  # noqa: BLE001 - a surface that cannot be photographed now may be later
                logger.debug("watching: no picture (%s)", why)
                taken = None
            picture = taken[0] if isinstance(taken, tuple) else taken
            if picture is not None:
                self.saw(picture, time.monotonic())
            await asyncio.sleep(max(0.0, period - (time.monotonic() - began)))

    def saw(self, picture: Any, at: float) -> None:
        """One picture of the surface, at ``at``: what changed since the one before, cell by cell."""
        small = _small(picture)
        if small is None:
            return
        if self._last is not None and self._last.shape == small.shape:
            changed = np.abs(small - self._last) > PIXEL_CHANGED
            wide, high = CELLS
            rows = np.array_split(changed, high, axis=0)
            cells = np.array([[part.mean() > CELL_CHANGED for part in np.array_split(row, wide, axis=1)] for row in rows])
            self.moments.append(_Moment(at, cells, float(changed.mean())))
        self._last = small

    def acted(self, act: str, at: float | None = None) -> None:
        """She did ``act`` to the surface, now."""
        self.acts.append((time.monotonic() if at is None else at, str(act)))

    # ── reading it ───────────────────────────────────────────────────────

    def _always(self) -> np.ndarray:
        """The cells that change all the time: part of the picture, not something happening."""
        if not self.moments:
            return np.zeros((CELLS[1], CELLS[0]), dtype=bool)
        recent = list(self.moments)[-int(20 * self.per_second):]
        return np.mean([m.cells for m in recent], axis=0) > ALWAYS

    def _hers(self, at: float) -> bool:
        """Whether a change at ``at`` came soon enough after one of her acts to be its answer."""
        return any(0.0 <= at - when <= ANSWERS_WITHIN_S for when, _act in self.acts)

    def _of_its_own(self, since: float) -> list[_Moment]:
        always = self._always()
        return [m for m in self.moments if m.at >= since and not self._hers(m.at) and (m.cells & ~always).any()]

    def answered(self, act_at: float, within_s: float = ANSWERS_WITHIN_S) -> bool | None:
        """Whether something changed after the act at ``act_at`` that was not already changing; None before the stream has seen enough."""
        before = [m for m in self.moments if act_at - 3.0 <= m.at < act_at]
        after = [m for m in self.moments if act_at <= m.at <= act_at + within_s]
        if not after or not before:
            return None
        always = self._always()
        busy = np.any([m.cells for m in before], axis=0) | always
        # Changed where it was not already changing, or all at once (a new screen).
        return any((m.cells & ~busy).any() or m.share > 0.5 for m in after)

    def playing_on_its_own(self, now: float | None = None) -> bool:
        """Whether something is going on by itself: a change of its own in the last few seconds."""
        now = time.monotonic() if now is None else now
        return bool(self._of_its_own(now - SCENE_ENDS_AFTER_S))

    def keeps_changing_on_its_own(self, box: tuple[float, float, float, float], since_s: float = 20.0) -> bool:
        """Whether the part ``box`` (left, top, width, height as shares) has changed by itself more than once lately."""
        left, top, width, height = box
        wide, high = CELLS
        c0, c1 = int(left * wide), min(wide, max(int(left * wide) + 1, int(np.ceil((left + width) * wide))))
        r0, r1 = int(top * high), min(high, max(int(top * high) + 1, int(np.ceil((top + height) * high))))
        times = [m.at for m in self._of_its_own(time.monotonic() - since_s) if m.cells[r0:r1, c0:c1].any()]
        # Separate changes, not the frames of one: more than a second apart.
        separate = sum(1 for a, b in zip(times, times[1:], strict=False) if b - a > 1.0) + (1 if times else 0)
        return separate >= 2

    async def watch(self, *, at_most_s: float = WATCH_AT_MOST_S, tell: Callable[[str], None] | None = None) -> float:
        """Wait while something plays by itself, until it has been still a while or ``at_most_s`` has gone; the seconds watched.

        Once watched to its bound, the same going-on is not waited for again for
        a minute: what never stops is part of the picture, and she acts.
        """
        now = time.monotonic()
        if now < self._watched_until or not self.playing_on_its_own(now):
            return 0.0
        if tell is not None:
            tell("Something is playing out on its own; watching it before I do anything.")
        began = now
        # What is waited on is read off the stream, which sets nothing to wait on.
        while time.monotonic() - began < at_most_s and self.playing_on_its_own():  # noqa: ASYNC110
            await asyncio.sleep(0.25)
        watched = time.monotonic() - began
        if watched >= at_most_s:
            self._watched_until = time.monotonic() + 60.0
        logger.info("watched it play out on its own for %.1fs", watched)
        return watched

    def says(self) -> str:
        """What the stream shows, in a line."""
        now = time.monotonic()
        own = self._of_its_own(now - 10.0)
        return f"{len(self.moments)} pictures in the last {REMEMBERED_S:.0f}s; {len(own)} changes of its own in the last 10s"


def _small(picture: Any) -> np.ndarray | None:
    pixels = np.asarray(picture, dtype=np.float32)
    if pixels.ndim == 3:
        pixels = pixels[..., :3].mean(axis=2)
    if pixels.ndim != 2 or min(pixels.shape) < 8:
        return None
    step = max(1, int(round(pixels.shape[1] / LOOKED_AT_WIDE)))
    return pixels[::step, ::step]

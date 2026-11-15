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
import math
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.verify.invariants import invariant

logger = logging.getLogger("Aura.Watching")

__all__ = ["Watching"]

#: Pictures a second the stream is sampled at.
PER_SECOND = 8.0

#: How long the timeline reaches back, in seconds.
REMEMBERED_S = 60.0

#: How much of the picture a change covers to be a whole new screen rather than something going on in it, and how
#: many pictures something must go on for to have gone anywhere.
NEW_SCREEN_SHARE = 0.5
GOING_FOR = 3

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

#: A cell that changed after this share of her other recent acts, whatever they were, changes whatever she does: it is
#: no act's answer. Judged over the last few acts, of more than one kind.
AFTER_ANY_ACT = 0.7
OTHER_ACTS_SEEN = 8

#: Over how long, and in what share of its frames, a picture changing by
#: itself is a world in motion (a game being played) rather than a scene.
IN_MOTION_OVER_S = 2.0
IN_MOTION = 0.6

#: How long a stretch with no change of its own ends a scene, and the longest
#: she watches one before acting anyway.
SCENE_ENDS_AFTER_S = 4.0
WATCH_AT_MOST_S = 30.0


@dataclass
class _Moment:
    at: float
    since: float
    cells: np.ndarray  # bool, CELLS[1] x CELLS[0]: which cells changed since the picture before
    share: float  # of the picture that changed


@dataclass
class _ActWindow:
    sequence: int
    started: float
    act: str
    finished: float | None = None
    delivered: bool | None = None


@dataclass
class Watching:
    """The stream of one surface: what changed when and where, and when she acted."""

    take: Callable[[], Awaitable[Any]]
    per_second: float = PER_SECOND
    moments: deque[_Moment] = field(default_factory=lambda: deque(maxlen=int(REMEMBERED_S * PER_SECOND)))
    acts: deque[tuple[float, str]] = field(default_factory=lambda: deque(maxlen=200))
    _windows: deque[_ActWindow] = field(default_factory=lambda: deque(maxlen=200))
    _act_sequence: int = 0
    _last: np.ndarray | None = None
    _last_at: float | None = None
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
            except asyncio.CancelledError:
                pass  # not a failure: the cancel is what stopping asked for
            except Exception as exc:  # noqa: BLE001 - a stream stopped is stopped however it ended; how is said
                logger.info("The stream was stopped, and had ended with %s: %s", type(exc).__name__, exc)

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
                captured = taken[1] if isinstance(taken, tuple) and len(taken) > 1 else time.monotonic()
                # A time a picture was taken may come with how surely it is known (a frame's capture receipt,
                # core/perception/a_clock_for_drawn_frames.py): it is still a time. Taken only as a plain float, every
                # streamed frame was dropped, nothing her acts did could be told, and every one was left unjudged
                # (offline 2026-10-13: 1,276 of 1,280 acts in a pass).
                if isinstance(captured, (int, float)) and not isinstance(captured, bool) and math.isfinite(captured):
                    self.saw(picture, float(captured))
            await asyncio.sleep(max(0.0, period - (time.monotonic() - began)))

    def saw(self, picture: Any, at: float) -> None:
        """One picture of the surface, at ``at``: what changed since the one before, cell by cell."""
        small = _small(picture)
        if (small is None or not math.isfinite(at)
                or (self._last_at is not None and at <= self._last_at)):
            return
        if self._last is not None and self._last.shape == small.shape:
            changed = np.abs(small - self._last) > PIXEL_CHANGED
            if changed.ndim == 3:
                changed = changed.any(axis=2)
            wide, high = CELLS
            rows = np.array_split(changed, high, axis=0)
            cells = np.array([[part.mean() > CELL_CHANGED for part in np.array_split(row, wide, axis=1)] for row in rows])
            self.moments.append(_Moment(at, self._last_at, cells, float(changed.mean())))
        self._last = small
        self._last_at = at

    def begin_act(self, act: str, at: float | None = None) -> _ActWindow:
        """Register ownership before input can change the surface."""
        started = time.monotonic() if at is None else at
        self._act_sequence += 1
        window = _ActWindow(self._act_sequence, started, str(act))
        self._windows.append(window)
        self.acts.append((started, str(act)))
        return window

    def finish_act(self, window: _ActWindow, delivered: bool, at: float | None = None) -> None:
        """Close the input interval without giving uncertain delivery response credit."""
        finished = time.monotonic() if at is None else at
        if window.finished is not None or not math.isfinite(finished) or finished < window.started:
            raise ValueError("an action window needs one ordered completion")
        window.finished, window.delivered = finished, bool(delivered)

    def acted(self, act: str, at: float | None = None) -> None:
        """She did ``act`` to the surface, now."""
        window = self.begin_act(act, at)
        self.finish_act(window, True, window.started)

    # ── reading it ───────────────────────────────────────────────────────

    def _always(self) -> np.ndarray:
        """The cells that change all the time: part of the picture, not something happening."""
        if not self.moments:
            return np.zeros((CELLS[1], CELLS[0]), dtype=bool)
        recent = [m for m in list(self.moments)[-int(20 * self.per_second):] if not self._hers(m.at)]
        if not recent:
            return np.zeros((CELLS[1], CELLS[0]), dtype=bool)
        return np.mean([m.cells for m in recent], axis=0) > ALWAYS

    def _hers(self, at: float) -> bool:
        """Whether a change at ``at`` came soon enough after one of her acts to be its answer."""
        return any(w.started <= at <= (w.finished if w.finished is not None else at) + ANSWERS_WITHIN_S
                   for w in self._windows)

    def _of_its_own(self, since: float) -> list[_Moment]:
        always = self._always()
        return [m for m in self.moments if m.at >= since and not self._hers(m.at) and (m.cells & ~always).any()]

    def answered(self, act_at: float, within_s: float = ANSWERS_WITHIN_S) -> bool | None:
        """The response within the owned input interval and its bounded aftermath.

        Prior owned changes cannot become autonomous background. A missing or
        interrupted capture interval supplies no negative response evidence.
        """
        window = next((w for w in reversed(self._windows) if w.started == act_at), None)
        if window is None or window.finished is None or window.delivered is not True:
            return None
        ends = window.finished + max(0.0, within_s)
        following = next((w.started for w in self._windows if w.started > act_at), None)
        before = [m for m in self.moments if act_at - 3.0 <= m.at < act_at]
        after = [m for m in self.moments if act_at <= m.at <= ends
                 and (following is None or m.at < following)]
        gap = max(0.5, 4.0 / max(0.5, self.per_second))
        if len(before) < 2 or not after or act_at - before[-1].at > gap:
            return None
        if any(m.at - m.since > gap for m in before[-2:] + after):
            return None
        background = [m.cells for m in self.moments if act_at - 20.0 <= m.at < act_at and not self._hers(m.at)]
        if len(background) < 2:
            return None
        autonomous = [m.cells for m in before if not self._hers(m.at)]
        busy = np.any(autonomous, axis=0) if autonomous else np.zeros((CELLS[1], CELLS[0]), dtype=bool)
        busy |= np.mean(background, axis=0) > ALWAYS
        busy |= self._after_any_act(act_at, window.act)
        if any((m.cells & ~busy).any() for m in after):
            return True
        # A negative needs the entire interval; the first still frame is early,
        # and a later action would make the remaining comparison ambiguous.
        if following is not None and following < ends:
            return None
        covered = next((m for m in self.moments if m.at >= ends), None)
        if covered is None or covered.at - covered.since > gap or covered.at - ends > gap:
            return None
        return False

    def _after_any_act(self, act_at: float, act: str) -> np.ndarray:
        """The cells that changed after most of her other recent acts, whatever each was: a change that follows every
        act alike is caused by none of them. LIVE 2026-10-11 a building game's characters idled while she acted every
        few seconds, so no picture was free of her acts to show it, and each press of an arrow key, which the game does
        not listen for, was taken to have done something."""
        none = np.zeros((CELLS[1], CELLS[0]), dtype=bool)
        others = [w for w in list(self._windows)[-OTHER_ACTS_SEEN - 1:]
                  if w.started != act_at and w.act != act and w.finished is not None and w.delivered]
        if len(others) < 3 or len({w.act for w in others}) < 2:
            return none
        followed = []
        for other in others:
            ends = other.finished + ANSWERS_WITHIN_S
            after = [m.cells for m in self.moments if other.started <= m.at <= ends]
            if after:
                followed.append(np.any(after, axis=0))
        if len(followed) < 3:
            return none
        return np.mean(followed, axis=0) >= AFTER_ANY_ACT

    def answer_window(self) -> dict[str, Any] | None:
        """Detached timing and delivery evidence for the latest attempted input."""
        if not self._windows:
            return None
        window = self._windows[-1]
        return {"sequence": window.sequence, "act": window.act,
                "started": window.started, "finished": window.finished,
                "delivered": window.delivered, "captured_through": self._last_at,
                "response_until": None if window.finished is None else window.finished + ANSWERS_WITHIN_S}

    def playing_on_its_own(self, now: float | None = None) -> bool:
        """Whether a scene is playing by itself: changes of its own in the last few seconds, in bursts with stillness between.

        Not a game in motion: a ball that moves every frame is something to
        play, not a scene to wait out, and waiting would give the game its
        points (``in_motion``).
        """
        now = time.monotonic() if now is None else now
        return bool(self._of_its_own(now - SCENE_ENDS_AFTER_S)) and not self.in_motion(now)

    def in_motion(self, now: float | None = None) -> bool:
        """Whether the picture is changing in most of its frames of late: a world moving, not a scene stepping. A world
        set going by her own press moves as much as one that was going already: offline 2026-10-13, a ball set going
        by her click on PLAY moved in every frame, each counted as the click's answer and not as the world moving, and
        she watched it fall three times before doing anything, game after game."""
        now = time.monotonic() if now is None else now
        recent = [m for m in self.moments if m.at >= now - IN_MOTION_OVER_S]
        if len(recent) < 4:
            return False
        always = self._always()
        moving = sum(1 for m in recent if m.share < NEW_SCREEN_SHARE and (m.cells & ~always).any())
        return moving >= IN_MOTION * len(recent)

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
        try:
            # What is waited on is read off the stream, which sets nothing to wait on.
            while time.monotonic() - began < at_most_s and self.playing_on_its_own():  # noqa: ASYNC110
                await asyncio.sleep(0.25)
        except asyncio.CancelledError:
            # Cut off by whoever was waiting on her is watched as far as she can: LIVE 2026-10-08 a title that never
            # stops moving was waited on inside an eight-second look, three looks running, and the game was left.
            self._watched_until = time.monotonic() + 60.0
            raise
        watched = time.monotonic() - began
        if watched >= at_most_s:
            self._watched_until = time.monotonic() + 60.0
        logger.info("watched it play out on its own for %.1fs", watched)
        return watched

    def where_it_came_to_rest(self, since: float) -> tuple[float, float] | None:
        """Where what changed after ``since`` was changing last before it stopped: the middle of the last cells to
        change, as shares across and down, leaving out what changes all the time and a whole new screen. Where a
        machine she built was set going, that is where it stopped. None where nothing went on for a few pictures."""
        always = self._always()
        going = [m for m in self.moments if m.at > since and m.share < NEW_SCREEN_SHARE and (m.cells & ~always).any()]
        if len(going) < GOING_FOR:
            return None
        rows, columns = np.nonzero(going[-1].cells & ~always)
        return (float(columns.mean()) + 0.5) / CELLS[0], (float(rows.mean()) + 0.5) / CELLS[1]

    def says(self) -> str:
        """What the stream shows, in a line."""
        now = time.monotonic()
        own = self._of_its_own(now - 10.0)
        return f"{len(self.moments)} pictures in the last {REMEMBERED_S:.0f}s; {len(own)} changes of its own in the last 10s"


def _small(picture: Any) -> np.ndarray | None:
    pixels = np.asarray(picture, dtype=np.float32)
    if pixels.ndim == 3:
        pixels = pixels[..., :3]
    if (pixels.ndim not in (2, 3) or pixels.shape[0] < 2 * CELLS[1] or pixels.shape[1] < 2 * CELLS[0]
            or (pixels.ndim == 3 and not pixels.shape[2])):
        return None  # smaller than the grid it is measured on: nothing to measure
    step = max(1, int(round(pixels.shape[1] / LOOKED_AT_WIDE)))
    return pixels[::step, ::step]


def _response_window_keeps_its_evidence() -> bool:
    """A repeat response remains owned, while an early still remains unknown."""
    async def no_picture() -> None:
        return None

    watching = Watching(take=no_picture)
    still = np.full((48, 64), 50, dtype=np.uint8)
    changed = still.copy()
    changed[16:28, 16:32] = 240
    for n in range(17):
        watching.saw(still, 10 + n / 8)
    first = watching.begin_act("select one", 12.05)
    watching.saw(changed, 12.125)
    watching.finish_act(first, True, 12.15)
    for n in range(2, 16):
        watching.saw(changed, 12 + n / 8)
    second = watching.begin_act("select another", 13.95)
    watching.saw(still, 14.0)
    watching.finish_act(second, True, 14.05)
    repeat_owned = watching.answered(second.started) is True
    latest = watching.begin_act("no response yet", 14.075)
    watching.saw(still, 14.125)
    watching.finish_act(latest, True, 14.15)
    return repeat_owned and watching.answered(latest.started) is None


@invariant("perception.action_response_uses_owned_capture_windows", scope="perception",
           owner="core/perception/watching_it_happen.py", observational=False)
def _action_response_window_invariant() -> tuple:
    assert _response_window_keeps_its_evidence(), "an owned response became background or an early still became failure"
    return ()

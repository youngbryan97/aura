"""Playing something that does not wait for her, at the speed it goes.

Her other ways of playing look, think, act once and look again. That is right
for a board, which waits, and hopeless for a ball, which is past her paddle in
the time one look takes. Here the looking and the acting are one loop that
runs as fast as pictures arrive, twenty to fifty times a second, and the
judgement in it is arithmetic over what she has measured in this run:

- which thing is hers (core/agency/which_one_answers_to_her.py), and what each
  key does to it;
- where every other thing is going (core/perception/what_moves_in_the_picture.py);
- what comes of meeting each kind of thing, or letting it by
  (core/agency/what_meeting_things_does.py).

From those she picks a place to be: where the next thing worth meeting will
cross her path, in line with the next thing worth shooting, clear of the path
of anything that costs her. Each picture she holds the key that brings her
nearest to that place without walking into anything, and lets it go when she
is there. Nothing in it names a game, a key or a colour.

A stretch of play ends when the picture stops moving: a new screen that holds
still, a game over, a menu. What to do about that screen is a question for
her slower way of playing, which reads it.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.agency.what_meeting_things_does import AVOID, CLICK, IGNORE, MEET, SHOOT, WhatMeetingDoes
from core.agency.what_the_rules_said import WhatTheRulesSaid
from core.agency.which_one_answers_to_her import WhichIsHers
from core.perception.how_things_move_here import HowThingsMoveHere
from core.perception.what_changed_and_stayed import WhatChangedAndStayed
from core.perception.what_moves_in_the_picture import WhatMoves

logger = logging.getLogger("Aura.PlayingAsItHappens")

__all__ = ["controls_named_in", "play_as_it_happens", "the_world_moves_on_its_own"]

#: How soon after something last moved a picture must have been taken for its
#: words to count as seen during play.
MOVING_WHEN_READ_S = 0.3

#: How long each key is held while she finds out what it does.
TRY_A_KEY_S = 0.45

#: How far ahead she looks for anything she might walk into.
DANGER_AHEAD_S = (0.08, 0.16, 0.28, 0.42, 0.6)

#: How often the counters on the screen are read. Reading takes a tenth of a
#: second off to one side and is never waited for.
READ_EVERY_S = 0.5

#: Nothing moving for this long ends a stretch of play.
STILL_FOR_S = 3.0

#: After a new screen, nothing moving for this long ends it sooner.
NEW_SCREEN_STILL_S = 1.5

#: Spoken lines are at least this far apart, so each can be read.
SAY_EVERY_S = 5.0

#: The least time between two clicks, and how far ahead of a moving thing a
#: click is aimed: the click's trip through the browser and the next frame.
CLICK_EVERY_S = 0.25
CLICK_LANDS_S = 0.08

#: How fast a thing that follows the pointer can be taken somewhere, in working
#: pixels a second: the width of a picture in a few frames.
POINTER_SPEED = 2000.0

#: How far the pointer goes in one picture while she sweeps it, as a share of the picture.
POINTER_STEP = 0.04

#: Where the pointer is taken while she finds out whether anything follows it.
_POINTER_TRIAL = ((0.2, 0.5), (0.8, 0.5), (0.5, 0.2), (0.5, 0.8), (0.3, 0.3), (0.7, 0.7))


@dataclass
class _Run:
    keys: list[str]
    began: float
    held: str = ""
    held_since: float = 0.0
    held_trying: bool = False
    trying: int = 0
    tried_at: float = 0.0
    #: When she last found the thing she took for hers was not.
    lost_at: float = -math.inf
    last_moving: float = 0.0
    new_screen_at: float = -math.inf
    clicked: dict[int, float] = field(default_factory=dict)
    last_click: float = -math.inf
    pointer: tuple[float, float] = (0.5, 0.5)
    pointed: int = 0
    pointer_first: bool = False
    taps: int = 0
    said_at: float = -math.inf
    said: set[str] = field(default_factory=set)
    lines: list[str] = field(default_factory=list)
    reading: asyncio.Task | None = None
    read_at: float = -math.inf
    counted: dict[str, int] = field(default_factory=dict)
    reported_at: float = 0.0
    stayed: WhatChangedAndStayed = field(default_factory=WhatChangedAndStayed)
    #: Where on her thing she meets things, as a share of its length from the
    #: middle, and for each: how many meetings, and how many were followed by a
    #: gain for her.
    meeting_with: dict[float, list[int]] = field(default_factory=lambda: {-0.7: [0, 0], 0.0: [0, 0], 0.7: [0, 0]})
    meeting_now: float = 0.0
    told_checked: set[int] = field(default_factory=set)
    #: Every line of writing read on the screen during the stretch.
    words_seen: set[str] = field(default_factory=set)
    situation: str = ""
    situation_at: float = -math.inf
    met: list[tuple[float, float]] = field(default_factory=list)
    credited_up_to: int = 0
    touching: set[int] = field(default_factory=set)
    pictures: int = 0
    gains: int = 0
    losses: int = 0


# -- what she was told ---------------------------------------------------------

#: Words that say a game is played with the pointer.
_POINTER_WORDS = ("mouse", "cursor", "pointer", "click", "drag", "aim", "trackpad")

#: What the usual names of keys mean.
_NAMED_KEYS = (
    (("arrow", "arrows", "cursor keys", "direction"), ("up", "down", "left", "right")),
    (("space", "spacebar", "space bar"), ("space",)),
    (("up",), ("up",)),
    (("down",), ("down",)),
    (("left",), ("left",)),
    (("right",), ("right",)),
    (("enter", "return"), ("return",)),
    (("shift",), ("shift",)),
)


def controls_named_in(text: str, *, keys_without_words: Sequence[str] = ("up", "down", "left", "right", "space")) -> tuple[list[str], bool]:
    """The keys a game's own words name, and whether they name the pointer.

    "Use the arrow keys to move and space to jump" names five keys; "Move the
    mouse to aim, click to throw" names the pointer. With no keys named, the
    keys most games use are tried, after the pointer when it is named.
    """
    import re

    from core.runtime.watched_goal import keys_named_in

    lowered = " ".join(str(text or "").lower().split())
    words = set(re.findall(r"[a-z]+", lowered))
    keys: list[str] = []
    for names, meant in _NAMED_KEYS:
        if any((name in words) if " " not in name else (name in lowered) for name in names):
            keys.extend(key for key in meant if key not in keys)
    keys.extend(key for key in keys_named_in(lowered) if key not in keys)
    pointer = any(word in words or word + "s" in words for word in _POINTER_WORDS)
    if not keys:
        keys = list(keys_without_words)
    return keys, pointer


# -- naming what she sees -----------------------------------------------------

_COLOURS = {
    "black": (20, 20, 20), "white": (240, 240, 240), "grey": (128, 128, 128),
    "red": (210, 40, 40), "orange": (240, 140, 30), "yellow": (240, 210, 50),
    "green": (60, 180, 70), "blue": (50, 110, 230), "cyan": (60, 210, 230),
    "purple": (150, 60, 200), "pink": (240, 110, 200), "brown": (140, 90, 45),
}


def colour_name(rgb: Sequence[int]) -> str:
    return min(_COLOURS, key=lambda name: sum((a - b) ** 2 for a, b in zip(rgb, _COLOURS[name], strict=True)))


def shape_name(w: float, h: float) -> str:
    long, short = max(w, h), max(1.0, min(w, h))
    if long / short >= 2.5:
        return "bar"
    return "thing"


def describe(moves: WhatMoves, kind: int, thing: Any = None) -> str:
    alike = [thing] if thing is not None else [t for t in moves.things.values() if t.kind == kind]
    if not alike or kind >= len(moves.kinds):
        return "other"
    sample = alike[0]
    return f"{colour_name(moves.kinds[kind].colour)} {shape_name(sample.w, sample.h)}"


def where_on_screen(moves: WhatMoves, x: float, y: float) -> str:
    sx, sy = moves.share(x, y)
    across = "left" if sx < 0.33 else "right" if sx > 0.67 else ""
    down = "top" if sy < 0.33 else "bottom" if sy > 0.67 else ""
    return " ".join(part for part in (down, across) if part) or "middle"


# -- deciding -----------------------------------------------------------------


def _ahead(thing: Any, after: float, low: float, high: float, axis: int) -> float:
    """Where a thing will be along ``axis`` after ``after`` seconds, bouncing between low and high."""
    start = (thing.x, thing.y)[axis]
    speed = (thing.vx, thing.vy)[axis]
    span = max(1e-6, high - low)
    travelled = (start - low) + speed * after
    folded = travelled % (2 * span)
    return low + (folded if folded <= span else 2 * span - folded)


def _range_of(moves: WhatMoves, kind: int, axis: int) -> tuple[float, float]:
    """The span things of one kind have been seen over along an axis: where they bounce."""
    tall, wide = moves.shape
    low, high = 0.0, float((wide, tall)[axis])
    seen = [p[1 + axis] for t in moves.things.values() if t.kind == kind for p in t.path]
    if len(seen) >= 20:
        low, high = max(low, min(seen) - 1.0), min(high, max(seen) + 1.0)
    return low, high


def _boxes_meet(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _moved_box(box: tuple[float, float, float, float], dx: float, dy: float, grow: float = 0.0) -> tuple[float, float, float, float]:
    return (box[0] + dx - grow, box[1] + dy - grow, box[2] + dx + grow, box[3] + dy + grow)


class _Choosing:
    """The arithmetic of one decision, from what she has measured so far."""

    def __init__(self, moves: WhatMoves, hers: WhichIsHers, meeting: WhatMeetingDoes, keys: list[str],
                 physics: HowThingsMoveHere | None = None) -> None:
        self.moves, self.hers, self.meeting = moves, hers, meeting
        self.physics = physics
        self.mine = hers.thing(moves)
        self.ways = hers.keys_that_move_her(keys)
        self.across, self.updown = hers.follows_along if hers.follows_pointer else hers.axes(keys)
        self.pointing = hers.follows_pointer
        shots = hers.makes
        self.shot = next(iter(shots.values()), None)

    def line(self) -> int | None:
        """The axis she cannot move along, when she moves along only one."""
        if self.across and not self.updown:
            return 1
        if self.updown and not self.across:
            return 0
        return None

    def others(self) -> list[Any]:
        shot_kinds = {made.kind for made in self.hers.makes.values()}
        # Only her own thing is left out, and still copies of it (a row of
        # lives). Another thing that looks like hers may be the other player.
        return [
            t for t in self.moves.things.values()
            if self.mine is not None and t.number != self.mine.number
            and not (t.kind == self.hers.kind and not t.moved)
            and t.kind not in shot_kinds and t.number not in self.meeting.writing
        ]

    def stance(self, thing: Any) -> str:
        return self.meeting.stance(thing.kind, fixture=not thing.moved)

    def speed(self, axis: int) -> float:
        if self.pointing:
            # A thing under the pointer is wherever the pointer is put.
            return POINTER_SPEED if (self.across, self.updown)[axis] else 1.0
        return max([abs(way[axis]) for way in self.ways.values()] + [1.0])

    def target(self) -> tuple[tuple[float | None, float | None], str, Any]:
        """Where she wants to be, why, and for which thing."""
        line = self.line()
        if line is not None:
            return self._target_on_a_line(line)
        return self._target_anywhere()

    def _target_on_a_line(self, line: int) -> tuple[tuple[float | None, float | None], str, Any]:
        free = 1 - line
        mine = self.mine
        her_line, her_free = (mine.x, mine.y)[line], (mine.x, mine.y)[free]
        best = None
        for thing in self.others():
            if self.stance(thing) not in (MEET, CLICK) or not thing.moved:
                continue
            gap = her_line - (thing.x, thing.y)[line]
            closing = (thing.vx, thing.vy)[line]
            if gap * closing <= 0 or abs(closing) < 5.0:
                continue
            when = gap / closing
            low, high = _range_of(self.moves, thing.kind, free)
            there = _ahead(thing, when, low, high, free)
            # Her own physics of this world, once she has some: gravity, where
            # the walls really are, how much a bounce keeps.
            imagined = self.physics.when_it_reaches(thing, line, her_line) if self.physics is not None else None
            if imagined is not None:
                when, there = imagined
            reachable = abs(there - her_free) <= self.speed(free) * when + (mine.w, mine.h)[free] / 2
            rank = (0 if reachable else 1, when)
            if best is None or rank < best[0]:
                best = (rank, there, thing)
        if best is not None:
            goal = [None, None]
            goal[free] = best[1]
            return (goal[0], goal[1]), "meet", best[2]
        aimed = self._aim_a_shot(line)
        if aimed is not None:
            return aimed
        low, high = self.hers.lowest[free], self.hers.highest[free]
        goal = [None, None]
        goal[free] = (low + high) / 2 if high > low else her_free
        return (goal[0], goal[1]), "wait", None

    def _aim_a_shot(self, line: int | None) -> tuple[tuple[float | None, float | None], str, Any] | None:
        if self.shot is None:
            return None
        along = 1 if abs(self.shot.vy) >= abs(self.shot.vx) else 0
        across = 1 - along
        flight = abs((self.shot.vx, self.shot.vy)[along])
        mine = self.mine
        best = None
        for thing in self.others():
            if self.stance(thing) == IGNORE or self.stance(thing) == MEET and self.meeting.known(thing.kind):
                continue
            ahead = ((thing.x, thing.y)[along] - (mine.x, mine.y)[along]) * (self.shot.vx, self.shot.vy)[along]
            if ahead <= 0:
                continue
            when = abs((thing.x, thing.y)[along] - (mine.x, mine.y)[along]) / max(1.0, flight)
            low, high = _range_of(self.moves, thing.kind, across)
            there = _ahead(thing, when, low, high, across)
            if best is None or when < best[0]:
                best = (when, there, thing)
        if best is None:
            return None
        goal: list[float | None] = [None, None]
        goal[across] = best[1]
        return (goal[0], goal[1]), "shoot", best[2]

    def _target_anywhere(self) -> tuple[tuple[float | None, float | None], str, Any]:
        mine = self.mine
        speed = max(self.speed(0), self.speed(1))
        best = None
        for thing in self.others():
            stance = self.stance(thing)
            if stance not in (MEET, CLICK):
                continue
            when = math.hypot(thing.x - mine.x, thing.y - mine.y) / speed
            for _ in range(3):
                px, py = thing.where_at(when)
                when = math.hypot(px - mine.x, py - mine.y) / speed
            curious = not thing.moved and not self.meeting.known(thing.kind)
            rank = when * (2.0 if curious else 1.0)
            if best is None or rank < best[0]:
                best = (rank, thing.where_at(when), thing)
        if best is not None:
            return best[1], "meet", best[2]
        aimed = self._aim_a_shot(None)
        if aimed is not None:
            return aimed
        return (None, None), "wait", None

    def danger(self, way: tuple[float, float]) -> float:
        mine = self.mine
        threats = [t for t in self.others() if self.stance(t) in (AVOID,) or self.stance(t) == SHOOT]
        if not threats:
            return 0.0
        tall, wide = self.moves.shape
        low_x, high_x = mine.w / 2, wide - mine.w / 2
        low_y, high_y = mine.h / 2, tall - mine.h / 2
        total = 0.0
        for after in DANGER_AHEAD_S:
            x = min(max(mine.x + way[0] * after, low_x), high_x) if high_x >= low_x else mine.x
            y = min(max(mine.y + way[1] * after, low_y), high_y) if high_y >= low_y else mine.y
            box = _moved_box(mine.box(), x - mine.x, y - mine.y, 2.0)
            for thing in threats:
                tx, ty = thing.where_at(after)
                if _boxes_meet(box, _moved_box(thing.box(), tx - thing.x, ty - thing.y, 2.0)):
                    total += 1.0 / (after + 0.1)
        return total

    def key(self, held: str, *, offset: tuple[float, float] = (0.0, 0.0)) -> tuple[str, str, Any]:
        """The key to hold now ("" for none), why, and the thing it is for.

        ``offset`` moves where she wants to be by that much: meeting a thing
        with the end of her paddle instead of its middle.
        """
        (gx, gy), why, aim = self.target()
        if why == "meet":
            gx = gx - offset[0] if gx is not None else None
            gy = gy - offset[1] if gy is not None else None
        mine = self.mine
        choices = {"": (0.0, 0.0), **self.ways}
        best_key, best_cost = held if held in choices else "", math.inf
        for key, way in choices.items():
            x, y = mine.x + way[0] * 0.12, mine.y + way[1] * 0.12
            cost = 0.0
            if gx is not None:
                cost += abs(gx - x) / self.speed(0)
            if gy is not None:
                cost += abs(gy - y) / self.speed(1)
            cost += 3.0 * self.danger(way)
            cost += 0.0 if key == held else 0.01
            if cost < best_cost:
                best_key, best_cost = key, cost
        return best_key, why, aim

    def fire(self, aim: Any, why: str) -> str | None:
        """The key that fires, when what she is lined up on is worth a shot."""
        if self.shot is None or aim is None or why != "shoot":
            return None
        along = 1 if abs(self.shot.vy) >= abs(self.shot.vx) else 0
        across = 1 - along
        flight = abs((self.shot.vx, self.shot.vy)[along])
        when = abs((aim.x, aim.y)[along] - (self.mine.x, self.mine.y)[along]) / max(1.0, flight)
        low, high = _range_of(self.moves, aim.kind, across)
        there = _ahead(aim, when, low, high, across)
        if abs(there - (self.mine.x, self.mine.y)[across]) <= (aim.w, aim.h)[across] / 2 + 2:
            return self.shot.key
        return None


# -- the loop -----------------------------------------------------------------


async def the_world_moves_on_its_own(look: Callable[[], Awaitable[Any]], *, seconds: float = 0.8) -> bool:
    """Whether things move in the picture while she does nothing."""
    moves = WhatMoves()
    began = time.monotonic()
    while time.monotonic() - began < seconds + 0.6:
        seen = await look()
        if seen is None:
            return False
        picture, at = seen
        moves.see(picture, at)
        if moves.moving(faster_than=8.0) and moves.pictures > 4:
            return True
    return False


def _say(run: _Run, say: Callable[[str], Any] | None, line: str, at: float, *, once: str = "") -> None:
    if not line.strip(" ."):
        return
    if once and once in run.said:
        return
    if at - run.said_at < SAY_EVERY_S and not once:
        return
    if once:
        run.said.add(once)
    run.said_at = at
    run.lines.append(line)
    logger.info("saying while playing: %s", line)
    if say is not None:
        try:
            say(line)
        except (RuntimeError, ValueError, TypeError, OSError) as why:
            logger.debug("a line said while playing did not reach her chat: %s", why)


async def _hold(hands: Any, run: _Run, hers: WhichIsHers, key: str, at: float, *, trying: bool = False) -> None:
    if key == run.held and trying == run.held_trying:
        return
    if key != run.held:
        if run.held:
            await hands.up(run.held)
        if key:
            await hands.down(key)
            hers.tapped(key, at)
    run.held, run.held_since, run.held_trying = key, at, trying
    hers.holding(key, at, trying=trying)


async def _keep_reading(run: _Run, meeting: WhatMeetingDoes, hers: WhichIsHers, moves: WhatMoves,
                        picture: Any, at: float, read_words: Callable[[Any], list[dict[str, Any]]] | None) -> None:
    """Read the counters off to one side; take in the last reading when it is done."""
    if read_words is None:
        return
    if run.reading is not None and run.reading.done():
        regions, when = run.reading.result()
        run.reading = None
        # Words on screen while it was moving are the game's furniture. The
        # still screen at the end is not: LIVE 2026-10-04 she read "Press
        # SPACE to play again" in the seconds the end screen stood before her
        # play noticed nothing moved, so the restart that ended a won game
        # was taken for furniture, and she started another game.
        while_moving = when - run.last_moving < MOVING_WHEN_READ_S
        for region in regions if while_moving else ():
            said = " ".join(str(region.get("text") or "").lower().split())
            if said and len(run.words_seen) < 200:
                run.words_seen.add(said)
        mine = hers.thing(moves)
        her_x = moves.share(mine.x, mine.y)[0] if mine is not None else None
        for verdict in meeting.read(regions, when, her_x):
            if verdict["what"] == "gain":
                run.gains += 1
            else:
                run.losses += 1
        meeting.writing = _what_is_writing(moves, regions)
    if run.reading is None and at - run.read_at >= READ_EVERY_S:
        run.read_at = at
        bgr = picture[:, :, ::-1].copy()

        async def _read() -> tuple[list[dict[str, Any]], float]:
            return await asyncio.to_thread(read_words, bgr), at

        run.reading = asyncio.ensure_future(_read())


def _what_is_writing(moves: WhatMoves, regions: list[dict[str, Any]]) -> set[int]:
    """The things that are writing on the screen: words are for reading, not for touching."""
    tall, wide = moves.shape
    boxes = [
        (float(r["x"]) * wide, float(r["y"]) * tall, (float(r["x"]) + float(r["width"])) * wide, (float(r["y"]) + float(r["height"])) * tall)
        for r in regions
    ]
    return {t.number for t in moves.things.values() if not t.moved and any(_boxes_meet(t.box(), _moved_box(b, 0, 0, 1.0)) for b in boxes)}


async def _try_the_keys(hands: Any, run: _Run, hers: WhichIsHers, at: float) -> None:
    """Hold each key in turn, with a rest between, to see what answers."""
    if at - run.tried_at < TRY_A_KEY_S:
        return
    run.tried_at = at
    turn = run.trying % (2 * len(run.keys))
    run.trying += 1
    key = run.keys[turn // 2] if turn % 2 == 0 else ""
    await _hold(hands, run, hers, key, at, trying=True)


async def _click_things(hands: Any, run: _Run, moves: WhatMoves, meeting: WhatMeetingDoes, at: float) -> None:
    """With no thing of her own, a click is how she meets things.

    She clicks the thing that is about to leave first, among the kinds she
    has not learned to leave alone, where it will be when the click lands.
    """
    if at - run.last_click < CLICK_EVERY_S:
        return
    tall, wide = moves.shape
    candidates = [
        t for t in moves.things.values()
        if t.moved and t.number not in meeting.writing and meeting.stance(t.kind) in (MEET, CLICK)
        and not meeting.clicked_lately(t.number, at)
    ]
    if not candidates:
        return

    def leaving(thing: Any) -> tuple[int, float]:
        exits = [
            (edge - position) / speed
            for position, speed, edge in ((thing.x, thing.vx, wide if thing.vx > 0 else 0.0), (thing.y, thing.vy, tall if thing.vy > 0 else 0.0))
            if abs(speed) > 1.0
        ]
        return (0 if meeting.known(thing.kind) else 1, min(exits, default=math.inf))

    thing = min(candidates, key=leaving)
    x, y = thing.where_at(CLICK_LANDS_S)
    sx, sy = moves.share(x, y)
    if 0.0 <= sx <= 1.0 and 0.0 <= sy <= 1.0:
        await hands.click(sx, sy)
        meeting.clicked(thing, at)
        run.last_click = at


async def _try_the_pointer(hands: Any, run: _Run, hers: WhichIsHers, moves: WhatMoves, at: float) -> None:
    """Sweep the pointer through a few places, to see whether anything follows it.

    In small steps, a picture at a time: a thing that jumps with a jumping
    pointer cannot be followed from one picture to the next.
    """
    goal = _POINTER_TRIAL[(run.pointed // 2) % len(_POINTER_TRIAL)]
    x, y = run.pointer
    dx, dy = goal[0] - x, goal[1] - y
    far = math.hypot(dx, dy)
    if far <= POINTER_STEP:
        x, y = goal
        run.pointed += 1
    else:
        x, y = x + dx * POINTER_STEP / far, y + dy * POINTER_STEP / far
    run.pointer = (x, y)
    await hands.point(x, y)
    tall, wide = moves.shape
    hers.pointed(x * wide, y * tall, at)


async def _point_at(hands: Any, run: _Run, hers: WhichIsHers, moves: WhatMoves, choosing: _Choosing, at: float) -> None:
    """Where a thing follows the pointer, the pointer goes where she wants it to be."""
    (gx, gy), _why, _aim = choosing.target()
    mine = choosing.mine
    tall, wide = moves.shape
    x = gx if gx is not None else mine.x
    y = gy if gy is not None else mine.y
    sx, sy = min(1.0, max(0.0, x / max(1, wide))), min(1.0, max(0.0, y / max(1, tall)))
    # A move of the pointer costs a trip to the browser; a hundredth of the
    # picture is closer than her thing can be put anyway.
    if math.dist((sx, sy), run.pointer) > 0.02:
        run.pointer = (sx, sy)
        await hands.point(sx, sy)
        hers.pointed(x, y, at)


def _over(run: _Run, moves: WhatMoves, happened: list[dict[str, Any]], at: float) -> str:
    if any(h.get("what") == "new screen" for h in happened):
        run.new_screen_at = at
    if moves.moving(faster_than=8.0):
        run.last_moving = at
    still = at - run.last_moving
    if at - run.new_screen_at < STILL_FOR_S and still >= NEW_SCREEN_STILL_S and at - run.began > 2.0:
        return "the screen changed and nothing on it moves"
    if still >= STILL_FOR_S and at - run.began > STILL_FOR_S:
        return "nothing on the screen has moved for a while"
    return ""


#: How long she plays a moving picture in which nothing answers to her before
#: handing it back: a title screen that animates is not a game.
NOTHING_ANSWERS_S = 10.0


def _nothing_answers(run: _Run, hers: WhichIsHers, meeting: WhatMeetingDoes, at: float) -> str:
    if at - run.began < NOTHING_ANSWERS_S or hers.kind is not None or meeting.verdicts:
        return ""
    if any(kept.touched for kept in meeting.evidence.values()):
        return ""
    return "nothing here answers to me while it moves"


def _report(run: _Run, getting_somewhere: Callable[[str], Any] | None, at: float) -> None:
    """Tell whoever holds a deadline over this that it is still getting somewhere."""
    if getting_somewhere is None or at - run.reported_at < 5.0:
        return
    run.reported_at = at
    try:
        getting_somewhere(f"playing as it happens: {run.pictures} pictures, {run.gains} gains, {run.losses} losses")
    except (RuntimeError, ValueError, TypeError, OSError) as why:
        logger.debug("progress while playing was not taken: %s", why)


def _what_the_rules_said_of(rules: WhatTheRulesSaid | None, moves: WhatMoves, meeting: WhatMeetingDoes,
                            run: _Run, say: Any, at: float) -> None:
    """Tie the rules' words to the kinds on screen by colour, as each kind first appears."""
    if rules is None:
        return
    for kind in moves.kinds:
        if kind.number in meeting.told or kind.number in run.told_checked:
            continue
        run.told_checked.add(kind.number)
        stance = rules.stance_for_colour(colour_name(kind.colour))
        if stance is not None:
            meeting.told[kind.number] = stance
            line = {
                MEET: "the {c} ones are to be got", AVOID: "the {c} ones are to be kept clear of",
                SHOOT: "the {c} ones are to be shot", CLICK: "the {c} ones are to be clicked",
            }.get(stance, "")
            if line:
                _say(run, say, "The rules say " + line.format(c=colour_name(kind.colour)) + ".", at, once=f"told {kind.number}")


def _counters_without_reading(run: _Run, moves: WhatMoves, hers: WhichIsHers, meeting: WhatMeetingDoes, at: float) -> None:
    mine = hers.thing(moves)
    her_x = moves.share(mine.x, mine.y)[0] if mine is not None else None
    never = {hers.number} if hers.number is not None else set()
    for when, x, y in run.stayed.see(getattr(moves, "_last", None), moves.things, at, never=never):
        meeting.changed_in_place(when, x, y, her_x)


#: How long between two accounts of what kind of game this is.
SITUATION_EVERY_S = 30.0


def _what_kind_of_game(run: _Run, say: Any, moves: WhatMoves, hers: WhichIsHers, meeting: WhatMeetingDoes,
                       physics: HowThingsMoveHere, at: float) -> None:
    """Say what kind of game this is, once there is something to say, and again when it changes."""
    from core.agency.what_kind_of_game_this_is import in_a_sentence

    if "me" not in run.said or at - run.situation_at < SITUATION_EVERY_S:
        return
    sentence = in_a_sentence(moves, hers, meeting, physics, run.keys)
    if not sentence or ";" not in sentence or sentence == run.situation:
        return
    run.situation, run.situation_at = sentence, at
    _say(run, say, sentence[0].upper() + sentence[1:], at, once=f"situation {len(run.lines)}")


def _what_she_says(run: _Run, say: Any, moves: WhatMoves, hers: WhichIsHers, meeting: WhatMeetingDoes, at: float) -> None:
    mine = hers.thing(moves)
    keys = sorted(hers.keys_that_move_her(run.keys))
    settled = all(hers.tried(k) >= 4 for k in run.keys) or at - run.began > 8.0
    if mine is not None and hers.kind is not None and hers.follows_pointer:
        _say(run, say, f"That's me: the {describe(moves, hers.kind, mine)} at the {where_on_screen(moves, mine.x, mine.y)}. It goes where the mouse goes.", at, once="me")
    elif mine is not None and hers.kind is not None and keys and settled:
        how = " and ".join(keys)
        _say(run, say, f"That's me: the {describe(moves, hers.kind, mine)} at the {where_on_screen(moves, mine.x, mine.y)}. {how.capitalize()} move it.", at, once="me")
    for key in hers.makes:
        _say(run, say, f"{key.capitalize()} fires.", at, once=f"fires {key}")
    for kind in list(meeting.evidence):
        if not meeting.known(kind) or not any(t.kind == kind for t in moves.things.values()):
            continue
        name = describe(moves, kind)
        stance = meeting.stance(kind)
        line = {
            MEET: f"The {name}s are worth getting to.",
            AVOID: f"The {name}s cost me. Keeping clear of them.",
            SHOOT: f"The {name}s are worth shooting.",
        }.get(stance)
        if line:
            _say(run, say, line, at, once=f"{kind} {stance}")
    counters = {name: value for name, value in meeting.readouts.values.items() if not name.startswith("number")}
    if counters and counters != run.counted and at - run.said_at > 25.0:
        run.counted = dict(counters)
        _say(run, say, ", ".join(f"{name} {value}" for name, value in list(counters.items())[:3]).capitalize() + ".", at)


async def play_as_it_happens(
    look: Callable[[], Awaitable[Any]],
    hands: Any,
    *,
    keys: Sequence[str],
    seconds: float,
    say: Callable[[str], Any] | None = None,
    read_words: Callable[[Any], list[dict[str, Any]]] | None = None,
    keep: dict[str, Any] | None = None,
    pointer_first: bool = False,
    getting_somewhere: Callable[[str], Any] | None = None,
    told: str = "",
) -> dict[str, Any]:
    """Play what ``look`` shows through ``hands`` until it stops moving or ``seconds`` pass.

    ``look`` returns an RGB picture and when it was taken, or None. ``hands``
    has ``down(key)``, ``up(key)``, ``tap(key)``, ``click(x, y)`` and
    ``point(x, y)``, positions as shares of the picture. ``keep`` carries what
    she learned from one stretch of play into the next one in the same game.
    """
    began = time.monotonic()
    keep = keep if keep is not None else {}
    moves = WhatMoves(kinds=keep.get("kinds"))
    physics: HowThingsMoveHere = keep.get("physics") or HowThingsMoveHere()
    rules = WhatTheRulesSaid.read(told) if told else None
    hers: WhichIsHers = keep.get("hers") or WhichIsHers()
    meeting: WhatMeetingDoes = keep.get("meeting") or WhatMeetingDoes()
    for kept in (physics, hers, meeting):
        kept.numbered_afresh()
    run = _Run(keys=list(keys), began=began, last_moving=began, pointer_first=pointer_first)
    if keep.get("meeting_with"):
        run.meeting_with = {float(part): list(counts) for part, counts in keep["meeting_with"].items()}
    ended = ""
    try:
        while not ended:
            if time.monotonic() - began >= seconds:
                ended = "out of time"
                break
            seen = await look()
            if seen is None:
                ended = "the picture could not be taken"
                break
            picture, at = seen
            run.pictures += 1
            happened = moves.see(picture, at)
            hers.saw(moves, happened, at)
            physics.saw(moves, hers, happened, at)
            _what_the_rules_said_of(rules, moves, meeting, run, say, at)
            choosing = _Choosing(moves, hers, meeting, run.keys, physics)
            meeting.saw(moves, hers, happened, at, choosing.line() if choosing.mine is not None else None)
            _counters_without_reading(run, moves, hers, meeting, at)
            await _keep_reading(run, meeting, hers, moves, picture, at, read_words)
            await _act(hands, run, moves, hers, meeting, choosing, at)
            _what_she_says(run, say, moves, hers, meeting, at)
            _what_kind_of_game(run, say, moves, hers, meeting, physics, at)
            _report(run, getting_somewhere, at)
            ended = _over(run, moves, happened, at) or _nothing_answers(run, hers, meeting, at)
    finally:
        if run.held:
            try:
                await hands.up(run.held)
            except (RuntimeError, OSError, ValueError, TypeError, AttributeError) as why:
                logger.debug("letting go of %s failed: %s", run.held, why)
        if run.reading is not None:
            run.reading.cancel()
    keep.update({"hers": hers, "meeting": meeting, "kinds": moves.kinds, "physics": physics, "meeting_with": run.meeting_with})
    return _what_it_came_to(run, moves, hers, meeting, ended, began)


async def _act(hands: Any, run: _Run, moves: WhatMoves, hers: WhichIsHers, meeting: WhatMeetingDoes,
               choosing: _Choosing, at: float) -> None:
    # A thing taken for hers that did not answer her keys: try them again.
    if hers.lost_at > run.lost_at:
        run.lost_at, run.trying = hers.lost_at, 0
    # Every key is tried once before any is chosen: a plan that knows one key
    # can only go one way.
    trying_the_pointer = run.pointer_first and run.pointed < 2 * len(_POINTER_TRIAL) and not hers.follows_pointer
    if not trying_the_pointer and run.trying < 2 * len(run.keys) and not hers.keys_known(run.keys) and not hers.follows_pointer:
        await _try_the_keys(hands, run, hers, at)
        return
    if choosing.mine is not None and hers.follows_pointer:
        await _point_at(hands, run, hers, moves, choosing, at)
        return
    if choosing.mine is None or not choosing.ways:
        pointer_trial = run.pointed < 2 * len(_POINTER_TRIAL) and hasattr(hands, "point")
        if pointer_trial and run.pointer_first:
            await _try_the_pointer(hands, run, hers, moves, at)
            return
        if run.trying < 4 * len(run.keys):
            await _try_the_keys(hands, run, hers, at)
            return
        await _hold(hands, run, hers, "", at)
        if pointer_trial:
            await _try_the_pointer(hands, run, hers, moves, at)
            return
        if hasattr(hands, "click"):
            await _click_things(hands, run, moves, meeting, at)
        return
    untried = [k for k in run.keys if hers.tried(k) < 4 and k not in choosing.ways]
    if untried and at - run.tried_at > 3.0 and choosing.danger((0.0, 0.0)) == 0.0:
        run.tried_at = at
        await _hold(hands, run, hers, untried[0], at, trying=True)
        return
    if at - run.tried_at < TRY_A_KEY_S and run.held in untried:
        return
    _where_to_meet_things(run, choosing, meeting, at)
    imagined = _the_return_they_cannot_reach(choosing)
    part = imagined if imagined is not None else run.meeting_now
    shift = part * ((choosing.mine.h if choosing.line() == 0 else choosing.mine.w) / 2 if choosing.mine is not None else 0.0)
    offset = (0.0, shift) if choosing.line() == 0 else (shift, 0.0) if choosing.line() == 1 else (0.0, 0.0)
    key, why, aim = choosing.key(run.held, offset=offset)
    await _hold(hands, run, hers, key, at)
    fire = choosing.fire(aim, why)
    if fire and fire != run.held:
        await hands.tap(fire)
        hers.tapped(fire, at)
        run.taps += 1


#: The parts of her thing she imagines meeting a thing with, from one end
#: (-1) through the middle to the other end (1).
_PARTS = (-0.8, -0.5, -0.25, 0.0, 0.25, 0.5, 0.8)


def _the_return_they_cannot_reach(choosing: _Choosing) -> float | None:
    """Where on her thing to meet what is coming, chosen by running each return forward in her head.

    For each part of her thing: how the thing would leave it (from the
    meetings she has seen), where it would then cross the far side's line, and
    how far the other side's player would have to go in that time against how
    fast it has been seen to go. The part that leaves them furthest short
    wins. None until she has seen enough meetings, or when there is nothing
    coming or nobody on the other side.
    """
    physics, mine, line = choosing.physics, choosing.mine, choosing.line()
    if physics is None or mine is None or line is None:
        return None
    (_gx, _gy), why, coming = choosing.target()
    if why != "meet" or coming is None:
        return None
    free = 1 - line
    arrival = physics.when_it_reaches(coming, line, (mine.x, mine.y)[line])
    if arrival is None:
        return None
    them = _the_other_side(choosing)
    if them is None:
        return None
    reach = max(1.0, physics.fastest(them.kind) or 1.0)
    best = None
    for part in _PARTS:
        leaving = physics.after_meeting(coming.kind, part, (coming.vx, coming.vy), across=line == 0)
        if leaving is None:
            return None
        start = [0.0, 0.0]
        start[line] = (mine.x, mine.y)[line]
        start[free] = arrival[1]
        crossing = physics.when_it_reaches(
            coming, line, (them.x, them.y)[line], start=(start[0], start[1], leaving[0], leaving[1])
        )
        if crossing is None:
            continue
        short = abs(crossing[1] - (them.x, them.y)[free]) - reach * crossing[0]
        if best is None or short > best[0]:
            best = (short, part)
    return best[1] if best is not None else None


def _the_other_side(choosing: _Choosing) -> Any:
    """The thing shaped like hers on the other half of the picture: the other side's player."""
    mine, moves = choosing.mine, choosing.moves
    tall, wide = moves.shape
    middle = (wide, tall)[choosing.line() or 0] / 2
    for thing in moves.things.values():
        if thing.number == mine.number:
            continue
        alike = abs(math.log(max(1.0, thing.w) / max(1.0, mine.w))) < 0.4 and abs(math.log(max(1.0, thing.h) / max(1.0, mine.h))) < 0.4
        across = ((thing.x, thing.y)[choosing.line() or 0] < middle) != ((mine.x, mine.y)[choosing.line() or 0] < middle)
        if alike and across:
            return thing
    return None


#: How long after a meeting a gain for her is put down to it.
GAIN_AFTER_A_MEETING_S = 4.0


def _where_to_meet_things(run: _Run, choosing: _Choosing, meeting: WhatMeetingDoes, at: float) -> None:
    """Learn which part of her thing to meet things with, from the gains that follow.

    Where a thing that meets hers is sent depends on where it met: off the end
    of a paddle a ball goes away steeply, off the middle it comes back the way
    it came. Which of those wins points is a fact about the world she is in,
    so it is measured: each meeting is put down to the part she met with, a
    gain soon after is put down to that meeting, and the next part is drawn
    from what each has earned so far (Thompson sampling over three arms).
    """
    import random

    if choosing.mine is None or choosing.line() is None:
        return
    touching = {t.number for t in choosing.others() if t.moved and _boxes_meet(_moved_box(choosing.mine.box(), 0, 0, 2.0), t.box())}
    if touching - run.touching:
        run.meeting_with[run.meeting_now][0] += 1
        run.met.append((at, run.meeting_now))
        run.meeting_now = max(
            run.meeting_with,
            key=lambda part: random.betavariate(
                1 + run.meeting_with[part][1], 1 + max(0, run.meeting_with[part][0] - run.meeting_with[part][1])
            ),
        )
    run.touching = touching
    for verdict in meeting.verdicts[run.credited_up_to :]:
        if verdict["what"] == "gain":
            earned = [part for when, part in run.met if 0 <= verdict["since"] - when <= GAIN_AFTER_A_MEETING_S]
            if earned and run.meeting_with[earned[-1]][1] < run.meeting_with[earned[-1]][0]:
                run.meeting_with[earned[-1]][1] += 1
    run.credited_up_to = len(meeting.verdicts)


def _what_it_came_to(run: _Run, moves: WhatMoves, hers: WhichIsHers, meeting: WhatMeetingDoes,
                     ended: str, began: float) -> dict[str, Any]:
    took = max(1e-6, time.monotonic() - began)
    learned = {}
    for kind in meeting.evidence:
        if meeting.known(kind) and kind < len(moves.kinds):
            learned[colour_name(moves.kinds[kind].colour)] = meeting.stance(kind)
    return {
        "seconds": round(took, 1),
        "pictures": run.pictures,
        "pictures_a_second": round(run.pictures / took, 1),
        "ended": ended,
        "hers": describe(moves, hers.kind, hers.thing(moves)) if hers.kind is not None else "",
        "keys_that_move_her": sorted(hers.keys_that_move_her(run.keys)),
        "fires": sorted(hers.makes),
        "learned": learned,
        "gains": run.gains,
        "losses": run.losses,
        "counters": dict(meeting.readouts.values),
        "said": list(run.lines),
        "words_seen": sorted(run.words_seen),
    }

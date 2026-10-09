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
import re
import statistics
import time
from collections import Counter, deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.agency.how_the_contest_stands import ContestStands
from core.agency.pressing_what_is_shown import PRESSED_FOR_S
from core.agency.what_meeting_things_does import AVOID, CLICK, IGNORE, MEET, SHOOT, WhatMeetingDoes
from core.agency.what_the_rules_said import WhatTheRulesSaid
from core.agency.which_one_answers_to_her import RESPONSE_S, WAYS, WhichIsHers
from core.perception.how_things_move_here import HowThingsMoveHere
from core.perception.what_changed_and_stayed import WhatChangedAndStayed
from core.perception.what_moves_in_the_picture import TOLD_FROM_STANDING_PX, WhatMoves

logger = logging.getLogger("Aura.PlayingAsItHappens")

__all__ = ["controls_named_in", "it_goes_while_held", "play_as_it_happens", "the_world_moves_on_its_own"]

#: How long each key is held while she finds out what it does.
TRY_A_KEY_S = 0.45

#: How far ahead she looks for anything she might walk into.
DANGER_AHEAD_S = (0.08, 0.16, 0.28, 0.42, 0.6)

def _times_to(lasts: float) -> tuple[float, ...]:
    """The moments a press's path is looked at: the near ones, then a tenth of a second apart to its end."""
    later = []
    step = DANGER_AHEAD_S[-1] + 0.1
    while step <= lasts + 1e-9:
        later.append(round(step, 3))
        step += 0.1
    return (*DANGER_AHEAD_S, *later)


#: How near in time to a loss a thing must have met her for the loss to be its doing, in seconds: a counter is read
#: every half second and the reading takes a moment.
TOUCHED_WITHIN_S = 1.0

#: How much less a press's path must meet than holding her course does, for the press to be made: half, so a press
#: is made for what it clears and not for a near tie.
CLEARER = 0.5

#: How often the counters on the screen are read. Reading takes a tenth of a
#: second off to one side and is never waited for.
READ_EVERY_S = 0.5

#: Nothing moving for this long ends a stretch of play.
STILL_FOR_S = 3.0

#: After a new screen, nothing moving for this long ends it sooner.
NEW_SCREEN_STILL_S = 1.5

#: Where the world waits for her, nothing moving for this long ends a stretch: she has tried her ways, and none goes.
WAITING_STILL_FOR_S = 8.0

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

#: Revisit inconclusive control experiments while the world keeps moving.
RECHECK_CONTROLS_S = 5.0

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
    #: Keys the words say to press in turn, fast (core/agency/playing_as_it_happens.py `keys_pressed_fast`), and when last.
    burst_keys: list[str] = field(default_factory=list)
    burst_at: float = -math.inf
    last_trigger_began: float = -math.inf
    #: Where in the picture a click at nothing in particular has paid (core/agency/where_clicks_pay.py).
    clicks_pay: Any = None
    pointer: tuple[float, float] = (0.5, 0.5)
    pointed: int = 0
    pointer_first: bool = False
    #: Whether the visible instructions give the pointer a separate trigger.
    pointer_trigger: bool = False
    taps: int = 0
    #: What a legend on the place's screens drew and said of each thing (core/perception/what_a_legend_shows.py).
    legend: tuple[Any, ...] = ()
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
    told_checked: dict[int, bool] = field(default_factory=dict)
    #: Every line of writing read on the screen during the stretch.
    #: When something was first seen moving in this stretch.
    first_moving: float = math.inf
    #: Every reading's words, with when its picture was taken.
    words_read: list[tuple[float, str]] = field(default_factory=list)
    #: The bars on the screen that fill and empty (core/perception/how_full_a_bar_is.py), the last words read with where
    #: each stood, and how much she has left: the lowest of her bars that are worse lower, 1 where none is known.
    bars: Any = None
    regions_read: list[dict[str, Any]] = field(default_factory=list)
    vitals: float = 1.0
    situation: str = ""
    #: What she has said of the game, each part without where things stood.
    situation_known: set[str] = field(default_factory=set)
    hers_description: str = ""
    hers_descriptions: Counter[str] = field(default_factory=Counter)
    situation_at: float = -math.inf
    met: list[tuple[float, float]] = field(default_factory=list)
    credited_up_to: int = 0
    touching: set[int] = field(default_factory=set)
    pictures: int = 0
    observation_sources: Counter[str] = field(default_factory=Counter)
    input_key_downs: Counter[str] = field(default_factory=Counter)
    responsive_pictures: int = 0
    #: Each line said in this game, and when: carried from one stretch to the next.
    lately: dict = field(default_factory=dict)
    #: Where the rules ask for a place to be gone over: the cells her thing has been over, and their size (set once).
    covered: set = field(default_factory=set)
    cell: float = 0.0
    #: Cells she held a way toward and did not go: out of her reach, as far as she knows.
    barred: set = field(default_factory=set)
    #: Whether the world stood still until she moved in it: then stillness is hers to break.
    waits_for_her: bool = False
    #: The way held, where her thing was, and since when, while it stays put under it.
    stuck: tuple[str, float, float, float] | None = None
    #: The place she is making for, the nearest she has got to it, and since when.
    making_for: tuple[tuple[int, int], float, float] | None = None
    control_probe_retries: int = 0
    control_attribution: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=128))
    attribution_changes: int = 0
    last_attribution: tuple[Any, ...] | None = None
    picture_at: float | None = None
    intervals: deque[float] = field(default_factory=lambda: deque(maxlen=12))
    responses: deque[float] = field(default_factory=lambda: deque(maxlen=12))
    motion_responses: deque[float] = field(default_factory=lambda: deque(maxlen=12))
    previous_control: tuple[Any, ...] | None = None
    gains: int = 0
    losses: int = 0
    #: How the contest stands (core/agency/how_the_contest_stands.py), and the last of it she said.
    contest: ContestStands = field(default_factory=ContestStands)
    contest_said: str = ""
    #: How it stood, and the counters, when she came in: what they were then is not news (LIVE 2026-10-07 "I have 0.").
    contest_first: str | None = None
    counters_first: dict[str, Any] | None = None
    #: Keys the screen shows as pictures in play, and whether it asks for them now (core/agency/pressing_what_is_shown.py).
    shown: Any = None
    #: What each press does over the moment after it (core/agency/what_a_press_does.py), the things in the last
    #: picture, and presses made beside the key held, each with when to let it go.
    presses: Any = None
    things: dict = field(default_factory=dict)
    #: Where no body answers her keys: what pressing has paid by where the moving things were (when_a_press_pays.py).
    timing: Any = None
    #: How far each kind reaches: the distance and side at which it has cost her untouched (how_far_a_thing_reaches.py).
    reach: Any = None
    letting_go: dict[str, float] = field(default_factory=dict)


# -- what she was told ---------------------------------------------------------

def _record_control_attribution(run: _Run, moves: Any, hers: WhichIsHers, at: float) -> None:
    """Retain bounded ownership changes from visible geometry and measured controls."""
    mine = hers.thing(moves)
    keys = tuple(sorted(hers.keys_that_move_her(run.keys))) if mine is not None else ()
    source = "pointer" if mine is not None and hers.follows_pointer else "keys" if keys else "unknown"
    state = (mine.number if mine is not None else None, source, keys)
    if state == run.last_attribution:
        return
    run.last_attribution = state
    run.attribution_changes += 1
    shape = getattr(moves, "shape", None)
    position = ([round(mine.x / shape[1], 4), round(mine.y / shape[0], 4)]
                if mine is not None and shape and shape[0] > 0 and shape[1] > 0 else None)
    run.control_attribution.append({"seconds": round(at - run.began, 3), "thing": state[0],
                                    "source": source, "keys": list(keys), "position": position})

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


#: Words beside a way that say a key is meant by it.
_KEY_CUES = frozenset({"key", "keys", "arrow", "arrows", "press", "pressing", "hold", "holding", "tap", "hit", "push"})


def _a_key_is_meant(lowered: str, way: str) -> bool:
    import re

    words = re.findall(r"[a-z]+", lowered)
    return any(word == way and _KEY_CUES & set(words[max(0, at - 2):at + 3])
               for at, word in enumerate(words))


#: A single letter named as a key: "the X key", "X key"; or a capital after a word for pressing, or before what it is
#: for: "press Z", "Z to shoot". Read in the case it was written: "press a button" names no key called A.
_A_LETTER_KEY = re.compile(
    r"\b(?:the\s+)?([A-Za-z])\s+key\b"
    r"|\b(?:press|hit|tap|hold|use|push)\s+(?:the\s+)?[\"'“]?([A-Z])[\"'”]?(?![\w'])(?:\s+(?:key\s+)?to\s+([a-z]+))?"
    r"|(?<![\w'])([A-Z])\s+to\s+([a-z]+)\b")

#: What a key pressed only to begin or begin again is for: the menu's, not play's.
_TO_BEGIN = frozenset({"start", "begin", "restart", "continue", "play", "pause", "quit"})


def _letter_keys(text: str, *, during_play: bool = False) -> list[str]:
    """Letter keys a game's own words name (LIVE 2026-10-09 "grenades (activated with the X key)" was not read as one)."""
    keys: list[str] = []
    for match in _A_LETTER_KEY.finditer(text):
        letter = match.group(1) or match.group(2) or match.group(4)
        purpose = (match.group(3) or match.group(5) or "").lower()
        if not letter or (match.group(4) and letter in "IA"):
            continue
        if during_play and purpose in _TO_BEGIN:
            continue
        if letter.lower() not in keys:
            keys.append(letter.lower())
    return keys


#: Words that say keys are pressed in turn and fast: "press left and right rapidly", "tap space repeatedly".
_FAST = re.compile(r"\b(?:rapidly|repeatedly|quickly|as fast as|mash\w*|alternat\w*|in turn|over and over|again and again)\b",
                   re.I)

#: How long a burst of presses lasts, how often one may come, and the time between presses in it, in seconds.
BURST_S = 1.5
BURST_EVERY_S = 3.0
BURST_TAP_S = 0.06


def keys_pressed_fast(text: str) -> list[str]:
    """The keys a sentence says to press in turn and fast, one or two of them (LIVE 2026-10-09 a game wanted left and
    right pressed in turn, fast, when she was caught, and she pressed each a second at a time)."""
    keys: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", str(text or "")):
        if not _FAST.search(sentence):
            continue
        named, _pointer = controls_named_in(sentence, keys_without_words=(), during_play=True)
        if 1 <= len(named) <= 2:
            keys += [key for key in named if key not in keys]
    return keys[:2]


async def _burst(hands: Any, run: _Run, at: float, *, say: Any = None) -> None:
    """The keys pressed in turn, as fast as a person can, for a moment."""
    if run.held:
        await hands.up(run.held)
        run.held = ""
    _say(run, say, f"Pressing {' and '.join(run.burst_keys)} in turn, fast, as I read I should.", at, once="burst")
    began, n = time.monotonic(), 0
    while time.monotonic() - began < BURST_S:
        await hands.tap(run.burst_keys[n % len(run.burst_keys)])
        n += 1
        await asyncio.sleep(BURST_TAP_S)
    run.burst_at = at


def controls_named_in(text: str, *, keys_without_words: Sequence[str] = ("up", "down", "left", "right", "space"),
                      during_play: bool = False) -> tuple[list[str], bool]:
    """The keys a game's own words name, and whether they name the pointer.

    "Use the arrow keys to move and space to jump" names five keys; "Move the
    mouse to aim, click to throw" names the pointer. With no keys named, the
    keys most games use are tried, after the pointer when it is named.
    """
    import re

    from core.runtime.watched_goal import keys_named_in

    lowered = " ".join(str(text or "").lower().split())
    if during_play:
        # A lifecycle command belongs to the menu, rather than to the active
        # controls. Keep other uses of the same key, such as space to jump.
        lowered = re.sub(r"\b(?:press|tap|hit)\s+[^.!?]{0,40}?\b(?:to\s+)?"
                         r"(?:start|begin|restart|play\s+again)\b", "", lowered)
    words = set(re.findall(r"[a-z]+", lowered))
    # Restrict generic arrow instructions only when directions qualify the
    # arrows themselves. "Pick up coins" says nothing about which arrows work.
    arrow_directions = re.findall(
        r"\b((?:(?:up|down|left|right)\s*(?:[,/&+-]|\band\b|\bor\b)?\s*)+)"
        r"(?:arrows?\b|arrow\s+keys?\b|cursor\s+keys?\b)", lowered)
    keys: list[str] = []
    for names, meant in _NAMED_KEYS:
        if meant == ("up", "down", "left", "right") and arrow_directions:
            continue
        if meant[0] in WAYS and len(meant) == 1:
            # A way said alone is a key only beside a word for keys or pressing: LIVE 2026-10-07 "when the hamster
            # lines up with the pillow" was read as the up key, and "left-click fires" as the left one.
            qualified = meant[0] in re.findall(r"up|down|left|right", " ".join(arrow_directions))
            if qualified or re.search(rf"\b{meant[0]}\b(?!-?\s?click)", lowered) and _a_key_is_meant(lowered, meant[0]):
                keys.extend(key for key in meant if key not in keys)
            continue
        if any((name in words) if " " not in name else (name in lowered) for name in names):
            keys.extend(key for key in meant if key not in keys)
    keys.extend(key for key in keys_named_in(lowered) if key not in keys)
    keys.extend(key for key in _letter_keys(str(text or ""), during_play=during_play) if key not in keys)
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


def _range_of(moves: WhatMoves, kind: int, axis: int, physics: HowThingsMoveHere | None = None) -> tuple[float, float]:
    """The picture's bounds, replaced by walls whose rebounds have been measured.

    A short path's minimum and maximum are places already visited, and give
    no evidence that the next step will turn there.
    """
    tall, wide = moves.shape
    low, high = 0.0, float((wide, tall)[axis])
    if physics is not None:
        names = ("left", "right") if axis == 0 else ("top", "bottom")
        lower, upper = (physics.edge(kind, name) for name in names)
        if lower[0] == "bounce" and lower[1] is not None:
            low = lower[1]
        if upper[0] == "bounce" and upper[1] is not None:
            high = upper[1]
    return low, high


def goes_with(thing: Any, mine: Any) -> bool:
    """Whether a thing beside hers has gone where hers went, picture for picture: its shadow, what it carries.

    LIVE 2026-10-07 a hero's shadow, a thing of its own to her eyes, was
    always just below it; she made for it, and held the arrow down against the
    bottom wall. Over their last pictures together the two keep one distance
    apart while hers goes somewhere.
    """
    size = max(float(mine.w), float(mine.h), 1.0)
    if math.dist((thing.x, thing.y), (mine.x, mine.y)) > 2.0 * size:
        return False
    theirs = {at: (x, y) for at, x, y in list(thing.path)[-8:]}
    together = [(at, x, y) for at, x, y in list(mine.path)[-8:] if at in theirs]
    if len(together) < 4:
        return False
    apart = [(x - theirs[at][0], y - theirs[at][1]) for at, x, y in together]
    went = math.dist(together[0][1:], together[-1][1:])
    return went > 0.5 * size and max(math.dist(gap, apart[0]) for gap in apart) < 0.25 * went + 2.0


def _boxes_meet(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _moved_box(box: tuple[float, float, float, float], dx: float, dy: float, grow: float = 0.0) -> tuple[float, float, float, float]:
    return (box[0] + dx - grow, box[1] + dy - grow, box[2] + dx + grow, box[3] + dy + grow)


def _contact_line(mine: Any, coming: Any, axis: int) -> float:
    """The incoming thing's centre when the two visible bodies first touch."""
    here, there = (mine.x, mine.y)[axis], (coming.x, coming.y)[axis]
    reach = ((mine.w, mine.h)[axis] + (coming.w, coming.h)[axis]) / 2
    return here + math.copysign(reach, there - here)


def _measure_response(run: _Run, mine: Any, at: float) -> None:
    """Fit the old command's remaining motion at a visible command transition."""
    previous, run.previous_control = run.previous_control, None
    if previous is None or mine is None:
        return
    number, x, y, then, before, after = previous
    dt = at - then
    if mine.number != number or not 0.0 < dt <= 0.25:
        return
    change = (before[0] - after[0], before[1] - after[1])
    norm = change[0] ** 2 + change[1] ** 2
    if norm < 20.0 ** 2:
        return
    error = (mine.x - x - after[0] * dt, mine.y - y - after[1] * dt)
    delay = (error[0] * change[0] + error[1] * change[1]) / norm
    # A result outside this interval does not identify this transition: a
    # boundary, a collision or another motion changed what the key could do.
    if -0.1 * dt <= delay <= 1.1 * dt:
        run.motion_responses.append(min(dt, max(0.0, delay)))


class _Choosing:
    """The arithmetic of one decision, from what she has measured so far."""

    #: How much wider a berth what costs her is given than usual: more as what she has left runs low.
    caution: float = 1.0

    def __init__(self, moves: WhatMoves, hers: WhichIsHers, meeting: WhatMeetingDoes, keys: list[str],
                 physics: HowThingsMoveHere | None = None, *, next_picture_s: float = 1 / 30,
                 response_s: float = 0.0, covered: tuple[set[tuple[int, int]], float] | None = None,
                 barred: tuple[set[tuple[int, int]], float] | None = None, presses: Any = None,
                 reach: Any = None) -> None:
        self.moves, self.hers, self.meeting = moves, hers, meeting
        #: Where the rules ask for a place to be gone over, the cells of it her thing has been over; else None.
        self.covered = covered
        #: The cells she has found out of her reach, and their size.
        self.barred = barred
        self.physics = physics
        self.next_picture_s = next_picture_s
        self.response_s = response_s
        self.mine = hers.thing(moves)
        self.ways = hers.keys_that_move_her(keys)
        self.reach = reach
        # A key that lifts her and lets her fall is pressed for what its press does, not held as a way to go.
        self.lifts = (presses.lifts(self.mine.kind, max(float(self.mine.w), float(self.mine.h)), self.mine.number)
                      if presses is not None and self.mine is not None else [])
        for lift in self.lifts:
            self.ways.pop(lift, None)
        self.across, self.updown = hers.follows_along if hers.follows_pointer else hers.axes(keys)
        self.pointing = hers.follows_pointer
        shots = hers.makes
        self.shot = next(iter(shots.values()), None)
        # Low on what she has left, what costs her is given a wider berth.
        self.caution = 1.0 + 2.0 * max(0.0, CAREFUL_BELOW - getattr(meeting, "vitals", 1.0)) / CAREFUL_BELOW

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
        # lives), and what goes wherever it goes (its shadow, what it holds).
        # Another thing that looks like hers may be the other player.
        return [
            t for t in self.moves.things.values()
            if self.mine is not None and t.number != self.mine.number
            and not (t.kind == self.hers.kind and not t.moved)
            and t.kind not in shot_kinds and t.number not in self.meeting.writing
            and not goes_with(t, self.mine)
        ]

    def stance(self, thing: Any) -> str:
        # A kind that has cost her from a distance is kept clear of, whatever meeting it has done.
        if self.reach is not None and self.reach.reach(int(thing.kind)) is not None:
            return AVOID
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
            contact = _contact_line(mine, thing, line)
            when = max(0.0, (contact - (thing.x, thing.y)[line]) / closing)
            low, high = _range_of(self.moves, thing.kind, free, self.physics)
            there = _ahead(thing, when, low, high, free)
            # Her own physics of this world, once she has some: gravity, where
            # the walls really are, how much a bounce keeps.
            imagined = self.physics.when_it_reaches(thing, line, contact) if self.physics is not None else None
            if imagined is not None:
                when, there = imagined
            # The next arrival is urgent even when she cannot get there in
            # time. A distant, reachable arrival must not pull her away from it.
            rank = when
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
            low, high = _range_of(self.moves, thing.kind, across, self.physics)
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
            # A still thing she is on already is not somewhere to go: what being on it does, it is doing.
            if stance not in (MEET, CLICK) or self._out_of_reach(thing) or not thing.moved and _boxes_meet(mine.box(), thing.box()):
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
        ground = self._ground_not_yet_covered()
        if ground is not None:
            return ground, "cover", None
        return (None, None), "wait", None

    def _out_of_reach(self, thing: Any) -> bool:
        """A thing standing where she has found she cannot go: behind a wall, up on the scenery."""
        if self.barred is None or thing.moved:
            return False
        cells, cell = self.barred
        return (int(thing.x // cell), int(thing.y // cell)) in cells

    def _ground_not_yet_covered(self) -> tuple[float, float] | None:
        """Where the rules ask for a place to be gone over, the middle of the nearest part of it her thing has not been over.

        Waiting is right for a paddle and wrong where the ground is the point:
        LIVE 2026-10-07 a painter told to give "the entire campground a fresh
        coat of paint" stood waiting for something to go to, and its paint ran
        out where it stood. Measured in cells of her own thing's size, over
        the picture; ties go to the way she is already going, so she sweeps.
        """
        mine = self.mine
        if self.covered is None or mine is None:
            return None
        covered, cell = self.covered
        tall, wide = self.moves.shape
        columns, rows = max(1, int(wide // cell)), max(1, int(tall // cell))
        heading = (self.hers.last_velocity[0], self.hers.last_velocity[1])
        best = None
        for row in range(rows):
            for column in range(columns):
                if (column, row) in covered:
                    continue
                x, y = (column + 0.5) * cell, (row + 0.5) * cell
                far = math.hypot(x - mine.x, y - mine.y)
                ahead = (x - mine.x) * heading[0] + (y - mine.y) * heading[1]
                rank = far - (0.25 * cell if ahead > 0 else 0.0)
                if best is None or rank < best[0]:
                    best = (rank, x, y)
        return (best[1], best[2]) if best is not None else None

    def danger(self, way: tuple[float, float]) -> float:
        return self.danger_along(lambda after: (way[0] * after, way[1] * after))

    def a_press_that_clears(self, presses: Any, held: str) -> str | None:
        """A key whose press takes her clear of what is coming where holding ``held`` on runs into it; else None."""
        way = self.ways.get(held, (0.0, 0.0))
        staying = self.danger(way)
        if staying <= 0.0 or presses is None:
            return None
        best: tuple[float, str] | None = None
        for key in self.lifts:
            path = presses.path(key, self.mine.kind, self.mine.number)
            if key == held or path is None:
                continue
            # Judged over all of what the press does, not only the next moment: a jump that clears what is near and
            # lands on it is no way clear (offline 2026-10-09: jumps made 0.7 s early landed on the block each time).
            lasts = max(DANGER_AHEAD_S[-1], presses.lasts(key, self.mine.kind, self.mine.number))
            risk = self.danger_along(lambda after, p=path: (way[0] * after + p(after)[0], way[1] * after + p(after)[1]),
                                     times=_times_to(lasts))
            if risk < CLEARER * staying and (best is None or risk < best[0]):
                best = (risk, key)
        return best[1] if best is not None else None

    def danger_along(self, path: Callable[[float], tuple[float, float]], times: Sequence[float] = DANGER_AHEAD_S) -> float:
        """How soon and how often her thing, taken along ``path`` (seconds to displacement), meets what to keep clear of,
        looked at ``times`` seconds ahead."""
        mine = self.mine
        threats = [t for t in self.others() if self.stance(t) in (AVOID,) or self.stance(t) == SHOOT]
        if not threats:
            return 0.0
        tall, wide = self.moves.shape
        low_x, high_x = mine.w / 2, wide - mine.w / 2
        low_y, high_y = mine.h / 2, tall - mine.h / 2
        total = 0.0
        for after in times:
            dx, dy = path(after)
            x = min(max(mine.x + dx, low_x), high_x) if high_x >= low_x else mine.x
            y = min(max(mine.y + dy, low_y), high_y) if high_y >= low_y else mine.y
            box = _moved_box(mine.box(), x - mine.x, y - mine.y, 2.0)
            for thing in threats:
                tx, ty = thing.where_at(after)
                if (_boxes_meet(box, _moved_box(thing.box(), tx - thing.x, ty - thing.y, 2.0))
                        or (self.reach is not None and self.reach.within(thing, x, y, (tx, ty)))):
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
        moving = choices.get(held, (0.0, 0.0))
        # The old command keeps moving her while the next one is delivered.
        # Its measured delay belongs in the prediction before the new command.
        here_x, here_y = mine.x + moving[0] * self.response_s, mine.y + moving[1] * self.response_s
        best_key, best_cost = held if held in choices else "", math.inf
        for key, way in choices.items():
            x, y = here_x + way[0] * self.next_picture_s, here_y + way[1] * self.next_picture_s
            cost = 0.0
            if gx is not None:
                cost += abs(gx - x) / self.speed(0)
            if gy is not None:
                cost += abs(gy - y) / self.speed(1)
            cost += 3.0 * self.caution * self.danger(way)
            cost += 0.0 if key == held else self.next_picture_s * 0.1
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
        low, high = _range_of(self.moves, aim.kind, across, self.physics)
        there = _ahead(aim, when, low, high, across)
        if abs(there - (self.mine.x, self.mine.y)[across]) <= (aim.w, aim.h)[across] / 2 + 2:
            return self.shot.key
        return None


# -- the loop -----------------------------------------------------------------


async def the_world_moves_on_its_own(look: Callable[[], Awaitable[Any]], *, seconds: float = 0.8, longest: float = 3.0) -> bool:
    """Whether things go somewhere in the picture while she does nothing.

    Moving in place is not going anywhere. LIVE 2026-10-06 a checkers game's
    figures swayed beside a board that waited for her move, and she played it
    with the arrow keys for five minutes as if it were an action game. A thing
    goes somewhere when it travels further than half its own size, mostly one
    way, or faster than two of its sizes a second; where things move and none
    has gone anywhere yet, she watches a while longer before saying the world
    waits for her. Measured on the pictures' own clock.
    """
    moves = WhatMoves()
    began: float | None = None
    while True:
        seen = await look()
        if seen is None:
            return False
        picture, at = seen
        began = at if began is None else began
        moves.see(picture, at)
        if moves.pictures <= 4:
            continue
        moving = moves.moving(faster_than=8.0)
        if any(_goes_somewhere(thing) for thing in moving):
            return True
        if at - began >= (longest if moving else seconds) + 0.6:
            return False


#: How long she holds a key down to see what it does, and how long after it goes down the jump at the press is over.
HELD_TO_SEE_S = 0.45
THE_PRESS_S = 0.1


async def it_goes_while_held(look: Callable[[], Awaitable[Any]], hands: Any, key: str, *, held: float = HELD_TO_SEE_S) -> bool:
    """``key`` pressed the way a person presses one to see what it does, held a moment while she watches: whether something goes on going while it is down.

    A world that stands still until she moves in it is played as it happens all
    the same: LIVE 2026-10-07 a top-down shooter's room held still, it was
    taken for a screen to be stepped through, and each tap of an arrow moved
    its hero a few pixels, which she took for nothing; for three hours she
    clicked the words on its scoreboard. A menu's highlight jumps once as the
    key goes down and is still for the rest of the hold; a thing she steers
    keeps going for as long as the key is down. So what moved in the first
    tenth of a second is not counted, and a thing goes on going when it is seen
    in three pictures after that and has travelled half its own size between
    them. She looks before she presses, so what is there already is known.
    Measured on the pictures' own clock. The key goes down once, so to whatever
    counts presses it is one press.
    """
    moves = WhatMoves()
    began: float | None = None
    while not moves.has_looked:
        seen = await look()
        if seen is None:
            return False
        began = seen[1] if began is None else began
        moves.see(*seen)
        if seen[1] - began > 3 * held:
            break
    await hands.down(key)
    first: float | None = None
    try:
        while True:
            seen = await look()
            if seen is None:
                return False
            picture, at = seen
            moves.see(picture, at)
            first = at if first is None else first
            if at - first >= held:
                break
    finally:
        await hands.up(key)
    since = first + THE_PRESS_S
    return any(_went_on(thing, since) for thing in moves.things.values())


def _went_on(thing: Any, since: float) -> bool:
    points = [(at, x, y) for at, x, y in thing.path if at >= since]
    if len(points) < 3:
        return False
    size = max(float(thing.w), float(thing.h), 1.0)
    return math.dist(points[0][1:], points[-1][1:]) > 0.5 * size


#: How long her thing stays put under a way held before what lies that way is taken to be out of her reach.
OUT_OF_REACH_S = 0.6

#: How long she makes for a place and gets no nearer before it is taken to be out of her reach.
NO_NEARER_S = 1.5


def _out_of_her_reach(run: _Run, choosing: _Choosing, at: float) -> None:
    """What she has found she cannot get to: the cells beyond what stops her, and places she gets no nearer to.

    A room has walls, and the ground she means to go over is only the ground
    she can get to: a person walking into a wall learns there is a wall that way
    and looks elsewhere. Where she holds a way and her thing does not go, the
    cells that way to the edge of the picture, across the band of her thing's
    own size. Where she makes for a place, by whatever ways, and for a while
    gets no nearer to it, that place: LIVE 2026-10-07 her hero went left and
    right under a spot on the wall above it a hundred and forty times.
    """
    mine, cell = choosing.mine, run.cell
    if mine is None or not cell:
        run.stuck = run.making_for = None
        return
    (gx, gy), why, _aim = choosing.target()
    if why in ("meet", "cover") and gx is not None and gy is not None:
        place, far = (int(gx // cell), int(gy // cell)), math.hypot(gx - mine.x, gy - mine.y)
        if run.making_for is None or run.making_for[0] != place or far < run.making_for[1] - 0.25 * cell:
            run.making_for = (place, far, at)
        elif at - run.making_for[2] >= NO_NEARER_S:
            run.barred.add(place)
            run.making_for = None
    else:
        run.making_for = None
    way = choosing.ways.get(run.held) if run.held else None
    if way is None:
        run.stuck = None
        return
    if run.stuck is None or run.stuck[0] != run.held or math.dist(run.stuck[1:3], (mine.x, mine.y)) > 0.25 * cell:
        run.stuck = (run.held, mine.x, mine.y, at)
        return
    if at - run.stuck[3] < OUT_OF_REACH_S:
        return
    tall, wide = choosing.moves.shape
    along = 0 if abs(way[0]) >= abs(way[1]) else 1
    step = 1 if way[along] > 0 else -1
    middle, half = (mine.y, mine.h / 2) if along == 0 else (mine.x, mine.w / 2)
    band = range(int((middle - half) // cell), int((middle + half) // cell) + 1)
    ahead = int((mine.x, mine.y)[along] // cell) + step
    while 0 <= ahead <= int((wide, tall)[along] // cell):
        for across in band:
            run.barred.add((ahead, across) if along == 0 else (across, ahead))
        ahead += step
    run.stuck = None


def _goes_somewhere(thing: Any) -> bool:
    """Whether a moving thing travels rather than moving in place, by its path and its speed against its own size."""
    size = max(float(thing.w), float(thing.h), 1.0)
    if math.hypot(thing.vx, thing.vy) > 2.0 * size:
        return True
    points = [(x, y) for _at, x, y in thing.path]
    if len(points) < 3:
        return False
    xs, ys = [x for x, _y in points], [y for _x, y in points]
    extent = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    return extent > 0.5 * size and math.dist(points[0], points[-1]) > 0.5 * extent


#: How long a line once said is not said again, word for word, in the same game: a watcher heard it.
REPEAT_AFTER_S = 90.0


def _say(run: _Run, say: Callable[[str], Any] | None, line: str, at: float, *, once: str = "") -> None:
    if not line.strip(" ."):
        return
    if once and once in run.said:
        return
    # The same words a moment ago, about another thing of the same look or in the stretch before: LIVE 2026-10-07
    # "The black things are worth shooting." three times in two seconds, and "Space fires." over and over.
    if at - run.lately.get(line, -REPEAT_AFTER_S) < REPEAT_AFTER_S:
        return
    if at - run.said_at < SAY_EVERY_S and not once:
        return
    if once:
        run.said.add(once)
    run.said_at = at
    run.lately[line] = at
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
            if run.presses is not None:
                run.presses.released(run.held, time.monotonic())
        dispatched = time.monotonic()
        if key:
            await hands.down(key)
            run.input_key_downs[key] += 1
            if run.presses is not None:
                run.presses.pressed(key, dispatched, run.things.values(), hers.number)
        run.responses.append(max(0.0, time.monotonic() - at))
    delivered = time.monotonic()
    if key and key != run.held:
        hers.tapped(key, delivered, began=dispatched)
    run.held, run.held_since, run.held_trying = key, delivered, trying
    hers.holding(key, delivered, trying=trying)


async def _keep_reading(run: _Run, meeting: WhatMeetingDoes, hers: WhichIsHers, moves: WhatMoves,
                        picture: Any, at: float, read_words: Callable[[Any], list[dict[str, Any]]] | None,
                        *, say: Any = None) -> None:
    """Read the counters off to one side, and the bars every picture; take in the last reading when it is done."""
    from core.perception.the_drawing_as_objects import words_in

    _read_the_bars(run, meeting, hers, moves, picture, at, say)
    rendered = words_in(picture)
    if rendered is not None:
        if run.reading is not None:
            run.reading.cancel()
            run.reading = None
        run.read_at = at
        _read_the_words(run, meeting, hers, moves, rendered, at)
        return
    if read_words is None:
        return
    if run.reading is not None and run.reading.done():
        regions, when = run.reading.result()
        run.reading = None
        _read_the_words(run, meeting, hers, moves, regions, when)
    if run.reading is None and at - run.read_at >= READ_EVERY_S:
        run.read_at = at
        bgr = picture[:, :, ::-1].copy()

        async def _read() -> tuple[list[dict[str, Any]], float]:
            return await asyncio.to_thread(read_words, bgr), at

        run.reading = asyncio.ensure_future(_read())


def _read_the_words(run: _Run, meeting: WhatMeetingDoes, hers: WhichIsHers,
                    moves: WhatMoves, regions: list[dict[str, Any]], when: float) -> None:
    """The same counter semantics for renderer text and text read from pixels."""
    run.regions_read = list(regions)
    for region in regions:
        said = " ".join(str(region.get("text") or "").lower().split())
        if said and len(run.words_read) < 400:
            run.words_read.append((when, said))
    mine = hers.thing(moves)
    her_x = moves.share(mine.x, mine.y)[0] if mine is not None else None
    for verdict in meeting.read(regions, when, her_x):
        if verdict["what"] == "gain":
            run.gains += 1
        else:
            run.losses += 1
            hers.lost_a_life(when)
            _lost_from_a_distance(run, meeting, mine, moves, verdict)
            # A loss sets her back and changes the place: where she seemed not to get to before it is not known to be
            # out of her reach (offline 2026-10-09, set back to the start each time, the way to a coin was marked out
            # of her reach and she waited in a corner while it stood there).
            run.barred.clear()
    run.contest.heard(" ".join(str(region.get("text") or "") for region in regions))
    run.contest.counted(meeting.readouts.current, meeting.readouts.where, her_x, when)
    meeting.writing = _what_is_writing(moves, regions)


#: How full her bars may be before she is careful: below this share of their fullest, what costs is given wider berth.
CAREFUL_BELOW = 0.5


def _what_a_bar_measures(bar: Any, regions: list[dict[str, Any]], wide: int, tall: int) -> tuple[str, str]:
    """The words written nearest a bar, before it on its line or just over it, and what a change in them means; a bar
    with no words is taken for something running out, worse lower."""
    from core.agency.what_meeting_things_does import _meaning

    left, top, right, bottom = bar.where(wide, tall)
    near = []
    for region in regions:
        said = " ".join(str(region.get("text") or "").split())
        x, y = float(region.get("center_x", -1.0)), float(region.get("center_y", -1.0))
        if not any(ch.isalpha() for ch in said):
            continue
        before = abs(y - (top + bottom) / 2) <= 0.06 and -0.02 <= left - x <= 0.25
        over = 0.0 <= top - y <= 0.1 and left - 0.05 <= x <= right + 0.05
        if before or over:
            near.append((abs(y - (top + bottom) / 2) + abs(x - left), said))
    label = min(near)[1] if near else ""
    return label, (_meaning(label) if label else "") or "down is bad"


def _read_the_bars(run: _Run, meeting: WhatMeetingDoes, hers: WhichIsHers, moves: WhatMoves, picture: Any, at: float,
                   say: Any) -> None:
    """Every bar's rise or fall, as a gain or a loss as its words say; and how much she has left, for how careful to be.

    A bar that empties as she is hit is the cost of what hit her, learned as a counter's fall is: LIVE 2026-10-09 three
    heroes shared a health bar, and she played as if nothing could hurt her.
    """
    from core.perception.how_full_a_bar_is import BarsOnTheScreen

    if run.bars is None:
        run.bars = BarsOnTheScreen()
    for change in run.bars.read(picture, at):
        bar = change["bar"]
        tall, wide = run.bars.shape
        if not bar.meaning:
            bar.label, bar.meaning = _what_a_bar_measures(bar, run.regions_read, wide, tall)
        if bar.meaning == "neither":
            continue
        fell = change["to"] < change["from"]
        lost = fell if bar.meaning == "down is bad" else not fell
        middle = ((bar.start + bar.end) / 2 / max(1, wide), bar.row / max(1, tall))
        where = where_on_screen(moves, middle[0] * moves.shape[1], middle[1] * moves.shape[0]) if moves.shape[0] else "middle"
        named = f"the {bar.label} bar" if bar.label else f"the bar at the {where}"
        verdict = {"what": "loss" if lost else "gain", "at": at, "since": change["since"], "counter": named, "by": -1 if lost else 1}
        meeting._verdict(verdict)
        if lost:
            run.losses += 1
            _lost_from_a_distance(run, meeting, hers.thing(moves), moves, verdict)
            _say(run, say, f"{named[0].upper()}{named[1:]} goes down as I'm hit: it's what I have left.", at, once=f"bar {named}")
        else:
            run.gains += 1
    left = [bar.full for bar in run.bars.bars() if bar.meaning == "down is bad"]
    run.vitals = meeting.vitals = min(left) if left else 1.0


def _lost_from_a_distance(run: _Run, meeting: WhatMeetingDoes, mine: Any, moves: WhatMoves, verdict: dict[str, Any]) -> None:
    """A loss with nothing having touched her lately is put down to the things near her when it came, as they stood then."""
    if run.reach is None or mine is None:
        return
    then = (float(verdict.get("from", verdict["at"])) + float(verdict["at"])) / 2
    if any(abs(when - then) <= TOUCHED_WITHIN_S for when in meeting._met.values()):
        return
    run.reach.lost_untouched(_as_it_stood(mine, then), [_as_it_stood(thing, then) for thing in moves.things.values()
                                                         if thing.number != mine.number and thing.number not in meeting.writing])


def _as_it_stood(thing: Any, then: float) -> Any:
    """A thing as it stood at ``then``, from where its way had it nearest that time."""
    from types import SimpleNamespace

    path = list(getattr(thing, "path", ()) or ())
    at, x, y = min(path, key=lambda step: abs(step[0] - then)) if path else (then, thing.x, thing.y)
    return SimpleNamespace(number=thing.number, kind=thing.kind, x=x, y=y, vx=thing.vx, vy=thing.vy, w=thing.w, h=thing.h)


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
    turn = run.trying % (2 * len(run.keys))
    run.trying += 1
    key = run.keys[turn // 2] if turn % 2 == 0 else ""
    await _hold(hands, run, hers, key, at, trying=True)
    run.tried_at = run.held_since


async def _click_things(hands: Any, run: _Run, moves: WhatMoves, meeting: WhatMeetingDoes, at: float,
                        *, only_named: bool = False) -> None:
    """With no thing of her own, a click is how she meets things.

    She clicks the thing that is about to leave first, among the kinds she
    has not learned to leave alone, where it will be when the click lands.
    """
    if at - run.last_click < CLICK_EVERY_S:
        return
    tall, wide = moves.shape
    candidates = [
        t for t in moves.things.values()
        if (t.moved or meeting.stance(t.kind) == CLICK)
        and t.number not in meeting.writing and meeting.stance(t.kind) in (MEET, CLICK)
        and (not only_named or meeting.stance(t.kind) == CLICK)
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


#: How often the picture itself is clicked, where the game is played with the pointer and nothing in it is to be clicked.
CLICK_THE_PICTURE_S = 1.5


async def _click_the_picture(hands: Any, run: _Run, moves: WhatMoves, at: float) -> None:
    """Where the game's words name the pointer and nothing she could click has come of it, a click on the picture itself.

    "Click your mouse button to start the hamster in motion... click again to
    launch": a click aimed at nothing is how such a game is played (LIVE
    2026-10-07, four minutes in one with nothing clicked). Paced, so what one
    did can be seen before the next. Motion holds back an unspecified click;
    a trigger named by the visible instructions still works during motion.
    Where in it is learned as she goes: the places a click
    paid are clicked more (core/agency/where_clicks_pay.py), and where such
    clicks have cost more than they paid, she stops making them.
    """
    from core.agency.where_clicks_pay import WhereClicksPay

    if not run.pointer_first or at - run.last_click < CLICK_THE_PICTURE_S:
        return
    if not run.pointer_trigger and any(thing.moved for thing in moves.things.values()):
        return
    run.clicks_pay = run.clicks_pay or WhereClicksPay()
    if run.clicks_pay.costing():
        return
    x, y = run.clicks_pay.where(run.gains, run.losses)
    await hands.click(x, y)
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
    hers.pointed(x * wide, y * tall, time.monotonic())


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
        hers.pointed(x, y, time.monotonic())


def _over(run: _Run, moves: WhatMoves, happened: list[dict[str, Any]], at: float) -> str:
    if any(h.get("what") == "new screen" for h in happened):
        run.new_screen_at = at
    # Motion seen in this picture, not a thing carried on along its last speed
    # because it was not seen.
    if any(thing.seen == at for thing in moves.moving(faster_than=8.0)):
        run.last_moving = at
        run.first_moving = min(run.first_moving, at)
    still = at - run.last_moving
    # A world that waits for her is still until she moves: its stillness ends a stretch only once her ways have had time.
    still_for = WAITING_STILL_FOR_S if run.waits_for_her else STILL_FOR_S
    if at - run.new_screen_at < still_for and still >= max(NEW_SCREEN_STILL_S, still_for - STILL_FOR_S) and at - run.began > 2.0:
        return "the screen changed and nothing on it moves"
    if still >= still_for and at - run.began > still_for:
        return "nothing on the screen has moved for a while"
    return ""


#: How long she plays a moving picture in which nothing answers to her before
#: handing it back: a title screen that animates is not a game.
NOTHING_ANSWERS_S = 10.0


#: How long things she touches may go on gaining and losing nothing before the picture is handed back.
TOUCHING_TO_NO_END_S = 20.0


def _nothing_answers(run: _Run, hers: WhichIsHers, meeting: WhatMeetingDoes, at: float) -> str:
    # Allow two complete experiments and their intervening wait before
    # declaring that no control answers. The run's own deadline still bounds it.
    experiment_time = 8 * len(run.keys) * TRY_A_KEY_S + RECHECK_CONTROLS_S if run.keys else 0.0
    if at - run.began < NOTHING_ANSWERS_S + experiment_time or hers.kind is not None or meeting.verdicts:
        return ""
    # Things she touched keep it going for a while, and only a while, where
    # touching them has gained and lost nothing: LIVE 2026-10-06 a game's
    # title screen, its picture moving, its START! drawn as words: she clicked
    # the moving figure for the whole five minutes and never left it for START.
    if any(kept.touched for kept in meeting.evidence.values()):
        if at - run.began < NOTHING_ANSWERS_S + experiment_time + TOUCHING_TO_NO_END_S:
            return ""
        return "nothing I touch here gains or loses anything"
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
                            run: _Run, say: Any, at: float, *, pointer_steers: bool = False) -> None:
    """Tie the rules' words to the kinds on screen by colour, and a legend's drawings by look, as each kind first appears."""
    if rules is None and not run.legend:
        return
    for kind in moves.kinds:
        if run.told_checked.get(kind.number) == pointer_steers:
            continue
        run.told_checked[kind.number] = pointer_steers
        stance = rules.stance_for_colour(colour_name(kind.colour), pointer_steers=pointer_steers) if rules is not None else None
        if stance is None and _as_a_legend_drew(run, kind, meeting, say, at):
            continue
        if stance is not None:
            meeting.told[kind.number] = stance
            line = {
                MEET: "the {c} ones are to be got", AVOID: "the {c} ones are to be kept clear of",
                SHOOT: "the {c} ones are to be shot", CLICK: "the {c} ones are to be clicked",
            }.get(stance, "")
            if line:
                _say(run, say, "The rules say " + line.format(c=colour_name(kind.colour)) + ".", at, once=f"told {kind.number} {stance}")


def _as_a_legend_drew(run: _Run, kind: Any, meeting: WhatMeetingDoes, say: Any, at: float) -> bool:
    """A kind that looks like a thing a legend drew is taken for what the legend said of it, until play says otherwise."""
    from core.perception.what_a_legend_shows import most_like

    drawn = most_like(kind.look, run.legend, colour=kind.colour)
    stance = {"meet": MEET, "avoid": AVOID, "shoot": SHOOT}.get(drawn.stance if drawn is not None else "")
    if drawn is None or stance is None:
        return False
    meeting.told[kind.number] = stance
    _say(run, say, f"The {colour_name(kind.colour)} thing is what the rules showed: \u201c{drawn.words}\u201d", at,
         once=f"legend {drawn.words}")
    return True


def _counters_without_reading(run: _Run, moves: WhatMoves, hers: WhichIsHers, meeting: WhatMeetingDoes, at: float) -> None:
    mine = hers.thing(moves)
    her_x = moves.share(mine.x, mine.y)[0] if mine is not None else None
    never = {hers.number} if hers.number is not None else set()
    for when, x, y in run.stayed.see(getattr(moves, "_last", None), moves.things, at, never=never):
        meeting.changed_in_place(when, x, y, her_x)


#: How long between two accounts of what kind of game this is.
SITUATION_EVERY_S = 30.0

#: Where a thing stands, as where_on_screen names it, inside an account of the game.
_WHERE_IT_STANDS = re.compile(r" at the (?:(?:top|bottom) )?(?:left|right)\b| at the (?:top|bottom|middle)\b")


def _what_kind_of_game(run: _Run, say: Any, moves: WhatMoves, hers: WhichIsHers, meeting: WhatMeetingDoes,
                       physics: HowThingsMoveHere, at: float) -> None:
    """Say what kind of game this is, once there is something to say, and again when it changes."""
    from core.agency.what_kind_of_game_this_is import in_a_sentence

    if "me" not in run.said or at - run.situation_at < SITUATION_EVERY_S:
        return
    sentence = in_a_sentence(moves, hers, meeting, physics, run.keys, me=run.hers_description)
    if not sentence or ";" not in sentence:
        return
    # Said again only for something learned: her paddle having moved to the
    # top, or the other side being still a moment, is not news. LIVE
    # 2026-10-06 the same account came every thirty seconds of a game.
    parts = {_WHERE_IT_STANDS.sub("", part).strip(" .") for part in sentence.split(";")}
    if parts <= run.situation_known:
        return
    run.situation_known |= parts
    run.situation, run.situation_at = sentence, at
    _say(run, say, sentence[0].upper() + sentence[1:], at, once=f"situation {len(run.lines)}")


def _what_she_says(run: _Run, say: Any, moves: WhatMoves, hers: WhichIsHers, meeting: WhatMeetingDoes, at: float) -> None:
    mine = hers.thing(moves)
    keys = sorted(hers.keys_that_move_her(run.keys))
    if mine is not None and hers.kind is not None and (keys or hers.follows_pointer):
        run.hers_descriptions[describe(moves, hers.kind, mine)] += 1
        # A brief overlap changes a connected region's outline. Describe the
        # control by its usual observed shape rather than that one picture.
        run.hers_description = run.hers_descriptions.most_common(1)[0][0]
    settled = all(hers.tried(k) >= 4 for k in run.keys) or at - run.began > 8.0
    if mine is not None and hers.kind is not None and hers.follows_pointer:
        _say(run, say, f"That's me: the {describe(moves, hers.kind, mine)} at the {where_on_screen(moves, mine.x, mine.y)}. It goes where the mouse goes.", at, once="me")
    elif mine is not None and hers.kind is not None and keys and settled:
        how = " and ".join(keys)
        verb = "moves" if len(keys) == 1 else "move"
        _say(run, say, f"That's me: the {describe(moves, hers.kind, mine)} at the {where_on_screen(moves, mine.x, mine.y)}. {how.capitalize()} {verb} it.",
             at, once="me")
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
    # How it stands, said when it changes: the score, what is left to win, what is left to lose.
    standing = run.contest.says()
    if standing and run.contest_first is None:
        run.contest_first = standing
    if standing and standing != run.contest_said and standing != run.contest_first:
        if at - run.said_at < SAY_EVERY_S:
            return
        run.contest_said = standing
        _say(run, say, standing[:1].upper() + standing[1:], at)
        return
    # The counters as read are said only where no standing could be made of
    # them: LIVE 2026-10-06 an end screen's "You 5 Computer 4" was read as one
    # counter, and "You computer 4." followed "4-4, level".
    # Said by the names a person reading the screen would say: words of the language, a misread letter mended
    # ("Leuel" is level); a name that is not one ("Ini", "iin", "x") is not said (core/language/words_of_the_language.py).
    from core.language.words_of_the_language import words_of

    counters = {said: value for name, value in meeting.readouts.values.items()
                if not name.startswith("number") and (said := words_of(name)) is not None}
    # Each counter as it first read is the baseline: read a few at a time, a set that grew was not a change.
    first = run.counters_first if run.counters_first is not None else {}
    for name, value in counters.items():
        first.setdefault(name, value)
    run.counters_first = first
    moved = any(first.get(name) != value for name, value in counters.items())
    if counters and moved and not run.contest_said and counters != run.counted and at - run.said_at > 25.0:
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
    waits_for_her: bool = False,
) -> dict[str, Any]:
    """Play what ``look`` shows through ``hands`` until it stops moving or ``seconds`` pass.

    ``look`` returns an RGB picture and when it was taken, or None. ``hands``
    has ``down(key)``, ``up(key)``, ``tap(key)``, ``click(x, y)`` and
    ``point(x, y)``, positions as shares of the picture. ``keep`` carries what
    she learned from one stretch of play into the next one in the same game.
    ``waits_for_her`` says the world stood still until she moved in it
    (``it_goes_while_held``): with nothing to go to, she goes where she has not been.
    """
    began = time.monotonic()
    keep = keep if keep is not None else {}
    from core.agency.when_motion_breaks_a_rule import MotionChecks

    motion_checks = MotionChecks(frozenset(keep.get("required_edges") or []), str(keep.get("edge_provenance") or ""))
    moves = WhatMoves(kinds=keep.get("kinds"))
    physics: HowThingsMoveHere = keep.get("physics") or HowThingsMoveHere()
    rules = WhatTheRulesSaid.read(told) if told else None
    hers: WhichIsHers = keep.get("hers") or WhichIsHers()
    meeting: WhatMeetingDoes = keep.get("meeting") or WhatMeetingDoes()
    for kept in (physics, hers, meeting):
        kept.numbered_afresh()
    run = _Run(keys=list(keys), began=began, last_moving=began, pointer_first=pointer_first, legend=tuple(keep.get("legend") or ()),
               contest=keep.get("contest") or ContestStands(), waits_for_her=waits_for_her)
    run.pointer_trigger = rules is not None and rules.a_click_is_a_shot
    run.burst_keys = keys_pressed_fast(told)
    run.contest.heard(told)
    run.situation_known = set(keep.get("situation_known") or ())
    run.clicks_pay = keep.get("where_clicks_pay")
    if run.clicks_pay is not None:
        run.clicks_pay.begin_stretch(run.gains, run.losses)
    run.lately = dict(keep.get("said_lately") or {})
    # What is said once ("That's me", what a kind of thing is worth) is said once a game, not once a stretch.
    run.said = set(keep.get("said_once") or ())
    from core.agency.pressing_what_is_shown import KeysShown

    run.shown = KeysShown(keep.get("keys_never_absent"))
    from core.agency.what_a_press_does import WhatAPressDoes

    run.presses = keep.get("presses") or WhatAPressDoes()
    from core.agency.when_a_press_pays import WhenAPressPays

    run.timing = keep.get("timing") or WhenAPressPays()
    from core.agency.how_far_a_thing_reaches import HowFarThingsReach

    run.reach = keep.get("reach") or HowFarThingsReach()
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
            run.observation_sources["canvas-paint-v1" if getattr(picture, "drawing_scene", None) is not None else "pixels"] += 1
            if run.picture_at is not None and 0.0 < at - run.picture_at <= 0.5:
                run.intervals.append(at - run.picture_at)
            run.picture_at = at
            run.pictures += 1
            happened = moves.see(picture, at)
            hers.saw(moves, happened, at)
            # A key named for a way moves her; it is taken to fire only where the game's words say it does: LIVE
            # 2026-10-07 in a shooter whose words said "SPACE press to fire", left, up and right were each said to fire,
            # shots turning up beside her whatever she pressed.
            for key in [k for k in hers.makes if k in WAYS and not (rules is not None and k in rules.fire_keys)]:
                del hers.makes[key]
            _record_control_attribution(run, moves, hers, at)
            _measure_response(run, hers.thing(moves), at)
            physics.saw(moves, hers, happened, at)
            _what_the_rules_said_of(rules, moves, meeting, run, say, at, pointer_steers=hers.follows_pointer)
            # The ground is the point where the rules say so, and where the world waits for her: with nothing coming and
            # nothing to go to, a person looks round the place for what is there (a way on, a thing to take).
            goes_over = (rules is not None and rules.covers) or waits_for_her
            if any(h.get("what") == "new screen" for h in happened):
                run.covered.clear()
                run.barred.clear()
            if hers.thing(moves) is not None:
                mine = hers.thing(moves)
                run.cell = run.cell or max(4.0, mine.w, mine.h)
                run.covered.add((int(mine.x // run.cell), int(mine.y // run.cell)))
            choosing = _Choosing(moves, hers, meeting, run.keys, physics,
                                 next_picture_s=statistics.median(run.intervals) if run.intervals else 1 / 30,
                                 response_s=(statistics.median(run.motion_responses) if len(run.motion_responses) >= 3
                                             else statistics.median(run.responses) if run.responses else 0.0),
                                 covered=(run.covered | run.barred, run.cell) if goes_over and run.cell else None,
                                 barred=(run.barred, run.cell) if run.cell else None, presses=run.presses,
                                 reach=run.reach)
            if choosing.mine is not None and (choosing.ways or choosing.lifts or choosing.pointing):
                run.responsive_pictures += 1
            run.things = moves.things
            run.presses.saw(moves.things, at)
            run.timing.saw(moves.things.values())
            _found_by_her_presses(run, moves, hers, at)
            await _let_go_of_presses(hands, run, at)
            meeting.saw(moves, hers, happened, at, choosing.line() if choosing.mine is not None else None)
            if getattr(picture, "drawing_scene", None) is None:
                _counters_without_reading(run, moves, hers, meeting, at)
            await _keep_reading(run, meeting, hers, moves, picture, at, read_words, say=say)
            run.shown.look(picture, at)
            violations = motion_checks.see(moves, happened, at, hers.number)
            if violations:
                ended = "runtime contract violated"
                _say(run, say, "This still behaves incorrectly: " + violations[0]["finding"] + ". I need to check the repair.", at, once="runtime_fault")
                break
            await _act(hands, run, moves, hers, meeting, choosing, at, say=say, rules=rules)
            _out_of_her_reach(run, choosing, at)
            if goes_over:
                if waits_for_her and choosing.mine is not None and choosing.ways and choosing.target()[1] == "cover":
                    _say(run, say, "Nothing's coming at me, so I'm looking around the place.", at, once="looking around")
            _what_she_says(run, say, moves, hers, meeting, at)
            _what_kind_of_game(run, say, moves, hers, meeting, physics, at)
            _report(run, getting_somewhere, at)
            ended = _over(run, moves, happened, at) or _nothing_answers(run, hers, meeting, at)
    finally:
        for key in [run.held, *run.letting_go] if run.held else list(run.letting_go):
            try:
                await hands.up(key)
            except (RuntimeError, OSError, ValueError, TypeError, AttributeError) as why:
                logger.debug("letting go of %s failed: %s", key, why)
        if run.reading is not None:
            run.reading.cancel()
        keep["keys_never_absent"] = run.shown.done()
    keep.update({"hers": hers, "meeting": meeting, "kinds": moves.kinds, "physics": physics, "meeting_with": run.meeting_with,
                 "contest": run.contest, "situation_known": run.situation_known, "said_lately": run.lately,
                 "where_clicks_pay": run.clicks_pay, "presses": run.presses, "timing": run.timing, "reach": run.reach,
                 "said_once": run.said})
    result = _what_it_came_to(run, moves, hers, meeting, ended, began)
    result["runtime_checks"] = {"required_edges": sorted(motion_checks.required_edges),
                                "provenance": motion_checks.provenance, "violations": motion_checks.violations,
                                "unconfirmed": motion_checks.unconfirmed}
    return result


async def _act(hands: Any, run: _Run, moves: WhatMoves, hers: WhichIsHers, meeting: WhatMeetingDoes,
               choosing: _Choosing, at: float, *, say: Any = None, rules: WhatTheRulesSaid | None = None) -> None:
    # Keys the screen puts up mid-play are asked for now, before anything she had in mind (pressing_what_is_shown.py).
    if run.shown is not None and run.shown.asks(at):
        if run.held:
            await hands.up(run.held)
            run.held = ""
        await run.shown.press(hands, say=lambda line: _say(run, say, line, at, once="keys shown"))
        return
    # A key pressed before she can see what it does tells her nothing: the first look at a screen takes its measure.
    if not getattr(moves, "seeing", True):
        return
    # A thing taken for hers that did not answer her keys: try them again.
    if hers.lost_at > run.lost_at:
        run.lost_at, run.trying = hers.lost_at, 0
    # An instruction to click a visible target already names the interaction.
    # It does not require discovering an avatar that follows the pointer.
    aimed_trigger = (choosing.mine is not None and hers.follows_pointer
                     and rules is not None and rules.a_click_is_a_shot)
    if run.pointer_first and not aimed_trigger and any(meeting.stance(t.kind) == CLICK for t in moves.things.values()):
        await _click_things(hands, run, moves, meeting, at, only_named=True)
        return
    # Every key is tried once before any is chosen: a plan that knows one key
    # can only go one way.
    trying_the_pointer = run.pointer_first and run.pointed < 2 * len(_POINTER_TRIAL) and not hers.follows_pointer
    # A game its words say is played with the pointer is clicked between the keys tried, not after all of them:
    # LIVE 2026-10-08 a stretch ended on its ten key trials before the click that would have started the game.
    if (run.pointer_first and not trying_the_pointer and not hers.follows_pointer and choosing.mine is None
            and hasattr(hands, "click") and at - run.last_click >= CLICK_THE_PICTURE_S):
        await _click_the_picture(hands, run, moves, at)
        if run.last_click == at:
            return
    if choosing.mine is None and not hers.follows_pointer:
        await _fire_while_finding_herself(hands, run, hers, moves, rules, at)
    if not trying_the_pointer and run.trying < 2 * len(run.keys) and not hers.keys_known(run.keys) and not hers.follows_pointer:
        await _try_the_keys(hands, run, hers, at)
        return
    if choosing.mine is not None and hers.follows_pointer:
        await _hold(hands, run, hers, "", at)
        _goal, why, aim = choosing.target()
        await _trigger(hands, run, hers, choosing, aim, why, at, rules=rules)
        await _point_at(hands, run, hers, moves, choosing, at)
        return
    if choosing.mine is None or not (choosing.ways or choosing.lifts):
        # Held, having moved before: where the words say to press keys in turn, fast, that is how she breaks free.
        if run.burst_keys and hers.kind is not None and at - run.burst_at >= BURST_EVERY_S:
            await _burst(hands, run, at, say=say)
            return
        # No press of her keys has moved anything on the screen, after all her trials of them, and things go on by
        # themselves: a press may count by when it is made, not by what it moves (core/agency/when_a_press_pays.py).
        # A press that moved something the same way each time is a body still to be found, not a clock to beat:
        # offline 2026-10-09 a runner missed by her first trials was timed against and never looked for again.
        # And timing goes on while it pays; where it has not paid once every place has been tried, she looks for a
        # body again.
        if (run.keys and not run.pointer_first and run.timing is not None and run.trying >= 4 * len(run.keys)
                and not run.presses.moves_anything(TOLD_FROM_STANDING_PX * 2)
                and (run.timing.pays() or at - run.tried_at < max(RECHECK_CONTROLS_S, run.timing.a_round_s()))
                and await _press_in_time(hands, run, moves, at, say)):
            return
        if run.keys and run.trying >= 4 * len(run.keys) and at - run.tried_at >= RECHECK_CONTROLS_S:
            run.trying = 0
            run.control_probe_retries += 1
            hers.recheck_controls()
            if run.pointer_first:
                run.pointed = 0
            # Said only where she had found which one was hers and lost it: on a title screen there was nothing to lose,
            # and LIVE 2026-10-07 the line was read out over one.
            if "me" in run.said:
                _say(run, say, "I've lost track of which one is me; trying the keys again to find out.", at, once="recheck_controls")
            else:
                logger.info("trying the keys again: nothing has answered to them yet")
        pointer_trial = run.pointed < 2 * len(_POINTER_TRIAL) and hasattr(hands, "point")
        if pointer_trial and run.pointer_first:
            await _try_the_pointer(hands, run, hers, moves, at)
            return
        # Played with the pointer: a click, paced, before the keys are gone through again.
        if run.pointer_first and hasattr(hands, "click") and at - run.last_click >= CLICK_THE_PICTURE_S:
            await _click_the_picture(hands, run, moves, at)
            if run.last_click == at:
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
            await _click_the_picture(hands, run, moves, at)
        return
    untried = [k for k in run.keys if hers.tried(k) < 4 and k not in choosing.ways]
    # Where she means to go and no way she knows takes her nearer, a way she has not learned yet is tried now: LIVE
    # 2026-10-07 her hero stood still beside the thing it meant to go to, left being the one arrow not yet learned.
    if untried and choosing.danger((0.0, 0.0)) == 0.0 and (
            at - run.tried_at > 3.0 or choosing.key(run.held)[:2] in {("", "meet"), ("", "cover")}):
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
    before = choosing.ways.get(run.held, (0.0, 0.0))
    after = choosing.ways.get(key, (0.0, 0.0))
    if key != run.held:
        mine = choosing.mine
        run.previous_control = (mine.number, mine.x, mine.y, at, before, after)
    await _hold(hands, run, hers, key, at)
    await _press_to_get_clear(hands, run, hers, choosing, key, at, say)
    await _trigger(hands, run, hers, choosing, aim, why, at, rules=rules)


async def _press_to_get_clear(hands: Any, run: _Run, hers: WhichIsHers, choosing: _Choosing, held: str, at: float,
                              say: Any) -> None:
    """A press beside the key held, where the path it takes her on (core/agency/what_a_press_does.py) is clear of what
    is coming and holding her course is not: over a thing running at her, across a gap, out of the way of a fall."""
    if run.letting_go or choosing.mine is None:
        return
    press = choosing.a_press_that_clears(run.presses, held)
    if press is None:
        return
    began = time.monotonic()
    await hands.down(press)
    run.input_key_downs[press] += 1
    run.presses.pressed(press, began, run.things.values(), choosing.mine.number)
    run.letting_go[press] = began + run.presses.held_for(press, THE_PRESS_S)
    hers.pressed_beside(began + max(run.presses.lasts(press, choosing.mine.kind, choosing.mine.number),
                                    run.letting_go[press] - began))
    _say(run, say, f"{press.capitalize()} lifts me clear of what comes at me; I'm using it when something is about to.",
         at, once="a press that clears")


def _found_by_her_presses(run: _Run, moves: WhatMoves, hers: WhichIsHers, at: float) -> None:
    """Where her trials have not found her, the one thing of a kind her presses move alike every time is hers."""
    if hers.number is not None or hers.follows_pointer or not run.presses.curves:
        return
    kinds = run.presses.kinds_it_moves(TOLD_FROM_STANDING_PX * 2)
    candidates = [thing for thing in moves.things.values() if thing.kind in kinds]
    if len(candidates) == 1:
        hers.moved_by_her_presses(candidates[0], at)


async def _press_in_time(hands: Any, run: _Run, moves: WhatMoves, at: float, say: Any) -> bool:
    """A press timed to where the things that keep moving will be; whether she is playing by timing presses."""
    ahead = statistics.median(run.responses) if run.responses else RESPONSE_S
    if not run.timing.press_now(moves.things.values(), at, ahead):
        return run.timing.has_something_to_time(moves.things.values())
    key = run.keys[0]
    run.timing.credit(run.gains, run.losses)
    _say(run, say, f"Nothing here moves when I press {key}; I'm timing each press to where the moving thing is, "
         "and keeping to the places where a press has paid.", at, once="timing")
    await hands.down(key)
    run.input_key_downs[key] += 1
    await asyncio.sleep(PRESSED_FOR_S)
    await hands.up(key)
    run.timing.pressed(moves.things.values(), at, ahead)
    return True


async def _let_go_of_presses(hands: Any, run: _Run, at: float) -> None:
    """Let go of each press made beside the held key once it has been held as long as it was when it was watched."""
    for key, when in list(run.letting_go.items()):
        if time.monotonic() >= when:
            await hands.up(key)
            run.presses.released(key, time.monotonic())
            del run.letting_go[key]


async def _trigger(hands: Any, run: _Run, hers: WhichIsHers, choosing: _Choosing,
                   aim: Any, why: str, at: float, *, rules: WhatTheRulesSaid | None = None) -> None:
    """Deliver a trigger alongside steering, using its own physical input.

    Steering and firing are compatible channels. A pointer-controlled object
    used to return before its trigger could run. Named triggers are tried at
    a measured pace until their projectiles can be aimed by learned motion.
    """
    fire = choosing.fire(aim, why)
    discovering = choosing.shot is None
    if fire is None and discovering and choosing.pointing and choosing.mine is not None:
        if rules is not None and rules.a_click_is_a_shot:
            fire = "mouse"
        elif rules is not None:
            fire = next((key for key in rules.fire_keys if key not in WAYS), None)
    window = None if discovering else hers.emission_window(choosing.shot.key, choosing.shot.kind)
    if discovering:
        next_trigger = run.last_click + CLICK_THE_PICTURE_S
    elif window is None:
        next_trigger = run.last_click + max(CLICK_EVERY_S, 0.4 + RESPONSE_S)
    else:
        next_trigger = max(run.last_click + CLICK_EVERY_S, run.last_trigger_began + window + RESPONSE_S)
    if not fire or fire == run.held or at < next_trigger:
        return
    if fire == "mouse":
        if not hasattr(hands, "click"):
            return
        if choosing.pointing:
            tall, wide = choosing.moves.shape
            run.pointer = (min(1.0, max(0.0, choosing.mine.x / max(1, wide))),
                           min(1.0, max(0.0, choosing.mine.y / max(1, tall))))
        dispatched = time.monotonic()
        await hands.click(*run.pointer)
    else:
        dispatched = time.monotonic()
        await hands.tap(fire)
    delivered = time.monotonic()
    if fire == "mouse" and choosing.pointing:
        hers.pointed(run.pointer[0] * wide, run.pointer[1] * tall, delivered)
    hers.tapped(fire, delivered, began=dispatched)
    run.last_trigger_began = dispatched
    run.last_click = delivered
    run.taps += 1


#: How often she fires while still finding which thing is hers, by a key the words named for it or a click that throws.
FIRE_WHILE_FINDING_S = 0.35
THROW_WHILE_FINDING_S = 0.6


async def _fire_while_finding_herself(hands: Any, run: _Run, hers: WhichIsHers, moves: WhatMoves,
                                      rules: WhatTheRulesSaid | None, at: float) -> None:
    """Fire from the first moment, as the words say to, while she is still finding which thing is hers.

    A person dropped into a game that says "SPACE to fire" fires at once and finds their feet as they go; one told
    "aim with your mouse and click to throw" throws at what moves. LIVE 2026-10-08 she spent a hundred seconds of a
    shooter holding one arrow at a time to see what moved, and never fired. A press of the fire key moves nothing, so
    the finding goes on as before, and what each press brings out beside her is how she learns it fires.
    """
    if rules is None:
        return
    fire = next((key for key in rules.fire_keys if key not in WAYS and key != run.held), None)
    if fire is not None and at - run.last_trigger_began >= FIRE_WHILE_FINDING_S:
        began = time.monotonic()
        await hands.tap(fire)
        hers.tapped(fire, time.monotonic(), began=began)
        run.last_trigger_began, run.taps = began, run.taps + 1
        return
    if not rules.a_click_is_a_shot or not hasattr(hands, "click") or at - run.last_click < THROW_WHILE_FINDING_S:
        return
    going = [t for t in moves.things.values() if math.hypot(t.vx, t.vy) > 8.0 and t.number != hers.number
             and t.kind not in {made.kind for made in hers.makes.values()}]
    if not going:
        return
    target = max(going, key=lambda t: t.size)
    tall, wide = moves.shape
    await hands.click(min(1.0, max(0.0, target.x / max(1, wide))), min(1.0, max(0.0, target.y / max(1, tall))))
    run.last_click = at


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
    contact = _contact_line(mine, coming, line)
    arrival = physics.when_it_reaches(coming, line, contact)
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
        start[line] = contact
        start[free] = arrival[1]
        crossing = physics.when_it_reaches(
            coming, line, _contact_line(them, coming, line), start=(start[0], start[1], leaving[0], leaving[1])
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


def _the_words_of_play(run: _Run, ended: str) -> set[str]:
    """The words read while the game was going, not those of the screen it ended on.

    A stretch that ended because nothing moved ended on a still screen, and
    whatever was read from the moment movement stopped (and a little before:
    readings are taken from pictures a step behind) is that screen's. LIVE
    2026-10-04 "to play again" was read in the seconds the end screen stood,
    kept as a word of play, and so the restart that ended a won game was taken
    for the game's furniture and she started another.
    """
    still = ended.startswith(("nothing on the screen", "the screen changed"))
    cutoff = run.last_moving - END_SCREEN_MARGIN_S if still else math.inf
    # Nor the screen it began on, read before anything had moved: a stretch
    # begun as the last game's end screen gave way is not that screen's game.
    return {said for when, said in run.words_read if run.first_moving <= when < cutoff}


#: How long before the last movement a reading may already show the end.
END_SCREEN_MARGIN_S = 0.5


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
        "hers": run.hers_description or (describe(moves, hers.kind, hers.thing(moves)) if hers.kind is not None else ""),
        "keys_that_move_her": sorted(hers.keys_that_move_her(run.keys)),
        "fires": sorted(hers.makes),
        "learned": learned,
        "gains": run.gains,
        "losses": run.losses,
        "counters": dict(meeting.readouts.values),
        "observations": dict(run.observation_sources),
        "input_key_downs": dict(run.input_key_downs),
        "responsive_pictures": run.responsive_pictures,
        "control_probe_retries": run.control_probe_retries,
        "control_attribution": list(run.control_attribution),
        "attribution_changes": run.attribution_changes,
        "identification_evidence": list(hers.identification.receipts),
        # How the contest stood when the stretch ended, and what its counters alone settle.
        "standing": run.contest.says(),
        "settled": run.contest.settled(),
        "stalled": run.contest.stalled(time.monotonic()),
        "said": list(run.lines),
        "words_seen": sorted(_the_words_of_play(run, ended)),
    }

"""A control used on the thing it is for, when she is beside it: told "Press SPACE to open doors", she presses space
when she comes up beside a door.

A game's words often name what a key does and what it is done to: open doors, climb ladders, talk to people, pick
up items. The guide to the place keeps each control with its act and the thing it is done to (``Control.on``,
core/cognition/a_guide_to_a_place.py), and her eyes name the things in the picture (``WhatSheSees``,
core/cognition/what_things_are.py). A person who has read the controls and comes up beside a door presses the key
that opens doors. If they walked past it, they would never learn what was behind it. The people in the recordings
of these games did just that: they stopped at each door and pressed the key, talked to everyone they met, and
climbed every ladder they reached.

What a thing is for may be known even when the game does not say what the key is done to. Told "press up to climb",
a person knows a ladder is a thing you climb. Her knowledge of such things anywhere (what a kind of thing usually is)
joins the two. A name for many things ("people", "items", "enemies") is matched by how a thing bears, as she knows
it: a character, a thing to get, a danger.

What followed a press made at a thing decides what she keeps. If the thing went away, moved or changed, or
something new came near it, or the screen changed, the press did something. She says so, and uses that key on that
kind wherever she meets it. If nothing changed after two presses at one kind of thing, she stops using that key on
that kind. Clicks go to the thing itself, wherever it is. Keys are pressed only beside it, the way the game's words
say.

Nothing here knows a game.
"""
from __future__ import annotations

import logging
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["UsingOnThings", "a_name_for", "a_place_to_go", "use_controls_on_things"]

logger = logging.getLogger("Aura.UsingAControlOnAThing")

#: How near a thing must be to be beside her, as a share of her own size, edge to edge.
BESIDE = 0.6
#: How soon after a press what it did is judged, and how long between presses at one thing.
JUDGED_AFTER_S = 0.7
AGAIN_AFTER_S = 2.5
#: Presses at one kind that did nothing before the key is no longer used on that kind.
NOTHING_TWICE = 2
#: Acts that are held while they last (a ladder is climbed for as long as the key is held), and how long.
_HELD_ACTS = frozenset({"climb", "push", "pull", "dig", "hold", "drag", "swim", "charge", "fill", "pump"})
HELD_FOR_S = 0.9
#: How long any other press is held, as a person's is: a game that asks each picture whether a key is down never sees
#: one let go as soon as it is pressed (offline 2026-10-10, three presses at doors that never opened).
PRESSED_FOR_S = 0.15
#: Acts that move her, not done to a thing beside her: the way she is moved is learned and played elsewhere.
_MOVING_ACTS = frozenset({"move", "steer", "go", "walk", "run", "guide", "control", "drive", "fly", "jump", "turn",
                          "aim", "dodge", "avoid", "navigate", "travel", "roll", "skate", "ride"})
#: Acts done to the game, not to a thing in it: pausing it, starting it, turning its sound off.
_GAME_ACTS = frozenset({"pause", "unpause", "start", "begin", "restart", "continue", "resume", "mute", "unmute", "skip",
                        "quit", "exit", "play", "select", "choose", "confirm", "toggle", "reset", "retry", "save", "load",
                        "zoom", "scroll", "view", "see", "read", "look", "show", "hide", "open the menu", "menu"})
#: Acts done from afar, aimed and timed where play shoots (core/agency/playing_as_it_happens.py), not beside a thing.
_FROM_AFAR = frozenset({"shoot", "fire", "throw", "launch", "toss", "blast", "zap", "lob", "hurl"})
#: Names for many things at once, and how a thing named one bears, as she knows it (core/cognition/what_things_are.py).
_MANY = {
    "a friend": frozenset({"person", "people", "character", "someone", "anyone", "everyone", "friend", "villager",
                           "citizen", "customer", "guest", "kid", "npc", "neighbor", "neighbour"}),
    "to get": frozenset({"item", "thing", "object", "stuff", "treasure", "goody", "goodie", "pickup", "powerup",
                         "power-up", "prize", "loot", "collectible", "bonus"}),
    "danger": frozenset({"enemy", "monster", "baddie", "baddy", "villain", "foe", "creature", "ghost", "opponent"}),
}
_IRREGULAR = {"people": "person", "men": "man", "women": "woman", "children": "child", "mice": "mouse", "feet": "foot",
              "teeth": "tooth", "geese": "goose", "leaves": "leaf", "knives": "knife", "shelves": "shelf"}


def _one(word: str) -> str:
    """One of a word for many: doors a door, boxes a box, enemies an enemy."""
    word = word.lower()
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "xes", "sses", "zes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _words(name: str) -> set[str]:
    return {_one(w) for w in re.findall(r"[a-z][a-z'-]+", str(name or "").lower()) if len(w) >= 3}


def a_name_for(on: str, seen_as: str, bears: str = "", usually: str = "", act: str = "", someone: bool = False) -> bool:
    """Whether a thing seen as ``seen_as`` is one the control is used on: one of the things named (``on``: "doors"), by
    its name, or by how it bears where ``on`` names many ("people", "items"); or, where the control names no thing,
    a thing that is usually what its act is done to ("climb", and a ladder is a thing you climb)."""
    seen = _words(seen_as)
    if on:
        wanted = _words(on)
        if wanted & seen:
            return True
        for bearing, names in _MANY.items():
            if wanted & names and (bears == bearing or (bearing == "a friend" and someone)):
                return True
        return False
    verb = (act or "").split()[0].lower() if act else ""
    return bool(verb) and len(verb) >= 3 and re.search(rf"\b{re.escape(verb)}(?:s|ed|ing|es)?\b", usually.lower()) is not None


@dataclass(frozen=True)
class _Spot:
    """A thing a control may be used on: one play follows (``number``), or one her eyes saw standing in the place
    (``landmark``); where it is in play's measure, and what it is."""

    name: str
    x: float
    y: float
    w: float
    h: float
    number: int | None = None
    kind: int | None = None
    landmark: Any = None
    bears: str = ""
    usually: str = ""
    someone: bool = False

    @property
    def mark(self) -> tuple[Any, ...]:
        return ("thing", self.number) if self.number is not None else ("seen", self.name, round(self.x), round(self.y))


@dataclass
class _Press:
    key: str
    spot: _Spot
    at: float
    near: frozenset[int]
    screens: int
    crop: Any = None
    hers: tuple[float, float, float, float] | None = None


@dataclass
class UsingOnThings:
    """What she has made of using controls on things in one game: kept from one stretch of play to the next."""

    #: Presses at a kind of thing (by its name) that did nothing, and those a key is known to do something to.
    nothing: dict[tuple[str, str], int] = field(default_factory=dict)
    worked: set[tuple[str, str]] = field(default_factory=set)
    #: When each thing was last pressed at.
    last: dict[tuple[Any, ...], float] = field(default_factory=dict)
    pending: _Press | None = None
    screens: int = 0
    screen_since: float | None = None
    #: What her eyes saw standing in this screen (core/perception/what_is_in_a_place.py), for which screen, and when
    #: they are asked again.
    landmarks: list[Any] = field(default_factory=list)
    surveyed: int = -1
    survey_again_at: float | None = None
    surveying: Any = None
    surveyed_at: float = 0.0

    def new_screen(self, happened: list[dict[str, Any]], at: float) -> None:
        if any(h.get("what") == "new screen" for h in happened):
            self.screens += 1
            self.screen_since = at
            self.landmarks = []
        elif self.screen_since is None:
            self.screen_since = at

    def judged(self, moves: Any, picture: Any, at: float) -> tuple[_Press, bool] | None:
        """The press made a moment ago and whether it did something; None while it is too soon, or none was made."""
        press = self.pending
        if press is None or at - press.at < JUDGED_AFTER_S:
            return None
        self.pending = None
        spot = press.spot
        size = max(spot.w, spot.h, 1.0)
        near_it = any(t.number not in press.near and abs(t.x - spot.x) <= 3 * size and abs(t.y - spot.y) <= 3 * size
                      for t in moves.things.values())
        if spot.number is not None:
            thing = moves.things.get(spot.number)
            did = (thing is None or thing.kind != spot.kind or abs(thing.x - spot.x) + abs(thing.y - spot.y) > 0.5 * size
                   or abs(thing.w * thing.h - spot.w * spot.h) > 0.3 * max(spot.w * spot.h, 1.0))
        else:
            did = _changed_there(press, moves, picture)
        did = did or near_it or self.screens != press.screens
        pair = (press.key, spot.name)
        if did:
            self.worked.add(pair)
            self.nothing.pop(pair, None)
            if spot.landmark is not None and spot.landmark in self.landmarks:
                # What it was is not what it is now: her eyes are asked again in a moment.
                self.landmarks.remove(spot.landmark)
                self.survey_again_at = at + RESURVEY_AFTER_USE_S
        else:
            self.nothing[pair] = self.nothing.get(pair, 0) + 1
        return press, did

    def place_to_go(self, guide: Any, moves: Any, hers: Any, at: float) -> tuple[tuple[float, float], _Spot, Any] | None:
        """The nearest thing standing still that a key she has been told of is for, not yet given up on: where it is,
        the thing, and the control. None where there is none."""
        mine = hers.thing(moves)
        if guide is None or mine is None:
            return None
        best = None
        for spot in _spots(guide, moves, mine, self.landmarks):
            if spot.number is not None and moves.things[spot.number].moved:
                continue
            control = _a_use_for(guide, self, spot, at, waiting=False)
            if control is None or control.key == "the pointer":
                continue
            far = abs(spot.x - mine.x) + abs(spot.y - mine.y)
            if best is None or far < best[0]:
                best = (far, spot, control)
        return ((best[1].x, best[1].y), best[1], best[2]) if best is not None else None

    def survey(self, guide: Any, moves: Any, picture: Any, at: float) -> None:
        """Her eyes asked, beside her play, what stands in this screen that a player could go to or use: once a screen,
        a moment after it comes, and again a moment after a thing in it was used."""
        import asyncio

        controls = _the_controls_with_a_use(guide)
        if not controls or not getattr(moves, "seeing", True) or (self.surveying is not None and not self.surveying.done()):
            return
        due = (self.surveyed != self.screens and self.screen_since is not None and at - self.screen_since >= SURVEY_AFTER_S
               or self.survey_again_at is not None and at >= self.survey_again_at
               # A place changes without a new screen (a door shut again, more come): looked over again now and then,
               # sooner where nothing in it is left to use.
               or at - self.surveyed_at >= (LOOK_AGAIN_WHEN_DONE_S if not self.landmarks else LOOK_AGAIN_S))
        if not due:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        import numpy as np

        from core.perception.what_is_in_a_place import what_is_here

        frame, screen = np.array(picture)[..., :3], self.screens
        speaks_of = [c.on for c in controls if c.on]
        self.survey_again_at = None
        self.surveyed, self.surveyed_at = screen, at

        async def looked() -> None:
            seen = await what_is_here(frame, speaks_of)
            if self.screens == screen:
                self.landmarks = seen

        self.surveying = loop.create_task(looked())


#: How long a screen is looked at by play before her eyes are asked what stands in it, and how soon after a thing in
#: it was used they are asked again.
SURVEY_AFTER_S = 1.0
RESURVEY_AFTER_USE_S = 1.2
LOOK_AGAIN_WHEN_DONE_S = 6.0
LOOK_AGAIN_S = 20.0
#: How unlike, on its most different channel, a pixel must be to have changed, and how much of a place must change for
#: a press at it to have done something.
CHANGED_BY = 40
CHANGED_SHARE = 0.25


def _crop(picture: Any, box: tuple[float, float, float, float]) -> tuple[Any, tuple[int, int, int, int]] | None:
    import numpy as np

    frame = np.asarray(picture)
    if frame.ndim != 3:
        return None
    tall, wide = frame.shape[:2]
    left, top = max(0, int(box[0] * wide)), max(0, int(box[1] * tall))
    right, bottom = min(wide, int(box[2] * wide) + 1), min(tall, int(box[3] * tall) + 1)
    if right - left < 2 or bottom - top < 2:
        return None
    return frame[top:bottom, left:right, :3].astype(np.int16), (left, top, right, bottom)


def _changed_there(press: _Press, moves: Any, picture: Any) -> bool:
    """Whether the place a press was made at looks different now, leaving out where her own body was and is."""
    import numpy as np

    if press.crop is None or press.spot.landmark is None:
        return False
    now = _crop(picture, press.spot.landmark.box)
    then, (left, top, right, bottom) = press.crop
    if now is None or now[0].shape != then.shape:
        return False
    changed = np.abs(now[0] - then).max(axis=2) > CHANGED_BY
    keep = np.ones(changed.shape, bool)
    tall, wide = np.asarray(picture).shape[:2]
    for box in (press.hers, _her_box(moves, None)):
        if box is None:
            continue
        x1, y1 = int(box[0] * wide) - left, int(box[1] * tall) - top
        x2, y2 = int(box[2] * wide) + 1 - left, int(box[3] * tall) + 1 - top
        keep[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = False
    return bool(keep.sum()) and float(changed[keep].mean()) > CHANGED_SHARE


def _her_box(moves: Any, mine: Any) -> tuple[float, float, float, float] | None:
    if mine is None:
        return None
    left, top = moves.share(mine.x - mine.w / 2, mine.y - mine.h / 2)
    right, bottom = moves.share(mine.x + mine.w / 2, mine.y + mine.h / 2)
    return left, top, right, bottom


def _beside(mine: Any, spot: Any, missed: int = 0) -> bool:
    """Whether a thing is beside her: within a share of her size, edge to edge; touching her, once a press made beside
    one of its kind has done nothing."""
    reach = BESIDE * max(mine.w, mine.h, 1.0) if not missed else -0.25 * min(mine.w, spot.w)
    gap_x = abs(mine.x - spot.x) - (mine.w + spot.w) / 2
    gap_y = abs(mine.y - spot.y) - (mine.h + spot.h) / 2
    return gap_x <= reach and gap_y <= reach


def _the_controls_with_a_use(guide: Any) -> list[Any]:
    return [c for c in getattr(guide, "controls", {}).values()
            if c.act and c.act.split()[0].lower() not in _MOVING_ACTS | _FROM_AFAR | _GAME_ACTS
            and (c.on or c.key != "the pointer")]


def _known(name: str) -> tuple[str, str]:
    from core.cognition.what_things_are import what_things_are

    known = what_things_are().of(name) if name else None
    return (known.bears, known.usually) if known else ("", "")


def _spots(guide: Any, moves: Any, mine: Any, landmarks: list[Any]) -> list[_Spot]:
    """What play follows that her eyes have named, and what they saw standing in the place, but for her own."""
    seen = guide.seen
    spots: list[_Spot] = []
    for thing in moves.things.values():
        if mine is not None and (thing.number == mine.number or thing.kind == mine.kind):
            continue
        # What is on the move is met, kept clear of or struck as play has learned to (a robot walking up is punched by
        # her blows, core/agency/how_far_her_blow_reaches.py); a control is used here on what stands still.
        if math.hypot(getattr(thing, "vx", 0.0), getattr(thing, "vy", 0.0)) > max(thing.w, thing.h, 1.0):
            continue
        name = seen.as_seen(thing.kind)
        if name:
            bears, usually = _known(seen.by_kind.get(thing.kind, "") or name)
            spots.append(_Spot(name, thing.x, thing.y, thing.w, thing.h, thing.number, thing.kind, None, bears, usually,
                               thing.kind in seen.who))
    tall, wide = moves.shape
    for landmark in landmarks:
        left, top, right, bottom = landmark.box
        x, y, w, h = (left + right) / 2 * wide, (top + bottom) / 2 * tall, (right - left) * wide, (bottom - top) * tall
        # Her own figure, or a thing play already follows, named once is enough.
        if mine is not None and abs(mine.x - x) < max(w, mine.w) / 2 and abs(mine.y - y) < max(h, mine.h) / 2:
            continue
        if any(abs(s.x - x) < max(w, s.w) / 2 and abs(s.y - y) < max(h, s.h) / 2 for s in spots if s.number is not None):
            continue
        words = landmark.what.split()
        name = " ".join([*words[:-1], _one(words[-1])])                       # her eyes may say "doors" of one
        bears, usually = _known(name)
        spots.append(_Spot(name, x, y, w, h, landmark=landmark, bears=bears, usually=usually))
    return spots


def _a_use_for(guide: Any, using: UsingOnThings, spot: _Spot, at: float, *, waiting: bool = True) -> Any | None:
    if waiting and at - using.last.get(spot.mark, -AGAIN_AFTER_S) < AGAIN_AFTER_S:
        return None
    for control in _the_controls_with_a_use(guide):
        # Given up on only where it has never done anything: a door that once opened to it is worth pressing at again.
        pair = (control.key, spot.name)
        if pair not in using.worked and using.nothing.get(pair, 0) >= NOTHING_TWICE:
            continue
        if a_name_for(control.on, spot.name, spot.bears, spot.usually, control.act, spot.someone):
            return control
    return None


async def use_controls_on_things(hands: Any, run: Any, moves: Any, hers: Any, happened: list[dict[str, Any]], at: float,
                                 say: Callable[[str, str], Any] | None = None, picture: Any = None) -> bool:
    """Press the key the game names for what is done to a thing beside her (or click the thing, where the pointer is
    named for it), and judge what came of the last such press. Whether a press or click was made now."""
    from core.cognition.a_guide_to_a_place import THE_GUIDE

    guide = THE_GUIDE.get()
    using: UsingOnThings | None = getattr(run, "using", None)
    if guide is None or using is None or getattr(guide, "seen", None) is None:
        return False
    using.new_screen(happened, at)
    if picture is not None:
        using.survey(guide, moves, picture, at)
    judged = using.judged(moves, picture, at)
    if judged is not None:
        _said_what_came_of(judged, say)
    mine = hers.thing(moves)
    if using.pending is not None:
        return False
    spots = _spots(guide, moves, mine, using.landmarks)
    for spot in sorted(spots, key=lambda s: abs(s.x - mine.x) + abs(s.y - mine.y) if mine else s.x):
        control = _a_use_for(guide, using, spot, at)
        if control is None:
            continue
        pointer = control.key == "the pointer"
        # A key is pressed beside the thing; a click goes to the thing itself, with or without a body of her own.
        if not pointer and (mine is None or not _beside(mine, spot, using.nothing.get((control.key, spot.name), 0))):
            continue
        if pointer and not hasattr(hands, "click"):
            continue
        crop = _crop(picture, spot.landmark.box) if picture is not None and spot.landmark is not None else None
        if not await _used(hands, run, moves, control, spot):
            continue
        using.last[spot.mark] = at
        using.pending = _Press(control.key, spot, at, frozenset(moves.things), using.screens, crop, _her_box(moves, mine))
        if say is not None and (control.key, spot.name) not in using.worked:
            what = f" {control.on}" if control.on else ""
            how = "clicking it" if pointer else f"pressing {control.key}"
            say(f"There's {_a(spot.name)} {'there' if pointer else 'next to me'}. "
                f"{'A click' if pointer else control.key.capitalize()} is to {control.act}{what}, so I'm {how}.",
                f"using:{control.key}:{spot.name}")
        return True
    return False


def a_place_to_go(run: Any, moves: Any, hers: Any, at: float,
                  say: Callable[[str, str], Any] | None = None) -> tuple[float, float] | None:
    """Where she goes when nothing more pressing is chosen: the nearest thing standing still that a key she has been
    told of is for ("Press SPACE to open doors", and a door down the hall). Said once a kind of thing."""
    from core.cognition.a_guide_to_a_place import THE_GUIDE

    using: UsingOnThings | None = getattr(run, "using", None)
    guide = THE_GUIDE.get()
    if using is None or guide is None or getattr(guide, "seen", None) is None:
        return None
    found = using.place_to_go(guide, moves, hers, at)
    if found is None:
        return None
    where, spot, control = found
    if say is not None:
        say(f"I'll go over to the {spot.name} and press {control.key}.", f"going:{control.key}:{spot.name}")
    return where


async def _used(hands: Any, run: Any, moves: Any, control: Any, spot: _Spot) -> bool:
    try:
        if control.key == "the pointer":
            await hands.click(*moves.share(spot.x, spot.y))
            run.last_click = time.monotonic()
        elif control.key == getattr(run, "held", "") or control.key in run.letting_go:
            return False
        elif hasattr(hands, "down"):
            await hands.down(control.key)
            held = HELD_FOR_S if control.act.split()[0].lower() in _HELD_ACTS else PRESSED_FOR_S
            run.letting_go[control.key] = time.monotonic() + held
        else:
            await hands.tap(control.key)
    except (RuntimeError, OSError, ValueError, TypeError, AttributeError) as why:
        logger.debug("using %s on a thing failed: %s", control.key, why)
        return False
    run.input_key_downs[control.key] += 1
    return True


def _said_what_came_of(judged: tuple[_Press, bool], say: Callable[[str, str], Any] | None) -> None:
    press, did = judged
    name = press.spot.name
    how = "Clicking" if press.key == "the pointer" else f"Pressing {press.key}"
    logger.info("%s at the %s %s", how.lower(), name, "did something" if did else "did nothing")
    if say is None:
        return
    if did:
        say(f"{how} at the {name} did something.", f"used:{press.key}:{name}")
    elif press.key != "the pointer":
        say(f"{how} at the {name} does nothing yet; I'll try it again closer.", f"unused:{press.key}:{name}")


def _a(name: str) -> str:
    if name[:1].isupper():
        return name
    return ("an " if name[:1].lower() in "aeiou" else "a ") + name

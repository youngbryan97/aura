"""What she notices while she plays: what follows what she does, what things do after she meets them, what is new.

Four kinds of noticing, each said for any place:

- after: a gain or a loss that comes within a moment of one of her acts more
  often than gains or losses come at all ("after I press space, I tend to
  score");
- what a thing does when met: one that stops for a moment after she hits it,
  every time;
- what is new: a kind of thing that turns up after play has gone on a while
  without it ("that's new: a gold star showed up");
- what goes on by itself: a count that goes up with nothing she did beside it
  ("my score goes up on its own").

Each is put in her notebook (core/cognition/what_she_notices.py), where most are
let go and some become theories; a theory that became one is said once.

Nothing here knows a game.
"""
from __future__ import annotations

import math
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

__all__ = ["NoticingInPlay", "a_key_that_pays", "noticed"]

#: How soon after an act an outcome is put down to it, in seconds; and how many of an act before it is judged.
FOLLOWS_WITHIN_S = 1.5
JUDGED_AFTER = 4
#: How much more often than by chance (the share of play an outcome's moment covers) an outcome must follow an act to be
#: noticed, and how many times at least.
MORE_THAN_CHANCE = 1.5
FOLLOWED_AT_LEAST = 3
#: Where acts of hers are followed by an outcome more often than this whatever they are, no one act is noticed for it:
#: the outcome comes on its own.
MOST_FOLLOWED = 0.6
#: How often a theory, as it becomes one, is said, in seconds.
SAY_A_THEORY_EVERY_S = 30.0
#: How long after meeting a thing its stopping is looked for, how slow it must go to have stopped, as a share of how
#: fast it went before, and how fast it must have gone to be said to stop.
STOPS_WITHIN_S = (0.2, 1.0)
STOPPED_SHARE = 0.2
WENT_AT_LEAST = 15.0
#: How long play must have gone on for a kind turning up to be new.
NEW_AFTER_S = 15.0


@dataclass
class NoticingInPlay:
    """What she has done, what came of it, and what she has met, watched picture by picture for what to notice."""

    acts: deque = field(default_factory=lambda: deque(maxlen=400))
    pressed: Counter = field(default_factory=Counter)
    followed: Counter = field(default_factory=Counter)
    outcomes: int = 0
    began: float | None = None
    _keys_down: Counter = field(default_factory=Counter)
    _held: str = ""
    _clicked_at: float = -math.inf
    _verdicts_read: int = 0
    _met_read: dict[int, float] = field(default_factory=dict)
    _watching: dict[int, tuple[int, float, float]] = field(default_factory=dict)
    met: Counter = field(default_factory=Counter)
    stopped: Counter = field(default_factory=Counter)
    kinds_seen: set[int] = field(default_factory=set)
    followed_any: Counter = field(default_factory=Counter)
    _credited: set = field(default_factory=set)
    _said_at: float = -math.inf

    def saw(self, run: Any, moves: Any, meeting: Any, at: float, notebook: Any, name: Any, say: Any = None) -> None:
        """One picture: what she did since the last, what came of it, what she met, what turned up. ``name(kind,
        thing, part)`` says what the place calls a thing; ``say`` is told of a theory as it becomes one."""
        if notebook is None:
            return
        self.began = at if self.began is None else self.began
        self._her_acts(run, at)
        for verdict in list(getattr(meeting, "verdicts", []))[self._verdicts_read:]:
            self._came_of(verdict, at, notebook, say)
        self._verdicts_read = len(getattr(meeting, "verdicts", []))
        self._met_things(moves, meeting, at, notebook, name, say)
        self._what_is_new(moves, at, notebook, name)

    # -- what she did, and what came of it ----------------------------------------------------------------------------

    def _her_acts(self, run: Any, at: float) -> None:
        held = str(getattr(run, "held", "") or "")
        if held and held != self._held:
            self._act(f"hold {held}", at)
        self._held = held
        for key, times in dict(getattr(run, "input_key_downs", {}) or {}).items():
            if times > self._keys_down[key]:
                self._act(f"press {key}", at)
            self._keys_down[key] = times
        clicked = float(getattr(run, "last_click", -math.inf))
        if clicked > self._clicked_at:
            self._act("click", at)
        self._clicked_at = clicked

    def _act(self, act: str, at: float) -> None:
        self.acts.append((at, act))
        self.pressed[act] += 1

    def _came_of(self, verdict: dict[str, Any], at: float, notebook: Any, say: Any) -> None:
        what = "gain" if verdict.get("what") == "gain" else "lose"
        when = float(verdict.get("at", at))
        # A count by its own name, not where on the screen it was read ("the score at 0.46").
        counter = re.sub(r"\s+at\s+[0-9.]+(?:\s*,\s*[0-9.]+)?$", "", str(verdict.get("counter") or "").strip())
        self.outcomes += 1
        recent = [(when_done, act) for when_done, act in self.acts if 0.0 <= when - when_done <= FOLLOWS_WITHIN_S]
        # A bar's level is what she has left or what fills, read by how bars are read (core/agency/what_she_has_left.py);
        # its flicker is no count going up by itself: LIVE 2026-10-10 "the bar at the top goes down: it's what I have
        # left" was followed by "the bar at the top goes up with nothing I did beside it".
        if not recent and what == "gain" and counter and not re.search(r"\bbar\b", counter, re.I):
            whose = counter if counter.lower().startswith(("the ", "my ")) else f"my {counter}"
            self._said(notebook.notice(f"by itself {counter}", f"{whose} goes up with nothing I did beside it", at,
                                       kind="by itself", about=counter, pays=1), say, at)
        # Each act done is followed by a kind of outcome once, however many come after it.
        for done_at, act in recent:
            if (done_at, act, what) not in self._credited:
                self._credited.add((done_at, act, what))
                self.followed[(act, what)] += 1
                self.followed_any[what] += 1
        if len(self._credited) > 2000:
            self._credited = {c for c in self._credited if at - c[0] <= 10 * FOLLOWS_WITHIN_S}
        # An act is noticed for being followed more often than her other acts are, or, where she has done little else,
        # than the outcome comes at all in the time.
        span = max(1.0, at - (self.began or at))
        by_time = min(1.0, self.outcomes / span * FOLLOWS_WITHIN_S)
        for act in {a for a, _w in self.followed} | {a for _t, a in recent}:
            done = self.pressed[act]
            if done < JUDGED_AFTER:
                continue
            others_done = sum(self.pressed.values()) - done
            chance = (min(1.0, (self.followed_any[what] - self.followed[(act, what)]) / others_done)
                      if others_done >= JUDGED_AFTER else by_time)
            share = min(1.0, self.followed[(act, what)] / done)
            held = (share >= MORE_THAN_CHANCE * chance and chance <= MOST_FOLLOWED and share >= 0.3
                    and self.followed[(act, what)] >= FOLLOWED_AT_LEAST)
            self._said(notebook.notice(f"after {act} {what}", f"after I {act}, I tend to {what}"
                                       + (f" ({counter})" if counter else ""), at, kind="after", held=held,
                                       about=act.split(" ", 1)[-1] if act != "click" else "click",
                                       pays=1 if what == "gain" else -1, bears=True), say, at)

    # -- what things do when met ---------------------------------------------------------------------------------------

    def _met_things(self, moves: Any, meeting: Any, at: float, notebook: Any, name: Any, say: Any) -> None:
        things = getattr(moves, "things", {}) or {}
        for number, when in dict(getattr(meeting, "_met", {}) or {}).items():
            if self._met_read.get(number) == when:
                continue
            self._met_read[number] = when
            thing = things.get(number)
            if thing is None:
                continue
            going = math.hypot(float(thing.vx), float(thing.vy))
            if going >= WENT_AT_LEAST:
                self._watching[number] = (int(thing.kind), when, going)
                self.met[int(thing.kind)] += 1
        for number, (kind, when, going) in list(self._watching.items()):
            if at - when > STOPS_WITHIN_S[1]:
                del self._watching[number]
                continue
            thing = things.get(number)
            if thing is None or at - when < STOPS_WITHIN_S[0]:
                continue
            if math.hypot(float(thing.vx), float(thing.vy)) <= STOPPED_SHARE * going:
                del self._watching[number]
                self.stopped[kind] += 1
                called = name(kind, thing, "") if name is not None else "thing"
                self._said(notebook.notice(f"stops when met {kind}", f"the {called} stops for a moment after I meet it",
                                           at, kind="when met", about=str(kind), bears=True), say, at)

    # -- what is new -----------------------------------------------------------------------------------------------------

    def _what_is_new(self, moves: Any, at: float, notebook: Any, name: Any) -> None:
        for thing in list((getattr(moves, "things", {}) or {}).values()):
            kind = int(thing.kind)
            # What goes about: a still thing turning up is the screen being drawn afresh as often as anything new.
            if kind in self.kinds_seen or not getattr(thing, "moved", False):
                continue
            self.kinds_seen.add(kind)
            if at - (self.began or at) >= NEW_AFTER_S:
                called = name(kind, thing, "") if name is not None else "thing"
                notebook.notice(f"new {called}", f"that's new: a {called} showed up", at, kind="new", about=str(kind))

    def _said(self, became: Any, say: Any, at: float) -> None:
        """A theory as it becomes one, said, one in a while: the rest are in her log and her reasoning."""
        if became is None or say is None or at - self._said_at < SAY_A_THEORY_EVERY_S:
            return
        self._said_at = at
        say(f"I think I've noticed something: {became.said}.")


def noticed(run: Any, moves: Any, meeting: Any, at: float, say: Any, named: Any) -> None:
    """One picture of play watched for what to notice, into the notebook of the guide to where she is, with things
    called by the place's names for the part each plays (``named(moves, kind, thing, part)``)."""
    from core.agency.what_meeting_things_does import AVOID, MEET, SHOOT
    from core.cognition.a_guide_to_a_place import THE_GUIDE

    guide = THE_GUIDE.get()
    if guide is None:
        return
    parts = {MEET: "get", AVOID: "avoid", SHOOT: "shoot"}

    def name(kind: int, thing: Any, _part: str) -> str:
        return named(moves, kind, thing, parts.get(meeting.stance(kind), ""))

    run.noticing.saw(run, moves, meeting, at, guide.notes, name, say)


def a_key_that_pays(keys: Any) -> str:
    """A key of hers she has come to think pays, from the notes of the guide to where she is; "" where none."""
    from core.cognition.a_guide_to_a_place import THE_GUIDE

    guide = THE_GUIDE.get()
    return next((key for key in keys if guide is not None and guide.notes.paying(key) > 0), "")

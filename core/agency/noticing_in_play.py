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
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

__all__ = ["NoticingInPlay", "noticed"]

#: How soon after an act an outcome is put down to it, in seconds; and how many of an act before it is judged.
FOLLOWS_WITHIN_S = 1.5
JUDGED_AFTER = 4
#: How much more often than by chance (the share of play an outcome's moment covers) an outcome must follow an act to be
#: noticed, and how many times at least.
MORE_THAN_CHANCE = 1.5
FOLLOWED_AT_LEAST = 3
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
        counter = str(verdict.get("counter") or "").strip()
        self.outcomes += 1
        recent = {act for when_done, act in self.acts if 0.0 <= when - when_done <= FOLLOWS_WITHIN_S}
        if not recent and what == "gain" and counter:
            self._said(notebook.notice(f"by itself {counter}", f"my {counter} goes up with nothing I did beside it", at,
                                       kind="by itself", about=counter, pays=1), say)
        for act in recent:
            self.followed[(act, what)] += 1
        span = max(1.0, at - (self.began or at))
        chance = min(1.0, self.outcomes / span * FOLLOWS_WITHIN_S)
        for act in {a for a, _w in self.followed} | recent:
            done = self.pressed[act]
            if done < JUDGED_AFTER:
                continue
            share = self.followed[(act, what)] / done
            held = share >= MORE_THAN_CHANCE * chance and share >= 0.3 and self.followed[(act, what)] >= FOLLOWED_AT_LEAST
            self._said(notebook.notice(f"after {act} {what}", f"after I {act}, I tend to {what}"
                                       + (f" ({counter})" if counter else ""), at, kind="after", held=held,
                                       about=act.split(" ", 1)[-1] if act != "click" else "click",
                                       pays=1 if what == "gain" else -1, bears=True), say)

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
                                           at, kind="when met", about=str(kind), bears=True), say)

    # -- what is new -----------------------------------------------------------------------------------------------------

    def _what_is_new(self, moves: Any, at: float, notebook: Any, name: Any) -> None:
        for thing in list((getattr(moves, "things", {}) or {}).values()):
            kind = int(thing.kind)
            if kind in self.kinds_seen:
                continue
            self.kinds_seen.add(kind)
            if at - (self.began or at) >= NEW_AFTER_S:
                called = name(kind, thing, "") if name is not None else "thing"
                notebook.notice(f"new {kind}", f"that's new: a {called} showed up", at, kind="new", about=str(kind))

    @staticmethod
    def _said(became: Any, say: Any) -> None:
        if became is not None and say is not None:
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

"""Which way a place asks to be played, and holding to it while it pays.

A person sat down at a game reads what it says, takes up the way it says it is
played, and plays that way: they throw where it says throw, they steer where it
says steer, again and again, watching whether it is going anywhere. They do not
press keys to see what they do in a game that said it is played with the mouse,
and they do not stop to look round between throws. Only when the way they took
up is getting them nowhere do they try something else.

LIVE 2026-10-09 a game said "click and hold the mouse button to aim, then
release to shoot". She threw for one round and scored, and in the next its
player's idle bobbing read as a world moving on its own: she steered him with
the mouse, never threw, and spent three minutes pressing arrows and clicking
shapes "to see what it does".

So the way is chosen from what the place says first (its words, and what taking
stock found out), from what has paid since, and only then from what the world
does by itself. Held to, it is played stretch after stretch for as long as it is
being learned or is paying: something counted went up, a screen not seen before
came up, a counter of how far she has got (a level, a distance) rose, a thing
sent went home, or the counters settle it won. A way that has had its turn and
paid nothing is let go, and not taken up again until the place says something
new.

Holding her own while steering what is hers shows she has the controls, not that
she is getting anywhere: a stretch of it, few losses and nothing gained, pays
only where the place's words make lasting the task ("survive", "as long as you
can").

Nothing here knows a game. The ways are the general ones in
core/agency/ways_of_playing.py.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["AS_IT_HAPPENS", "FOLLOW", "SEND", "TYPE", "HeldTo", "held_her_own", "progress", "ways_asked"]

#: The ways the reflexes play: typing what is asked, doing again what was shown, sending by a press held or pulled and
#: let go, and playing what moves as it happens (steering, shooting, catching), by keys or the pointer.
TYPE, FOLLOW, SEND, AS_IT_HAPPENS = "type", "copy a sequence", "send", "as it happens"

#: Stretches a way is given to be learned before it must pay, and how many of its latest are looked at for pay.
LEARNING = 3
PAYING_OVER = 3

#: How long she must have steered what is hers in a stretch for it to count as holding her own, in seconds.
HELD_HER_OWN_S = 8.0

#: Counters of how far she has got rather than how well: one that rose since the stretch before is progress.
_HOW_FAR = re.compile(r"\b(level|lvl|stage|wave|round|distance|lap|floor|zone|area|world|miles?|met(?:er|re)s?|km)\b", re.I)
#: Words that make lasting the task, where holding her own is getting somewhere. Words that only ask her to keep clear
#: of things ("avoid the bombs") do not: there something else is the task.
_LASTING_IS_THE_TASK = re.compile(
    r"\b(survive|surviving|stay alive|keep alive|as long as (?:you can|possible)|last (?:as long|until|till)|"
    r"hold (?:out|on) (?:as long|until|till|for))\b", re.I)


def ways_asked(read: str, last_screen: str = "") -> list[str]:
    """The ways the place's words ask to be played, most particular first; then the ways it looks to be played by.

    ``read`` is everything the place has said and what taking stock found; ``last_screen`` the screen in front of her.
    What the place says comes first; after it, what the reading of the place says it is done by (a slingshot and
    hamsters are launched, a court is played on, core/cognition/what_this_place_is.py), which is a reading, not its
    word.
    """
    from core.agency.doing_again_what_was_shown import asks_to_follow
    from core.agency.playing_as_it_happens import controls_named_in
    from core.agency.playing_by_shots import sends_by_letting_go
    from core.language.words_asked_for import words_asked_for

    asked: list[str] = []
    if last_screen and words_asked_for(last_screen) is not None:
        asked.append(TYPE)
    if asks_to_follow(read):
        asked.append(FOLLOW)
    if sends_by_letting_go(read):
        asked.append(SEND)
    keys, pointer = controls_named_in(read, keys_without_words=(), during_play=True)
    if keys or pointer:
        asked.append(AS_IT_HAPPENS)
    from core.cognition.a_guide_to_a_place import THE_GUIDE
    from core.cognition.what_this_place_is import ways_it_is_played_by

    guide = THE_GUIDE.get()
    looks = ways_it_is_played_by(guide) if guide is not None else []
    return asked + [way for way in looks if way not in asked and way != TYPE]


def held_her_own(stretch: dict[str, Any]) -> bool:
    """Whether she steered what is hers through a stretch and lost little: she has the controls, whatever came of it."""
    held = stretch.get("hers") and float(stretch.get("seconds") or 0.0) >= HELD_HER_OWN_S
    return bool(held and int(stretch.get("losses") or 0) <= int(stretch.get("gains") or 0) + 1)


def _got_further(stretch: dict[str, Any], before: dict[str, Any] | None) -> bool:
    """Whether a counter of how far she has got (a level, a stage, a distance) rose since the stretch before."""
    if not before:
        return False
    then = before.get("counters") or {}
    return any(_HOW_FAR.search(str(name)) and name in then and int(value) > int(then[name])
               for name, value in (stretch.get("counters") or {}).items())


def progress(stretch: dict[str, Any], before: dict[str, Any] | None = None, task: str = "") -> str:
    """What a stretch shows of getting on with the task, named; "" where it shows none.

    ``before`` is the stretch played before it, for counters of how far she has got; ``task`` what the place says its
    task is, for whether lasting is it. Holding her own counts only there: anywhere else it shows she has the controls.
    """
    if int(stretch.get("gains") or 0) > 0:
        return "counted"
    if int(stretch.get("new_screens") or 0) > 0:
        return "a new screen"
    if int((stretch.get("by_shots") or {}).get("gains") or 0) > 0:
        return "sent home"
    if stretch.get("typed"):
        return "typed"
    if stretch.get("followed"):
        return "followed"
    if stretch.get("settled") == "won":
        return "won"
    if _got_further(stretch, before):
        return "got further"
    if task and _LASTING_IS_THE_TASK.search(task) and held_her_own(stretch):
        return "lasted"
    return ""


def _the_task(read: str) -> str:
    """What the place says its task is: its words, and the goals the guide to it holds."""
    from core.cognition.a_guide_to_a_place import THE_GUIDE

    guide = THE_GUIDE.get()
    return " ".join([read, *(guide.goals if guide is not None else [])])


@dataclass
class HeldTo:
    """The way she has taken up here, the stretches it has had, and the ways let go."""

    way: str = ""
    since: float = 0.0
    stretches: list[dict[str, Any]] = field(default_factory=list)
    let_go: set[str] = field(default_factory=set)
    #: What the place had said when each way was let go: a way is taken up again once it says something new.
    let_go_on: dict[str, str] = field(default_factory=dict)
    #: What the place says its task is, as last heard: lasting counts as progress only where it is the task.
    task: str = ""

    def choose(self, asked: list[str], read: str) -> str:
        """The way to play now: the one held to while it may still pay, else the first asked and not let go."""
        self.task = _the_task(read)
        for way in list(self.let_go):
            if self.let_go_on.get(way) != read[-400:]:
                self.let_go.discard(way)
        if self.way and self.way not in self.let_go and (self.way in asked or self.paying()):
            return self.way
        for way in asked:
            if way not in self.let_go:
                if way != self.way:
                    self.way, self.since, self.stretches = way, time.monotonic(), []
                return way
        return ""

    def took(self, way: str, stretch: dict[str, Any], read: str) -> None:
        """A stretch played the way she holds to; let it go where it has had its turn and paid nothing."""
        if way != self.way:
            return
        self.task = _the_task(read)
        self.stretches.append(stretch)
        if not self.paying():
            self.let_go.add(way)
            self.let_go_on[way] = read[-400:]
            self.way = ""

    def paying(self) -> bool:
        """Whether the way held to is still being learned, or has shown progress in its latest stretches."""
        if not self.way:
            return False
        if len(self.stretches) < LEARNING:
            return True
        latest = len(self.stretches) - PAYING_OVER
        return any(progress(stretch, self.stretches[at - 1] if at > 0 else None, self.task)
                   for at, stretch in enumerate(self.stretches) if at >= latest)

    def holding(self) -> bool:
        """Whether a way is held to and may still pay: play goes on with it, without stopping to look round."""
        return bool(self.way) and self.way not in self.let_go and self.paying()

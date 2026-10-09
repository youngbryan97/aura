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
came up, a thing sent went somewhere, or she held her own steering what is hers.
A way that has had its turn and paid nothing is let go, and not taken up again
until the place says something new.

Nothing here knows a game. The ways are the general ones in
core/agency/ways_of_playing.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["AS_IT_HAPPENS", "FOLLOW", "SEND", "TYPE", "HeldTo", "ways_asked"]

#: The ways the reflexes play: typing what is asked, doing again what was shown, sending by a press held or pulled and
#: let go, and playing what moves as it happens (steering, shooting, catching), by keys or the pointer.
TYPE, FOLLOW, SEND, AS_IT_HAPPENS = "type", "copy a sequence", "send", "as it happens"

#: Stretches a way is given to be learned before it must pay, and how many of its latest are looked at for pay.
LEARNING = 3
PAYING_OVER = 3

#: How long she must have steered what is hers in a stretch for it to count as holding her own, in seconds.
HELD_HER_OWN_S = 8.0


def ways_asked(read: str, last_screen: str = "") -> list[str]:
    """The ways the place's words ask to be played, most particular first.

    ``read`` is everything the place has said and what taking stock found; ``last_screen`` the screen in front of her.
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
    return asked


def _paid(stretch: dict[str, Any]) -> bool:
    """Whether a stretch of play got anywhere: counted, reached, sent, followed, typed, or held her own."""
    if int(stretch.get("gains") or 0) > 0 or int(stretch.get("new_screens") or 0) > 0:
        return True
    shots = stretch.get("by_shots") or {}
    if int(shots.get("gains") or 0) > 0:
        return True
    if stretch.get("typed") or stretch.get("followed"):
        return True
    held_own = stretch.get("hers") and float(stretch.get("seconds") or 0.0) >= HELD_HER_OWN_S
    return bool(held_own and int(stretch.get("losses") or 0) <= int(stretch.get("gains") or 0) + 1)


@dataclass
class HeldTo:
    """The way she has taken up here, the stretches it has had, and the ways let go."""

    way: str = ""
    since: float = 0.0
    stretches: list[dict[str, Any]] = field(default_factory=list)
    let_go: set[str] = field(default_factory=set)
    #: What the place had said when each way was let go: a way is taken up again once it says something new.
    let_go_on: dict[str, str] = field(default_factory=dict)

    def choose(self, asked: list[str], read: str) -> str:
        """The way to play now: the one held to while it may still pay, else the first asked and not let go."""
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
        self.stretches.append(stretch)
        if not self.paying():
            self.let_go.add(way)
            self.let_go_on[way] = read[-400:]
            self.way = ""

    def paying(self) -> bool:
        """Whether the way held to is still being learned, or has paid in its latest stretches."""
        if not self.way:
            return False
        if len(self.stretches) < LEARNING:
            return True
        return any(_paid(stretch) for stretch in self.stretches[-PAYING_OVER:])

    def holding(self) -> bool:
        """Whether a way is held to and may still pay: play goes on with it, without stopping to look round."""
        return bool(self.way) and self.way not in self.let_go and self.paying()

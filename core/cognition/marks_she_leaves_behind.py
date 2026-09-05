"""Making the world remember, so she does not have to.

Everybody who plays Minecraft learns the same lesson the same way: they walk
off in a straight line, turn round, and cannot get home. What they do about it
is not to remember harder. They put a torch down every fifty blocks, or swap a
dirt block for a stone one every ten steps, and after that the way back is not
in their head at all — it is on the ground, and it stays there while they are
doing something else.

The other half is the bed. Dying sends you back to where you last slept, so
sleeping somewhere is choosing where failure will put you. Nobody thinks of
that as memory and it is exactly memory: a fact about the future, placed
deliberately, in the world rather than in the head.

She has memory and it is all inside her. So a route she worked out is a thing
she has to keep holding, competing with everything else she is holding, and
lost with the process — while a mark on the ground costs nothing to keep, is
found again by walking into it, and works just as well when she has forgotten
everything about why she left it.

Three things, and none of them clever. Leave a mark where a thing was, so a
place can be recognised rather than recalled. Follow them back the way they
were laid. And say where she should restart from, so that when something goes
wrong the ground she loses is ground she chose to risk.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["MarksOnTheGround"]


@dataclass
class MarksOnTheGround:
    """Marks she has left, in the order she left them."""

    #: The places themselves, as she was given them — not written down as
    #: text. A mark handed back as the repr of a place is a mark the caller
    #: has to unpick before it can be walked to.
    trail: list[tuple[Any, str]] = field(default_factory=list)
    #: Where she has said she would rather come back to.
    _back_to: Any = None
    _said_back_to: bool = False
    #: What each mark was for, if she said.
    said: dict[str, str] = field(default_factory=dict)

    def she_marked(self, where: Hashable, *, saying: str = "") -> None:
        """Leave a mark here.

        Marking the same place twice is one mark. A trail is about places and
        not about how many times she walked over them, and a trail that grows
        every time she paces is not a trail.
        """
        if self.trail and self.trail[-1][0] == where:
            return
        self.trail.append((where, saying))
        if saying:
            self.said[repr(where)] = saying

    def has_been_here(self, where: Hashable) -> bool:
        """Whether she left a mark here — recognised rather than recalled."""
        return any(one == where for one, _ in self.trail)

    def what_she_said_here(self, where: Hashable) -> str:
        return self.said.get(repr(where), "")

    def the_way_back(self, *, to: Any = None) -> tuple[Any, ...]:
        """The marks between here and there, newest first.

        Back the way they were laid, because that is the way that is known to
        work. A shorter way may exist and she has not walked it.
        """
        want = to if to is not None else (self._back_to if self._said_back_to else None)
        places = [one for one, _ in self.trail]
        if want is None:
            return tuple(reversed(places))
        if want not in places:
            return ()
        at = len(places) - 1 - places[::-1].index(want)
        return tuple(reversed(places[at:]))

    def come_back_to(self, where: Hashable) -> None:
        """Say where she would rather start again from.

        Failure has to put her somewhere. Saying where means the ground she
        loses is ground she chose to risk, and not everything she has done.
        """
        self._back_to = where
        self._said_back_to = True

    @property
    def starts_again_at(self) -> Any:
        return self._back_to

    def how_far_from_safety(self, *, now: Hashable) -> int:
        """How many marks between here and where she would restart.

        The number that says whether to press on or go back, and it is a
        measurement rather than a nerve.
        """
        way = self.the_way_back()
        if not way or not self._said_back_to:
            return 0
        here = now
        places = [one for one, _ in self.trail]
        if here not in places or self._back_to not in places:
            return 0
        return abs(places.index(here) - places.index(self._back_to))

    def describe(self) -> str:
        if not self.trail:
            return "nothing left behind yet"
        said = f"{len(self.trail)} mark(s)"
        return f"{said}, coming back to {self._back_to!r}" if self._said_back_to else said

    def as_memory(self) -> dict[str, Any]:
        return {
            "trail": [[str(place), what] for place, what in self.trail],
            "back_to": str(self._back_to) if self._said_back_to else "",
        }

    @classmethod
    def from_memory(cls, held: Any) -> MarksOnTheGround:
        if not isinstance(held, dict):
            return cls()
        trail: list[tuple[str, str]] = []
        for one in held.get("trail") or ():
            try:
                trail.append((str(one[0]), str(one[1])))
            except (TypeError, IndexError):
                # not a failure: a mark without a place is not a mark.
                continue
        back = str(held.get("back_to") or "")
        got = cls(trail=trail, _back_to=back, _said_back_to=bool(back))
        got.said = {repr(place): what for place, what in trail if what}
        return got


def worth_marking(seen: Sequence[Hashable], marks: MarksOnTheGround) -> tuple[Hashable, ...]:
    """Places she has not marked and has now been to.

    Nothing decides for her whether a place is worth a mark; this only says
    which are unmarked, because the ones she has already marked are the ones
    she will recognise.
    """
    return tuple(one for one in seen if not marks.has_been_here(one))

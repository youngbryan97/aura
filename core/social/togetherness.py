"""A "we" both sides are saying, and the "they" that draws its edge.

A line about all of them coming from the same runaway place lands twelve times
over a record that is 37 % first person plural and 25 % third person plural and asks nothing. The belonging in it
is stated as a shared origin, and the boundary of the group is carried by the
other pronoun: a we is a we because there is a they.

She measured the other person's register and never her own, so nothing could
tell an exchange where both of them had started saying "we" from one where only
one had. Belonging moved on safety and on nothing that was said.

    together   min(her plural share, their plural share): a we both are saying
    edge       their third person share while there is a we to have an edge
    belonging  floored at together, the way safety floors it

The minimum is the reading because a we one side says and the other does not is
not shared. The edge is kept as its own reading and does not raise belonging:
naming who is outside the group is how the group is drawn, and turning that
into warmth would reward it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["Togetherness", "last_said", "read_togetherness"]


@dataclass(frozen=True)
class Togetherness:
    """How much of a we the exchange is carrying, and what draws its edge."""

    together: float = 0.0
    edge: float = 0.0
    hers: float = 0.0
    theirs: float = 0.0
    measured: bool = False
    why: str = "neither of them has said anything with a person in it yet"

    def as_dict(self) -> dict[str, Any]:
        return {
            "together": round(self.together, 6),
            "edge": round(self.edge, 6),
            "hers": round(self.hers, 6),
            "theirs": round(self.theirs, 6),
            "measured": self.measured,
            "why": self.why,
        }


def last_said(messages: Sequence[Any], roles: tuple[str, ...]) -> str:
    """The newest non-empty message from one of these roles, or ""."""
    return _last(messages, roles)


def _last(messages: Sequence[Any], roles: tuple[str, ...]) -> str:
    for entry in reversed(list(messages or ())):
        if isinstance(entry, Mapping) and str(entry.get("role", "")).strip().lower() in roles:
            text = str(entry.get("content", "") or "")
            if text.strip():
                return text
    return ""


def read_togetherness(messages: Sequence[Any]) -> Togetherness:
    """Read the we off her last turn and theirs."""
    from core.expression.register import comparable, read

    mine = read(_last(messages, ("assistant", "aura")))
    theirs = read(_last(messages, ("user",)))
    if not comparable(mine, theirs):
        return Togetherness(why="one of them has not said enough to have a shape")
    if not (mine.placed and theirs.placed):
        return Togetherness(
            measured=True,
            why="one of them said nothing with a person in it",
        )
    together = min(mine.plural, theirs.plural)
    edge = theirs.third if together > 0.0 else 0.0
    if together > 0.0:
        why = f"both saying we, {together:.2f} of the placed pronouns on the smaller side"
    elif theirs.plural > 0.0:
        why = "they are saying we and she is not"
    elif mine.plural > 0.0:
        why = "she is saying we and they are not"
    else:
        why = "neither of them is saying we"
    return Togetherness(
        together=together,
        edge=edge,
        hers=mine.plural,
        theirs=theirs.plural,
        measured=True,
        why=why,
    )

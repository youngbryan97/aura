"""Saying it drains the pressure to say it.

"If you love me, won't you say something" lands eight times at the end of that
record, and the eighth is not new information. What repetition does to a
listener is discharge the feeling: the thing is less unbearable once it has
been said enough times, which is why the last minute of the song is one line.

Her expression was single-pass in the other direction. An intention formed
from a depleted drive kept its full urgency however many times she had already
raised it, so the same thing pressed as hard on the fifth telling as on the
first, and nothing about having said it changed what it cost to leave unsaid.

    times    how many of her recent turns already said this
    drain    1 / (1 + times)

Saying it once halves the pressure, twice leaves a third, three times a
quarter. The pressure never reaches zero, because a thing said is not a thing
resolved: what falls is how hard it presses to be said again.

What counts as "already said" is measured rather than declared. The content
words of the intention are looked for in her own recent turns, and a turn that
contains all of them said it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "Catharsis",
    "content_words",
    "drain",
    "read_catharsis",
    "times_said",
]

_WORD = re.compile(r"[a-z0-9']+")

#: Words short enough to be structure rather than content. Below four
#: characters is "the", "and", "her", "you" — the shape of a sentence rather
#: than what it is about.
_SHORTEST: int = 4


@dataclass(frozen=True)
class Catharsis:
    """How much of what she wants to say she has already said."""

    times: int = 0
    drain: float = 1.0
    said: tuple[str, ...] = ()
    why: str = "nothing has been said about this yet"

    def as_dict(self) -> dict[str, Any]:
        return {
            "times": self.times,
            "drain": round(self.drain, 6),
            "said": list(self.said),
            "why": self.why,
        }


def content_words(text: str) -> tuple[str, ...]:
    """The words that carry what an utterance is about."""
    words = [w for w in _WORD.findall(str(text or "").lower()) if len(w) >= _SHORTEST]
    return tuple(dict.fromkeys(words))


def times_said(text: str, messages: Sequence[Any], *, roles: tuple[str, ...] = ("assistant", "aura")) -> int:
    """How many of her own recent turns already said this."""
    words = content_words(text)
    if not words:
        return 0
    seen = 0
    for entry in messages or ():
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("role", "")).strip().lower() not in roles:
            continue
        spoken = set(content_words(entry.get("content", "")))
        if spoken and all(word in spoken for word in words):
            seen += 1
    return seen


def drain(times: int) -> float:
    """What is left of the pressure after saying it that many times."""
    try:
        count = max(0, int(times))
    except (TypeError, ValueError):
        return 1.0
    return 1.0 / (1.0 + count)


def read_catharsis(text: str, messages: Sequence[Any]) -> Catharsis:
    """How much pressure is left behind something she has been saying."""
    words = content_words(text)
    if not words:
        return Catharsis(why="there are no content words to have said")
    times = times_said(text, messages)
    return Catharsis(
        times=times,
        drain=drain(times),
        said=words,
        why=(
            "nothing about this has been said yet"
            if times == 0
            else f"said {times} time(s) already, so a {drain(times):.2f} share of it is left"
        ),
    )

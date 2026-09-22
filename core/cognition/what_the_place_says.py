"""What a place tells her, and how that bears on what she was asked.

Somewhere you have not been before usually tells you what it is for. A form
says which fields it needs. A wizard says what finishing looks like. An error
says what would make it go away. An empty state says what to make first. A
board says to join the numbers and get to the 2048 tile. None of that is
decoration: it is a statement about the task, made by the thing that would
know, and it is ordinarily better than anything you could infer from the
shape of the buttons.

Reading it is not the whole of it. The reading has to be RELATED to what she
was asked, because the two are rarely the same sentence and rarely disagree
either. Somebody says "play it" and the place says what winning is: the place
supplies the part the request left out. Somebody says "get me to 256" and the
place says 2048: hers is a waypoint on the way to the place's, and hers is
still the finish. Somebody says "close this" and the place says to subscribe:
they are about different things and the person's wins.

That relation is the understanding. A thing that read the words and acted on
them without being able to say how they bore on the task would be following
instructions off a screen, which is a different and worse thing — it is how
an agent ends up doing what a page told it to instead of what its person
asked for. So the relation is named, it is sayable, and where the two conflict
the person is the one who meant it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ENOUGH_WORDS_TO_BE_SAYING_SOMETHING",
    "WhatThePlaceSays",
    "sentences_in",
    "what_this_place_tells_her",
]

logger = logging.getLogger("Aura.WhatThePlaceSays")

#: How many words a line needs before it could be a sentence rather than a
#: label. Four, because that is the fewest that can carry a verb, a thing, and
#: a relation between them — "get to the tile" — and three is a heading.
ENOUGH_WORDS_TO_BE_SAYING_SOMETHING = 4

#: How the thing's own statement bears on what she was asked to do.
SUPPLIES = "supplies the finish I was not given"
AGREES = "says the same thing I was asked for"
ON_THE_WAY = "is further than what I was asked for, so mine is on the way to it"
DISAGREES = "is about something else, so I am going by what I was asked"
SAYS_NOTHING = "does not say what it is for"


@dataclass(frozen=True)
class WhatThePlaceSays:
    """What the thing in front of her states, and what she does about it."""

    #: The sentence she read it out of, verbatim.
    said: str = ""
    #: The objective it states, in something measurable. Empty when it states
    #: nothing that can be measured.
    states: str = ""
    #: How that bears on what she was asked.
    bearing: str = SAYS_NOTHING
    #: What she is therefore playing for.
    aim: str = ""

    @property
    def worth_saying(self) -> bool:
        """Whether this changed anything she would otherwise have done."""
        return bool(self.states) and self.bearing in {SUPPLIES, ON_THE_WAY}

    def said_out_loud(self) -> str:
        """The relation, in a sentence, because understanding is sayable."""
        if not self.states:
            return ""
        return f"It says {self.said.strip()!r} — that {self.bearing}."


def sentences_in(words: Any) -> list[str]:
    """The lines of a reading that are prose rather than furniture.

    A screen is mostly labels and only sometimes a sentence, and telling them
    apart is what makes reading one possible at all. By shape rather than by
    vocabulary: a sentence is mostly words, and a panel is a number for every
    label it has. "2048 SCORE 504 BEST 5292" is two words and three numbers;
    "Join the numbers and get to the 2048 tile" is eight and one.

    Most sentence-like first, so the line most clearly saying something is the
    one she reads first.
    """
    found: list[tuple[float, str]] = []
    for line in str(words or "").splitlines():
        spoken = " ".join(line.split())
        alphabetic = re.findall(r"[A-Za-z][A-Za-z'-]*", spoken)
        numbers = re.findall(r"\d[\d,.]*", spoken)
        if len(alphabetic) < ENOUGH_WORDS_TO_BE_SAYING_SOMETHING:
            continue
        if len(alphabetic) <= len(numbers):
            continue
        found.append((len(numbers) / len(alphabetic), spoken))
    return [spoken for _how_much_furniture, spoken in sorted(found)]


def _measurably(said: str) -> str:
    """What a sentence names as its end, read the way a request's end is read.

    Asked of the finishing test directly. Read as a whole request, a sentence
    needed a word for carrying on before its end counted, and "Join the numbers
    and get to the 2048 tile!" has none: it was understood only on a machine
    where the game's own name happened to be installed as an application.
    """
    try:
        from core.runtime.watched_goal import _best_finishing_test  # noqa: PLC0415

        return str(_best_finishing_test(said) or "")
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return ""


def _as_a_number(said: str) -> float | None:
    digits = re.findall(r"\d[\d,]*", str(said or ""))
    if not digits:
        return None
    try:
        return float(digits[0].replace(",", ""))
    # not a failure: digits that will not become a number are not a number.
    except ValueError:
        return None


def what_this_place_tells_her(
    seeing: Any, *, asked: str = "", success_when: str = ""
) -> WhatThePlaceSays:
    """What the thing in front of her says it is for, and how that bears.

    ``seeing`` is anything that can render itself as text — the reading she is
    already acting on. ``success_when`` is the finish she was given, if she was
    given one; ``asked`` is the request in the words it was made in.

    Reading the place never overrides the person. Where they name different
    ends, hers stands and the disagreement is said rather than resolved
    quietly: an agent that takes its objective off a screen is doing what the
    screen said instead of what it was asked, and that is worth being loud
    about.
    """
    said_by = getattr(seeing, "as_text", None)
    words = said_by() if callable(said_by) else str(seeing or "")
    mine = str(success_when or "").strip()

    for spoken in sentences_in(words):
        states = _measurably(spoken)
        if not states:
            continue
        if not mine:
            logger.info("the place says what it is for: %r", spoken[:90])
            return WhatThePlaceSays(spoken, states, SUPPLIES, states)
        if states == mine:
            return WhatThePlaceSays(spoken, states, AGREES, mine)
        theirs, ours = _as_a_number(states), _as_a_number(mine)
        if theirs is not None and ours is not None and theirs > ours:
            # Hers is nearer, so it is a waypoint on the way to the place's
            # and still the thing she stops at.
            return WhatThePlaceSays(spoken, states, ON_THE_WAY, mine)
        logger.info(
            "the place says %r and she was asked for %r; going by what she was asked",
            states, mine,
        )
        return WhatThePlaceSays(spoken, states, DISAGREES, mine)

    return WhatThePlaceSays("", "", SAYS_NOTHING, mine)

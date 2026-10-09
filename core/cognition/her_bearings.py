"""Her bearings in a place: why she is there, what she is to do, what the place tells her, what stands out, what she
can act on, and whether she should.

A person walking into somewhere new asks themselves, without saying it: why am
I here; what am I supposed to be doing; is anything here telling me what to
do; does anything stand out, or call for my attention; how much of this can I
touch; should I? Most of the answers are in front of them. Some are not, and
those are the questions worth asking someone (core/cognition/taking_stock.py).

She asks the same of every place, from what she perceives of it: what she was
asked; the place's own words; the controls on it and which of them stand apart
from the others; whether something moves there by itself; what she knows moves
her. Nothing here knows whether the place is a game, a form or a document.

Should she act? Where acting serves what she was asked. A control whose words
say it takes her away from the thing (to more of the site's games, a download, a
purchase, an account, a share) does not, unless she was asked for exactly that,
and is left alone: it is not a way on in the thing she was sent to.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from core.perception.shapes_that_look_pressable import STANDS_OUT

__all__ = ["Bearings", "Place", "instructions_in", "leads_away", "take_bearings"]

#: A control's words that say it leads away from the thing it is on.
_AWAY = re.compile(
    r"\b(more games?|other games?|play more|download|install|get the app|buy|purchase|shop|store|subscribe|sign ?up|"
    r"log ?in|sign ?in|register|share|tweet|facebook|follow us|visit|website|privacy|terms|advert\w*|sponsor\w*)\b",
    re.I)

#: A sentence that tells a person what to do: it opens with an act, or says what is to be done, or with what.
_TELLS = re.compile(
    r"^(?:to\s+\w+[^,]{0,40},\s*)?(?:press|hold|click|tap|use|move|jump|avoid|collect|catch|grab|shoot|fire|throw|aim|"
    r"steer|drag|drop|select|choose|match|find|get|help|guide|keep|stay|don'?t|do not|try|make|build|fill|enter|type|"
    r"answer|pick|put|place|watch out|look for)\b"
    r"|\b(?:you (?:must|need to|have to|should|can)|your (?:goal|job|mission|task) is|the (?:goal|object|aim) (?:is|of))\b"
    r"|\b(?:press|click|use|tap|hold)\b[^.]{0,40}\bto\b",
    re.I)


def instructions_in(says: str) -> list[str]:
    """The sentences of a place's words that tell a person what to do."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", " ".join(str(says or "").split())) if len(s.split()) >= 3]
    return [s for s in sentences if _TELLS.search(s)]


def leads_away(label: str, asked: str = "") -> bool:
    """Whether a control's words say it takes her away from the thing, and she was not asked for where it goes."""
    found = _AWAY.search(str(label or ""))
    return bool(found) and found.group(0).lower() not in str(asked or "").lower()


@dataclass(frozen=True)
class Place:
    """What she perceives of where she is."""

    #: What she was asked, in the person's words.
    asked: str = ""
    #: The place's own prose.
    says: str = ""
    #: What can be clicked, by its words (or its place, for a wordless shape).
    labels: tuple[str, ...] = ()
    #: Of those, the ones that stand apart from the controls like them (core/perception/shapes_that_look_pressable.py),
    #: and the ones whose own words are a way on (Start, Play, Next).
    stands_out: tuple[str, ...] = ()
    #: Whether something in it moves by itself.
    moving: bool = False
    #: The keys she knows to act with here, named or found.
    keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class Bearings:
    """Her answers to the questions anyone asks of a place, and the questions left open."""

    why: str
    instructions: tuple[str, ...]
    sticks_out: tuple[str, ...]
    can_act_on: tuple[str, ...]
    leaving_alone: tuple[str, ...]
    moving: bool
    open_questions: tuple[str, ...]

    @property
    def should_act(self) -> bool:
        """Whether there is anything here to act on that serves what she was asked."""
        return bool(self.can_act_on)

    def said(self) -> str:
        """Her bearings in a sentence, the parts a reading of the place's words does not already say: what stands out,
        and what she leaves alone."""
        parts = []
        if self.sticks_out:
            # A shape already named for standing out is named by where it is: LIVE 2026-10-09 'The one that stands out
            # at 85% across, 30% down stands out.'
            named = [label.replace(STANDS_OUT, "the one at", 1) for label in self.sticks_out[:2]]
            parts.append(f"{_and(named)} stand{'s' if len(named) == 1 else ''} out")
        if self.leaving_alone:
            parts.append(f"{_and(self.leaving_alone[:2])} would take me away from it, so I leave {'it' if len(self.leaving_alone) == 1 else 'them'} be")
        if not parts:
            return ""
        line = "; ".join(parts)
        return line[:1].upper() + line[1:] + "."


def _short(text: str, most: int = 110) -> str:
    said = " ".join(str(text or "").split())
    return said if len(said) <= most else said[: most - 1].rsplit(" ", 1)[0] + "…"


def _and(items: tuple[str, ...] | list[str]) -> str:
    items = [f"“{i}”" if not str(i).startswith("the ") else str(i) for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def take_bearings(place: Place) -> Bearings:
    """Her bearings in ``place``: what she can answer from it, and what she cannot."""
    instructions = tuple(instructions_in(place.says))
    away = tuple(label for label in place.labels if leads_away(label, place.asked))
    sticks_out = tuple(label for label in place.stands_out if label not in away)
    can_act_on = tuple(label for label in place.labels if label not in away) + tuple(place.keys)
    why = f"I was asked to {place.asked.strip().rstrip('.')[:1].lower()}{place.asked.strip().rstrip('.')[1:]}." if place.asked else ""
    asked: list[str] = []
    if not instructions and not place.keys:
        asked.append("how is this done here")
    if not can_act_on:
        asked.append("what can I do here")
    return Bearings(why, instructions, sticks_out, can_act_on, away, place.moving, tuple(asked))

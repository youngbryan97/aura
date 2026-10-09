"""Typing what a screen asks for: the right words, in the place it points to, sent the way it says.

A screen that asks for words (core/language/words_asked_for.py) is answered as a
person answers it: their name where a name is asked, a question where it asks
for one, an answer where something is asked of them and they know it, nothing
where they do not (a code nobody gave them). The place is where the words say
("type your question above"), else beside or under the words that ask, else the
words themselves, for a field often shows its own ask until it is typed in.
What was typed is looked for on the screen afterwards; where it is not there,
the next place is tried. It is sent by the control the words name, else one
beside the field that sends things (Ask, OK, Go), else the Return key.

Nothing here knows a screen, a game or a program: a form, a search box, a
quiz, an oracle and a high-score table are all asked and answered alike.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

from core.language.words_asked_for import ANSWER, NAME, QUESTION, SEARCH, SENDERS, WordsAsked

logger = logging.getLogger("Aura.TypingWhatIsAsked")

__all__ = ["places_to_type", "type_what_is_asked", "what_to_type"]

#: How far, as a share of the picture, past the words that ask the field is looked for.
A_LINE_AWAY = 0.07

#: How far below the words that ask a box drawn for the answer may be, as a share of the picture; and how tall it must
#: be, as a share of the asking words' own line, to hold a line of writing.
FIELD_WITHIN = 0.15
FIELD_AS_TALL = 0.6


#: Her name, typed where a screen asks for one.
HER_NAME = "Aura"


def _a_question_about(goal: str) -> str:
    """A yes-or-no question about what she is doing, from the person's own words."""
    lowered = goal.lower()
    if re.search(r"\bwin\b|\bbeat\b", lowered):
        return "Will I win?"
    if re.search(r"\bscore\b|\bbest\b", lowered):
        return "Will I get a good score?"
    return "Will this go well?"


def _what_is_sought(goal: str) -> str:
    """What the person asked her to look for, where they said ("find X", "look up X", "search for X")."""
    found = re.search(r"\b(?:find|look up|search for|look for)\s+(?:the\s+)?(.+?)(?:[.,;]|$)", goal, re.I)
    return found.group(1).strip() if found else ""


async def what_to_type(asked: WordsAsked, *, goal: str = "",
                       answer: Callable[[str], Awaitable[str | None]] | None = None) -> str | None:
    """The words to type for what was asked, or None where she has none to give."""
    if asked.kind == NAME:
        return HER_NAME
    if asked.kind == QUESTION:
        return _a_question_about(goal)
    if asked.kind == SEARCH:
        return _what_is_sought(goal) or None
    if asked.kind == ANSWER and answer is not None:
        said = await answer(asked.asked_by)
        return " ".join(str(said).split())[:60] if said else None
    return None


def _region_of(regions: list[dict[str, Any]], words: str) -> dict[str, Any] | None:
    """The piece of writing that holds the most of ``words``."""
    wanted = set(re.findall(r"[a-z]{3,}", words.lower()))
    best, most = None, 0
    for region in regions:
        have = set(re.findall(r"[a-z]{3,}", str(region.get("text") or "").lower()))
        overlap = len(wanted & have)
        if overlap > most:
            best, most = region, overlap
    return best


def _fields_near(picture: Any, region: dict[str, Any]) -> list[tuple[float, float]]:
    """Plain boxes drawn just below or beside the words that ask, nearest first: a field to type in is an empty strip.

    LIVE 2026-10-09 a golf game's name box sat half a line under "Enter your name."; clicked a line and more below the
    words, and to their right, the name went nowhere, and the game would not go on without it.
    """
    from core.perception.how_full_a_bar_is import _strips

    if picture is None:
        return []
    import numpy as np

    pixels = np.asarray(picture)
    tall, wide = pixels.shape[:2]
    x, y = float(region.get("center_x", 0.5)), float(region.get("center_y", 0.5))
    line = float(region.get("height", 0.03) or 0.03)
    found = []
    for strip in _strips(pixels):
        if strip.thick < FIELD_AS_TALL * line * tall:
            continue  # a rule or an underline, not a box a line of writing goes in
        cx, cy = (strip.start + strip.end) / 2 / wide, strip.row / tall
        below = 0.0 < cy - y <= FIELD_WITHIN and abs(cx - x) <= 0.25
        beside = abs(cy - y) <= 0.04 and 0.0 < cx - x <= 0.4
        if below or beside:
            found.append((abs(cy - y) + abs(cx - x) / 4, (round(cx, 4), round(cy, 4))))
    return [place for _distance, place in sorted(found)]


def places_to_type(asked: WordsAsked, regions: list[dict[str, Any]], picture: Any = None) -> list[tuple[float, float]]:
    """Where to click before typing, in order, as shares of the picture: a field seen near the ask first."""
    region = _region_of(regions, asked.asked_by)
    if region is None:
        return [(0.5, 0.5)]
    x, y = float(region.get("center_x", 0.5)), float(region.get("center_y", 0.5))
    tall = max(float(region.get("height", 0.03) or 0.03), 0.02)
    wide = max(float(region.get("width", 0.2) or 0.2), 0.05)
    above, below = (x, max(0.0, y - tall - A_LINE_AWAY)), (x, min(1.0, y + tall + A_LINE_AWAY))
    right = (min(1.0, x + wide / 2 + A_LINE_AWAY), y)
    said = {"above": [above], "over this": [above], "below": [below], "beneath": [below], "underneath": [below],
            "to the right": [right], "here": [(x, y)]}.get(asked.where)
    order = _fields_near(picture, region) + (said or [right, below, (x, y), above])
    return list(dict.fromkeys(order))


def _the_sender(asked: WordsAsked, regions: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The control that sends what was typed: the one the words name, else one that says it sends."""
    wanted = [asked.sent_by.lower()] if asked.sent_by else []
    wanted += [word for word in SENDERS if word not in wanted]
    for want in wanted:
        for region in regions:
            said = " ".join(re.findall(r"[a-z]+", str(region.get("text") or "").lower()))
            if said == want:
                return region
    return None


async def type_what_is_asked(
    asked: WordsAsked,
    look: Callable[[], Awaitable[Any]],
    read_words: Callable[[Any], list[dict[str, Any]]],
    hands: Any,
    *,
    goal: str = "",
    answer: Callable[[str], Awaitable[str | None]] | None = None,
    say: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Type what ``asked`` asks for where the screen points, and send it. ``hands`` has ``click(x, y)``,
    ``type_text(text)`` and ``tap(key)``; ``read_words`` reads a picture's writing as regions."""
    text = await what_to_type(asked, goal=goal, answer=answer)
    if not text:
        logger.info("asked for %s, and I have none to give", asked.kind)
        return {"typed": "", "why": f"asked for {asked.kind}, and I have none to give"}
    seen = await look()
    if seen is None:
        return {"typed": "", "why": "the picture could not be taken"}
    regions = read_words(seen[0])
    landed_at = None
    for place in places_to_type(asked, regions, seen[0]):
        await hands.click(*place)
        await hands.type_text(text)
        after = await look()
        written = " ".join(str(region.get("text") or "") for region in read_words(after[0])) if after else ""
        if _shows(written, text):
            landed_at = place
            break
        for _ in text:
            await hands.tap("backspace")
    if landed_at is None:
        logger.info("typed %r in every place the screen points to, and saw it in none", text)
    if say is not None:
        say(f"It asks for {asked.kind}; I'm typing {text!r}.")
    sender = _the_sender(asked, regions)
    if sender is not None:
        await hands.click(float(sender.get("center_x", 0.5)), float(sender.get("center_y", 0.5)))
        sent = str(sender.get("text") or "").strip()
    else:
        await hands.tap(asked.sent_by_key or "return")
        sent = asked.sent_by_key or "return"
    return {"typed": text, "kind": asked.kind, "landed_at": landed_at, "sent_by": sent}


def _shows(written: str, text: str) -> bool:
    """Whether a screen's writing shows what was typed (most of its words)."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    seen = set(re.findall(r"[a-z0-9]+", written.lower()))
    return bool(words) and sum(word in seen for word in words) >= max(1, (2 * len(words)) // 3)

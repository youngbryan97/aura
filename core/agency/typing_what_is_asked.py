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
    if asked.kind == ANSWER:
        given = the_answer_it_gives(asked.said or asked.asked_by)
        if given:
            return given
    if asked.kind == ANSWER and answer is not None:
        said = await answer(asked.asked_by)
        return " ".join(str(said).split())[:60] if said else None
    return None


#: A question a place asks and answers itself: "Answer? YOUR NAME!", "The answer is: a shadow".
_IT_ANSWERS = re.compile(r"(?:(?<!your )(?<!an )(?<!the )\banswer\s*[?:]|\bthe answer is\s*:?|\bsolution\s*:)\s*"
                         r"([A-Za-z][A-Za-z' ]{0,38}?)\s*[.!]", re.I)
#: An answer that is the player's own: what she types for it is hers.
_HERS = re.compile(r"^(?:your|my) (?:own )?name$|^(?:you|yourself|me|myself)$", re.I)


def the_answer_it_gives(said: str, *, as_given: bool = False) -> str:
    """Where a place asks a question and gives the answer itself, that answer, the player's own resolved to hers (a
    riddle whose answer is "your name" is answered with her name); "" where it gives none. Offline 2026-10-11 the
    Riddler's machine asked "What is everything to you and nothing to anyone else? Answer? YOUR NAME!", a riddle that
    is its instruction, and she said she had no answer to give."""
    found = _IT_ANSWERS.search(" ".join(str(said or "").split()))
    if not found:
        return ""
    given = " ".join(found.group(1).split()).strip(" '")
    if not given or given.lower() in ("the", "a", "an", "your", "is"):
        return ""
    return given if as_given or not _HERS.match(given) else HER_NAME


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
    # "Here" in a heading ("YOUR NAME HERE") labels the box drawn under it: offline 2026-10-11 the Riddler's heading was
    # clicked, the name typed there went nowhere, and a click a line lower, in the box, took it.
    under = (x, min(1.0, y + tall))
    said = {"above": [above], "over this": [above], "below": [below], "beneath": [below], "underneath": [below],
            "to the right": [right], "here": [under, (x, y)]}.get(asked.where)
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
    given = the_answer_it_gives(asked.said or asked.asked_by) if asked.kind == ANSWER else ""
    if given and not _fields_near(seen[0], _region_of(regions, asked.asked_by) or {}):
        # A question it asks and answers itself, with nowhere to type on the screen, is told, not answered: what it
        # comes to is what is asked for next (a riddle whose answer is your name, before the box for a name).
        if say is not None:
            raw = the_answer_it_gives(asked.said or asked.asked_by, as_given=True).lower()
            say(f"It asks a riddle, {asked.asked_by!r}, and answers it itself: {raw}."
                + (f" So it wants mine: {given}." if raw != given.lower() else ""))
        return {"typed": "", "why": "it answered its own question", "it_answered": text}
    landed_at = None
    places = places_to_type(asked, regions, seen[0])
    for place in places:
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
        # Writing in a field may be too small or too strange a face to read back: it is typed again where a field is
        # likeliest and left there, not taken out everywhere. Offline 2026-10-11 a name typed into the right box, in
        # green segments, was read as nothing, rubbed out, and the box sent empty.
        logger.info("typed %r in every place the screen points to, and saw it in none; leaving it in the first", text)
        await hands.click(*places[0])
        await hands.type_text(text)
    if say is not None:
        say(f"It asks for {asked.kind}; I'm typing {text!r}.")
    sender = _the_sender(asked, regions)
    if sender is not None:
        await hands.click(float(sender.get("center_x", 0.5)), float(sender.get("center_y", 0.5)))
        sent = str(sender.get("text") or "").strip()
    else:
        await hands.tap(asked.sent_by_key or "return")
        sent = asked.sent_by_key or "return"
        if not asked.sent_by_key:
            sent = await _sent_by_the_button_beside(look, hands, landed_at or places[0], sent)
    return {"typed": text, "kind": asked.kind, "landed_at": landed_at, "sent_by": sent}


#: How near the field, as a share of the picture, a control that sends what was typed is drawn; and how much of the
#: picture must change for Return to have done something.
SENDER_WITHIN = 0.3
SCREEN_CHANGED = 0.2
#: How many of the controls beside a field are tried in turn, how much more of the picture than was changing anyway
#: must change, away from the control itself, for one to have sent it, and how far round the control its own lighting
#: up reaches.
SENDERS_TRIED = 3
SENT_SHOWS = 0.003
LIT_ITSELF = 0.08


async def _sent_by_the_button_beside(look: Callable[[], Awaitable[Any]], hands: Any, field: tuple[float, float],
                                     sent: str) -> str:
    """Where Return changed nothing, the controls drawn beside the field are pressed in turn until one does something:
    offline 2026-10-11 a name machine's GO was read run into the words drawn behind it ("ETRYGOAIN"), no control was
    found by its name, Return did nothing, and the name sat in its box unsent. How it was sent.

    What sends a field comes after it, below it or to its right, as a person reads: LIVE 2026-10-12 the nearest
    control to a name box was the lever on its left, it was pulled three times and did nothing, and the GO under the
    box was never pressed, so no secret identity was ever given."""
    import asyncio

    import numpy as np

    before = await look()
    await asyncio.sleep(0.8)
    after = await look()
    if before is None or after is None:
        return sent
    # Changed is most of the picture changed: a caret blinking or a figure breathing is the same screen.
    a, b = (np.asarray(seen[0])[:, :, :3].astype(np.int16) for seen in (before, after))
    if a.shape != b.shape or (np.abs(a - b).max(axis=2) > 40).mean() > SCREEN_CHANGED:
        return sent
    # What changes on its own between two looks is no sign of a control having done anything.
    anyway = _changed_away_from(before[0], after[0], field)
    for place in _the_controls_beside(after[0], hands, field)[:SENDERS_TRIED]:
        await hands.click(*place)
        await asyncio.sleep(0.8)
        now = await look()
        if now is None or _changed_away_from(after[0], now[0], place) > anyway + SENT_SHOWS:
            return f"the control beside it at {place}"
        after = now
    return sent


def _the_controls_beside(picture: Any, hands: Any, field: tuple[float, float]) -> list[tuple[float, float]]:
    """The controls drawn near a field, those after it first, each in order of nearness."""
    import math

    from core.perception.shapes_that_look_pressable import pressable_shapes

    # Not the box typed into: offline 2026-10-11 the shape "beside" a name was the box the name was in, and its GO, a
    # line below, was never pressed by it.
    shapes = [(float(shape.get("center_x", 0.5)), float(shape.get("center_y", 0.5))) for shape in pressable_shapes(picture)
              if not _holds(shape, field)]
    # Where the pointer turns into a hand is surer than how a thing looks: offline 2026-10-11 a machine's GO, bright
    # letters drawn over others, was no shape that looked pressable, and the pointer turned into a hand over it.
    hands_over = [tuple(place) for place in getattr(hands, "answer_the_pointer", ()) or ()]
    near = [place for place in (hands_over or shapes) if 0.03 < math.dist(place, field) <= SENDER_WITHIN]
    return sorted(near, key=lambda place: (not _after(place, field), math.dist(place, field)))


def _after(place: tuple[float, float], field: tuple[float, float]) -> bool:
    """Whether a place comes after a field as a person reads: below it, or to its right on its line."""
    return place[1] > field[1] + 0.03 or (abs(place[1] - field[1]) <= 0.03 and place[0] > field[0])


def _changed_away_from(before: Any, after: Any, place: tuple[float, float]) -> float:
    """How much of the picture changed between two looks, leaving out round a place: a control lit up by being
    pressed has not by that done anything."""
    import numpy as np

    a, b = (np.asarray(seen)[:, :, :3].astype(np.int16) for seen in (before, after))
    if a.shape != b.shape:
        return 1.0
    changed = np.abs(a - b).max(axis=2) > 40
    tall, wide = changed.shape
    left, top = max(0, int((place[0] - LIT_ITSELF) * wide)), max(0, int((place[1] - LIT_ITSELF) * tall))
    changed[top:max(top, int((place[1] + LIT_ITSELF) * tall)), left:max(left, int((place[0] + LIT_ITSELF) * wide))] = False
    return float(changed.mean())


def _holds(shape: dict[str, Any], place: tuple[float, float]) -> bool:
    """Whether a shape's box holds a place: the box itself, not a control beside it."""
    try:
        x, y = float(shape["center_x"]), float(shape["center_y"])
        half_w, half_h = float(shape.get("width", 0.0)) / 2, float(shape.get("height", 0.0)) / 2
    # not a failure: a shape that is not numbers holds no place
    except (KeyError, TypeError, ValueError):
        return False
    return abs(place[0] - x) <= half_w and abs(place[1] - y) <= half_h


def _shows(written: str, text: str) -> bool:
    """Whether a screen's writing shows what was typed (most of its words)."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    seen = set(re.findall(r"[a-z0-9]+", written.lower()))
    return bool(words) and sum(word in seen for word in words) >= max(1, (2 * len(words)) // 3)

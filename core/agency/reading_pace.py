"""How long a line she says needs to stay up before the next one replaces it.

A commentary a person asked for is for them to read. A loop that narrates and
acts in the same breath puts the next line up before the last was read, and
what a watcher sees is a blur they can only reconstruct from the log
afterwards — which is the thing narration exists to avoid.

The pace is the reading, not the acting. When the person said how long to
wait ("wait 3 seconds before the next one"), that is the wait: the number is
theirs, read from what they asked. When they did not, the time comes from how
much there is to read, at an ordinary adult silent reading rate for prose.

It belongs to the loops whose tempo is hers to set — a page waits for her
click, so a watcher may as well be given time to read why it came. It does not
belong to a loop with a clock of its own: a live game moves whether or not
anyone has finished reading, and holding a move for five seconds would be
playing the narration instead of the game. So this is offered here and taken by
the loop that can afford it, rather than imposed by the thing that does the
saying.
"""
from __future__ import annotations

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

#: Ordinary adult silent reading of prose, in words per minute. The rate is a
#: property of the reader, not of what is being narrated, so it is one number
#: for every loop rather than a knob per caller.
WORDS_PER_MINUTE = 200.0

#: The longest a line stays up when nobody said how long, so one unusually
#: long line cannot stall a run. A wait the person named is not cut to this:
#: they asked for it.
AT_MOST_S = 30.0

#: The wait the request being worked on asked for, if it named one. Task-local,
#: so two pursuits running at once each keep the pace their own person set.
_ASKED: ContextVar[float | None] = ContextVar("aura_asked_pause", default=None)

#: Words for a count, as people write them in a request.
_COUNT_WORDS = {
    "a": 1.0, "an": 1.0, "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0,
    "five": 5.0, "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0,
    "ten": 10.0, "fifteen": 15.0, "twenty": 20.0, "thirty": 30.0,
    "half": 0.5, "couple": 2.0,
}

_UNIT_S = {"s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
           "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0}

def _amount(tag: str) -> str:
    """An amount of time, its count and unit captured under names ending ``tag``."""
    return (
        r"(?:~|about\s+|around\s+|roughly\s+|approximately\s+|maybe\s+|like\s+)?"
        r"(?:a\s+)?(?P<count" + tag + r">\d+(?:\.\d+)?|"
        + "|".join(sorted(_COUNT_WORDS, key=len, reverse=True)) + r")"
        r"(?:\s+of)?[\s-]*(?P<unit" + tag + r">seconds?|secs?|s|minutes?|mins?)\b"
    )


#: A wait the person asked for: a verb of waiting and an amount of time near
#: it, in either order ("wait 3 seconds", "pause ~5 s to give time to read",
#: "give me 3 seconds to read each", "a 3-second pause between them").
#: An amount with no waiting near it ("play for ten minutes") is not a pause.
_WAITING = r"(?:wait|waiting|pause|pausing|hold|linger|give\s+(?:me|us|people|them|the\s+\w+|time)|leave\s+(?:it|each|them|that)\b[^.?!]{0,30}?\bup|let\s+(?:me|us|people)\s+read|break|gap|delay|beat)"
_ASKED_WAIT = re.compile(
    _WAITING + r"\b[^.?!\d]{0,60}?\b" + _amount("") + r"|" + _amount("_first")
    + r"[\s-]+(?:long\s+)?(?:pause|break|gap|delay|wait|beat|hold)\b",
    re.IGNORECASE,
)

_WORD = re.compile(r"\S+")

#: A sentence, for cutting a line at one rather than at a character count.
_SENTENCE = re.compile(r"[^.!?]*[.!?]+[\s]*|[^.!?]+$")


def pause_asked_for(request: object) -> float | None:
    """The wait between lines the person asked for, in seconds, or None.

    The five-second floor this replaces was a number from one request, frozen
    into every later run: the next person to ask for three seconds got five.
    The request is where the number is, so it is read from there each time.
    """
    said = " ".join(str(request or "").split())
    if not said:
        return None
    match = _ASKED_WAIT.search(said)
    if match is None:
        return None
    count = match.group("count") or match.group("count_first")
    unit = match.group("unit") or match.group("unit_first")
    if not count or not unit:
        return None
    count = count.lower()
    amount = _COUNT_WORDS.get(count)
    if amount is None:
        try:
            amount = float(count)
        except ValueError:
            return None
    seconds = amount * _UNIT_S.get(unit.lower(), 0.0)
    return seconds if seconds > 0.0 else None


@contextmanager
def paced_as_asked(request: object) -> Iterator[float | None]:
    """Hold every line said inside this block for the wait ``request`` named."""
    token = _ASKED.set(pause_asked_for(request))
    try:
        yield _ASKED.get()
    finally:
        _ASKED.reset(token)


def time_to_read(text: str) -> float:
    """Seconds to leave ``text`` up before saying the next thing.

    Zero for nothing to read. The wait the request named, inside
    `paced_as_asked`; otherwise the time it takes to read. ``AURA_NARRATION_PACE``
    scales either — 0 turns pacing off for a run nobody is watching, which is
    what a batch of tests is.
    """
    words = len(_WORD.findall(str(text or "")))
    if not words:
        return 0.0
    scale = _pace_scale()
    if scale <= 0.0:
        return 0.0
    asked = _ASKED.get()
    if asked is not None:
        return asked * scale
    reading = words / WORDS_PER_MINUTE * 60.0
    return min(AT_MOST_S, reading) * scale


def as_much_as_can_be_read(text: str) -> str:
    """``text`` cut to whole sentences, at what a watcher can take in at once.

    The same policy as the pause above, used for the other half of the same
    question. A line she says was being cut at four hundred characters — a
    number written for a game, where one move is one short sentence — and a
    reason with two or three sentences in it lost its end mid-clause. What a
    watcher can take in is how long they are given to read, which is
    ``AT_MOST_S`` at the rate above, so the two come from one place.

    Cut at a sentence, never mid-word: a reason that stops before its point is
    worse than a shorter reason that reaches one. Where the first sentence alone
    is already past the bound it is kept whole, because half a sentence says
    less than none.
    """
    said = " ".join(str(text or "").split())
    if not said:
        return ""
    most = int(AT_MOST_S / 60.0 * WORDS_PER_MINUTE)
    words = _WORD.findall(said)
    if len(words) <= most:
        return said
    kept: list[str] = []
    for sentence in _SENTENCE.findall(said):
        ahead = kept + [sentence]
        if kept and len(_WORD.findall(" ".join(ahead))) > most:
            break
        kept.append(sentence)
    return " ".join(kept).strip() or said


def _pace_scale() -> float:
    raw = str(os.getenv("AURA_NARRATION_PACE", "") or "").strip()
    if not raw:
        return 1.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 1.0

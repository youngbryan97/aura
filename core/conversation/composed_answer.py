"""Serve a computed fact without dropping the rest of the question.

LIVE, 2026-08-22. "Morning. How long have you been up, and what have you got
going on today?" came back as:

    Awake 61.6 days in total, across 2199 sessions, 6 minutes of it in this
    one, and you have said 1 things to me today.

Every figure is right and half the message went unanswered. The lifetime
channel matched, and a channel that matches returns its reading in place of
the whole reply, so the second question left no trace.

A measured fact should displace a guess about the same thing, and nothing
else. This decides which case a turn is in: when the message asks one thing,
the reading is the answer; when it asks several and the channel covers some of
them, the reading is joined to what was already written for the rest.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

from core.language.asking_clauses import asking_clauses

_LOG = logging.getLogger("Aura.ComposedAnswer")

__all__ = ["compose_measured", "coverage_of",
    "shapes_the_answer"]


#: A clause saying what the answer should look like rather than asking
#: anything: "give me a number", "keep it short", "in one sentence", "no
#: fluff", "plain language". It constrains the answer to the clause before it.
_SHAPES_THE_ANSWER = re.compile(
    r"^\s*(?:and\s+|but\s+)?(?:please\s+)?(?:"
    r"give\s+me\s+(?:a|an|the)\s+(?:number|figure|percentage|straight\s+answer|"
    r"one[\s-]?word\s+answer)|"
    r"keep\s+it\s+(?:short|brief|tight|simple)|"
    r"in\s+(?:one|a\s+single|two|three)\s+(?:sentence|line|word|paragraph)s?|"
    r"no\s+(?:fluff|preamble|hedging|waffle|marketing)|"
    r"plain\s+(?:language|english)|"
    r"be\s+(?:brief|concise|specific|honest)|"
    r"just\s+the\s+(?:number|answer|facts?)"
    r")\b",
    re.IGNORECASE,
)


def shapes_the_answer(clause: object) -> bool:
    """Whether this clause says how to answer rather than asking something.

    LIVE, 2026-08-25: "How hard is the machine you run on working right now?
    Give me a number you can stand behind." The second sentence counted as a
    request the reading did not cover, so the measured line was joined to "I'd
    rather not guess at that" — a number and a refusal to give one, in that
    order. It is a constraint on the first sentence, not a second question.
    """
    return bool(_SHAPES_THE_ANSWER.match(str(clause or "").strip()))


def coverage_of(
    user_message: object, matches: Callable[[str], bool]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The asking clauses this channel covers, and the ones it does not."""
    clauses = asking_clauses(str(user_message or ""))
    covered: list[str] = []
    uncovered: list[str] = []
    for clause in clauses:
        # A clause about the shape of the answer is not a question left over.
        if shapes_the_answer(clause):
            continue
        try:
            hit = bool(matches(clause))
        except (RuntimeError, TypeError, ValueError):
            hit = False
        (covered if hit else uncovered).append(clause)
    return tuple(covered), tuple(uncovered)


def compose_measured(
    user_message: object,
    reply: object,
    measured: str,
    matches: Callable[[str], bool],
    refute: Callable[[object], tuple[str, str | None]] | None = None,
) -> str:
    """The reading, alone or joined to the answer for what it does not cover.

    `refute` lets the channel police its own quantity in the half it did not
    write. Three minutes after a restart, directly beneath a measured line
    saying so, she wrote "I've been up since 0600" — the reading was in the
    messages she was given, and evidence informs rather than enforces. A
    channel that can compute a number can contradict a claim about it.
    """
    reading = str(measured or "").strip()
    if not reading:
        return str(reply or "")
    covered, uncovered = coverage_of(user_message, matches)

    # One request, or none that could be separated: the reading is the answer.
    if not uncovered:
        return reading
    # The channel covered nothing it could name, so this is the old case: it
    # matched the message as a whole and the message is about one thing.
    if not covered:
        return reading

    written = str(reply or "").strip()
    if refute is not None and written:
        try:
            kept, wrong = refute(written)
        except (RuntimeError, TypeError, ValueError):
            kept, wrong = written, None
        if wrong:
            _LOG.info("Struck a claim the record refutes: %s", wrong)
            written = str(kept or "").strip()
    if not written:
        return reading
    # The reading first. It is the part that is known rather than composed,
    # and putting it second reads as an afterthought to a guess.
    return f"{reading}\n\n{written}"

"""Split a message into the parts that ask, and the parts that describe.

Two defects came from reading a whole message as one lump.

A message was tested for being a question against all of it, and for what it
was about against all of it, so a question at the end licensed a topic match in
any sentence — the rules of an invented game were answered with the
maintenance queue, because "you cannot move ... directly next" put "you"
within sixty characters of "next".

And a message that asks two things was answered with a computed fact covering
one of them, which then replaced the whole reply. "How long have you been up,
and what have you got going on today?" came back as an uptime figure and
nothing else.

Both need the same thing: the clauses, separately.
"""

from __future__ import annotations

import re

__all__ = ["asking_clauses", "asking_part", "asks_more_than_one_thing"]

#: Openings that make a clause a request even without a question mark.
_ASKING = re.compile(
    r"^\s*(?:what|when|where|which|who|whom|whose|how|why|whats|what's|"
    r"anything|any\s+more|is\s+there|are\s+there|do\s+you|did\s+you|does\s+it|"
    r"have\s+you|has\s+it|will\s+you|would\s+you|can\s+you|could\s+you|"
    r"got\s+anything|tell\s+me|show\s+me|list|give\s+me|explain|describe)\b",
    re.IGNORECASE,
)

#: Inside one sentence, these join two separate requests: "how long have you
#: been up, AND what have you got going on today?"
_JOINED = re.compile(r"(?:,|;)?\s+(?:and|but|also|then)\s+(?=\w)", re.IGNORECASE)


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.?!])\s+", str(text or "")) if part.strip()]


def asking_clauses(text: str) -> tuple[str, ...]:
    """Every clause in this message that asks for something.

    A sentence carries its question mark to each of its halves, so both halves
    of "how long have you been up, and what have you got going on today?" are
    returned as requests rather than one being read as a trailing remark.
    """
    found: list[str] = []
    for sentence in _sentences(text):
        interrogative = sentence.endswith("?")
        pieces = [part.strip() for part in _JOINED.split(sentence) if part.strip()]
        for index, piece in enumerate(pieces):
            asks = bool(_ASKING.match(piece))
            # The first piece keeps the sentence's own punctuation; a later
            # piece counts only if it opens like a request of its own.
            if asks or (index == 0 and interrogative):
                found.append(piece if piece.endswith("?") or not interrogative else piece + "?")
    return tuple(dict.fromkeys(found))


def asking_part(text: str) -> str:
    """The asking clauses as one string, or the whole message if none stand out."""
    clauses = asking_clauses(text)
    return " ".join(clauses) if clauses else str(text or "")


def asks_more_than_one_thing(text: str) -> bool:
    return len(asking_clauses(text)) > 1

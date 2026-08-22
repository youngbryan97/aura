"""Decide whether a turn needs evidence from outside this machine.

LIVE, 2026-08-22. "what can you tell me about the company Hugging Face?
founders, what they sell, rough size. link your sources." ran no search. The
grounding taken was her own source code, the cortex answered from memory in
twenty-six seconds, and the reply read "It was founded by <NAME> and <NAME>"
with no citations — to a question that had asked for them in as many words.

Asked instead as "can you LOOK UP Hugging Face", the same question searched.
The decision rested on pattern lists, and three ordinary ways of asking about
a company matched none of them.

Two things are decided here, and neither is a list of topics or verbs.

The first is an instruction: somebody who asks for sources has asked for
evidence, and an answer without any is a broken promise rather than a style
choice.

The second is structural: a question about a named thing that is not her, not
on this disk, and not a matter of opinion cannot be answered from what she
happens to remember. Names are recognised by their shape, so this works for a
company nobody has heard of as well as for one that ships in a pattern list.
"""

from __future__ import annotations

import re

__all__ = ["asks_for_sources", "asks_about_a_named_thing", "wants_outside_evidence"]

#: Asking where something came from. An instruction, not a topic.
_WANTS_SOURCES = re.compile(
    r"\b(?:cite|citation|citations|sources?|referenced?|references|"
    r"link (?:me )?(?:the|your|to)|show me where|where did you (?:get|find|read)|"
    r"back (?:it|that) up with|with links?)\b",
    re.IGNORECASE,
)

#: Asking for facts rather than an opinion or a feeling.
_ASKS_FOR_FACTS = re.compile(
    r"\b(?:who|what|when|where|which|how many|how much|how big|how old|"
    r"tell me about|what do you know about|what can you tell me about|"
    r"give me (?:a )?(?:rundown|summary|overview|background)|"
    r"look (?:up|into)|research|find out|background on|"
    r"founded|founders?|headquarters|revenue|employees|valuation|"
    r"ceo|founder|owner|acquired|launched|released)\b",
    re.IGNORECASE,
)

#: A name, by its shape rather than by being on a list: several capitalised
#: words, an acronym, or a single capitalised word that is not merely the
#: first word of its sentence.
_NAME_SHAPE = re.compile(
    r"\b(?:[A-Z][a-z0-9&.'-]+(?:\s+[A-Z][a-z0-9&.'-]+)+"
    r"|[A-Z]{2,}[A-Za-z0-9]*"
    r"|[A-Z][a-z0-9&.'-]+)"
)

#: Capitalised because of where they sit, not because they name anything.
_CAPITALISED_BUT_ORDINARY = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december", "today",
    "tomorrow", "yesterday", "morning", "afternoon", "evening", "tonight",
    "ok", "okay", "hi", "hey", "hello", "thanks", "please", "yes", "no",
}

#: Words that look like names but are the subject of the conversation itself.
_NOT_A_SUBJECT = {
    "i", "you", "we", "us", "me", "my", "your", "our", "yourself", "myself",
    "aura", "luna", "aura luna",
}

#: Openings that are about her rather than about the world.
_ABOUT_HER = re.compile(
    r"\b(?:your(?:self)?|you'?re|you are|you have|you feel|you think|"
    r"how are you|who are you|what are you)\b",
    re.IGNORECASE,
)


def asks_for_sources(message: object) -> bool:
    """Whether the person asked where the answer comes from."""
    return bool(_WANTS_SOURCES.search(str(message or "")))


def _names(message: str) -> list[str]:
    """Candidate names, with the sentence-initial word discounted."""
    found: list[str] = []
    for sentence in re.split(r"(?<=[.?!])\s+", message):
        stripped = sentence.strip()
        if not stripped:
            continue
        first = stripped.split(" ", 1)[0]
        for match in _NAME_SHAPE.finditer(stripped):
            text = match.group(0).strip()
            if text.lower() in _NOT_A_SUBJECT or text.lower() in _CAPITALISED_BUT_ORDINARY:
                continue
            # A single capitalised word that only leads the sentence is
            # capitalisation, not a name.
            if match.start() == 0 and text == first.rstrip(",;:") and " " not in text:
                continue
            found.append(text)
    return found


def asks_about_a_named_thing(message: object) -> bool:
    """A factual question about something named, that is not her."""
    text = str(message or "")
    if not text.strip() or _ABOUT_HER.search(text):
        return False
    if not _ASKS_FOR_FACTS.search(text):
        return False
    return bool(_names(text))


def wants_outside_evidence(message: object) -> bool:
    """Whether this turn should not be answered from memory alone."""
    text = str(message or "")
    if not text.strip():
        return False
    if asks_for_sources(text):
        return True
    return asks_about_a_named_thing(text)

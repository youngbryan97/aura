"""Is this person asking to see Aura's OWN source?

One definition, because two layers need the answer and they must not be able
to disagree.

The conversational floor uses it to answer from the real source tree rather
than from the model's weights. The desktop-objective router uses it to keep
its hands off: "show me a piece of your own code and tell me which file it
lives in" contains an action word ("show me") and a surface word ("file"), so
the generic classifier read it as a request to operate the computer, sent it
to os_automation, and got back "refused to act because the objective has no
complete observable acceptance contract" — because reading out her own source
has no observable desktop effect to verify. Measured live 2026-08-03: the
floor produced a real 1999-character excerpt for that exact sentence and the
person never saw it, because the desktop lane answered first.

Kept in core/utils deliberately. core/runtime may not import cognition, and a
second copy of this predicate living over there is precisely how the two
answers drift apart.
"""
from __future__ import annotations

import re
from typing import Any

#: Ways of asking to be shown something.
#:
#: Retained as the literal list it always was, for the callers that import it.
#: The MATCH now goes through _SHOW_CUE_RE below, because a phrase list is
#: always one phrasing behind — "can you read your own source?" and "what part
#: of your code do you find interesting?" both missed every entry here, and a
#: miss means she denies a capability she has. That exact failure shape hit the
#: screen-observation router twice, weeks apart.
SOURCE_SHOW_MARKERS: tuple[str, ...] = (
    "show me",
    "show a",
    "can you show",
    "let me see",
    "let's see",
    "display",
    "print out",
    "paste",
)

#: A request to be shown something, as a CUE CLASS rather than a phrase list.
#: Verbs of displaying and of reading both count: asking her to READ her source
#: to you is asking to be shown it.
_SHOW_CUE_RE = re.compile(
    r"\b(?:show|see|display|print|paste|read|open|pull\s+up|look\s+at|"
    r"walk\s+me\s+through|which\s+part|what\s+part|which\s+file|what\s+file|"
    r"which\s+piece|what\s+piece)\b",
    re.IGNORECASE,
)

#: "your ... code" with any adjectives between — "your actual codebase",
#: "your own real source". Substring lists missed exactly the phrasings a
#: person uses, which is how "show me a snippet of code from your actual
#: codebase" fell through to the model.
OWN_SOURCE_RE = re.compile(
    r"\byour\s+(?:\w+\s+){0,3}(?:code|codebase|source|implementation|architecture)\b",
    re.IGNORECASE,
)

#: A subject named right after the code phrase, other than Aura herself.
NAMES_ANOTHER_SUBJECT_RE = re.compile(
    r"\s+(?:for|of|in|from|behind)\s+(?!your\b|yourself\b|you\b|aura\b)\w",
    re.IGNORECASE,
)

#: "the actual code" means HERS only when nothing else is named.
ACTUAL_SOURCE_RE = re.compile(
    r"\b(?:the|some)\s+(?:actual|real|genuine|true)\s+(?:code|codebase|source)\b",
    re.IGNORECASE,
)

# Semantic similarity can identify that a sentence is about implementation,
# but it cannot invent whose implementation it is. This structural subject
# test supplies that missing relation for the embedding route. It deliberately
# excludes ordinary second-person requests such as "can you explain X" where
# Aura is the addressee rather than the object being inspected.
_OWN_IMPLEMENTATION_SUBJECT_RE = re.compile(
    r"\b(?:your|yours|yourself|aura(?:'s)?)\b"
    r"|\bhow\s+(?:do|does|did|would|can)\s+(?:you|aura)\b"
    r"|\b(?:you(?:'re|'re)|you\s+are|aura\s+is)\b[^.?!]{0,50}"
    r"\b(?:built|made|implemented|wired|structured|organized|interested|"
    r"using|running|maintaining)\b"
    r"|\bwhat\b[^.?!]{0,30}\b(?:do|would)\s+you\b[^.?!]{0,30}"
    r"\b(?:use|run|maintain|find\s+interesting)\b",
    re.IGNORECASE,
)


def refers_to_own_implementation(user_message: Any) -> bool:
    """Whether Aura is the grammatical subject of an implementation question."""

    return bool(_OWN_IMPLEMENTATION_SUBJECT_RE.search(str(user_message or "")))


def _contains_show_marker(text: str) -> bool:
    if _SHOW_CUE_RE.search(text):
        return True
    lowered = text.lower()
    return any(marker in lowered for marker in SOURCE_SHOW_MARKERS)


def asks_for_own_source(user_message: Any) -> bool:
    """True when the request is to be shown Aura's own source.

    Requires BOTH a request to be shown something and a reference to her code.
    "What language are you written in?" is a question about her source and not
    a request to see it; "show me the file" without naming whose is not one
    either.
    """

    raw = str(user_message or "")
    if not raw.strip():
        return False
    if not _contains_show_marker(raw):
        return False
    if OWN_SOURCE_RE.search(raw):
        return True
    actual = ACTUAL_SOURCE_RE.search(raw)
    if not actual:
        return False
    # "the actual code for numpy" is a question about numpy, and answering it
    # with a piece of Aura would be its own kind of made-up answer.
    return not NAMES_ANOTHER_SUBJECT_RE.match(raw[actual.end():])


__all__ = [
    "ACTUAL_SOURCE_RE",
    "NAMES_ANOTHER_SUBJECT_RE",
    "OWN_SOURCE_RE",
    "SOURCE_SHOW_MARKERS",
    "asks_for_own_source",
    "refers_to_own_implementation",
]

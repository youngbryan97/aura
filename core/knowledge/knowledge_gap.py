"""Notice not knowing, then go and look — the way a person reaches for a phone.

Aura has an offline Wikipedia index, a web-search skill, and several checkers.
None of them run because she is asked to research. They run when something
decides they are needed, and nothing was making that decision during ordinary
conversation: web_search fired for explicit research requests, and every other
turn was answered from weights, hedges and all.

So she would say "I think it's around 1969, but I'm not certain" and stop
there, holding a search skill she never reached for. A person who says that
picks up their phone. The gap between those two behaviours is not knowledge
and not capability — it is that nothing converted the hedge into an action.

This module is that conversion. It reads a draft the model has already
produced, finds the places where the draft itself admits uncertainty, and
returns the query that would settle it.

Why read the DRAFT rather than the question: the draft is where the
uncertainty actually lives. "When did the Apollo 11 landing happen" looks
confident as a question and may be answered confidently or not; only the answer
knows. This also means the signal is the model's own — a hedge it chose to
write — rather than a guess made about the question from outside.

Deliberately narrow about what counts as a gap. Every false positive spends a
lookup and delays a turn that was already fine, and "I think you'd enjoy that
film" is not a knowledge gap: it is an opinion, correctly hedged. Hedges about
preference, feeling, and the user's own life are excluded, because no reference
work settles them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "KnowledgeGap",
    "detect_knowledge_gap",
    "gap_query",
]

# The model admitting it does not know, or knows only approximately. These are
# the phrasings that should send a person to their phone.
_UNCERTAINTY = re.compile(
    r"\b(?:"
    r"i'?m not (?:sure|certain|positive)|i am not (?:sure|certain)|"
    r"i don'?t (?:know|recall|remember)|i do not (?:know|recall)|"
    r"i can'?t recall|i cannot recall|"
    r"not (?:entirely|totally|completely) sure|"
    r"if i (?:recall|remember) correctly|"
    r"as far as i (?:know|recall)|to my knowledge|"
    r"i (?:believe|think) it(?:'?s| is| was)|i'?d guess|i would guess|"
    r"may be (?:wrong|mistaken)|might be (?:wrong|mistaken)|"
    r"off the top of my head|from memory|"
    r"i'?m fuzzy on|hazy on|"
    r"don'?t quote me|correct me if i'?m wrong|"
    r"something like|roughly|approximately|around (?=\d)"
    r")\b",
    re.IGNORECASE,
)

# Hedges that no reference work settles. A lookup here is pure cost.
_NOT_A_KNOWLEDGE_GAP = re.compile(
    r"\b(?:"
    r"i (?:think|believe|feel) (?:you|we|that you|it would|it'?d|i would|i'?d)|"
    r"in my (?:opinion|view|experience)|"
    r"i'?m not sure (?:how you|what you|why you|if you)|"
    r"how you (?:feel|felt)|what you (?:want|meant|prefer)|"
    r"your (?:mood|preference|plan|schedule|file|screen)"
    r")\b",
    re.IGNORECASE,
)

# Content the corpus and the web can actually settle: named things, numbers,
# dates. A hedge with none of these has nothing to look up.
_LOOKUPABLE = re.compile(r"\b(?:[A-Z][a-z]{2,}|\d{3,4})\b")

_QUERY_STOPWORDS = frozenset(
    """
    a about an and are as at be been but by can could did do does for from had
    has have i if in into is it its me my not of on or so that the their them
    then there these they this those to was we were what when where which who
    why will with would you your sure certain think believe guess recall
    remember know knowledge top head memory fuzzy hazy quote wrong mistaken
    correct roughly approximately around something like exactly entirely
    off actually honestly really certain positive sure recall
    """.split()
)


@dataclass(frozen=True, slots=True)
class KnowledgeGap:
    """One admission of not knowing, and what would settle it."""

    hedge: str
    sentence: str
    query: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.hedge!r} -> {self.query!r}"


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", str(text or "")) if s.strip()]


def gap_query(sentence: str, user_message: Any = "") -> str:
    """The lookup that would settle this sentence.

    Built from the hedged sentence, falling back to the question, because the
    sentence names the thing in doubt and the question supplies the topic when
    the hedge is bare ("I'm not sure, actually").
    """

    def terms(text: str) -> list[str]:
        words = re.findall(r"[A-Za-z][A-Za-z'-]*|\d{3,4}", str(text or ""))
        kept: list[str] = []
        for word in words:
            # "it's" and "I'm" survive a plain stopword check and then poison
            # the query; compare on the bare stem, not the written form.
            stem = word.split("'")[0].lower()
            if not stem or stem in _QUERY_STOPWORDS:
                continue
            if word[0].isdigit():
                kept.append(word)
                continue
            if len(stem) <= 2:
                continue
            kept.append(word)
        return kept

    kept = terms(sentence)
    if len(kept) < 2:
        kept = terms(user_message) or kept
    return " ".join(kept[:6])


def detect_knowledge_gap(draft: Any, user_message: Any = "") -> KnowledgeGap | None:
    """The first sentence of `draft` that admits a settleable uncertainty.

    Returns None when the draft is confident, when the hedge is about taste or
    the user's own life, or when there is nothing nameable to look up.
    """

    for sentence in _sentences(draft):
        if _NOT_A_KNOWLEDGE_GAP.search(sentence):
            continue
        match = _UNCERTAINTY.search(sentence)
        if not match:
            continue
        query = gap_query(sentence, user_message)
        if not query:
            continue
        # A hedge with no nameable content ("I'm not sure I follow") has
        # nothing a reference work could return.
        if not _LOOKUPABLE.search(sentence) and not _LOOKUPABLE.search(str(user_message or "")):
            continue
        return KnowledgeGap(
            hedge=match.group(0), sentence=sentence, query=query
        )
    return None

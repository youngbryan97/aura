"""Ask the local corpus before answering a question it can answer.

LIVE 2026-08-17. Asked to explain correlation versus causation to a twelve
year old, she said "correlation means two things happen together without any
clear relationship between them", which is not what correlation means. The
answer went out unverified.

Meanwhile /Users/bryan/.aura/knowledge/corpus.db holds a BM25 index over ~7M
Wikipedia pages, and the query "correlation does not imply causation" returns
the article of that exact name in 13 milliseconds, opening:

    The phrase "correlation does not imply causation" refers to the inability
    to legitimately deduce a cause-and-effect relationship between two events
    or variables solely on the basis of an observed association...

She had the right answer, locally, in 13ms, and nothing looked. The corpus had
a reader (local_corpus.LocalCorpusStore.search) and the response path had a
grounding channel (_build_active_grounding_message), and no wire ran between
them: grounding was only ever built from a web_search or sovereign_browser
skill result, so a question answered from the model's own weights was never
grounded at all.

This is the wire. It is deliberately NOT a prompt instruction to "look things
up" — it retrieves real passages and feeds them into the grounding channel
that already exists, the same one web_search fills, marked with the same
authority.

Two things it must not do, both of which make a corpus lookup worse than none:

  * fire on turns that are not about the world. "How are you doing", "what did
    I ask you first today", "open my notes" have no corpus answer, and a
    lookup on them spends the turn's latency to inject irrelevant text that
    competes with the real context. The eligibility test is conservative.
  * cost unbounded time. Over a 7M-page index the any-term fallback for a
    conversational phrase took THREE SECONDS to return nothing, which every
    caller on the conversation lane would pay. The deadline here is tight and
    a miss is silent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "CorpusGrounding",
    "corpus_grounding_for",
    "is_corpus_groundable",
]

#: Tight on purpose. A reference lookup that does not answer fast is not worth
#: a conversational turn's latency; a real topic answers in 8-80ms.
SEARCH_DEADLINE_S = 0.35

#: How many passages to carry. More than a few crowds out the live context.
DEFAULT_LIMIT = 3

# Turns that are about her, the user, or an action to take. A corpus cannot
# answer these, and injecting encyclopaedia text into them actively hurts.
_NOT_ABOUT_THE_WORLD = re.compile(
    r"\b(?:"
    r"how are you|how'?s your|are you (?:ok|okay|alright|there)|"
    r"what did i|did i (?:ask|say|tell)|do you remember|remind me|"
    r"open |close |run |launch |click |type |save |delete |install |"
    r"my (?:screen|clipboard|file|folder|notes|calendar|email)|"
    r"your (?:mood|feeling|state|head|mind|memory stores)"
    r")\b",
    re.IGNORECASE,
)

# A question that reaches for a fact, a definition, a mechanism, or a name.
_ABOUT_THE_WORLD = re.compile(
    r"\b(?:"
    r"what (?:is|are|was|were|does|do|causes?)|who (?:is|was|were)|"
    r"why (?:is|are|does|do|did)|how (?:does|do|did|is|are)|"
    r"when (?:did|was|were)|where (?:is|was|are)|"
    r"explain|define|definition|difference between|compare|"
    r"the history of|tell me about|what happens when"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class CorpusGrounding:
    """Passages the local corpus offered for one question."""

    query: str
    passages: tuple[tuple[str, str], ...]  # (title, text)

    @property
    def grounded(self) -> bool:
        return bool(self.passages)

    def render(self, *, max_chars: int = 700) -> list[str]:
        """Lines for the grounding block, shortest-useful first."""
        lines: list[str] = []
        for title, text in self.passages:
            body = " ".join(str(text or "").split())[:max_chars]
            if not body:
                continue
            lines.append(f"{title}: {body}" if title else body)
        return lines


def is_corpus_groundable(objective: Any) -> bool:
    """True when a reference lookup could plausibly answer this turn."""

    text = " ".join(str(objective or "").split())
    if len(text.split()) < 3:
        return False
    if _NOT_ABOUT_THE_WORLD.search(text):
        return False
    return bool(_ABOUT_THE_WORLD.search(text))


# Scaffolding that carries the QUESTION but not the TOPIC. Passing the whole
# sentence to an FTS index is the difference between an answer and a timeout:
# "correlation does not imply causation" returns the right article in 13ms,
# while "Explain the difference between correlation and causation to a smart 12
# year old." AND-matches nothing, falls back to any-term over 7M pages, and
# spends the entire deadline returning empty.
_QUESTION_SCAFFOLDING = frozenset(
    """
    a about an and are as at be been between but by can could define definition
    describe did difference do does explain for from get give had has have how
    i if in into is it its like make me mean means more most much my of on one
    or please so some tell than that the their them then there these they this
    those to told understand upon us use used using was way we were what when
    where which who why will with would you your simple simply smart year old
    kid child teenager terms layman laymans plain english quick quickly short
    briefly like im am not just really actually
    """.split()
)


def _topical_query(text: str) -> str:
    """The content terms of a question, in order, without its scaffolding."""

    words = re.findall(r"[A-Za-z][A-Za-z'-]+|\d{3,4}", str(text or ""))
    kept = [w for w in words if w.lower() not in _QUESTION_SCAFFOLDING and len(w) > 2]
    # Keep it short: a long AND-query over an encyclopaedia matches nothing.
    return " ".join(kept[:6])


def _passage_of(hit: Any) -> tuple[str, str]:
    data = getattr(hit, "__dict__", None) or {}
    title = str(data.get("title") or data.get("doc_id") or "")
    text = str(data.get("text") or data.get("snippet") or data.get("body") or "")
    return title, text


def corpus_grounding_for(
    objective: Any,
    *,
    limit: int = DEFAULT_LIMIT,
    deadline_s: float = SEARCH_DEADLINE_S,
    store: Any = None,
) -> CorpusGrounding:
    """Passages for `objective`, or an empty grounding when it cannot help.

    Never raises. A missing corpus, a slow query, or an ineligible turn all
    produce the same empty result, because to the caller they are the same
    thing: no reference to stand on.
    """

    query = " ".join(str(objective or "").split())
    if not is_corpus_groundable(query):
        return CorpusGrounding(query=query, passages=())
    try:
        if store is None:
            from core.knowledge.local_corpus import LocalCorpusStore

            store = LocalCorpusStore()
        topical = _topical_query(query) or query
        hits = store.search(topical, limit=max(1, int(limit)), deadline_s=deadline_s)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return CorpusGrounding(query=query, passages=())
    passages: list[tuple[str, str]] = []
    for hit in hits or ():
        title, text = _passage_of(hit)
        if text.strip():
            passages.append((title, text))
    return CorpusGrounding(query=query, passages=tuple(passages))

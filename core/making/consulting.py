"""What her sources say of what she is making: her library, the web and her own model, asked as one asks advisers.

Bryan, 27 November 2026: "she should be able to use the internet, her own llms, and her wikipedia corpus for advice, context,
information, solutions, research, definitions, how-tos, etc. not always as authoritative ofc. but as consulting rather than only
relying on her python."

What is asked is written by her model from what she is making: what a word means, how a thing is usually worked out, what people
expect of a program of its kind. Each question goes to her library on this machine (the encyclopaedia, textbooks and manuals) and
to the web, through her own browser as the finding of things does; and what her own model would advise is asked of it once. Each
answer is kept with where it came from, and what goes to her is a reading of them, said to be advice. What is so of the program
is still what its tests and checks show, which no source is asked.
"""
from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = ["ANSWERS", "Consulted", "Said", "ToAsk", "answering", "consult"]

logger = logging.getLogger("Making.Consulting")

#: How many questions are asked, how many answers of each source are kept for each, and how much of an answer is read.
QUESTIONS, OF_EACH, AN_ANSWER = 4, 3, 420
#: How long the web is given for all of them, in seconds, and her library for each: it answers in well under a second.
WEB_S, LIBRARY_S = 90.0, 10.0


#: A question of making something, and a passage of an encyclopedia that answers it. How near her encoder holds the least near
#: of these is how near a passage of her library has to be to a question to be read as answering it. LIVE 28 November 2026,
#: asked how a reading tracker should keep its data, her library gave Ember.js, Google Cloud Datastore and Object REXX.
ANSWERS: tuple[tuple[str, str], ...] = (
    ("How should a small app keep what is put in it between one use and the next?",
     "Web storage provides web applications with methods for storing data in the browser. Data kept with localStorage stays after "
     "the browser is closed and is there the next time the page is opened."),
    ("How can a search find a title even when it is spelled wrongly?",
     "Approximate string matching, often called fuzzy string searching, finds strings that match a pattern approximately rather than "
     "exactly, allowing for letters left out, added or swapped."),
    ("What makes a progress bar easy to read at a glance?",
     "A progress bar is a graphical control element that shows how far an operation has gone. The filled part grows from one end, and "
     "it is often shown with the percentage in text beside it."),
    ("How should a list with nothing in it yet be shown?",
     "An empty state is what an interface shows when there is no content yet. A good one says what will appear there and offers the "
     "action that puts the first thing in."),
    ("How should dates be written so that people read them easily?",
     "ISO 8601 writes dates year, month, day, for machines to sort. For people, applications usually show the month as a word and "
     "follow the conventions of the reader's locale."),
    ("Which colours can people with colour blindness tell apart?",
     "People with red-green colour blindness confuse reds and greens of similar lightness, so designers pair colour with shape, "
     "pattern or text rather than letting colour alone carry the meaning."),
    ("How does a jump in a platform game come to feel right?",
     "In platform games a jump is usually a parabola: the character leaves the ground with an upward speed that gravity takes away each "
     "frame, and many games cut the jump short when the button is let go."),
)
#: Her encoder, for some texts: [] where it is not loaded.
Vectors = Callable[[Iterable[str]], list[list[float]]]
_BOUND: dict[Any, float] = {}


def _vectors(texts: Iterable[str]) -> list[list[float]]:
    from core.cognition.evidence_relevance import vectors_if_ready

    return vectors_if_ready(list(texts))


def _as_asked(question: str) -> str:
    from core.memory.embedding_model import query_prompt

    return query_prompt("document") + " ".join(str(question).split())


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    top, na, nb = sum(x * y for x, y in zip(a, b, strict=False)), math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return top / (na * nb) if na and nb else 0.0


def _bound(vectors: Vectors) -> float | None:
    if vectors not in _BOUND:
        made = vectors([text for question, passage in ANSWERS for text in (_as_asked(question), passage)])
        if len(made) != 2 * len(ANSWERS):
            return None
        _BOUND[vectors] = min(_cosine(made[2 * i], made[2 * i + 1]) for i in range(len(ANSWERS)))
    return _BOUND[vectors]


def answering(question: str, passages: Sequence[str], vectors: Vectors | None = None) -> list[bool]:
    """For each passage, whether it answers the question as near as the passages on record answer theirs; every one, where her
    encoder is not there to read them."""
    vectors = vectors or _vectors
    texts = [" ".join(str(p or "").split())[:AN_ANSWER * 2] for p in passages]
    bound = _bound(vectors) if texts else None
    made = vectors([_as_asked(question), *texts]) if bound is not None else []
    if bound is None or len(made) != len(texts) + 1:
        return [True] * len(texts)
    return [_cosine(made[0], vector) >= bound for vector in made[1:]]


class AQuestion(BaseModel):
    question: str = Field(description="something worth knowing to make it well, as a question that makes sense read alone")
    search: str = Field(description="the words to look it up by")


class ToAsk(BaseModel):
    questions: list[AQuestion] = Field(default_factory=list, max_length=QUESTIONS, description="the most worth asking first")


@dataclass(frozen=True)
class Said:
    """One answer, and where it came from."""

    question: str
    source: Literal["library", "web", "her model"]
    where: str
    text: str


@dataclass
class Consulted:
    asked: list[str]
    said: list[Said]

    def reading(self) -> str:
        """What her sources said, to weigh: each with where it came from, none of it checked against what she is making."""
        if not self.said:
            return ""
        lines = ["What her sources say, which is advice to weigh and not what is so of it:"]
        for question in self.asked:
            answers = [s for s in self.said if s.question == question]
            if answers:
                lines.append(f"\n{question}")
                lines += [f"- {s.source}, {s.where}: {s.text}" for s in answers]
        advice = [s for s in self.said if s.source == "her model"]
        lines += [f"\nWhat her own model would advise, not looked up: {s.text}" for s in advice]
        return "\n".join(lines)


Ask = Callable[[str, type[BaseModel], int], Awaitable[BaseModel | None]]
Write = Callable[[str, int], Awaitable[str]]
Library = Callable[[str], Sequence[tuple[str, str, str]]]
Web = Callable[[Sequence[str]], Awaitable[Sequence[tuple[str, str, str, str]]]]


def _short(text: str, most: int = AN_ANSWER) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= most else text[:most].rsplit(" ", 1)[0] + "…"


def her_library(query: str) -> list[tuple[str, str, str]]:
    """Her library's passages for a query: (title, shelf, words)."""
    from core.knowledge.local_corpus import get_local_corpus_store

    return [(hit.title, hit.source, hit.snippet) for hit in get_local_corpus_store().search(query, OF_EACH)]


async def the_web(queries: Sequence[str]) -> list[tuple[str, str, str, str]]:
    """What the web gives for each query, in her own browser: (query, address, title, words around it)."""
    from core.search.engines_in_her_browser import search_in_her_browser

    found = await search_in_her_browser(list(queries), at_most=OF_EACH, principal="research_pipeline")
    return [(query, url, title, around) for query in queries for url, title, around in (found.get(query) or [])[:OF_EACH]]


async def consult(about: str, *, ask: Ask, write: Write, library: Library = her_library, web: Web = the_web,
                  advice_on: str = "") -> Consulted:
    """Ask her sources what is worth knowing about ``about``; and, where ``advice_on`` is given, her own model what it would
    advise of that. Nothing that does not answer holds anything up: what comes back by its time is what was said."""
    read = await ask(f"{about}\n\nWhat would be worth looking up, to make it well?", ToAsk, 500)
    questions = [q for q in (read.questions if isinstance(read, ToAsk) else []) if q.question.strip() and q.search.strip()][:QUESTIONS]
    said: list[Said] = []
    for q in questions:
        try:
            hits = await asyncio.wait_for(asyncio.to_thread(library, q.search), timeout=LIBRARY_S)
        except (TimeoutError, OSError, RuntimeError, ValueError) as why:
            logger.info("Her library was not read for %r: %s", q.search, why)
            hits = []
        # What does not answer the question is left on the shelf.
        kept = await asyncio.to_thread(answering, q.question, [words for _title, _shelf, words in hits]) if hits else []
        said += [Said(q.question, "library", f"{title} ({shelf})", _short(words)) for (title, shelf, words), keep in zip(hits, kept, strict=True) if keep]
    if questions:
        try:
            found = await asyncio.wait_for(web([q.search for q in questions]), timeout=WEB_S)
        except (TimeoutError, OSError, RuntimeError, ValueError) as why:
            logger.info("The web was not read: %s", why)
            found = []
        by_search = {q.search: q.question for q in questions}
        said += [Said(by_search.get(query, query), "web", f"{_short(title, 90)} <{url}>", _short(around)) for query, url, title, around in found if around.strip()]
    if advice_on:
        words = await write(f"{advice_on}\n\nWhat would you advise, as somebody who has made many of these: what makes one feel finished and "
                            "well made, and what is easy to get wrong?", 700)
        if words.strip():
            said.append(Said("", "her model", "asked", _short(words, 1600)))
    logger.info("Consulted on %d question(s): %d from her library, %d from the web, %d from her model.", len(questions),
                sum(s.source == "library" for s in said), sum(s.source == "web" for s in said), sum(s.source == "her model" for s in said))
    return Consulted([q.question for q in questions], said)

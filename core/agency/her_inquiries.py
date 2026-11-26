"""Looking into something of her own: the questions it comes to, taken one at a time, and whether what she read answers them.

A want was looked into by searching its name. LIVE 25 November 2026, 02:06, what she wanted to find out more of was "chatting":
the word was searched, every page that came back was one she had read, and the act came to nothing, as the one before it had.
A goal was looked into by searching its whole sentence, and was called done when any page came back.

What there is to find out is a list of questions, each with the words to search for it. Her model writes them, from what it is
about, what she has found out of it already and what she asked and got no answer to. A pass takes the first that is open,
reads the pages it finds, and an answer stands only where the words that give it are on the page. A question nothing answered
twice is let go. What has been asked and answered is kept, so the next pass goes on from there; where all of it has been, what
is asked next is asked knowing the answers.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = ["Answered", "Inquiry", "LET_GO_AFTER", "NotNowError", "Page", "Pages", "Question", "look_into_one"]

logger = logging.getLogger("Aura.HerInquiries")

#: How often a question is taken up before it is let go, and how many inquiries are kept.
LET_GO_AFTER, KEPT = 2, 64
#: How many pages one question is read for, how much of each, and how much of what is known is given to write the next from.
PAGES, PAGE_ROOM, KNOWN_ROOM = 4, 9000, 2400

ANSWERED, PARTLY, OPEN, LET_GO = "answered", "partly", "open", "let go"


class NotNowError(RuntimeError):
    """What a pass needs is held for something else: it is put down, and says nothing of what it was about."""


class ToAsk(BaseModel):
    question: str = Field(description="one thing that is not known here and that a page could say, as a question that makes sense read alone")
    search: str = Field(description="the words to search for it")


class ToFindOut(BaseModel):
    questions: list[ToAsk] = Field(default_factory=list, description="what would have to be found out, the first of it first")


class Says(BaseModel):
    answers: Literal["yes", "partly", "no"]
    answer: str = Field("", description="what the page says to the question, in a sentence or two")
    quote: str = Field("", description="the words on the page that say it, copied exactly; empty where it does not")


@dataclass(frozen=True)
class Page:
    url: str
    title: str
    text: str


#: Pages found for some words, one read at a time; and her model, asked for data of a shape.
Pages = Callable[[str], Awaitable[Sequence[Callable[[], Awaitable[Page | None]]]]]
Ask = Callable[[str, type[BaseModel], int], Awaitable[BaseModel | None]]


@dataclass
class Question:
    question: str
    search: str
    stands: str = OPEN
    answer: str = ""
    quote: str = ""
    where: str = ""
    tries: int = 0


@dataclass(frozen=True)
class Answered:
    """What one pass came to."""

    question: Question
    read: tuple[str, ...] = ()

    @property
    def got(self) -> float:
        return {ANSWERED: 1.0, PARTLY: 0.5}.get(self.question.stands, 0.0)


@dataclass
class Inquiry:
    """What she is finding out about one thing."""

    about: str
    questions: list[Question] = field(default_factory=list)
    changed_at: float = 0.0

    @staticmethod
    def of(kept: Any) -> "Inquiry | None":
        try:
            held = dict(kept)
            return Inquiry(str(held["about"]), [Question(**q) for q in held.get("questions") or ()], float(held.get("changed_at") or 0.0))
        # not a failure: what cannot be read as an inquiry of this shape is begun again when it is next taken up.
        except (KeyError, TypeError, ValueError):
            return None

    def kept(self) -> dict[str, Any]:
        return asdict(self)

    def next(self) -> Question | None:
        """The question to take up: the first open one, those tried before last."""
        return min((q for q in self.questions if q.stands in (OPEN, PARTLY)), key=lambda q: q.tries, default=None)

    @property
    def is_done(self) -> bool:
        """Every question it came to has been answered, or let go, and one at least was answered."""
        return bool(self.questions) and self.next() is None and any(q.stands == ANSWERED for q in self.questions)

    @property
    def met(self) -> float:
        """How much of what it came to has been answered."""
        return sum({ANSWERED: 1.0, PARTLY: 0.5}.get(q.stands, 0.0) for q in self.questions) / len(self.questions) if self.questions else 0.0

    def known(self) -> list[str]:
        return [f"{q.question} {q.answer}" for q in self.questions if q.answer]

    async def ask_what_next(self, ask: Ask) -> bool:
        """What there is to find out, written from what has been; whether there is anything."""
        unanswered = [q.question for q in self.questions if q.stands == LET_GO]
        told = "\n".join(f"- {line}" for line in self.known())[-KNOWN_ROOM:]
        given = [f"What it is about: {self.about}", f"Found out already:\n{told}" if told else "",
                 "Asked, and answered nowhere that was read:\n" + "\n".join(f"- {q}" for q in unanswered[-6:]) if unanswered else ""]
        read = await ask("\n\n".join(part for part in given if part) + "\n\nWhat would have to be found out next?", ToFindOut, 600)
        if not isinstance(read, ToFindOut):
            return False
        asked = {_plain(q.question) for q in self.questions}
        new = [Question(q.question.strip(), (q.search or q.question).strip()) for q in read.questions if q.question.strip() and _plain(q.question) not in asked]
        self.questions += new
        self.changed_at = time.time()
        return bool(new)


def _plain(words: str) -> str:
    return " ".join(str(words).casefold().split())


async def look_into_one(inquiry: Inquiry, *, ask: Ask, pages: Pages, seen: Callable[[Iterable[str]], Any]) -> Answered | None:
    """One pass: the next question, taken to the pages it finds. None where there is nothing to ask of it."""
    from core.understanding.carriers.finding import quoted_in

    question = inquiry.next()
    if question is None and await inquiry.ask_what_next(ask):
        question = inquiry.next()
    if question is None:
        return None
    question.tries += 1
    inquiry.changed_at = time.time()
    read: list[str] = []
    best: tuple[int, Says, Page] | None = None
    for open_page in list(await pages(question.search))[:PAGES]:
        page = await open_page()
        if page is None or not page.text.strip():
            continue
        read.append(page.url)
        said = await ask(f"A question: {question.question}\n\nA page, {page.url}\nTitle: {page.title}\nText:\n{page.text[:PAGE_ROOM]}\n\nWhat does the page say to the question?", Says, 400)
        if not isinstance(said, Says) or said.answers == "no" or not quoted_in(said.quote, f"{page.title}\n{page.text}"):
            continue
        rank = 2 if said.answers == "yes" else 1
        if best is None or rank > best[0]:
            best = (rank, said, page)
        if rank == 2:
            break
    seen(read)
    if best is not None:
        rank, said, page = best
        question.stands = ANSWERED if rank == 2 else PARTLY
        question.answer, question.quote, question.where = said.answer.strip()[:600], said.quote.strip()[:400], page.url
    if question.stands != ANSWERED and question.tries >= LET_GO_AFTER:
        question.stands = LET_GO if question.stands == OPEN else ANSWERED
    logger.info("She asked %r: %s, from %d pages%s", question.question[:160], question.stands, len(read), f" ({question.where[:120]})" if question.where else "")
    return Answered(question, tuple(read))

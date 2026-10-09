"""Taking stock: stopping to ask what she knows, when she is stuck, keeps failing, has a question, or before she begins.

A person who keeps losing at something does not press the same keys harder.
They stop, and ask: how do you win this, what do you do when that happens, what
is the thing in the corner? They ask themselves first (have I been here, or
somewhere like it?), then what they have read, then someone who might know, and
they weigh what they hear: what several say, and what says what to do, counts
for more than one voice saying something vague. Then they try it, and remember
whether it helped.

So does she, with what she has:

- what helped her before, in this thing or one of its name (``WhatHelped``);
- her own copy of Wikipedia (``from_her_corpus``);
- the web, where she can reach it (``from_the_web``, given a way to search and
  read pages: in a browser, a tab of her own);
- her model, as one witness among the others, asked only where its answer can
  come in the time there is (``from_her_model``).

Each is asked the same few questions, made from the situation (``questions_for``),
all at once and each within the time there is. What comes back is sentences; the
ones kept are about the thing, say what to do, and, better, are said by more
than one source (``what_to_take``). What is kept is counsel she reads the way
she reads a screen's own instructions; it is never a command, and the screen
in front of her still decides. Nothing here knows what kind of thing she is in.

Only when there is time: a thing that is waiting for her (an end screen, a menu,
a still page) has time; a thing in motion does not, and is not stopped for.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "BEFORE", "FAILING", "QUESTION", "STUCK", "Counsel", "Heard", "Situation", "WhatHelped",
    "HELPED_BEFORE", "from_her_corpus", "from_her_memory", "from_her_model", "from_the_web", "questions_for", "take_stock",
    "what_to_take",
]

logger = logging.getLogger("Aura.TakingStock")

#: Why she stops to take stock.
STUCK = "I'm getting nowhere"
FAILING = "I keep losing"
QUESTION = "there is something here I don't understand"
BEFORE = "before I begin"

#: What helped her before is called this when she says where counsel came from.
HELPED_BEFORE = "what helped me before"

#: The most questions asked at once, and the most sentences of counsel kept.
MOST_QUESTIONS = 4
MOST_KEPT = 3

#: Words that carry no subject of their own, left out of what a question or a sentence is about.
_PLAIN = frozenset("""a an the of and or to in on at by for with from as is are was be it its this that these those
you your i me my we our they their he she his her them there here what which who how when where why do does did can
could would should will just then than so if not no yes all any some more most very much one two three game games
play playing played""".split())

#: What a sentence that says what to do sounds like: an act, a control, a manner of acting.
_SAYS_WHAT_TO_DO = re.compile(
    r"\b(press|hold|release|click|tap|type|drag|drop|use|move|jump|duck|dodge|avoid|collect|catch|grab|shoot|fire|throw|aim|"
    r"steer|turn|rotate|select|choose|match|stack|build|land|keep|stay|wait|watch for|time|alternate|repeatedly|rapidly|"
    r"quickly|as fast as|arrow keys?|space ?bar|spacebar|mouse|left|right|up|down|enter|shift|ctrl|key|button|"
    r"win|wins|lose|before|until|when|if)\b", re.I)

#: The acts and controls a sentence names: what two sources agree on, when they agree.
_A_CUE = re.compile(r"\b(press|hold|click|tap|drag|jump|dodge|avoid|collect|catch|shoot|throw|aim|steer|alternate|"
                    r"repeatedly|rapidly|quickly|arrow keys?|space ?bar|spacebar|mouse|left|right|up|down|enter)\b", re.I)


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", str(text or "").lower().replace("’", "'")) if len(w) > 2 and w not in _PLAIN}


@dataclass(frozen=True)
class Situation:
    """Where she is and how it is going, as the questions are made from it."""

    #: The thing she is in, by its own name ("Regular Show: All-Nighter", "the expenses form").
    thing: str
    #: What she was asked to do with it, in the person's words.
    task: str = ""
    #: What the place has said lately: its instructions, its labels, the words its last end showed.
    said: tuple[str, ...] = ()
    #: How the last try ended, in its own words ("OUCH! CRASHES: 4", "Game over").
    ended: str = ""
    #: Why she is taking stock (STUCK, FAILING, QUESTION, BEFORE).
    why: str = STUCK


#: The verb of a task, for the question of how it is done: "Play this game and win it" asks how to win.
_THE_AIM = re.compile(r"\b(win|beat|finish|complete|solve|clear|reach|escape|fill(?: in| out)?|submit|build|make|find|fix)\b", re.I)


def _the_aim(task: str) -> str:
    found = _THE_AIM.findall(str(task or ""))
    return found[-1].lower() if found else "do well at"


def _what_ended_it(ended: str, thing: str) -> str:
    """The words of an end that say what happened, without the screen's furniture or the thing's own name."""
    furniture = _words("score points level lives time try again play continue retry restart menu best total rating new start "
                       "over quit back next yes ok your you got have has were ouch oops sorry uh-oh whoa wow nope")
    named = _words(thing)
    # What happened is said by what was done to her: "captured", "crashed", "caught", "fell".
    kept = [w for w in re.findall(r"[A-Za-z][A-Za-z'’-]+", str(ended or "")) if w.lower() not in _PLAIN
            and w.lower() not in furniture and w.lower() not in named and len(w) > 2
            and (w.lower().endswith("ed") or w.lower() in _HAPPENED)]
    return " ".join(dict.fromkeys(w.lower() for w in kept[:2]))


#: Words of what happened that do not end as most do.
_HAPPENED = frozenset("caught fell died dead eaten beaten stuck broke broken burnt burned hit lost sunk drowned".split())


def _unfamiliar_terms(said: Sequence[str], thing: str) -> list[str]:
    """Names on the place's screens that are its own (capitalised phrases not of the language's commonest words)."""
    named = _words(thing)
    found: list[str] = []
    for text in said:
        for phrase in re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}|[A-Z]{2,}(?:\s+[A-Z]{2,}){0,2})\b", str(text)):
            plain = _words(phrase)
            if plain and not plain <= named and len(phrase) >= 4 and phrase.lower() not in ("how to play", "press start"):
                found.append(phrase.strip())
    return list(dict.fromkeys(found))[:2]


def questions_for(situation: Situation) -> list[str]:
    """The few questions a person in her place would ask, in the order they matter."""
    thing, aim = situation.thing.strip(), _the_aim(situation.task)
    asked = [f"how to {aim} {thing}"]
    happened = _what_ended_it(situation.ended, thing)
    if happened and situation.why in (FAILING, STUCK):
        asked.append(f"{thing} what to do when {happened}")
    asked.append(f"{thing} controls how to play")
    if situation.why in (QUESTION, STUCK):
        asked += [f"what is {term} in {thing}" for term in _unfamiliar_terms(situation.said, thing)]
    return list(dict.fromkeys(asked))[:MOST_QUESTIONS]


@dataclass(frozen=True)
class Heard:
    """One thing one source said."""

    #: Who said it, as she says it: "the web", "my own copy of Wikipedia", "what helped me before", "my model".
    source: str
    #: Where exactly ("“Operation Z.E.R.O.” on a fan wiki", an article's title), for saying so.
    where: str
    text: str


#: A source: the questions and the seconds there are -> what it says.
Source = Callable[[list[str], float], Awaitable[list[Heard]]]


@dataclass(frozen=True)
class Counsel:
    """What taking stock came to: what to go by, where it came from, and a line for whoever is watching."""

    questions: tuple[str, ...] = ()
    kept: tuple[Heard, ...] = ()
    asked_of: tuple[str, ...] = ()
    took_s: float = 0.0

    @property
    def told(self) -> str:
        """The counsel as words to read, as a screen's own instructions are read."""
        return " ".join(h.text for h in self.kept)

    def __bool__(self) -> bool:
        return bool(self.kept)

    def said(self) -> str:
        """In a sentence or two: what she found, and from where; or that nothing said more than the place itself."""
        if not self.kept:
            asked = ", ".join(self.asked_of) or "what I have"
            return f"I asked {asked}, and nothing said more than this place itself does."
        first = self.kept[0]
        others = sorted({h.source for h in self.kept[1:] if h.source != first.source})
        also = f" ({' and '.join(others)} agree{'s' if len(others) == 1 else ''})" if others else ""
        return f"From {first.source}{_where(first)}: “{_short(first.text)}”{also}. I'll go by that."


def _where(heard: Heard) -> str:
    return f", {heard.where}" if heard.where and heard.where != heard.source else ""


def _short(text: str, most: int = 160) -> str:
    said = " ".join(str(text or "").split())
    return said if len(said) <= most else said[: most - 1].rsplit(" ", 1)[0] + "…"


def _sentences(text: str) -> list[str]:
    """A text's sentences of prose: of a sentence's length, and mostly words in lower case, as prose is. A page's menus
    and footers ("About Press Copyright Contact us Creators") run together as one long line of capitals, and
    "Press" in them is not an instruction."""
    plain = " ".join(str(text or "").split())
    found = []
    for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z\"“(])", plain):
        words = re.findall(r"[A-Za-z][A-Za-z'’-]*", sentence)
        if not 6 <= len(sentence.split()) <= 45 or not words:
            continue
        if sum(w[0].islower() for w in words) < 0.5 * len(words):
            continue
        found.append(sentence.strip())
    return found


def what_to_take(heard: Sequence[Heard], situation: Situation, questions: Sequence[str]) -> list[Heard]:
    """The sentences worth going by: about this thing and what was asked, saying what to do, and better where more than
    one source says the same acts and controls. One sentence a source at most, the best first."""
    about = _words(situation.thing) | set().union(*(_words(q) for q in questions)) if questions else _words(situation.thing)
    said_on_screen = set().union(*(_words(s) for s in situation.said)) if situation.said else set()
    candidates: list[tuple[float, Heard, set[str]]] = []
    for one in heard:
        for sentence in _sentences(one.text):
            words = _words(sentence)
            cues = {c.lower() for c in _A_CUE.findall(sentence)}
            # Each source is already kept to what is about the thing (a page or an article whose title names it, an
            # answer to the question asked); a sentence of it need not name the thing again to be about it.
            if not _SAYS_WHAT_TO_DO.search(sentence):
                continue
            if not cues and len(words & about) < 2:
                continue  # neither an act or a control named, nor plainly about what was asked: a site's own chatter
            score = len(words & about) + 2.0 * len(cues) + (4.0 if one.source == HELPED_BEFORE else 0.0)
            # What the place already says is not news; a sentence only repeating it is worth less.
            if words and len(words & said_on_screen) > 0.8 * len(words):
                score *= 0.4
            candidates.append((score, Heard(one.source, one.where, sentence), cues))
    by_cue: dict[str, set[str]] = {}
    for _score, one, cues in candidates:
        for cue in cues:
            by_cue.setdefault(cue, set()).add(one.source)
    ranked = sorted(((score + 1.5 * sum(len(by_cue[c]) - 1 for c in cues), one) for score, one, cues in candidates),
                    key=lambda pair: -pair[0])
    kept: list[Heard] = []
    for _score, one in ranked:
        # One sentence from each place it was found: two pages of the web are two witnesses, one page is one.
        if all((one.source, one.where) != (k.source, k.where) for k in kept) and all(one.text != k.text for k in kept):
            kept.append(one)
        if len(kept) >= MOST_KEPT:
            break
    return kept


async def take_stock(situation: Situation, sources: dict[str, Source], *, seconds: float) -> Counsel:
    """Ask every source the situation's questions at once, each within ``seconds``, and keep what is worth going by."""
    began = time.monotonic()
    questions = questions_for(situation)

    async def ask(name: str, source: Source) -> list[Heard]:
        try:
            return list(await asyncio.wait_for(source(questions, seconds), timeout=seconds) or [])
        except TimeoutError:
            logger.info("taking stock: %s did not answer in %.0fs", name, seconds)
        except Exception as why:  # noqa: BLE001 - a source that fails has said nothing
            logger.info("taking stock: %s could not be asked: %s", name, str(why)[:160])
        return []

    answers = await asyncio.gather(*(ask(name, source) for name, source in sources.items()))
    heard = [h for one in answers for h in one]
    kept = what_to_take(heard, situation, questions)
    counsel = Counsel(tuple(questions), tuple(kept), tuple(sources), time.monotonic() - began)
    logger.info("taking stock (%s): asked %s of %s; heard %d; kept %s", situation.why, questions, list(sources), len(heard),
                [(h.source, h.text[:100]) for h in kept])
    return counsel


# ── the sources ──────────────────────────────────────────────────────────


def from_her_corpus(corpus: Any = None, *, thing: str = "", names_it: Callable[[str, str], bool] | None = None) -> Source:
    """Her own copy of Wikipedia: the passages its search finds for each question, from articles whose titles name the
    thing (an article about something else, whatever it says, is not about this)."""

    async def ask(questions: list[str], seconds: float) -> list[Heard]:
        store = corpus
        if store is None:
            from core.knowledge.local_corpus import get_local_corpus_store

            store = get_local_corpus_store()

        def search() -> list[Heard]:
            found: list[Heard] = []
            for question in questions:
                for hit in store.search(question, limit=3, deadline_s=min(1.0, seconds / 4)) or []:
                    if names_it is not None and thing and not names_it(thing, str(hit.title)):
                        continue
                    body = store.body(hit.doc_id, max_chars=6000) if hasattr(store, "body") else hit.snippet
                    found.append(Heard("my own copy of Wikipedia", f"“{hit.title}”", f"{hit.snippet} {body}"))
            return found

        return await asyncio.to_thread(search)

    return ask


def from_her_memory(memory: Any = None) -> Source:
    """What she remembers that bears on the questions: her own semantic memory, searched for each."""

    async def ask(questions: list[str], seconds: float) -> list[Heard]:
        store = memory
        if store is None:
            from core.runtime.service_registry import get_runtime_service

            store = get_runtime_service("semantic_memory", default=None)
        if store is None or not hasattr(store, "search_memories"):
            return []

        def search() -> list[Heard]:
            found: list[Heard] = []
            for question in questions:
                for hit in store.search_memories(question, top_k=2) or []:
                    text = str(hit.get("text") or hit.get("content") or "") if isinstance(hit, dict) else str(hit)
                    if text:
                        found.append(Heard("what I remember", "", text[:1200]))
            return found

        return await asyncio.to_thread(search)

    return ask


def from_her_model(ask_typed: Callable[..., Awaitable[Any]], *, tokens: int = 220) -> Source:
    """Her model, asked the questions together for a short answer to each; only where it can answer in the time there is.

    ``ask_typed(prompt, schema, max_tokens)`` is how the caller's layer asks her model for typed data."""
    from pydantic import BaseModel, Field

    class _Answer(BaseModel):
        question: str = Field(default="", max_length=200)
        answer: str = Field(default="", max_length=400, description="what to do, in a sentence or two; empty if you do not know")

    class _Answers(BaseModel):
        answers: list[_Answer] = Field(default_factory=list, max_length=MOST_QUESTIONS)

    async def ask(questions: list[str], seconds: float) -> list[Heard]:
        from core.brain.llm.thinking_reserve import seconds_to_decode

        needs = float(seconds_to_decode(tokens) or 0.0)
        if needs and needs > seconds:
            logger.info("taking stock: my model needs about %.0fs and there are %.0fs; not asked", needs, seconds)
            return []
        prompt = ("Answer each question from what you know, in a sentence or two that says what to do. "
                  "Where you do not know, leave the answer empty.\n" + "\n".join(f"- {q}" for q in questions))
        got = await ask_typed(prompt, _Answers, tokens)
        return [Heard("my model", "", a.answer) for a in getattr(got, "answers", []) or [] if a.answer.strip()]

    return ask


#: Search and read: a query -> [(the page's title, its address, its words)], each page the search found and could read.
Reader = Callable[[str, float], Awaitable[list[tuple[str, str, str]]]]


def from_the_web(search_and_read: Reader, *, thing: str = "", names_it: Callable[[str, str], bool] | None = None) -> Source:
    """The web: each question searched, and the pages read whose titles name the thing (a page about something else,
    whatever it says, is not about this)."""

    async def ask(questions: list[str], seconds: float) -> list[Heard]:
        found: list[Heard] = []
        per = max(4.0, seconds / max(1, len(questions[:2])))
        for question in questions[:2]:
            for title, where, text in await search_and_read(question, per) or []:
                if names_it is not None and thing and not names_it(thing, title):
                    continue
                site = re.sub(r"^https?://(www\.)?", "", where).split("/")[0]
                found.append(Heard("the web", f"“{_short(title, 70)}” on {site}", text))
        return found

    return ask


# ── what helped ──────────────────────────────────────────────────────────


@dataclass
class WhatHelped:
    """Counsel she went by, and whether going by it helped, kept per thing between sittings."""

    helped: list[str] = field(default_factory=list)
    did_not: list[str] = field(default_factory=list)

    @staticmethod
    def _world(thing: str) -> str:
        from core.runtime.what_she_learned import named

        return named("what helped", thing)

    @classmethod
    def of(cls, thing: str) -> WhatHelped:
        from core.runtime.what_she_learned import recall

        held = recall(cls._world(thing)) or {}
        return cls(list(held.get("helped") or []), list(held.get("did_not") or []))

    def came_of(self, thing: str, counsel: Counsel, helped: bool) -> None:
        """Remember whether going by ``counsel`` helped, and keep it so."""
        from core.runtime.what_she_learned import remember

        for one in counsel.kept:
            (self.helped if helped else self.did_not).append(one.text)
            other = self.did_not if helped else self.helped
            if one.text in other:
                other.remove(one.text)
        self.helped, self.did_not = list(dict.fromkeys(self.helped))[-12:], list(dict.fromkeys(self.did_not))[-24:]
        remember(self._world(thing), {"helped": self.helped, "did_not": self.did_not})

    def source(self) -> Source:
        """What helped before, asked first: it was tried here, and it worked."""

        async def ask(_questions: list[str], _seconds: float) -> list[Heard]:
            return [Heard(HELPED_BEFORE, "", text) for text in self.helped]

        return ask

    def without_what_did_not(self, counsel: Counsel) -> Counsel:
        """The counsel less what was tried here before and did not help."""
        kept = tuple(h for h in counsel.kept if h.text not in self.did_not)
        return Counsel(counsel.questions, kept, counsel.asked_of, counsel.took_s)

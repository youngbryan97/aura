"""What is known of a program, read in depth: what it does, and what each thing it does rests on.

Two articles say what a program is; they do not say how a flood fill decides where to stop, what a PNG holds, or
how far back an undo goes. A person rebuilding a program reads on: from the program to its kind, from each thing
it does to the articles on what that rests on (an algorithm, a file format, a data structure, a standard), and
whatever else is written that puts the program and that thing together. That is read here, from her own corpus:
its articles by title, and its hundred million paragraphs by passage.

What a sentence says the program does is marked by rule first (what lets a person do what, what happens when, the
numbers and the limits), and only those sentences go on: asked to turn whole documents into specifications at once,
models oversimplify and invent (ICPC 2025, 603 specifications from 37 documents); marking the sentences first, then
converting those, raised correct extractions by 29%. Each sentence kept says where it was read. Nothing here knows
any program.

Rules over a program's own two articles find little of what it rests on: they are mostly its history. Her model
knows more of what such programs are made of, and is the least sure witness, so it only proposes
(``what_she_would_look_up``): each name it gives is taken only where an article of her corpus is titled by it,
opens by saying it is an algorithm, a format or the like, and speaks of the same work (``deepened``). What no
article bears out is kept as her model's word alone and read from nowhere.
"""
from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

__all__ = ["Known", "Read", "behaviour_stated", "deepened", "known_of", "known_of_async", "rests_on",
           "what_she_would_look_up"]

#: Sentences that say what a program does, by the shapes such sentences take.
_DOES = re.compile(
    r"\b(?:allows?|lets?|enables?|permits?)\s+(?:the\s+)?(?:users?|you|people|one)\s+to\b|"
    r"\b(?:supports?|provides?|includes?|features?|offers?|contains?|comes with|has an?)\b|"
    r"\b(?:can|could|may)\s+(?:be\s+)?(?:used|set|changed|saved|opened|exported|imported|drawn|selected|edited|"
    r"undone|filled|resized|rotated|zoomed|printed|inserted)\b|"
    r"\b(?:when|if|once|whenever)\s+(?:the\s+)?(?:user|you|a\s+\w+)\b|"
    r"\b(?:click|select|press|choose|drag|type|hold)\b\s+(?:the|a|on|and)\b|"
    r"\b(?:up to|at most|maximum of|minimum of|limited to|no more than)\s+\d", re.I)
#: The first sentence of an article about something a program can rest on: it says the thing is one of these.
_KINDS = (r"algorithm|method|technique|procedure|process|(?:file |image |data |document )?format|standard|"
          r"specification|protocol|data structure|data type|encoding|compression|representation|operation|transform|"
          r"filter|effect|tool|feature|notation|grammar|convention|mechanism|pattern|scheme|model|function|"
          r"interface|editor|graphics?|matrix|array|syntax|formula|table|language|implementation")
#: What a found phrase's article must open with: "X is a ... algorithm", "X is the process of ...".
_SAYS_IT_IS_ONE = re.compile(rf"\b(?:is|are|refers to)\s+(?:a|an|the|one of the)\b[^.;]{{0,90}}?\b(?:{_KINDS})\b", re.I)
#: What a name her model gave must say of itself somewhere in its opening: the same kinds, anywhere.
_A_THING_TO_REST_ON = re.compile(rf"\b(?:{_KINDS})\b", re.I)
#: An article that only lists others of the name.
_A_LIST_OF_MEANINGS = re.compile(r"\bmay (?:also )?refer to\b|\bcan (?:also )?refer to\b", re.I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9+#'-]*")
_COMMON = frozenset("""a an and are as at be been by can for from has have in into is it its of on or that the their
this to was were which with you your users user also other such these those using used use may more most some""".split())

#: How many articles of what the program rests on are read, and how many passages beside them.
MOST_RESTS_ON = 24
PASSAGES_EACH = 3
#: How many phrases are looked up as titles, at most: each is a read of her corpus.
MOST_LOOKUPS = 300
#: The most of one article kept, in characters.
ARTICLE_CHARS = 12_000


@dataclass(frozen=True)
class Read:
    """One sentence read, and where: an article's title in her corpus, a passage of another article."""

    text: str
    where: str
    about: str = ""


@dataclass
class Known:
    """What is known of a program: the sentences that say what it does, and what each thing it rests on is."""

    program: str
    does: list[Read] = field(default_factory=list)
    rests_on: dict[str, list[Read]] = field(default_factory=dict)
    read: list[str] = field(default_factory=list)
    #: What her model named for her to look up that no article of her corpus bears out: her model's word alone.
    unborne: list[str] = field(default_factory=list)

    def of(self, words: str, *, at_most: int = 6) -> list[Read]:
        """What was read about a feature, by the words it shares with the feature: what the program does with it,
        then what it rests on."""
        wanted = _terms(words)
        if not wanted:
            return []
        scored = [(len(wanted & _terms(r.text + " " + r.about)), r) for r in self.does]
        scored += [(len(wanted & _terms(name)) * 2 + len(wanted & _terms(r.text)), r)
                   for name, reads in self.rests_on.items() for r in reads]
        return [r for score, r in sorted(scored, key=lambda pair: -pair[0]) if score > 0][:at_most]


def _terms(text: str) -> set[str]:
    return {w.lower().rstrip("s") for w in _WORD.findall(text) if w.lower() not in _COMMON and len(w) > 2}


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(" ".join(str(text or "").split())) if len(s.strip()) > 20]


def behaviour_stated(sentence: str) -> bool:
    """Whether a sentence says what a program does, by its shape alone."""
    return bool(_DOES.search(sentence))


def _phrases(sentence: str) -> Iterable[str]:
    """What in a sentence could be the title of a thing the program rests on: a quoted name ("pencil"), a run of two or
    three words, and a word of one only where it is an acronym (BMP) or a name inside the sentence. A plain word of
    one ("list", "free") names too much to be what a feature rests on."""
    yield from re.findall(r"[\"“]([^\"”]{3,40})[\"”]", sentence)
    words = _WORD.findall(sentence)
    for at, word in enumerate(words):
        if re.fullmatch(r"[A-Z][A-Z0-9]{1,5}", word) or (at > 0 and word[:1].isupper() and len(word) > 3
                                                         and word.lower() not in _COMMON):
            yield word
    for size in (3, 2):
        for start in range(len(words) - size + 1):
            run = words[start:start + size]
            if run[0].lower() not in _COMMON and run[-1].lower() not in _COMMON:
                yield " ".join(run)


#: Words too general to say two things are about the same work.
_GENERAL = frozenset("software program programs application applications computer computers version versions "
                     "developed company system systems used first released".split())


def rests_on(sentences_read: Sequence[str], corpus: Any, *, at_most: int = MOST_RESTS_ON,
             known_titles: Iterable[str] = (), about: str = "") -> dict[str, str]:
    """The articles in her corpus that what the program does rests on: a phrase of a sentence that says what it
    does that is an article's title, whose article opens by saying it is an algorithm, a format, a structure or the
    like, and speaks of the same work as ``about`` (the program, its kind, its article's opening). Title to its
    opening paragraph, in the order the sentences gave them."""
    found: dict[str, str] = {}
    tried: set[str] = {t.lower() for t in known_titles}
    work = _terms(about) - _GENERAL
    for sentence in sentences_read:
        for phrase in _phrases(sentence):
            if len(found) >= at_most or len(tried) >= MOST_LOOKUPS:
                return found
            key = phrase.lower()
            if key in tried:
                continue
            tried.add(key)
            hit, opening = _a_thing_titled(phrase, corpus, work)
            if hit is not None and hit.title.lower() not in {t.lower() for t in (*found, *known_titles)}:
                found[hit.title] = opening
    return found


def _bare(title: str) -> str:
    """A title as its plain name: "BMP file format" and "Zoom (software)" as "bmp" and "zoom"."""
    plain = re.sub(r"\s*\([^)]*\)", "", title)
    plain = re.sub(r"\s+(?:file format|format|algorithm|data structure)$", "", plain, flags=re.I)
    return " ".join(plain.lower().replace("’", "'").split())


def _a_thing_titled(phrase: str, corpus: Any, work: set[str] = frozenset(), *, related_or_plain: bool = False
                    ) -> tuple[Any, str]:
    """The article a phrase names, however the encyclopedia titles it ("BMP file format", "Flood fill", "PNG" for
    "Portable Network Graphics"), whose opening says it is a thing a program can rest on; of several, the one that
    speaks most of the same ``work``. One that speaks of none of it is taken only where ``related_or_plain`` and its
    title carries no qualifier: "Undo" is about undoing in any program, "Zoom (software)" is a video call.
    (None, "") where none is."""
    name, said = _bare(phrase), " ".join(phrase.lower().split())
    kind = _A_THING_TO_REST_ON if related_or_plain else _SAYS_IT_IS_ONE
    candidates = list(corpus.search_titles(phrase, limit=12) or [])
    if related_or_plain and len(phrase.split()) > 1:
        # A name whose article is titled otherwise ("Portable Network Graphics" is "PNG"): its opening says the name.
        candidates += list(corpus.search(phrase, limit=5) or [])
    best, best_opening, best_score = None, "", -1
    for hit in candidates:
        opening = corpus.body(hit.doc_id, max_chars=800)
        first = " ".join(sentences(opening)[:2] if related_or_plain else sentences(opening)[:1])
        named = _bare(hit.title) == name or (related_or_plain and _bare(first[:len(phrase) + 40]).startswith(said))
        if not named or not first or _A_LIST_OF_MEANINGS.search(opening) or not kind.search(first):
            continue
        score = len(work & _terms(opening)) if work else 1
        if score == 0 and not (related_or_plain and "(" not in hit.title):
            continue
        if score > best_score:
            best, best_opening, best_score = hit, opening, score
    return best, best_opening


def _titled(phrase: str, corpus: Any) -> Any:
    return _a_thing_titled(phrase, corpus)[0] or corpus.by_title(phrase)


def known_of(program: str, sources: Sequence[Any], *, corpus: Any = None, passages: Any = None,
             at_most: int = MOST_RESTS_ON) -> Known:
    """What is known of ``program``, from the articles already read about it (``sources``, each with a title and its
    text) and on into what it rests on."""
    if corpus is None:
        from core.knowledge.local_corpus import get_local_corpus_store

        corpus = get_local_corpus_store()
    known = Known(program=program, read=[s.title for s in sources])
    stated: list[str] = []
    for source in sources:
        for sentence in sentences(source.text):
            if behaviour_stated(sentence):
                known.does.append(Read(sentence, f"the article {source.title}"))
                stated.append(sentence)
    about = " ".join([program, *(s.title for s in sources), *(" ".join(sentences(s.text)[:1]) for s in sources)])
    for title, opening in rests_on(stated, corpus, at_most=at_most, known_titles=known.read, about=about).items():
        body = corpus.body(_titled(title, corpus).doc_id, max_chars=ARTICLE_CHARS) or opening
        known.rests_on[title] = [Read(s, f"the article {title}", about=title) for s in sentences(body)
                                 if behaviour_stated(s) or s in sentences(opening)]
        known.read.append(title)
        for passage in _together(program, title, passages):
            known.rests_on[title].append(Read(passage.paragraph, f"a passage of {passage.title}", about=title))
    return known


def _together(program: str, title: str, passages: Any) -> list[Any]:
    """Paragraphs anywhere in her corpus that speak of the program and of what it rests on together."""
    if passages is None:
        try:
            from core.knowledge.passage_index import PassageIndex

            passages = PassageIndex()
        except ImportError:
            return []
    try:
        found = passages.search(f"{program} {title}", limit=PASSAGES_EACH * 3)
    except (OSError, RuntimeError, ValueError):
        return []
    program_words, title_words = _terms(program), _terms(title)
    return [p for p in found if program_words & _terms(p.paragraph) and title_words & _terms(p.paragraph)][:PASSAGES_EACH]


async def known_of_async(program: str, sources: Sequence[Any], **kwargs: Any) -> Known:
    """``known_of`` off the event loop: every lookup in it is a read of her corpus."""
    return await asyncio.to_thread(known_of, program, sources, **kwargs)


class _ToLookUp(BaseModel):
    names: list[str] = Field(default_factory=list, description="titles of encyclopedia articles, one thing each")


async def what_she_would_look_up(program: str, known: Known, ask: Any, *, at_most: int = MOST_RESTS_ON) -> list[str]:
    """What her model says a program like this one is built on, as the titles of articles to read: proposals, each
    taken only where her corpus bears it out (``deepened``)."""
    stated = "\n".join(f"- {r.text}" for r in known.does[:20])
    prompt = (f"The program: {program}. What is written of what it does:\n{stated or '(nothing)'}\n\n"
              f"Name at most {at_most} things a program like this is built on (its algorithms, the file formats it "
              "reads and writes, its data structures, the techniques its features use), each as the title of the "
              "encyclopedia article about that thing.")
    said = await ask(prompt, _ToLookUp, 600)
    names = [n.strip() for n in getattr(said, "names", []) or [] if isinstance(n, str) and n.strip()]
    return list(dict.fromkeys(names))[:at_most]


def deepened(known: Known, names: Sequence[str], *, corpus: Any = None, passages: Any = None) -> Known:
    """``known`` with each name her corpus bears out read in as what the program rests on; the rest kept as her
    model's word alone."""
    if corpus is None:
        from core.knowledge.local_corpus import get_local_corpus_store

        corpus = get_local_corpus_store()
    work = _terms(" ".join([known.program, *known.read])) - _GENERAL
    for name in names:
        if any(_bare(title) == " ".join(name.lower().split()) for title in known.rests_on):
            continue
        hit, opening = _a_thing_titled(name, corpus, work, related_or_plain=True)
        if hit is None:
            known.unborne.append(name)
            continue
        body = corpus.body(hit.doc_id, max_chars=ARTICLE_CHARS) or opening
        known.rests_on[hit.title] = [Read(s, f"the article {hit.title}, named by her model", about=hit.title)
                                     for s in sentences(body) if behaviour_stated(s) or s in sentences(opening)]
        known.read.append(hit.title)
        for passage in _together(known.program, hit.title, passages):
            known.rests_on[hit.title].append(Read(passage.paragraph, f"a passage of {passage.title}", about=hit.title))
    return known

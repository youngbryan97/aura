"""What an exchange is about: the things it names that her library knows, each read by what the exchange means.

Her wants were made of the words an exchange used more often than her exchanges had before (Dunning's log-likelihood
ratio, core/conation/what_her_talk_is_about.py), and a word is not a thing. LIVE 27 November 2026 the six things she
most wanted were "parameter", "hash", "quiet hours", "this reply", "flights" and "too cool". Her own time asked the
web "What are the valid values or types for this parameter?", and asked whether she would set out on something, she
said quiet hours, flights, too cool and this reply "aren't source changes. They're about pace." Bryan: "I dont want
to be keywords. We need real systems engineering there."

What an exchange is about is read the way a linker reads a text against an encyclopedia (TagMe, Ferragina and Scaiella
2010; BLINK, Wu et al. 2020), with what she has:

- What could name a thing comes from the grammar of what was said: its noun phrases and named entities (spaCy). A
  phrase of pronouns names nothing, and neither does an amount or a time. A phrase whose determiner points at
  something at hand, "this reply" or "your page", names a particular thing in the conversation, not a subject. A
  person is kept as a person elsewhere, not as the page about a name.
- What it could name is a document of her library titled by it, or by it with what tells one sense from another
  ("Parameter (computer programming)"). A phrase of several words is also tried without what comes before its head
  ("standard quiet hours", "quiet hours"), never down to the head alone, which names a broader kind than was said.
  A page that lists the senses of a name is where the senses are, never one of them.
- Which of them it names: the encyclopedia's primary sense, the one titled by the name alone, stands unless the
  exchange sets another sense apart from it, read by her encoder, by more than two documents drawn by chance differ
  by. Matched by nearness alone, "Lisbon" beside flights was a 1999 film and "dinner" a film too, since a film's page
  says what it is about in the words an exchange uses. Each sense is read as BLINK reads an entity: its title and the
  first 128 tokens of what it is.
- That the exchange is about it, and does not only pass it, is two things. It comes nearer the exchange than
  documents of her library drawn by chance come, beyond the conventional level, the level divided among the phrases
  that could name something. And it comes nearer than the documents that use its name do, on the whole: a name is
  meant as the document it names where the exchange is nearer that than the name's ordinary company, as TagMe asks how
  often a name is meant as a link. Measured on 28 November 2026, "Lisbon" beside flights came to 0.48 and the documents
  using the name 0.34; "Browser (cat)" in a game made in a browser came to 0.33 and they 0.37.

Where her library has no title index (tools/index_corpus_titles.py) or her encoder is not loaded, nothing is read as
a subject: a word is not one.
"""
from __future__ import annotations

import logging
import math
import random
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.WhatItIsAbout")

__all__ = ["CHANCE_DRAWN", "OPENING_CHARS", "READ_AS", "Subject", "WhatItIsAbout", "WRONG", "name_of", "shares_of"]

#: How what an exchange is about is read here, as her record of what she has met keeps it.
READ_AS = "things her library knows"

#: The chance of calling a thing a subject that is not one: the conventional level.
WRONG = 0.05
#: How many documents of her library are drawn by chance, to say how near a document comes to an exchange by chance:
#: the fewest whose spread, estimated from them, brings a one-sided test's critical value within five in a hundred of
#: the value a known spread would give (Student's t at 20 degrees of freedom, 1.725, against 1.645).
CHANCE_DRAWN = 21
#: How much of a document's start is read as what it is: the 128 tokens BLINK reads, at the 4.17 characters a token
#: her encoder's tokenizer gave the openings of 200 documents of her library drawn by chance (28 November 2026;
#: quartiles 3.78 and 4.55).
OPENING_CHARS = round(128 * 4.17)
#: The grammar model whose noun phrases and named entities say what could name a thing: the large one. The medium one
#: read "a hedgehog rolls downhill" with "rolls" a noun, so "hedgehog" was never tried; the large reads it a verb (28
#: November 2026), in the same 4 ms.
GRAMMAR = "en_core_web_lg"
#: How many documents a name's senses are read from: its primary sense and the 33 senses that cover all but one name in
#: a thousand of her library whole (400,163 names with senses on 28 November 2026: a median of 1, the 99.9th
#: percentile 33, the most 483).
SENSES = 1 + 33
#: How many names' company is kept once found: as many as the things she keeps (core/conation/what_she_wants.py).
NAMES_KEPT = 4096
#: What her encoder is asked of an exchange: the instruction is part of the query and never of a document, as her
#: encoder reads every task (core/memory/embedding_model.py).
TASK = "what_it_is_about"
#: The kinds of named entity that are amounts and times: nothing to know more of as a thing.
_AMOUNTS_AND_TIMES = frozenset({"DATE", "TIME", "CARDINAL", "ORDINAL", "QUANTITY", "PERCENT", "MONEY"})
#: People: kept as people, not as the page about a name.
_PEOPLE = frozenset({"PERSON"})
#: A title's last part in brackets, which tells one sense of a name from another.
_SENSE = re.compile(r"\s*\(([^)]*)\)\s*$")
#: How the encyclopedia marks a page that lists the senses of a name: in its title, and in its opening.
_LISTS_SENSES = "disambiguation"
_MAY_REFER_TO = re.compile(r"^\S.{0,200}?\bmay (?:also )?refer to\b", re.S)

#: Her encoder's vectors for some texts, only where it is loaded already; [] otherwise.
Vectors = Callable[[list[str]], list[Any]]


@dataclass(frozen=True)
class Subject:
    """A thing an exchange is about: a document of her library, and what of the exchange named it."""

    title: str
    doc_id: int
    #: How near her encoder reads it to the exchange, as a cosine.
    near: float
    #: How far beyond chance that is, in Student's t, and the chance of coming as near by chance.
    beyond_chance: float
    chance: float
    #: How the exchange said it.
    said_as: str
    #: Which of the texts read named it, by their place.
    named_in: frozenset[int]


def name_of(title: str) -> str:
    """What a document's title calls the thing, without what tells its sense from another's: "Parameter" of
    "Parameter (computer programming)", "Monterey" of "Monterey, California"."""
    return _REGION.sub("", _SENSE.sub("", str(title or ""))).strip()


#: How the encyclopedia titles a place by where it is: "Monterey, California".
_REGION = re.compile(r",\s[^,]*$")


def shares_of(subjects: Sequence[Subject]) -> dict[str, float]:
    """Each subject's share of how far beyond chance the subjects of one exchange came, all of them together."""
    held = {subject.title: max(0.0, subject.beyond_chance) for subject in subjects}
    if not held:
        return {}
    total = sum(value for value in held.values() if math.isfinite(value))
    if not total or any(not math.isfinite(value) for value in held.values()):
        return {title: 1.0 / len(held) for title in held}
    return {title: value / total for title, value in held.items()}


def _her_vectors(texts: list[str]) -> list[Any]:
    from core.cognition.evidence_relevance import vectors_if_ready

    return vectors_if_ready(texts)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _t_beyond(value: float, drawn: Sequence[float]) -> tuple[float, float]:
    """How far ``value`` lies beyond what was ``drawn`` by chance, as a new draw would: Student's t for one more draw
    from what the drawn ones say of chance, and the one-sided chance of a draw coming as far."""
    from scipy.stats import t as student

    n = len(drawn)
    mean = sum(drawn) / n
    spread = math.sqrt(sum((x - mean) ** 2 for x in drawn) / (n - 1))
    beyond = (value - mean) / (spread * math.sqrt(1.0 + 1.0 / n)) if spread > 0.0 else math.inf
    return beyond, float(student.sf(beyond, n - 1))


def _chance_apart(nearer: float, than: float, drawn: Sequence[float]) -> float:
    """The one-sided chance that one document comes ``nearer`` an exchange than another comes (``than``) by as much as
    it does, by chance: the difference of two draws has the spread of one draw times the root of two."""
    from scipy.stats import t as student

    n = len(drawn)
    mean = sum(drawn) / n
    spread = math.sqrt(sum((x - mean) ** 2 for x in drawn) / (n - 1))
    if spread <= 0.0:
        return 0.0 if nearer > than else 1.0
    return float(student.sf((nearer - than) / (spread * math.sqrt(2.0)), n - 1))


def _lists_senses(title: str, opening: str) -> bool:
    sense = _SENSE.search(title)
    return bool(sense and sense.group(1).strip().casefold() == _LISTS_SENSES) or bool(_MAY_REFER_TO.match(opening))


def _points_at_what_is_at_hand(chunk: Any) -> bool:
    """Whether a noun phrase is determined by something pointing at a particular thing at hand: "this reply", "that
    page", "your app", "my code". Read off the grammar of its determiners (spaCy's morphology), so any word that
    points that way counts and no word is listed."""
    for token in chunk:
        if token.i >= chunk.root.i:
            break
        morph = token.morph
        if "Dem" in morph.get("PronType"):
            return True
        if "Yes" in morph.get("Poss") and set(morph.get("Person")) & {"1", "2"}:
            return True
    return False


class WhatItIsAbout:
    """What an exchange is about, read against her library by her encoder."""

    def __init__(self, *, store: Any = None, vectors: Vectors | None = None, grammar: Any = None) -> None:
        self._store = store
        self._vectors = vectors or _her_vectors
        self._grammar = grammar
        self._lock = checked_lock("conation.what_an_exchange_is_about", reentrant=True)
        self._chance: list[Any] | None = None
        #: For each name, the openings of the documents that use it, found once.
        self._company: dict[str, list[str]] = {}
        #: What each exchange read came to, by its texts, the latest last; and whether one is being read beside her.
        self._read_before: dict[tuple[str, ...], list[Subject]] = {}
        self._reading = False
        self._last: list[Subject] = []
        self._said_why_not = ""

    # ── what she has to read with ────────────────────────────────────

    def store(self) -> Any:
        if self._store is None:
            from core.knowledge.local_corpus import get_local_corpus_store

            self._store = get_local_corpus_store()
        return self._store

    def grammar(self) -> Any:
        with self._lock:
            if self._grammar is None:
                import spacy

                self._grammar = spacy.load(GRAMMAR)
            return self._grammar

    def why_not(self) -> str:
        """Why nothing can be read as a subject now; "" where something can."""
        try:
            if not self.store().has_title_index():
                return "her library has no title index (tools/index_corpus_titles.py builds it)"
        except (AttributeError, OSError, RuntimeError, ValueError) as exc:
            return f"her library cannot be read: {exc}"
        if not self._vectors(["what an exchange is about"]):
            return "her encoder is not loaded"
        return ""

    def _opening(self, doc_id: int, title: str) -> str:
        return f"{title}: " + " ".join(str(self.store().body(doc_id, max_chars=OPENING_CHARS) or "").split())

    def _drawn_by_chance(self) -> list[Any]:
        """The vectors of CHANCE_DRAWN documents of her library drawn by chance, drawn once: what nearness by chance is."""
        with self._lock:
            if self._chance is not None:
                return self._chance
        store, rng = self.store(), random.Random(CHANCE_DRAWN)
        top = store.largest_id()
        openings: list[str] = []
        tries = 0
        while top and len(openings) < CHANCE_DRAWN and tries < 50 * CHANCE_DRAWN:
            tries += 1
            found = store.documents([rng.randint(1, top)])
            if not found:
                continue
            opening = self._opening(found[0].doc_id, found[0].title)
            if len(opening) > len(found[0].title) + 2 and not _lists_senses(found[0].title, opening):
                openings.append(opening)
        vectors = self._vectors(openings) if len(openings) == CHANCE_DRAWN else []
        with self._lock:
            # Kept only once read whole: an encoder not loaded yet is asked again on the next exchange.
            if len(vectors) == CHANCE_DRAWN:
                self._chance = list(vectors)
            return list(vectors)

    def _company_of(self, name: str, named: int) -> list[str]:
        """The openings of the documents of her library that use ``name`` most, as many as are drawn by chance, but
        the one it names and pages that list senses."""
        key = name.casefold()
        with self._lock:
            held = self._company.get(key)
        if held is None:
            held = []
            for hit in self.store().search(name, limit=CHANCE_DRAWN + 2):
                opening = self._opening(hit.doc_id, hit.title)
                if not _lists_senses(hit.title, opening):
                    held.append(f"{hit.doc_id}\t{opening}")
            with self._lock:
                self._company[key] = held
                while len(self._company) > NAMES_KEPT:
                    self._company.pop(next(iter(self._company)))
        return [line.split("\t", 1)[1] for line in held if int(line.split("\t", 1)[0]) != named][:CHANCE_DRAWN]

    # ── reading ──────────────────────────────────────────────────────

    def _phrases(self, text: str) -> dict[str, list[str]]:
        """What in a text could name a thing, each with the ways it is tried: as said, in the base forms of its words,
        and, for several words, without what comes before its head."""
        doc = self.grammar()(text)
        unnamed = {token.i for ent in doc.ents if ent.label_ in _AMOUNTS_AND_TIMES | _PEOPLE for token in ent}
        out: dict[str, list[str]] = {}
        for ent in doc.ents:
            if ent.label_ not in _AMOUNTS_AND_TIMES | _PEOPLE:
                out.setdefault(ent.text, [ent.text])
        for chunk in doc.noun_chunks:
            if any(token.i in unnamed for token in chunk) or _points_at_what_is_at_hand(chunk):
                continue
            words = [token for token in chunk if token.pos_ not in {"DET", "PRON", "PUNCT"}]
            if not any(token.pos_ in {"NOUN", "PROPN"} for token in words):
                continue
            forms: list[str] = []
            for start in range(len(words)):
                tail = words[start:]
                if start and len(tail) < 2:
                    break
                forms += [" ".join(token.text for token in tail), " ".join(token.lemma_ for token in tail)]
            out.setdefault(" ".join(token.text for token in words), list(dict.fromkeys(forms)))
        return out

    def read(self, *texts: str) -> list[Subject]:
        """The subjects of an exchange of ``texts``, most beyond chance first; [] where nothing can be read as one."""
        said = [str(text or "") for text in texts]
        whole = " ".join(" ".join(text.split()) for text in said if text.strip())
        if not whole:
            return []
        why_not = self.why_not()
        if why_not:
            if why_not != self._said_why_not:
                self._said_why_not = why_not
                record_degradation("what_an_exchange_is_about", RuntimeError(why_not), severity="info",
                                   action="nothing was read as a subject of what was said")
            return []
        try:
            found = self._read(said, whole)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            record_degradation("what_an_exchange_is_about", exc, severity="warning",
                               action="nothing was read as a subject of what was said")
            return []
        if found is None:
            return []
        with self._lock:
            self._read_before[self._key(said)] = found
            while len(self._read_before) > NAMES_KEPT:
                self._read_before.pop(next(iter(self._read_before)))
        return found

    @staticmethod
    def _key(texts: Sequence[str]) -> tuple[str, ...]:
        return tuple(" ".join(str(text or "").split()) for text in texts)

    def read_already(self, *texts: str) -> list[Subject] | None:
        """What these texts were read to be about, where they have been read; None where they have not. Reads nothing,
        so it can be asked where somebody is waiting."""
        with self._lock:
            held = self._read_before.get(self._key(texts))
        return list(held) if held is not None else None

    def read_soon(self, *texts: str) -> list[Subject] | None:
        """What was read of these texts already; where nothing was, a reading of them is begun beside her, one at a
        time, and None now."""
        held = self.read_already(*texts)
        if held is not None:
            return held
        with self._lock:
            if self._reading:
                return None
            self._reading = True

        def reading() -> None:
            try:
                self.read(*texts)
            finally:
                with self._lock:
                    self._reading = False

        threading.Thread(target=reading, name="what_an_exchange_is_about", daemon=True).start()
        return None

    def _senses(self, form: str) -> tuple[Any, list[tuple[Any, str]]]:
        """What ``form`` could name: the document titled by it alone, the encyclopedia's primary sense, where there is
        one, first; and the senses titled by it with what tells them apart. Each with its opening; a page that lists
        senses is none of them."""
        family = self.store().titled(form, limit=SENSES)
        primary = next((hit for hit in family if hit.title.casefold() == form.casefold()), None)
        held: list[tuple[Any, str]] = []
        for hit in sorted(family, key=lambda hit: hit is not primary):
            opening = self._opening(hit.doc_id, hit.title)
            if not _lists_senses(hit.title, opening):
                held.append((hit, opening))
        return (primary if held and held[0][0] is primary else None), held

    def _read(self, said: list[str], whole: str) -> list[Subject] | None:
        """The subjects of an exchange; None where it could not be read now (her encoder gave no vectors)."""
        named: dict[str, set[int]] = {}
        forms: dict[str, list[str]] = {}
        for place, text in enumerate(said):
            for phrase, tried in self._phrases(text).items() if text.strip() else ():
                named.setdefault(phrase, set()).add(place)
                forms.setdefault(phrase, tried)
        senses: dict[str, tuple[Any, list[tuple[Any, str]]]] = {}
        for phrase, tried in forms.items():
            # The first way of saying it that has a primary sense, or failing that the first with any: "apples" is
            # titled only as a card game, and "apple" as the fruit.
            found = [(primary, held) for primary, held in (self._senses(form) for form in tried) if held]
            if found:
                senses[phrase] = next((pair for pair in found if pair[0] is not None), found[0])
        if not senses:
            return []
        chance = self._drawn_by_chance()
        if len(chance) != CHANCE_DRAWN:
            return None
        openings = [opening for _primary, held in senses.values() for _hit, opening in held]
        from core.memory.embedding_model import query_prompt

        vectors = self._vectors([query_prompt(TASK) + whole, *openings])
        if len(vectors) != len(openings) + 1:
            return None
        exchange, read = vectors[0], iter(vectors[1:])
        by_chance = [_cosine(exchange, vector) for vector in chance]
        # The level is divided among the phrases that could name something: the more there are, the further beyond
        # chance each must come.
        level = WRONG / len(senses)
        subjects: dict[int, Subject] = {}
        for phrase, (primary, held) in senses.items():
            near = [(_cosine(exchange, next(read)), hit) for hit, _opening in held]
            ranked = sorted(near, key=lambda pair: -pair[0])
            if primary is not None:
                # The primary sense stands unless the exchange sets another apart from it by more than two documents
                # drawn by chance differ by, the level divided among the senses it is set against.
                chosen, other = near[0], max(near[1:], key=lambda pair: pair[0], default=None)
                if other is not None and _chance_apart(other[0], chosen[0], by_chance) < WRONG / (len(near) - 1):
                    chosen = other
            elif len(ranked) == 1 or _chance_apart(ranked[0][0], ranked[1][0], by_chance) < WRONG / (len(ranked) - 1):
                # With no primary sense, the one the exchange sets apart from the next nearest, or the only one there is.
                chosen = ranked[0]
            else:
                continue
            beyond, chance_of_it = _t_beyond(chosen[0], by_chance)
            if chance_of_it >= level:
                continue
            company = self._company_of(phrase, chosen[1].doc_id)
            if len(company) > 1:
                theirs = self._vectors(company)
                if len(theirs) == len(company) and chosen[0] <= sum(_cosine(exchange, v) for v in theirs) / len(theirs):
                    continue
            hit, held_before = chosen[1], subjects.get(chosen[1].doc_id)
            said_in = frozenset(named[phrase] | (held_before.named_in if held_before else set()))
            if held_before is None or chosen[0] > held_before.near:
                subjects[hit.doc_id] = Subject(hit.title, hit.doc_id, chosen[0], beyond, chance_of_it, phrase, said_in)
            else:
                subjects[hit.doc_id] = Subject(held_before.title, held_before.doc_id, held_before.near,
                                               held_before.beyond_chance, held_before.chance, held_before.said_as, said_in)
        return sorted(subjects.values(), key=lambda subject: -subject.beyond_chance)

    # ── as her record of what she meets reads it ────────────────────

    def about(self, *texts: str) -> list[str]:
        """The titles of what an exchange is about, most beyond chance first; kept as the last exchange read."""
        subjects = self.read(*texts)
        with self._lock:
            self._last = subjects
        return [subject.title for subject in subjects]

    def last(self) -> list[Subject]:
        with self._lock:
            return list(self._last)

    def shares(self) -> dict[str, float]:
        """Each subject's share of how far beyond chance the last exchange read came, all of them together."""
        return shares_of(self.last())

    def subjects_of(self, *texts: str) -> list[str]:
        """What ``about`` would say of these texts, without keeping them as the last exchange read."""
        return [subject.title for subject in self.read(*texts)]

"""What her corpus says about who and what a conversation is about, whether it is asked or said.

Her chat looked things up only for a question about the world. In a
conversation about sport (5 October) she misstated a well-known player's
awards, and she disputed, as unrecorded, an accurate and documented claim the
person made about another player. Her corpus settled both. A statement names
its subjects as well as a question does, and "he" names the subject of the
turns before it.

Mentions are a turn's names (core/knowledge/passages_about.named_spans) in its
prose, paths and addresses left out, a name ending at a comma or other break.
A sentence's first word counts only where the person capitalises it elsewhere
too, mid-sentence or inside a longer name; "I" and its contractions, and a
speaker's own name, never do.
The names whose words the conversation uses most come first, a whole name
before its parts.

A whole name of several words is linked to its own page. Otherwise a mention
is linked to the encyclopedia article, holding it in its title, that shares
the most of the turn's other words, so "Kobe" beside "81 point game" is
the player and not the city, and to none when no such article shares any. An
abbreviation is linked only to its own page, and a disambiguation page is
about no one. A turn that names no one but speaks of "he", "she" or "they"
carries the subjects of the person's own turns before it, then of hers.

With her embedder, every paragraph of the subjects' articles and the corpus's
paragraphs about the subjects as they are talked about compete by meaning for
the turn's few places. Without it, each subject's opening paragraph, where an
article sums up its subject, comes first, then its paragraphs holding the
turn's rarer words.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Sequence
from typing import Any

#: A sentence ends at its stop or at a line break.
_SENTENCE = re.compile(r"(?<=[.!?…])\s+|\n+")
_FIRST_WORD = re.compile(r"[^\W_][\w'’.-]*", re.UNICODE)
#: Where a run of capitalised words stops being one name.
_CLAUSE = re.compile(r"[,;:&()\[\]/#]+|\s[—–]\s|\s-\s")
#: Addresses and file paths: text a turn carries, not prose it speaks.
_NOT_PROSE = re.compile(r"\b[a-z][a-z0-9+.-]*://\S+|(?<!\w)(?:~|\.{1,2})?/\S+|\b\S+\.(?:html?|py|md|json|txt|js|css)\b",
                        re.IGNORECASE)
#: "I" and its contractions are capitalised by grammar.
_FIRST_PERSON = re.compile(r"^I(?:['’][a-z]+)?$")
#: Pronouns that point back to a person named before.
_THIRD_PERSON = re.compile(r"\b(?:he|him|his|she|her|hers|they|them|their)\b", re.IGNORECASE)
#: The opening of a page that only lists the subjects sharing a name.
_DISAMBIGUATION = re.compile(r"\bmay (?:also )?refer to\b", re.IGNORECASE)

Rerank = Callable[[str, list[str]], Sequence[float]]
Turn = tuple[str, str]  # (role, text): "user" for the person, anything else for her


def _prose(text: str) -> str:
    return _NOT_PROSE.sub(" ", str(text or ""))


def _capitalised_by_choice(texts: Sequence[str]) -> set[str]:
    """Words capitalised where grammar does not require it: past a sentence's first word, or inside a longer name."""
    from core.knowledge.passages_about import _POSSESSIVE, _is_capitalised, named_spans

    chosen: set[str] = set()
    for text in texts:
        for sentence in _SENTENCE.split(_prose(text)):
            for word in _FIRST_WORD.findall(sentence)[1:]:
                bare = _POSSESSIVE.sub("", word).strip(".’'")
                if len(bare) > 1 and _is_capitalised(bare) and not _FIRST_PERSON.match(bare):
                    chosen.add(bare)
            for clause in _CLAUSE.split(sentence):
                chosen.update(w for span in named_spans(clause) if len(span.split()) > 1 for w in span.split())
    return chosen


def mentions(text: str, *, speakers: Sequence[str] = (), conversation: Sequence[str] = (),
             evidence: Sequence[str] = ()) -> list[str]:
    """The names a text uses, those the conversation uses most first, then the longest.

    ``evidence`` is the text whose capitals count as evidence of names besides
    ``text`` itself: the person's own turns. Her replies are left out of it,
    because their headings and lists capitalise words that name nothing.
    """
    from core.knowledge.passages_about import _POSSESSIVE, named_spans

    own = {s.lower() for s in speakers}
    by_choice = _capitalised_by_choice([text, *evidence])
    found: list[str] = []
    for sentence in _SENTENCE.split(_prose(text)):
        words = _FIRST_WORD.findall(sentence)
        first = _POSSESSIVE.sub("", words[0]).strip(".’'") if words else ""
        for clause in _CLAUSE.split(sentence):
            for span in named_spans(clause):
                parts = [p for p in span.split() if not _FIRST_PERSON.match(p)]
                if parts and parts[0] == first and first not in by_choice:
                    parts = parts[1:]
                if not parts or parts[0][:1].isdigit() or (len(parts) == 1 and len(parts[0]) < 2):
                    continue
                name = " ".join(parts)
                if name.lower() not in own:
                    found.append(name)
    # A subject is what keeps being talked about: by how often its most used word comes up, the whole name first.
    said = re.findall(r"\w+", " ".join(_prose(t) for t in (text, *conversation)).lower())
    uses = {word: said.count(word) for word in {w.lower() for name in found for w in re.findall(r"\w+", name)}}
    return sorted(dict.fromkeys(found), key=lambda s: (
        -max((uses.get(w.lower(), 0) for w in re.findall(r"\w+", s)), default=0), -len(s.split())))


def _is_abbreviation(name: str) -> bool:
    return len(name.split()) == 1 and name.isupper()


def _about_one_subject(store: Any, hit: Any) -> bool:
    """An encyclopedia article about one subject: not a note she saved, not a list of namesakes."""
    from core.knowledge.local_corpus import RETAINED_SOURCE

    return hit.source != RETAINED_SOURCE and not _DISAMBIGUATION.search(store.body(hit.doc_id, max_chars=400))


def _link(store: Any, mention: str, words: set[str], *, deadline_s: float, everywhere: Any = None) -> Any:
    """The encyclopedia article for ``mention`` that the words around it point to; None when there is none.

    With the index of every paragraph, one search: the mention's words in the
    title, ranked by the other words (passage_index.articles_titled). Without
    it, the article holding the mention in its title that shares the most of
    the words.
    """
    from core.knowledge.local_corpus import RETAINED_SOURCE
    from core.knowledge.passages_about import _terms

    expires = time.monotonic() + max(0.01, deadline_s)
    own = {w.lower() for w in mention.split()}
    other = {w for w in words if w not in own}
    if everywhere is not None:
        from core.knowledge.corpus_grounding import _CANDIDATE_POOL_FACTOR

        if _is_abbreviation(mention):
            exact = store.by_title(mention, deadline_s=max(0.01, expires - time.monotonic()))
            return exact if exact is not None and _about_one_subject(store, exact) else None
        found = store.documents(everywhere.articles_titled(mention, sorted(other),
                                                           limit=_CANDIDATE_POOL_FACTOR * _CANDIDATE_POOL_FACTOR))
        whole = [hit for hit in found if len(mention.split()) > 1 and hit.title.lower() == mention.lower()]
        for hit in [*whole, *found]:
            if _about_one_subject(store, hit):
                return hit
        return None
    exact = store.by_title(mention, deadline_s=max(0.01, expires - time.monotonic()))
    # An abbreviation names many things: it counts only where its own page is about one of them.
    titled = [] if _is_abbreviation(mention) else store.search_titles(
        mention, limit=5, deadline_s=max(0.01, expires - time.monotonic()), about=sorted(other))
    best, best_shared = None, 0
    for hit in [*([exact] if exact is not None else []), *titled]:
        if hit.source == RETAINED_SOURCE:
            continue
        body = store.body(hit.doc_id, max_chars=10**7)
        if not body or _DISAMBIGUATION.search(body[:400]):
            continue
        if hit is exact and len(mention.split()) > 1:
            # Two subjects rarely share a whole name: its own page is the one meant.
            return hit
        shared = len(other & _terms(body))
        if shared > best_shared:
            best, best_shared = hit, shared
    return best


def _carried_from(before: Sequence[Turn]) -> list[str]:
    """Where a pronoun looks for its person: the person's own turns, newest first, then hers.

    "He" in what the person says most often goes on from what the person last
    named; her replies name many things in passing.
    """
    theirs = [text for role, text in reversed(before) if role == "user"]
    hers = [text for role, text in reversed(before) if role != "user"]
    return [*theirs, *hers]


def points_back(turn: str) -> bool:
    """Whether a turn speaks of "he", "she" or "they", a person its own words may not name."""
    return bool(_THIRD_PERSON.search(_prose(turn)))


def subjects(turn: str, before: Sequence[Turn], *, store: Any, deadline_s: float, most: int,
             speakers: Sequence[str] = (), everywhere: Any = None) -> list[Any]:
    """Articles for what the turn names, or for whom its pronouns point back to; at most ``most``."""
    from core.knowledge.passages_about import _terms

    expires = time.monotonic() + max(0.0, deadline_s)
    texts = [text for _role, text in before]
    theirs = [text for role, text in before if role == "user"]
    looking = [turn, *(_carried_from(before) if _THIRD_PERSON.search(_prose(turn)) else ())]
    for index, text in enumerate(looking):
        # A name is linked by the words said around it: the turn's own, and for a carried one, its turn's too.
        words = _terms(_prose(turn)) | (_terms(_prose(text)) if index else set())
        linked: list[Any] = []
        for mention in mentions(text, speakers=speakers, conversation=[turn, *texts], evidence=theirs):
            left = expires - time.monotonic()
            if left <= 0 or len(linked) >= most:
                return linked
            hit = _link(store, mention, words, deadline_s=left, everywhere=everywhere)
            if hit is not None and all(hit.doc_id != s.doc_id for s in linked):
                linked.append(hit)
        if linked:
            return linked
    return []


def said_about(turn: str, before: Sequence[Turn] = (), *, store: Any = None, limit: int, deadline_s: float,
               rerank: Rerank | None = None, speakers: Sequence[str] = (),
               everywhere: Any = None) -> list[tuple[str, str]]:
    """(title, paragraph) about the subjects of the conversation, at most ``limit``.

    With ``rerank`` (her embedder) and ``everywhere``
    (core/knowledge/passage_index.PassageIndex), the corpus's paragraphs about
    the subjects as they are talked about compete with the subjects' own:
    what is said about a subject often turns on others it does not name, as a
    ranking of players names the one above them. ``everywhere`` also links a
    mention to its article in one search.
    """
    from core.knowledge.passages_about import _paragraphs

    if store is None:
        from core.knowledge.local_corpus import LocalCorpusStore

        store = LocalCorpusStore()
    expires = time.monotonic() + max(0.0, deadline_s)
    found = subjects(turn, before, store=store, deadline_s=deadline_s, most=max(1, limit), speakers=speakers,
                     everywhere=everywhere)
    if not found:
        return []
    said = " ".join(str(turn or "").split())
    split = {hit.doc_id: list(_paragraphs(store.body(hit.doc_id, max_chars=10**7))) for hit in found}
    found = [hit for hit in found if split[hit.doc_id]]
    if not found:
        return []
    if rerank is None:
        # By words alone, each subject's opening, then its own paragraphs holding the turn's rarer words. The
        # rest of the corpus is left out: a chatty turn's common words find short pages of their own.
        openings = [(hit.title, split[hit.doc_id][0]) for hit in found][:max(1, limit)]
        rest = [(hit.title, p) for hit in found for p in split[hit.doc_id][1:]]
        return openings + _by_words(turn, rest)[:max(0, limit - len(openings))]
    # By meaning, everything competes: each subject's paragraphs, its opening among them, and the paragraphs
    # of the whole corpus about the subjects as they are being talked about. A stray link loses to them.
    titles = "; ".join(hit.title for hit in found)
    pool = [(hit.title, p) for hit in found for p in split[hit.doc_id]]
    if everywhere is not None and time.monotonic() < expires:
        from core.knowledge.corpus_grounding import _CANDIDATE_POOL_FACTOR

        pool += [(p.title, p.paragraph) for p in everywhere.search(f"{titles} {said}",
                                                                   limit=limit * _CANDIDATE_POOL_FACTOR)
                 if (p.title, p.paragraph) not in pool]
    similarity = list(rerank(f"{titles}: {said}", [f"{title}: {paragraph}" for title, paragraph in pool]))
    order = sorted(range(len(pool)), key=lambda i: -similarity[i])
    return [pool[i] for i in order[:max(1, limit)]]


def _by_words(turn: str, paragraphs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """The paragraphs holding the turn's words, rarer words counting for more among these paragraphs."""
    from core.knowledge.passages_about import _terms

    asked = _terms(turn)
    held = [asked & _terms(p) for _t, p in paragraphs]
    df = {term: sum(term in h for h in held) for term in asked}
    weights = [sum(math.log(len(paragraphs) / df[t]) for t in h) for h in held]
    order = sorted((i for i in range(len(paragraphs)) if weights[i] > 0), key=lambda i: -weights[i])
    return [paragraphs[i] for i in order]


__all__ = ["mentions", "points_back", "said_about", "subjects"]

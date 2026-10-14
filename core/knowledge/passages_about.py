"""The paragraphs of her corpus about what a question names, for answering a knowledge question with care.

Her conversational grounding (core/knowledge/corpus_grounding.py) is built for
a chat turn's latency: a quarter-second search, a conservative gate on the
question's wording, three passages. Measured on 60 development questions of
SimpleQA (14 October), it returned passages for 9, and the answer was in them
for 2, though the questions' sources are Wikipedia pages her corpus holds.
Half the questions never reached a search ("In what year did conductor
David Randolph marry Mildred Greenberg?" is not worded the way the gate
expects), and the rest searched on the question's first six content words,
which for a long question leaves out its subject.

A knowledge question is almost always about something it names. So this reads
the names first: each run of capitalised words, a name with its joining words
("Architectural League of America", "Ahmad al-Kabīr"), and anything quoted.
For each name the article of that title is taken; else, among the articles
whose title holds every word of the name, the one whose title adds least to it
(the rule her conversational grounding orders titles by); else the search hit
whose title shares most of the name. Inside those articles each paragraph is
scored by the question's content words it carries, each weighted by how rare
it is among the article's paragraphs, so the subject's own name, which is in
every paragraph, decides nothing. The best paragraphs are returned with their
titles. No gate on wording: the caller decides a question deserves care.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Sequence
from typing import Any

__all__ = ["named_spans", "passages_about"]

#: Words that join the parts of one name ("Bank of England", "Leonardo da Vinci", "Ibn al-Haytham").
_JOINING = frozenset("of de da del della di du la le von van der den al el bin ibn y and for the".split())
_WORD = re.compile(r"[^\W_][\w'’.-]*", re.UNICODE)
_QUOTED = re.compile(r"[\"“]([^\"”]{2,80})[\"”]")
_POSSESSIVE = re.compile(r"['’]s$")


def _is_capitalised(word: str) -> bool:
    """A word that starts a name: capitalised, a code like "F2A-3", or a particle joined to a name ("al-Kabīr")."""
    if word[:1].isupper() or (word[:1].isdigit() and any(c.isalpha() for c in word)):
        return True
    particle, _, rest = word.partition("-")
    return particle.lower() in _JOINING and rest[:1].isupper()


def named_spans(question: str) -> list[str]:
    """Names in a question, longest first: quoted strings and runs of capitalised words with their joining words."""
    text = str(question or "")
    spans = [m.strip() for m in _QUOTED.findall(text)]
    words = _WORD.findall(text)
    current: list[str] = []
    for index, word in enumerate(words):
        bare = _POSSESSIVE.sub("", word).strip(".’'")
        if _is_capitalised(bare) and not (index == 0 and bare.lower() in _OPENERS):
            current.append(bare)
        elif current and bare.lower() in _JOINING and index + 1 < len(words) and _is_capitalised(words[index + 1]):
            current.append(bare)
        else:
            if current:
                spans.append(" ".join(current))
            current = []
    if current:
        spans.append(" ".join(current))
    cleaned = [s.strip(" ,;:.") for s in spans]
    unique = list(dict.fromkeys(s for s in cleaned if s and s.lower() not in _OPENERS))
    return sorted(unique, key=lambda s: -len(s.split()))


#: A question's first word is capitalised by grammar, not because it names anything.
_OPENERS = frozenset("what which who whom whose when where why how in on during at for by from to a an the is was "
                     "are were did does do name give list".split())


def _terms(text: str) -> set[str]:
    from core.knowledge.corpus_grounding import _QUESTION_SCAFFOLDING

    return {w.lower() for w in _WORD.findall(str(text or "")) if len(w) > 2 and w.lower() not in _QUESTION_SCAFFOLDING}


def _article_for(store: Any, name: str, deadline_s: float) -> Any:
    from core.knowledge.corpus_grounding import _title_distance

    started = time.monotonic()
    hit = store.by_title(name, deadline_s=deadline_s)
    if hit is not None:
        return hit
    deadline_s -= time.monotonic() - started
    if deadline_s <= 0:
        return None
    titled = store.search_titles(name, limit=10, deadline_s=deadline_s) if hasattr(store, "search_titles") else []
    if titled:
        return min(titled, key=lambda candidate: _title_distance(name, candidate.title))
    wanted = _terms(name) or {w.lower() for w in name.split()}
    best, best_shared = None, 0
    for candidate in store.search(name, limit=5, deadline_s=deadline_s) or ():
        shared = len(wanted & _terms(candidate.title))
        if shared > best_shared:
            best, best_shared = candidate, shared
    return best


Rerank = Callable[[str, list[str]], Sequence[float]]


def passages_about(question: str, *, store: Any = None, limit: int | None = None,
                   deadline_s: float | None = None, rerank: Rerank | None = None) -> list[tuple[str, str]]:
    """(title, paragraph) pairs about what ``question`` names, best first; [] when it names nothing found.

    With ``rerank`` (her embedder's similarity of each paragraph to the
    question), the best paragraphs by words are a candidate pool, as large as
    her conversational grounding fetches for the passages it carries, and the
    pool is ordered by meaning before ``limit`` are taken: an answer is often
    worded otherwise than its question.

    ``deadline_s`` bounds the whole lookup, not each search: a name not
    reached in time is not looked up, and what was found by then is returned.
    """
    from core.knowledge.corpus_grounding import _CANDIDATE_POOL_FACTOR, DEFAULT_LIMIT
    from core.knowledge.local_corpus import SEARCH_DEADLINE_S

    if store is None:
        from core.knowledge.local_corpus import LocalCorpusStore

        store = LocalCorpusStore()
    limit = DEFAULT_LIMIT if limit is None else limit
    deadline_s = SEARCH_DEADLINE_S if deadline_s is None else deadline_s
    expires = time.monotonic() + max(0.0, float(deadline_s))
    asked = _terms(question)
    articles: list[Any] = []
    for name in named_spans(question):
        left = expires - time.monotonic()
        if left <= 0:
            break
        article = _article_for(store, name, left)
        if article is not None and all(a.doc_id != article.doc_id for a in articles):
            articles.append(article)
    scored: list[tuple[float, int, str, str]] = []
    for order, article in enumerate(articles):
        if order and time.monotonic() > expires:
            break
        # The whole article: an answer is often well past its opening.
        paragraphs = list(_paragraphs(store.body(article.doc_id, max_chars=10**7)))
        held = [asked & _terms(p) for p in paragraphs]
        df = {term: sum(term in h for h in held) for term in asked}
        for position, (paragraph, carried) in enumerate(zip(paragraphs, held, strict=True)):
            weight = sum(math.log(len(paragraphs) / df[term]) for term in carried)
            if weight > 0:
                scored.append((-weight, order * 10**6 + position, article.title, paragraph))
    scored.sort()
    if rerank is None or not scored:
        return [(title, paragraph) for _carried, _where, title, paragraph in scored[:max(1, limit)]]
    pool = scored[:max(1, limit) * _CANDIDATE_POOL_FACTOR]
    similarity = list(rerank(question, [f"{title}: {paragraph}" for _c, _w, title, paragraph in pool]))
    order = sorted(range(len(pool)), key=lambda i: -similarity[i])
    return [(pool[i][2], pool[i][3]) for i in order[:max(1, limit)]]


def _paragraphs(body: str) -> Sequence[str]:
    return [p.strip() for p in re.split(r"\n+", str(body or "")) if len(p.split()) > 3]

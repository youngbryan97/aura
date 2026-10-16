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
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

__all__ = [
    "CorpusGrounding",
    "conversation_this_turn_continues",
    "corpus_grounding_for",
    "is_corpus_groundable",
    "the_conversation_this_turn_continues",
]

#: The turns before the one being drafted, and who speaks in them, while it is drafted.
_THIS_CONVERSATION: ContextVar[tuple[tuple[tuple[str, str], ...], tuple[str, ...]] | None] = ContextVar(
    "aura_corpus_conversation", default=None
)


@contextmanager
def the_conversation_this_turn_continues(
    turns: Sequence[tuple[str, str]], *, speakers: Sequence[str] = ()
) -> Iterator[None]:
    """While a reply is drafted, the (role, text) turns before it, for a reading taken without them in hand.

    Her desktop chat's corpus reading is taken by the inference gate from the
    request alone (core/brain/observable_registry.py), so "how many did he
    win?" could not know who "he" was. The path that answers her chat sets
    this around its call; the reading looks here.
    """
    token = _THIS_CONVERSATION.set((tuple((str(r), str(x)) for r, x in turns), tuple(speakers)))
    try:
        yield
    finally:
        _THIS_CONVERSATION.reset(token)


def conversation_this_turn_continues() -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]] | None:
    """(turns, speakers) set by the_conversation_this_turn_continues, or None outside a reply."""
    return _THIS_CONVERSATION.get()

#: Tight on purpose. A reference lookup that does not answer fast is not worth
#: a conversational turn's latency. Measured cold over 6.5M pages, the
#: slowest of six fresh-process lookups took 243ms.
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


#: A question addressed to her or to the person, or about now or what is to come, is
#: conversation or forecast; an encyclopaedia answers neither.
_ADDRESSED_OR_DATED = re.compile(
    r"\b(?:i|me|my|mine|you|your|yours|we|us|our|ours|today|tonight|tomorrow|yesterday|now|currently|"
    r"this (?:week|month|year|morning|evening)|next (?:week|month|year))\b",
    re.IGNORECASE,
)

#: How a question opens when it asks about something rather than to do something.
_ASKS = re.compile(
    r"^(?:what|which|who|whom|whose|when|where|why|how|in what|in which|on what|on which|during which|"
    r"at what|for what|name the|was|were|is|are|did|does)\b",
    re.IGNORECASE,
)


def asks_about_named_things(objective: Any) -> bool:
    """A question that names something and is not about her, the person, or an action to take.

    The wording gate above (``is_corpus_groundable``) lets through "what is",
    "who was" and the like. Measured on SimpleQA's development questions, it
    refused half of them: "In what year did conductor David Randolph marry
    Mildred Greenberg?" is a question about named things worded otherwise.
    """
    text = " ".join(str(objective or "").split())
    if len(text.split()) < 3 or _NOT_ABOUT_THE_WORLD.search(text) or _ADDRESSED_OR_DATED.search(text):
        return False
    if not (text.endswith("?") or _ASKS.search(text)):
        return False
    from core.knowledge.passages_about import named_spans

    return bool(named_spans(text))


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
    conversation: Sequence[tuple[str, str]] | None = None,
    rerank: Callable[[str, list[str]], Sequence[float]] | None = None,
    speakers: Sequence[str] = (),
) -> CorpusGrounding:
    """Passages for `objective`, or an empty grounding when it cannot help.

    Never raises. A missing corpus, a slow query, or an ineligible turn all
    produce the same empty result, because to the caller they are the same
    thing: no reference to stand on.

    ``conversation`` is the (role, text) turns before this one, given by a
    caller answering a person. A turn the question paths do not ground then
    gets what her corpus says about whom the conversation is about
    (core/knowledge/talked_about.py), ordered by ``rerank`` when given.
    """

    query = " ".join(str(objective or "").split())
    started = time.monotonic()
    found = _for_a_question(query, limit=limit, deadline_s=deadline_s, store=store, rerank=rerank)
    if not found and conversation is not None and not _NOT_ABOUT_THE_WORLD.search(query):
        left = deadline_s - (time.monotonic() - started)
        if left > 0:
            try:
                from core.knowledge.talked_about import said_about

                found = said_about(query, conversation, store=store, limit=limit, deadline_s=left, rerank=rerank,
                                   speakers=speakers, everywhere=_paragraph_index_beside(store))
            except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
                found = []
    return CorpusGrounding(query=query, passages=tuple(found))


def _paragraph_index_beside(store: Any) -> Any:
    """The index of every paragraph built from this corpus (core/knowledge/passage_index.py), or None."""
    from core.knowledge.passage_index import PassageIndex

    corpus = getattr(store, "db_path", None)
    index = PassageIndex(corpus.with_name("paragraphs.tantivy"), corpus) if corpus is not None else PassageIndex()
    return index if index.available() else None


def _for_a_question(query: str, *, limit: int, deadline_s: float, store: Any,
                    rerank: Callable[[str, list[str]], Sequence[float]] | None = None) -> list[tuple[str, str]]:
    """The passages for a question about the world.

    With the index of every paragraph (core/knowledge/passage_index.py), the
    careful lookup's passages and the index's are joined by reciprocal rank
    fusion, or ordered by ``rerank`` (her embedder) when given. Measured on
    400 SimpleQA development questions at a chat's 0.25 s (15 October, two
    draws that share none): the answer was within three passages for 128
    fused, 123 by every paragraph alone, 103 by the careful lookup alone;
    within five, 157 against 116; median 160 ms. Without the index, this
    module's search and then the careful lookup, as before.
    """
    started = time.monotonic()
    groundable = is_corpus_groundable(query)
    named = asks_about_named_things(query)
    if not groundable and not named:
        return []
    index = _paragraph_index_beside(store)
    if index is not None:
        return _fused_for(query, index, named=named, limit=limit, deadline_s=deadline_s, store=store, rerank=rerank)
    found = _conversational_passages(query, limit=limit, deadline_s=deadline_s, store=store) if groundable else []
    if not found and named:
        # The careful lookup (core/knowledge/passages_about.py): the articles the
        # question names, their paragraphs that carry its rarer words, in what
        # is left of this lookup's time. Measured on 200 SimpleQA development
        # questions at this deadline (14 October): the answer was in the
        # passages for 45, against 3 for this module's search alone; median
        # 43 ms, 95th percentile 351 ms. Of 40 ordinary chat turns, one more
        # was grounded than before.
        left = deadline_s - (time.monotonic() - started)
        if left > 0:
            try:
                from core.knowledge.passages_about import passages_about

                found = passages_about(query, store=store, limit=limit, deadline_s=left)
            except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
                found = []
    return found


def _fused_for(query: str, index: Any, *, named: bool, limit: int, deadline_s: float, store: Any,
               rerank: Callable[[str, list[str]], Sequence[float]] | None) -> list[tuple[str, str]]:
    from core.knowledge.passage_index import fused

    pool = max(1, limit) * _CANDIDATE_POOL_FACTOR
    started = time.monotonic()
    every = [(p.title, p.paragraph) for p in index.search(query, limit=pool)]
    careful: list[tuple[str, str]] = []
    left = deadline_s - (time.monotonic() - started)
    if named and left > 0:
        try:
            from core.knowledge.passages_about import passages_about

            careful = passages_about(query, store=store, limit=pool, deadline_s=left)
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            careful = []
    if rerank is None:
        return fused([careful, every], limit=limit)
    candidates = fused([careful, every], limit=pool)
    if not candidates:
        return []
    similarity = list(rerank(query, [f"{title}: {paragraph}" for title, paragraph in candidates]))
    order = sorted(range(len(candidates)), key=lambda i: -similarity[i])
    return [candidates[i] for i in order[:limit]]


def _conversational_passages(query: str, *, limit: int, deadline_s: float, store: Any) -> list[tuple[str, str]]:
    """This module's own search: the question's topical words, BM25, relevance, and titles closest to the subject."""
    try:
        if store is None:
            from core.knowledge.local_corpus import LocalCorpusStore

            store = LocalCorpusStore()
        topical = _topical_query(query) or query
        # Ask for more than will be carried. The pool used to equal the limit,
        # so the three passages kept were simply the three BM25 returned, in
        # its order: "when did the berlin wall fall" grounded on "The Berlin
        # Wall (video game)" because that row happened to rank first. Ranking
        # needs candidates to rank, and the relevance filter below discards
        # some of them, so the pool has to be larger than the answer.
        pool = max(1, int(limit)) * _CANDIDATE_POOL_FACTOR
        hits = store.search(topical, limit=pool, deadline_s=deadline_s)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return []
    passages: list[tuple[str, str]] = []
    for hit in hits or ():
        title, text = _passage_of(hit)
        if not text.strip():
            # Title-only rows carry no passage to ground anything with.
            continue
        if not _is_relevant(topical, title, text):
            continue
        passages.append((title, text))
    passages.sort(key=lambda p: _title_distance(topical, p[0]))
    return passages[: max(1, int(limit))]


#: Candidates fetched per passage carried. Ranking cannot improve on an
#: ordering it is handed whole, so the pool must exceed the answer; four
#: leaves room for the relevance filter to reject most of a page of hits.
_CANDIDATE_POOL_FACTOR = 4

#: English function words, which carry no subject matter in a page title.
_FUNCTION_WORDS = frozenset(
    "a an the of in on at to for from and or with by as is are was were".split()
)

_DISAMBIGUATED_RE = re.compile(r"\s*\([^)]*\)\s*$")
_AGGREGATE_TITLE_RE = re.compile(
    r"^(?:list of|index of|outline of|timeline of|glossary of|"
    r"comparison of|history of)\b",
    re.IGNORECASE,
)


def _title_distance(query: str, title: str) -> tuple[int, int, int]:
    """How far a page title sits from the subject that was asked about.

    BM25 ranks by term statistics, which puts "The Berlin Wall (video game)"
    and "Ada Lovelace Award" above the articles a person meant. Both are real
    pages about something adjacent, and the filter above cannot tell them
    apart because they genuinely carry the asked-about words.

    What separates them is the TITLE: an encyclopaedia names its main article
    after the subject and everything else after the subject plus a qualifier.
    So a title is ordered by how much it adds to what was asked, aggregates
    and parenthetical variants last. Sorted, not filtered — a qualified page
    is still the best grounding available when nothing plainer exists.
    """
    bare = _DISAMBIGUATED_RE.sub("", str(title or "")).strip()
    lowered = bare.lower()
    asked = " ".join(str(query or "").lower().split())
    qualified = int(bool(_DISAMBIGUATED_RE.search(str(title or ""))))
    aggregate = int(bool(_AGGREGATE_TITLE_RE.match(lowered)))
    asked_terms = set(re.findall(r"[a-z0-9]+", asked))
    if lowered == asked or set(re.findall(r"[a-z0-9]+", lowered)) == asked_terms:
        return (0, aggregate, qualified)
    # Function words are not added subject matter. Counting them ranked
    # "Beloved Berlin Wall" above "Fall of the Berlin Wall", because "of" and
    # "the" scored as two extra topics.
    extra = len([
        t
        for t in re.findall(r"[a-z0-9]+", lowered)
        if t not in asked_terms and t not in _FUNCTION_WORDS
    ])
    return (1 + extra, aggregate, qualified)


def _is_relevant(query: str, title: str, text: str) -> bool:
    """Does this hit actually concern the query, or did BM25 reach?

    The corpus search falls back to ANY-term matching when the full conjunction
    misses, and over 7M pages that fallback will always return something. It
    returned "Pee-wee Herman" for "wrote Rust borrow checker originally" and
    "Django Haskins" for "Django released 2005" — matching on the incidental
    words while missing the subject entirely.

    An irrelevant passage presented as authoritative grounding is strictly
    worse than no grounding: it does not merely fail to help, it argues for the
    wrong answer with the weight of a citation. So a hit has to earn its place
    by carrying most of what was asked about, and a near-miss is discarded in
    favour of the model answering unaided.
    """

    terms = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2]
    if not terms:
        return False
    haystack = f"{title} {text}".lower()
    matched = sum(1 for t in terms if t in haystack)
    # Most of the asked-about terms, and never on a single incidental word.
    #
    # The ratio matters more than it looks. At half, "Django released 2005"
    # still admitted "Django Haskins" — it matched the name and the verb while
    # missing "web" and "framework", which are the two words that said WHICH
    # Django. The discriminating terms are usually the ones a wrong-sense hit
    # lacks, so the bar has to sit above half.
    #
    # The floor cannot exceed what the query HAS. "what is a semiconductor"
    # reduces to the single term "semiconductor", so a floor of two demanded a
    # match this query could never supply: three good hits arrived, all three
    # were discarded, and the offline corpus answered nothing at all for every
    # one-word subject — semiconductor, photosynthesis, gravity. When a query
    # carries one term, that term IS the subject rather than an incidental
    # word, so requiring it is already the strictest bar available.
    needed = min(len(terms), max(2, -(-len(terms) * 3 // 5)))
    return matched >= needed

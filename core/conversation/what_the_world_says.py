"""What the world says about what a turn names, gathered before she answers.

LIVE 2026-10-03, between 00:29 and 00:46, five turns in a row:

* "Which was better, Bam's 83 point game or Kobe's 81 point game?" Her cortex
  answered "It didn't happen. Bam Adebayo never scored 83 in a game". Her
  offline Wikipedia holds the article "Bam Adebayo's 83-point game", found in
  7 milliseconds, opening "On March 10, 2026, at Kaseya Center in Miami,
  Florida, Bam Adebayo scored 83 points". Nothing looked, because whether to
  look was decided from the wording of the question, and "which was better"
  reads as asking for an opinion.
* "Google it. Bam had an 83 point game" searched for the film "It", and the
  next request opened a results page and reported "Done". Nothing was read.
* A Wikipedia link was answered "I can't open that link".
* "What did you learn" had nothing to draw on.

Each of these asks the same thing of her: find out what the world says about
what was named, and read it, before answering. That happens here, in three
ways, none of which depends on how the request is phrased:

* a link the person sends is fetched and read, because sending one hands her a
  document; a Wikipedia link is also read from the offline copy when the
  network cannot reach it;
* a request to find something out ("google it", "look that up") is read by
  the language substrate (core/language/search_request.py) for what it is
  about, from the request or, when it only says "it", from what was asked
  before, then searched and its pages read;
* a question about what was found ("what did you learn") gets the sources the
  conversation's most recent lookup read, since it names nothing of its own;
* the offline corpus is consulted for every phrase of the request that names
  something. It is local and private, and a lookup takes milliseconds. The
  web is consulted beyond an explicit request where the existing
  outside-evidence decision asks for it, and for a factual question about a
  named thing that the offline copy does not answer, or answers only in part:
  its best source scores below the median the judge's own matched pairs
  reach. A remark that asks nothing is never sent to a search engine.

What is admitted is decided by the calibrated evidence-alignment judge in
core/cognition/evidence_relevance.py, not by shared words, and what reaches her
is the sources themselves, each labelled with where it came from. Nothing here
tells her how to answer.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import unquote, urlparse

from core.runtime.lockdep import checked_lock
from core.utils.readable_text import SENTENCE_END as _SENTENCE_END
from core.utils.readable_text import prose_paragraphs as _prose_paragraphs

#: A web address as people paste them.
URL_RE = re.compile(r"https?://[^\s<>\"'`)\]]+", re.IGNORECASE)

#: Where a request divides into the separate things it names.
_BETWEEN_REFERENTS = re.compile(r"[,;:?!.]+|\s+(?:or|and|vs\.?|versus|than)\s+", re.IGNORECASE)

#: Characters of one source carried into the turn.
_PASSAGE_CHARS = 1400
#: Characters of all sources together.
_EVIDENCE_CHARS = 7000
#: Sources read per search.
_PAGES_PER_SEARCH = 4
#: Conversations whose last reading is kept. A person holds a few at once; this
#: bounds the memory without reaching any real use.
_SESSIONS_REMEMBERED = 32


@dataclass(frozen=True)
class WorldSource:
    """One passage, with where it came from."""

    origin: str
    title: str
    location: str
    text: str
    score: float | None = None
    #: The whole document the passage was taken from, never shown: what a
    #: reply that goes beyond the passage is checked against
    #: (core/conversation/claims_against_sources.py).
    full_text: str = ""


@dataclass
class WorldEvidence:
    """What was looked up for a turn, what was read, and what was admitted."""

    sources: list[WorldSource] = field(default_factory=list)
    searched: list[str] = field(default_factory=list)
    read: list[str] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)
    #: Whether what a search found was saved to her memory.
    saved: bool = False
    seconds: float = 0.0

    @property
    def found_anything(self) -> bool:
        return bool(self.sources)

    def render(self) -> str:
        """The sources as she reads them: labelled, fenced, and nothing else.

        A fetched page can carry text written to be read as instructions; the
        fence (core/security/prompt_fencing.py) keeps it a quotation.
        """
        from core.security.prompt_fencing import fence

        lines: list[str] = []
        for number, source in enumerate(self.sources, start=1):
            where = f", {source.location}" if source.location else ""
            label = f"source {number}: {source.origin}, \"{source.title}\"{where}"
            lines.append(fence(source.text, label=label))
        for query in self.searched:
            kept = "; saved to memory" if self.saved else ""
            lines.append(f"[SEARCHED the web for \"{query}\"; {len(self.read)} page(s) read{kept}]")
        for url in self.unreadable:
            lines.append(f"[COULD NOT READ {url}]")
        return "\n\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sources": [
                {"origin": s.origin, "title": s.title, "location": s.location, "score": s.score}
                for s in self.sources
            ],
            "searched": list(self.searched),
            "read": list(self.read),
            "unreadable": list(self.unreadable),
            "saved": self.saved,
            "seconds": round(self.seconds, 3),
        }


_last_read: OrderedDict[str, tuple[WorldSource, ...]] = OrderedDict()
_last_read_lock = checked_lock("core.conversation.what_the_world_says.last_read")


def remember_reading(session_id: str, evidence: WorldEvidence) -> None:
    """Keep what this conversation's latest lookup read, for a later "what did you learn"."""
    if not session_id or not evidence.sources:
        return
    with _last_read_lock:
        _last_read[session_id] = tuple(evidence.sources)
        _last_read.move_to_end(session_id)
        while len(_last_read) > _SESSIONS_REMEMBERED:
            _last_read.popitem(last=False)


def answer_from_earlier_reading(evidence: WorldEvidence, request: str, session_id: str) -> bool:
    """Give a question about what was found, or about her grounds, the sources the last lookup read.

    The language substrate (core/language/search_request.py) reads the
    request. A question about what was found takes them when this turn found
    nothing of its own. A question about her grounds ("how sure are you about
    that?") takes them first whatever this turn found, because they are what
    the answer in question was read from; LIVE 2026-10-04 they were not looked
    at again, and the turn read five pages about something else.
    """
    if not session_id:
        return False
    from core.language.search_request import asks_about_grounds, asks_what_was_found

    grounds = asks_about_grounds(request) is True
    if not grounds and (evidence.sources or asks_what_was_found(request) is not True):
        return False
    with _last_read_lock:
        earlier = _last_read.get(session_id, ())
    label = "read for the answer being asked about" if grounds else "read for an earlier turn"
    evidence.sources = _within_the_evidence_budget([
        WorldSource(f"{label} ({source.origin})", source.title, source.location, source.text, source.score, source.full_text)
        for source in earlier
    ] + evidence.sources)
    if evidence.sources:
        from core.language.search_request import teach_from_the_floor

        teach_from_the_floor(request)
    return bool(evidence.sources)


#: Where one claim in an answer ends and the next begins. An en dash between
#: two numbers is a range ("1999–2021"), not a break.
_BETWEEN_CLAIMS = re.compile(r"(?<=[.!?])\s+|\s*[;:—]\s*|\s+–\s+")


def claim_asked_about(request: str, previous_reply: str, previous_request: str = "") -> str:
    """The part of her last answer that a question about her grounds points at, as a query.

    The clauses that carry a name the question uses ("the FTX part" names FTX),
    after what was asked about when the clause does not say it ("It opened
    with a Gloria Estefan concert" is about the Kaseya Center). A question
    that names nothing ("where did that come from?") puts the whole answer in
    question, and the query is what was asked.
    """
    named = {
        word.lower()
        for phrase in referent_phrases(request)
        for word in phrase.split()
        if word[:1].isupper() or word.isdigit()
    }
    if named:
        clauses = [
            clause.strip(" .!?")
            for clause in _BETWEEN_CLAIMS.split(str(previous_reply or ""))
            if named & {word.lower() for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9'’-]*", clause)}
        ]
        if clauses:
            claim = " ".join(clauses)
            topic = " ".join(referent_phrases(previous_request))
            said = {word.lower() for word in claim.split()}
            if topic and not {word.lower() for word in topic.split()} <= said:
                claim = f"{topic}: {claim}"
            return claim[:300]
    return str(previous_request or "").strip()


def links_in(request: str) -> list[str]:
    return [url.rstrip(".,;") for url in URL_RE.findall(str(request or ""))]


def referent_phrases(text: str) -> list[str]:
    """The separate things a request names, as corpus queries.

    Numbers are kept: "83" is what tells Bam Adebayo's 83-point game from his
    career. Question scaffolding is dropped, so a phrase that names nothing
    gives no query. Addresses are not words: LIVE 2026-10-04 "The Pong game
    at /Users/bryan/aura-demos/pong/pong.html is broken" searched the
    encyclopedia for "Pong game Users bryan aura-demos pong".
    """
    from core.intent.opaque_spans import without_absolute_paths
    from core.knowledge.corpus_grounding import _QUESTION_SCAFFOLDING

    phrases: list[str] = []
    for part in _BETWEEN_REFERENTS.split(without_absolute_paths(URL_RE.sub(" ", str(text or "")))):
        words = re.findall(r"[A-Za-z][A-Za-z'’-]*|\d+", part)
        kept = []
        for word in words:
            plain = re.sub(r"['’]s$", "", word)
            if plain.lower() in _QUESTION_SCAFFOLDING or (len(plain) < 3 and not plain.isdigit()):
                continue
            kept.append(plain)
        named = any(word[:1].isupper() or word.isdigit() for word in kept)
        if kept and named and " ".join(kept) not in phrases:
            phrases.append(" ".join(kept[:6]))
    return phrases


def wikipedia_title(url: str) -> str | None:
    """The article a Wikipedia link names."""
    parsed = urlparse(str(url or ""))
    if not parsed.netloc.lower().endswith("wikipedia.org") or not parsed.path.startswith("/wiki/"):
        return None
    title = unquote(parsed.path[len("/wiki/"):]).replace("_", " ").strip()
    return title or None


def _paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n|\n(?=[A-Z][^\n]{0,60}\.\n)", str(text or "")) if len(part.strip()) > 40]


def passage_of(text: str, terms: Sequence[str] = (), *, limit: int = _PASSAGE_CHARS) -> str:
    """The opening of a document, then the paragraphs that carry the request's terms.

    The opening is the first paragraph of more than one sentence, so a caption
    is not taken for a lead. With no terms (a bare link names nothing), the
    document's own order follows the opening.
    """
    paragraphs = _prose_paragraphs(text) or _paragraphs(text) or [str(text or "").strip()]
    opening = next(
        (index for index, paragraph in enumerate(paragraphs) if len(_SENTENCE_END.findall(paragraph)) > 1), 0
    )
    chosen = [paragraphs[opening]]
    rest = paragraphs[opening + 1 :]
    wanted = {term.lower() for term in terms if term}
    if wanted:
        def density(paragraph: str) -> float:
            lowered = paragraph.lower()
            return sum(lowered.count(term) for term in wanted) / max(1, len(paragraph))

        chosen.extend(paragraph for paragraph in sorted(rest, key=density, reverse=True)[:2] if density(paragraph) > 0)
    else:
        chosen.extend(rest[:2])
    passage = "\n\n".join(chosen)
    if len(passage) <= limit:
        return passage
    cut = passage[:limit]
    end = max(cut.rfind(". "), cut.rfind(".\n"))
    return cut[: end + 1] if end > limit // 2 else cut


Fetch = Callable[[str], Awaitable[dict[str, Any] | None]]
Search = Callable[[str], Awaitable[list[dict[str, Any]]]]


def _judge(question: str, candidates: list[WorldSource], *, solicited: bool) -> list[WorldSource]:
    """The candidates the evidence-alignment judge says bear on the question.

    What the person asked for (a search, a page) is admitted at the judge's
    boundary. What nobody asked for has to score where the judge's matched
    pairs did, and is not admitted at all when the judge could not measure.
    """
    if not candidates:
        return []
    if not question.strip():
        return candidates
    from core.cognition.evidence_relevance import EVIDENCE_MATCHED_FLOOR, assess_evidence_alignments

    verdicts = assess_evidence_alignments(question, [f"{c.title}\n{c.text}" for c in candidates])
    admitted = [
        WorldSource(c.origin, c.title, c.location, c.text, verdict.score, c.full_text)
        for c, verdict in zip(candidates, verdicts, strict=True)
        if verdict.relevant
        and (solicited or (verdict.measured and verdict.score is not None and verdict.score >= EVIDENCE_MATCHED_FLOOR))
    ]
    return sorted(admitted, key=lambda c: -(c.score if c.score is not None else 0.0))


def _as_of(store: Any) -> str:
    """", as of <date>" for an offline copy that records when it was taken.

    Without the date she cannot tell an event the copy predates from one that
    never happened; with it, the web is where anything later is found.
    """
    try:
        date = str(store.snapshot_date() or "")
    except (AttributeError, OSError, ValueError):
        date = ""
    return f", as of {date}" if date else ""


def _corpus_origin(hit: Any, body: str, as_of: str) -> tuple[str, str]:
    """Where a corpus document came from and when, and the page it was read from.

    The corpus holds the encyclopedia dump and what her own searches kept
    (core/search/research_pipeline.py, source ``web_retained``). LIVE
    2026-10-03 a kept search came back labelled "offline copy (web_retained),
    as of 2026-07-03": the dump's date on something read from the web that
    evening, with no page named.
    """
    if hit.source == "wikipedia":
        return "offline Wikipedia" + as_of, ""
    stamp = float(getattr(hit, "ingested_at", 0.0) or 0.0)
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(stamp)) if stamp > 0 else ""
    if hit.source != "web_retained":
        return f"offline copy ({hit.source})" + (f", added {when}" if when else ""), ""
    _, _, cited = body.partition("\nSources:")
    page = URL_RE.search(cited)
    return "kept from a web search" + (f", read {when}" if when else ""), page.group(0) if page else ""


def _from_corpus(
    phrases: Sequence[str], terms: Sequence[str], *, store: Any, read_since: float = 0.0
) -> list[WorldSource]:
    """The corpus documents the phrases name, at the conversation lane's deadline.

    The offline deadline (5 s) lets a phrase that no document matches whole run
    its any-term fallback over seven million pages. Every chat turn from 4
    October 16:24 UTC spent 7.3 to 8.2 s here and found nothing; a real topic
    answers in under 0.1 s.

    The encyclopedia and what her searches kept are searched as separate
    shelves. A kept note is short and full of its query's words, so it
    outranks the article it was read from: LIVE 2026-10-04 "When did the
    Kaseya Center open?" took two kept notes and never the offline article. A
    note kept at or after ``read_since`` is this turn's own reading, already
    among its sources; it came back as a second copy of the page it read.
    """
    from core.knowledge.local_corpus import CONVERSATION_SEARCH_DEADLINE_S, RETAINED_SOURCE

    candidates: list[WorldSource] = []
    as_of = _as_of(store)
    seen: set[int] = set()
    shelves = ({"not_source": RETAINED_SOURCE}, {"source": RETAINED_SOURCE})
    for phrase, shelf in ((phrase, shelf) for phrase in phrases[:4] for shelf in shelves):
        for hit in store.search(phrase, limit=2, deadline_s=CONVERSATION_SEARCH_DEADLINE_S, **shelf):
            if hit.doc_id in seen:
                continue
            seen.add(hit.doc_id)
            if hit.source == RETAINED_SOURCE and read_since and getattr(hit, "ingested_at", 0.0) >= read_since:
                continue
            body = store.body(hit.doc_id)
            if body:
                origin, location = _corpus_origin(hit, body, as_of)
                candidates.append(WorldSource(origin, hit.title, location, passage_of(body, terms), full_text=body))
    return candidates


def _answers_only_in_part(question: str, local: Sequence[WorldSource]) -> bool:
    """A factual question about a named thing that the offline copy leaves open.

    LIVE 2026-10-03: "Who designed the Kaseya Center, and what did it cost to
    build?" was answered from the offline article (0.69, the bottom of the
    judge's matched range); the cost was not in it, and she said she would
    want to verify it rather than look. Bryan: something should decide to
    look elsewhere, or double-check online.
    """
    from core.conversation.asks_about_the_world import asks_about_a_named_thing

    if not asks_about_a_named_thing(question):
        return False
    from core.cognition.evidence_relevance import matched_alignment_median

    median = matched_alignment_median()
    if median is None:
        return False
    best = max((source.score for source in local if source.score is not None), default=None)
    return best is None or best < median


async def gather_world_evidence(
    request: str,
    *,
    previous_request: str = "",
    fetch: Fetch | None = None,
    search: Search | None = None,
    store: Any = None,
    outside_wanted: bool = False,
    deadline_s: float = 45.0,
    read_since: float = 0.0,
    previous_reply: str = "",
) -> WorldEvidence:
    """Read what the request names, from the page sent, a search asked for, or the offline corpus.

    ``read_since`` is when the turn began (epoch seconds): what its own
    searches kept since then is not read back as a separate source.
    """
    started = time.monotonic()
    evidence = WorldEvidence()
    question = URL_RE.sub(" ", str(request or "")).strip()
    terms = [word for phrase in referent_phrases(question) for word in phrase.split()]
    if store is None:
        from core.knowledge.local_corpus import get_local_corpus_store

        store = get_local_corpus_store()

    def remaining() -> float:
        return max(1.0, deadline_s - (time.monotonic() - started))

    # A link is a document handed over: read it whatever else the request says.
    for url in links_in(request)[:3]:
        page = None
        if fetch is not None:
            try:
                page = await asyncio.wait_for(fetch(url), timeout=min(20.0, remaining()))
            except (TimeoutError, OSError, RuntimeError, ValueError):
                page = None
        if page and str(page.get("text") or "").strip():
            evidence.read.append(url)
            evidence.sources.append(
                WorldSource("the page you sent", str(page.get("title") or url), url, passage_of(page["text"], terms, limit=2 * _PASSAGE_CHARS), full_text=page["text"])
            )
            continue
        title = wikipedia_title(url)
        hit = await asyncio.to_thread(store.by_title, title) if title else None
        body = await asyncio.to_thread(store.body, hit.doc_id) if hit else ""
        if body:
            evidence.sources.append(WorldSource(f"offline Wikipedia copy of the page you sent{_as_of(store)}", hit.title, url, passage_of(body, terms, limit=2 * _PASSAGE_CHARS), full_text=body))
        else:
            evidence.unreadable.append(url)

    from core.language.search_request import read_search_request

    asked = read_search_request(request, previous_request)
    asked_for = asked.query if asked.about_the_world and asked.query else None
    if asked_for is None and previous_reply:
        from core.language.search_request import asks_about_grounds

        if asks_about_grounds(question) is True:
            # Checked against the world: what the question points at in her
            # last answer, not the question's own words.
            asked_for = claim_asked_about(question, previous_reply, previous_request) or None
    local = _judge(
        question,
        await asyncio.to_thread(
            _from_corpus, referent_phrases(asked_for or question), terms, store=store, read_since=read_since
        ),
        solicited=asked_for is not None,
    )
    evidence.sources.extend(local[:3])

    if search is not None and (asked_for or (outside_wanted and not local) or _answers_only_in_part(question, local)):
        query = asked_for or question
        try:
            results = await asyncio.wait_for(search(query), timeout=remaining())
        except (TimeoutError, OSError, RuntimeError, ValueError):
            results = []
        evidence.searched.append(query)
        candidates = []
        for result in results[: 2 * _PAGES_PER_SEARCH]:
            text = str(result.get("text") or result.get("snippet") or "").strip()
            if not text:
                continue
            url = str(result.get("url") or "")
            if result.get("text"):
                evidence.read.append(url)
            candidates.append(WorldSource("web", str(result.get("title") or url), url, passage_of(text, terms), full_text=text))
        evidence.sources.extend(_judge(asked_for or question, candidates, solicited=True)[:_PAGES_PER_SEARCH])
        if asked.decided_by == "instruction":
            from core.language.search_request import teach_from_the_floor

            teach_from_the_floor(question)

    evidence.sources = _within_the_evidence_budget(evidence.sources)
    evidence.seconds = time.monotonic() - started
    return evidence


def _within_the_evidence_budget(sources: Sequence[WorldSource]) -> list[WorldSource]:
    total = 0
    kept: list[WorldSource] = []
    for source in sources:
        if total + len(source.text) > _EVIDENCE_CHARS and kept:
            break
        kept.append(source)
        total += len(source.text)
    return kept


__all__ = [
    "URL_RE",
    "WorldEvidence",
    "WorldSource",
    "answer_from_earlier_reading",
    "claim_asked_about",
    "gather_world_evidence",
    "links_in",
    "passage_of",
    "referent_phrases",
    "remember_reading",
    "wikipedia_title",
]

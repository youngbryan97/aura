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
* the offline corpus is consulted for every phrase of the request that names
  something. It is local and private, and a lookup takes milliseconds. The
  web is consulted beyond an explicit request only where the existing
  outside-evidence decision asks for it, so nothing a person says is sent to a
  search engine unasked.

What is admitted is decided by the calibrated evidence-alignment judge in
core/cognition/evidence_relevance.py, not by shared words, and what reaches her
is the sources themselves, each labelled with where it came from. Nothing here
tells her how to answer.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import unquote, urlparse

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


@dataclass(frozen=True)
class WorldSource:
    """One passage, with where it came from."""

    origin: str
    title: str
    location: str
    text: str
    score: float | None = None


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


def links_in(request: str) -> list[str]:
    return [url.rstrip(".,;") for url in URL_RE.findall(str(request or ""))]


def referent_phrases(text: str) -> list[str]:
    """The separate things a request names, as corpus queries.

    Numbers are kept: "83" is what tells Bam Adebayo's 83-point game from his
    career. Question scaffolding is dropped, so a phrase that names nothing
    gives no query.
    """
    from core.knowledge.corpus_grounding import _QUESTION_SCAFFOLDING

    phrases: list[str] = []
    for part in _BETWEEN_REFERENTS.split(URL_RE.sub(" ", str(text or ""))):
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
    """The opening of a document, then the paragraphs that carry the request's terms."""
    paragraphs = _paragraphs(text) or [str(text or "").strip()]
    chosen = [paragraphs[0]]
    wanted = {term.lower() for term in terms if term}
    if wanted:
        ranked = sorted(
            paragraphs[1:],
            key=lambda paragraph: -sum(term in paragraph.lower() for term in wanted),
        )
        chosen.extend(paragraph for paragraph in ranked[:2] if any(term in paragraph.lower() for term in wanted))
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
        WorldSource(c.origin, c.title, c.location, c.text, verdict.score)
        for c, verdict in zip(candidates, verdicts, strict=True)
        if verdict.relevant
        and (solicited or (verdict.measured and verdict.score is not None and verdict.score >= EVIDENCE_MATCHED_FLOOR))
    ]
    return sorted(admitted, key=lambda c: -(c.score if c.score is not None else 0.0))


def _from_corpus(phrases: Sequence[str], terms: Sequence[str], *, store: Any) -> list[WorldSource]:
    candidates: list[WorldSource] = []
    seen: set[int] = set()
    for phrase in phrases[:4]:
        for hit in store.search(phrase, limit=2):
            if hit.doc_id in seen:
                continue
            seen.add(hit.doc_id)
            body = store.body(hit.doc_id)
            if body:
                origin = "offline Wikipedia" if hit.source == "wikipedia" else f"offline copy ({hit.source})"
                candidates.append(WorldSource(origin, hit.title, "", passage_of(body, terms)))
    return candidates


async def gather_world_evidence(
    request: str,
    *,
    previous_request: str = "",
    fetch: Fetch | None = None,
    search: Search | None = None,
    store: Any = None,
    outside_wanted: bool = False,
    deadline_s: float = 45.0,
) -> WorldEvidence:
    """Read what the request names, from the page sent, a search asked for, or the offline corpus."""
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
                WorldSource("the page you sent", str(page.get("title") or url), url, passage_of(page["text"], terms, limit=2 * _PASSAGE_CHARS))
            )
            continue
        title = wikipedia_title(url)
        hit = await asyncio.to_thread(store.by_title, title) if title else None
        body = await asyncio.to_thread(store.body, hit.doc_id) if hit else ""
        if body:
            evidence.sources.append(WorldSource("offline Wikipedia copy of the page you sent", hit.title, url, passage_of(body, terms, limit=2 * _PASSAGE_CHARS)))
        else:
            evidence.unreadable.append(url)

    from core.language.search_request import read_search_request

    asked = read_search_request(request, previous_request)
    asked_for = asked.query if asked.about_the_world and asked.query else None
    local = _judge(
        question,
        await asyncio.to_thread(_from_corpus, referent_phrases(asked_for or question), terms, store=store),
        solicited=asked_for is not None,
    )
    evidence.sources.extend(local[:3])

    if search is not None and (asked_for or (outside_wanted and not local)):
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
            candidates.append(WorldSource("web", str(result.get("title") or url), url, passage_of(text, terms)))
        evidence.sources.extend(_judge(asked_for or question, candidates, solicited=True)[:_PAGES_PER_SEARCH])

    total = 0
    kept: list[WorldSource] = []
    for source in evidence.sources:
        if total + len(source.text) > _EVIDENCE_CHARS and kept:
            break
        kept.append(source)
        total += len(source.text)
    evidence.sources = kept
    evidence.seconds = time.monotonic() - started
    return evidence


__all__ = [
    "URL_RE",
    "WorldEvidence",
    "WorldSource",
    "gather_world_evidence",
    "links_in",
    "passage_of",
    "referent_phrases",
    "wikipedia_title",
]

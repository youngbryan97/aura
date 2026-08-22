from __future__ import annotations

import asyncio
import hashlib
import html
import json
import logging
import os
import re
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from core.conversation.word_markers import names_any
from core.runtime.errors import record_degradation
from core.runtime.network_gateway import get_network_gateway
from core.runtime.state_ownership import state_root
from core.thought_stream import get_emitter
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.SearchPipeline")

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

_CURRENTNESS_TERMS = {
    "current",
    "currently",
    "today",
    "latest",
    "recent",
    "recently",
    "news",
    "now",
    "price",
    "weather",
    "stock",
    "score",
    "schedule",
    "release date",
    # Temporal deictics. "today" and "now" were here; the rest of the family
    # was not, so "who won the election yesterday" and "what happened last
    # night" were classified as timeless and could be answered from a dated
    # snapshot. Anything anchored to the speaker's clock needs a live source.
    "yesterday",
    "tomorrow",
    "tonight",
    "this morning",
    "this afternoon",
    "this evening",
    "last night",
    "this week",
    "last week",
    "this month",
    "last month",
    "this year",
    "so far this",
    "right now",
    "at the moment",
    "as of",
    "breaking",
    "live",
    "today's",
    "upcoming",
    "just announced",
    "still",
}

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "who",
    "why",
    "with",
}

_PROMPT_FILLERS = (
    "can you",
    "could you",
    "would you",
    "please",
    "look up",
    "find out",
    "search for",
    "search the web for",
    "google",
    "check online",
    "tell me",
)

_SOURCE_DOCUMENT_TERMS = (
    "story",
    "article",
    "post",
    "page",
    "thread",
    "document",
    "source",
    "text",
    "link",
    "paper",
    "report",
    "guide",
    "entry",
)

_SOURCE_SUMMARY_TERMS = (
    "what happens",
    "summary",
    "summarize",
    "summarise",
    "recap",
    "plot",
    "ending",
    "how does it end",
    "what does it say",
    "read it",
    "read this",
    "read that",
)

_NOISY_RESULT_HOST_TERMS = (
    "youtube.com",
    "youtu.be",
    "google.com",
    "bing.com",
    "duckduckgo.com",
    "googleadservices.com",
)

_SEARCH_HIT_CACHE_TTL_SECONDS = 10 * 60
_SEARCH_HIT_CACHE: dict[str, tuple[float, list[SearchHit]]] = {}


def _ddgs_enabled() -> bool:
    """Gate the native DDGS/primp lane behind an explicit opt-in on macOS.

    We have observed hard SIGABRT crashes inside primp on macOS 26.x. The
    legacy HTML search is less capable, but it fails gracefully instead of
    taking down the whole interpreter. Operators can opt back in explicitly
    once that upstream stack is stable again.
    """
    flag = os.getenv("AURA_ENABLE_UNSAFE_DDGS", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    if sys.platform == "darwin":
        return False
    return True


def _normalize_text(text: str, *, limit: int = 0) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if limit > 0 and len(cleaned) > limit:
        return cleaned[: limit - 1].rstrip() + "…"
    return cleaned


def _normalize_query(text: str) -> str:
    cleaned = str(text or "").strip().lower()
    for filler in _PROMPT_FILLERS:
        if cleaned.startswith(filler + " "):
            cleaned = cleaned[len(filler) + 1 :].strip()
    cleaned = re.sub(r"[\"'`]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" .?!,")


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(token) >= 3 and token not in _STOP_WORDS
    ]


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def _now() -> float:
    return time.time()


def _freshness_window(query: str) -> int:
    # Word boundaries, not containment: "now" sits inside "know", so "i dont
    # know what to do next" was treated as a question about the last hour.
    lowered = str(query or "").lower()
    if names_any(lowered, _CURRENTNESS_TERMS):
        return 60 * 60
    if names_any(lowered, ("version", "release", "pricing")):
        return 6 * 60 * 60
    return 14 * 24 * 60 * 60


def freshness_window_for_query(query: str) -> int:
    """Public wrapper for choosing a retention window for a query."""
    return _freshness_window(query)


def _query_is_current(query: str) -> bool:
    return names_any(query, _CURRENTNESS_TERMS)


def _quoted_phrases(text: str) -> list[str]:
    phrases: list[str] = []
    # Boundary-guarded: a contraction is not a quotation.
    for group in re.findall(
        r"[\"“”]([^\"“”]{4,200})[\"“”]|(?<![A-Za-z])'([^']{4,200})'(?![A-Za-z])",
        str(text or ""),
    ):
        phrase = group[0] or group[1]
        cleaned = _normalize_text(phrase)
        if cleaned and cleaned not in phrases:
            phrases.append(cleaned)
    return phrases


def query_requires_source_reading(query: str) -> bool:
    lowered = _normalize_query(query)
    if not lowered:
        return False
    if _quoted_phrases(query):
        return True
    if any(term in lowered for term in _SOURCE_SUMMARY_TERMS):
        return True
    if any(term in lowered for term in _SOURCE_DOCUMENT_TERMS) and any(
        marker in lowered
        for marker in (
            "what happens",
            "summary",
            "summarize",
            "summarise",
            "ending",
            "read",
            "tell me",
            "what does it say",
            "look up",
            "search for",
            "find",
        )
    ):
        return True
    if re.search(r"\btitle\b", lowered) or re.search(r"\burl\b", lowered):
        return True
    return False


def _normalized_match_text(*parts: str) -> str:
    combined = " ".join(str(part or "") for part in parts if str(part or "").strip())
    combined = combined.lower()
    combined = re.sub(r"[^a-z0-9]+", " ", combined)
    return re.sub(r"\s+", " ", combined).strip()


def _html_to_text(raw_html: str) -> str:
    text = re.sub(
        r"(?is)<(script|style|noscript|svg|iframe|canvas).*?>.*?</\1>",
        " ",
        raw_html,
    )
    text = re.sub(
        r"(?i)</?(article|section|div|p|li|ul|ol|table|tr|td|th|h[1-6]|blockquote|br)[^>]*>",
        "\n",
        text,
    )
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html.unescape(text)
    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in text.splitlines()
    ]
    lines = [line for line in lines if len(line) >= 3]
    return "\n".join(lines)


def _extract_title(raw_html: str) -> str:
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw_html)
    return _normalize_text(html.unescape(match.group(1))) if match else ""


@dataclass(slots=True)
class SearchHit:
    title: str
    url: str
    snippet: str = ""
    source_engine: str = ""
    position: int = 0


@dataclass(slots=True)
class SearchPage:
    url: str
    title: str
    text: str
    snippet: str = ""
    source_engine: str = ""
    position: int = 0
    fetched_at: float = field(default_factory=_now)


@dataclass(slots=True)
class SearchArtifact:
    artifact_id: str
    query: str
    normalized_query: str
    answer: str
    summary: str
    facts: list[str]
    citations: list[dict[str, str]]
    evidence: list[dict[str, Any]]
    created_at: float
    updated_at: float
    freshness_seconds: int
    confidence: float
    current: bool
    source: str
    skill: str = "web_search"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SearchArtifactStore:
    """Small local persistence layer for retained web learnings."""

    def __init__(self, path: Path | None = None):
        self.path = path or self._default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _default_path() -> Path:
        try:
            from core.config import config

            return config.paths.data_dir / "search" / "web_artifacts.jsonl"
        except (ImportError, AttributeError):
            return state_root() / "data" / "search" / "web_artifacts.jsonl"

    def _read_all(self) -> list[SearchArtifact]:
        if not self.path.exists():
            return []
        records: list[SearchArtifact] = []
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    payload = str(line or "").strip()
                    if not payload:
                        continue
                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    try:
                        records.append(SearchArtifact(**data))
                    except TypeError:
                        continue
        except OSError as exc:
            logger.debug("SearchArtifactStore read failed: %s", exc)
        return records

    def append(self, artifact: SearchArtifact) -> None:
        existing = {item.artifact_id: item for item in self._read_all()}
        if artifact.artifact_id in existing:
            current = existing[artifact.artifact_id]
            if current.updated_at >= artifact.updated_at:
                return

        existing[artifact.artifact_id] = artifact
        
        try:
            # Component C2: Compaction threshold
            if len(existing) > 500:
                with self.path.open("w", encoding="utf-8") as handle:
                    for item in existing.values():
                        handle.write(json.dumps(item.to_dict(), ensure_ascii=True) + "\n")
            else:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(artifact.to_dict(), ensure_ascii=True) + "\n")
        except OSError as exc:
            logger.debug("SearchArtifactStore append failed: %s", exc)

    def find_best(
        self,
        query: str,
        *,
        freshness_seconds: int | None = None,
        allow_stale: bool = False,
        force_refresh: bool = False,
    ) -> SearchArtifact | None:
        if force_refresh:
            return None
        normalized_query = _normalize_query(query)
        if not normalized_query:
            return None
        query_tokens = set(_tokenize(normalized_query))
        if not query_tokens:
            return None

        freshest = freshness_seconds if freshness_seconds is not None else _freshness_window(query)
        now = _now()
        best: SearchArtifact | None = None
        best_score = 0.0

        for artifact in reversed(self._read_all()):
            if not allow_stale and (now - float(artifact.updated_at or artifact.created_at)) > freshest:
                continue

            artifact_tokens = set(_tokenize(" ".join([artifact.query, artifact.answer, " ".join(artifact.facts)])))
            if not artifact_tokens:
                continue

            overlap = len(query_tokens & artifact_tokens) / max(1, len(query_tokens))
            if artifact.normalized_query == normalized_query:
                overlap += 1.0
            elif normalized_query in artifact.normalized_query or artifact.normalized_query in normalized_query:
                overlap += 0.4

            recency = max(0.0, 1.0 - ((now - float(artifact.updated_at or artifact.created_at)) / max(1, freshest)))
            score = overlap + (recency * 0.25)
            if score > best_score:
                best_score = score
                best = artifact

        # Component A5: Raise confidence threshold to avoid serving garbage
        return best if (best_score >= 0.55 and (best.confidence >= 0.4 or allow_stale)) else None


class ResearchSearchPipeline:
    """Hybrid web retrieval with chunking, synthesis, and retention."""

    def __init__(self, artifact_store: SearchArtifactStore | None = None):
        self.artifact_store = artifact_store or SearchArtifactStore()

    async def search(
        self,
        query: str,
        *,
        num_results: int = 5,
        deep: bool = False,
        retain: bool | None = None,
        context: dict[str, Any] | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        search_started = time.perf_counter()
        timing_ms: dict[str, float] = {}
        emitter = get_emitter()
        ctx = dict(context or {})
        cleaned_query = _normalize_query(query)
        if not cleaned_query:
            return {"ok": False, "error": "No search query provided."}

        freshness_seconds = _freshness_window(cleaned_query)
        cached = self.artifact_store.find_best(cleaned_query, freshness_seconds=freshness_seconds, force_refresh=force_refresh)
        if cached is not None:
            emitter.emit("✅ Knowledge Retrieved", f"Found verified answers for '{cleaned_query}' in persistent memory.", level="success", category="Research")
            result = self._artifact_to_result(cached, cached=True)
            if retain:
                result["retained"] = True
            return result

        emitter.emit("🔍 Searching...", f"Gathering sources for: {cleaned_query}", category="Research")
        stage_started = time.perf_counter()
        expanded_queries = await self._expand_queries(cleaned_query, ctx)
        timing_ms["query_expansion"] = round(
            (time.perf_counter() - stage_started) * 1000.0,
            1,
        )
        source_reading = query_requires_source_reading(cleaned_query)
        evidence_only = bool(ctx.get("evidence_only"))
        candidate_count = (
            max(1, num_results)
            if evidence_only and source_reading
            else max(num_results, 5)
        )
        stage_started = time.perf_counter()
        hits = await self._search_candidates(
            expanded_queries,
            num_results=candidate_count,
        )
        timing_ms["candidate_search"] = round(
            (time.perf_counter() - stage_started) * 1000.0,
            1,
        )
        hits = self._rerank_hits(cleaned_query, hits)
        if not hits:
            emitter.emit("⚠️ Search Failed", f"No results found for '{cleaned_query}'.", level="warning", category="Research")
            return {
                "ok": False,
                "error": f"No results found for '{cleaned_query}'.",
                "query": cleaned_query,
                "results": [],
                "result": "",
                "source": "none",
                "mode": "standard",
            }

        if source_reading:
            # A caller asking for three read sources needs three read sources,
            # not six speculative page loads.  Source-count repair belongs to
            # the caller after it can inspect which pages were actually usable.
            max_pages = max(1, min(len(hits), num_results))
        else:
            max_pages = min(len(hits), 6 if deep else 3)
        emitter.emit("📖 Reading Sources", f"Fetching top {max_pages} pages...", category="Research")
        
        stage_started = time.perf_counter()
        pages = await self._fetch_pages(hits[:max_pages], deep=deep)
        timing_ms["source_fetch"] = round(
            (time.perf_counter() - stage_started) * 1000.0,
            1,
        )

        # Record what is in hand BEFORE the summary is attempted.
        #
        # LIVE, 2026-08-22: five pages were fetched, the synthesis ran past the
        # caller's 35-second budget, and asyncio.wait_for cancelled the task —
        # taking the five pages with it. The turn ended in "I couldn't get to
        # an answer I'd stand behind" while the sources sat in memory.
        try:
            from core.search.gathered_sources import record_gathered

            record_gathered(cleaned_query, pages or hits[:max_pages])
        except (ImportError, AttributeError, TypeError, ValueError):
            pass

        if pages:
            emitter.emit("🧠 Synthesizing", f"Cross-referencing {len(pages)} sources to synthesize an accurate answer...", category="Research")
        else:
            emitter.emit("⚠️ Fallback Synthesis", "No full pages could be extracted. Attempting synthesis from search snippets...", level="warning", category="Research")

        chunks = self._rerank_chunks(cleaned_query, expanded_queries, pages, hits)
        stage_started = time.perf_counter()
        synthesized = await self._synthesize_answer(
            cleaned_query,
            chunks,
            ctx,
            allow_model=not evidence_only,
        )
        timing_ms["synthesis"] = round(
            (time.perf_counter() - stage_started) * 1000.0,
            1,
        )

        # Component A2: Cross-validation warning for low confidence
        if deep and synthesized["confidence"] < 0.5:
            emitter.emit("⚠️ Low Confidence", "Synthesized answer has low confidence. Discrepancies exist across sources.", level="warning", category="Research")
        else:
            emitter.emit("✅ Research Complete", f"Compiled concise answer with {len(synthesized['citations'])} citations.", level="success", category="Research")

        result = {
            "ok": True,
            "query": cleaned_query,
            "expanded_queries": expanded_queries,
            "results": [asdict(hit) for hit in hits[:num_results]],
            "answer": synthesized["answer"],
            "facts": synthesized["facts"],
            "confidence": synthesized["confidence"],
            "citations": synthesized["citations"],
            "summary": synthesized["answer"] or hits[0].snippet,
            "source": synthesized["citations"][0]["url"] if synthesized["citations"] else hits[0].url,
            "mode": "deep" if deep else "standard",
            "synthesis_mode": (
                "deterministic_evidence" if evidence_only else "model_with_deterministic_fallback"
            ),
            "timing_ms": timing_ms,
            "count": len(hits[:num_results]),
            "chunks": synthesized["evidence"],
            "content": "\n\n".join(item["text"] for item in synthesized["evidence"][:3]),
        }
        result["result"] = result["answer"] or result["content"] or self._format_hits(hits[:num_results])
        result["message"] = self._format_message(cleaned_query, result)
        timing_ms["total"] = round(
            (time.perf_counter() - search_started) * 1000.0,
            1,
        )
        logger.info(
            "Research stages for %r: query=%.1fms candidates=%.1fms fetch=%.1fms "
            "synthesis=%.1fms total=%.1fms mode=%s",
            cleaned_query[:100],
            timing_ms["query_expansion"],
            timing_ms["candidate_search"],
            timing_ms["source_fetch"],
            timing_ms["synthesis"],
            timing_ms["total"],
            result["synthesis_mode"],
        )

        should_retain = self._should_retain(cleaned_query, deep=deep, retain=retain, context=ctx, result=result)
        if should_retain:
            artifact = self._result_to_artifact(result, freshness_seconds=freshness_seconds)
            await self._retain_artifact(artifact, ctx)
            result["retained"] = True
            result["artifact_id"] = artifact.artifact_id

        return result

    async def _expand_queries(self, query: str, context: dict[str, Any]) -> list[str]:
        expansions = [query]

        # Document workflows already carry a concrete, source-reading query and
        # perform their own semantic synthesis after retrieval.  Spending a
        # model call to paraphrase that query adds seconds and often broadens it
        # away from the exact freshness/source-count contract.
        if bool(context.get("evidence_only")) and query_requires_source_reading(query):
            return expansions

        llm_prompt = (
            "Rewrite this web research query into 3 concise search variants.\n"
            "Return ONLY a JSON array of strings.\n"
            f"Query: {query}"
        )
        llm_output = await self._reason(llm_prompt, context=context, timeout_seconds=6.0)
        if llm_output:
            try:
                start = llm_output.find("[")
                end = llm_output.rfind("]") + 1
                if start != -1 and end > start:
                    parsed = json.loads(llm_output[start:end])
                    if isinstance(parsed, list):
                        for item in parsed:
                            cleaned = _normalize_query(str(item or ""))
                            if cleaned and cleaned not in expansions:
                                expansions.append(cleaned)
            except (json.JSONDecodeError, TypeError, ValueError):
                logger.debug("Search query expansion parse failed for %s", query)

        base = _normalize_query(query)
        if base not in expansions:
            expansions.insert(0, base)
        if _query_is_current(query):
            recent_variant = _normalize_query(f"{base} latest updates")
            if recent_variant not in expansions:
                expansions.append(recent_variant)
        elif not any(token in base for token in {"overview", "explained", "guide"}):
            overview_variant = _normalize_query(f"{base} overview")
            if overview_variant not in expansions:
                expansions.append(overview_variant)
        return expansions[:4]

    async def _search_candidates(self, queries: Iterable[str], *, num_results: int) -> list[SearchHit]:
        aggregated: dict[str, SearchHit] = {}
        ordered: list[SearchHit] = []
        use_ddgs = _ddgs_enabled()

        async def _search_one(query: str) -> list[SearchHit]:
            cached_hits = self._load_cached_search_hits(query, limit=num_results)
            if cached_hits:
                return cached_hits
            if use_ddgs:
                ddgs_hits = await asyncio.to_thread(self._ddgs_search, query, num_results)
                if ddgs_hits:
                    return ddgs_hits
            return await asyncio.to_thread(self._legacy_html_search, query, num_results)

        query_list = list(queries)
        batches = await asyncio.gather(
            *(_search_one(query) for query in query_list),
            return_exceptions=True,
        )
        for query, batch in zip(query_list, batches, strict=True):
            if isinstance(batch, asyncio.CancelledError):
                raise batch
            if isinstance(batch, Exception):
                record_degradation("research_pipeline.search_candidates", batch)
                logger.debug("Candidate search failed for %s: %s", query, batch)
                continue
            hits = batch
            for hit in hits:
                normalized_url = hit.url.rstrip("/")
                if not normalized_url or normalized_url in aggregated:
                    continue
                aggregated[normalized_url] = hit
                ordered.append(hit)
                if len(ordered) >= max(8, num_results):
                    return ordered
        return ordered

    def _ddgs_search(self, query: str, num_results: int) -> list[SearchHit]:
        if not _ddgs_enabled():
            return []

        ddgs_cls = None
        try:
            from ddgs import DDGS

            ddgs_cls = DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS

                ddgs_cls = DDGS
            except ImportError:
                ddgs_cls = None

        if ddgs_cls is None:
            return []

        hits: list[SearchHit] = []
        try:
            with ddgs_cls(timeout=8) as client:
                methods = []
                if _query_is_current(query) and hasattr(client, "news"):
                    methods.append(("news", {"max_results": max(4, num_results)}))
                methods.append(("text", {"max_results": max(4, num_results)}))

                for method_name, kwargs in methods:
                    method = getattr(client, method_name, None)
                    if method is None:
                        continue
                    try:
                        rows = method(query, **kwargs)
                    except TypeError:
                        rows = method(query)
                    for idx, row in enumerate(list(rows or [])[:num_results], start=1):
                        title = _normalize_text(str(row.get("title") or row.get("body") or ""), limit=220)
                        url = _normalize_text(str(row.get("href") or row.get("url") or ""), limit=500)
                        snippet = _normalize_text(str(row.get("body") or row.get("snippet") or ""), limit=360)
                        if title and url:
                            hits.append(
                                SearchHit(
                                    title=title,
                                    url=url,
                                    snippet=snippet,
                                    source_engine=f"ddgs:{method_name}",
                                    position=idx,
                                )
                            )
        except (ConnectionError, OSError, RuntimeError, TimeoutError, TypeError) as exc:
            record_degradation('research_pipeline', exc)
            logger.debug("DDGS search failed for %s: %s", query, exc)
            return self._load_cached_search_hits(query, limit=num_results)

        deduped = self._dedupe_hits(hits)
        if deduped:
            self._store_cached_search_hits(query, deduped)
        return deduped

    def _legacy_html_search(self, query: str, num_results: int) -> list[SearchHit]:
        import urllib.parse

        from bs4 import BeautifulSoup

        encoded = urllib.parse.quote_plus(query)
        endpoints = (
            ("https://html.duckduckgo.com/html/?q=", "ddg_html", self._parse_ddg_html_results),
            ("https://lite.duckduckgo.com/lite/?q=", "ddg_lite", self._parse_ddg_lite_results),
            ("https://www.bing.com/search?format=rss&q=", "bing_rss", self._parse_bing_rss_results),
            ("https://www.bing.com/search?q=", "bing_html", self._parse_bing_html_results),
        )

        aggregated: list[SearchHit] = []
        seen_urls: set[str] = set()

        for base_url, source_engine, parser in endpoints:
            url = f"{base_url}{encoded}"

            try:
                response = get_network_gateway().request(
                    "GET",
                    url,
                    headers=_HEADERS,
                    timeout=10,
                    source=f"research_pipeline.legacy_html_search.{source_engine}",
                    read_only=True,
                )
                if not response.get("ok"):
                    raise OSError(str(response.get("error") or "legacy search request failed"))
                raw_html = bytes(response.get("content") or b"").decode("utf-8", errors="replace")
            except (ConnectionError, OSError, TimeoutError) as exc:
                record_degradation('research_pipeline', exc)
                logger.debug("Legacy HTML search failed for %s via %s: %s", query, source_engine, exc)
                continue

            soup = None if source_engine == "bing_rss" else BeautifulSoup(raw_html, "html.parser")
            results = parser(raw_html, soup, source_engine=source_engine, limit=num_results)
            for hit in results:
                normalized_url = hit.url.rstrip("/")
                if not normalized_url or normalized_url in seen_urls:
                    continue
                seen_urls.add(normalized_url)
                aggregated.append(
                    SearchHit(
                        title=hit.title,
                        url=hit.url,
                        snippet=hit.snippet,
                        source_engine=hit.source_engine,
                        position=len(aggregated) + 1,
                    )
                )
                if len(aggregated) >= num_results:
                    self._store_cached_search_hits(query, aggregated)
                    return aggregated

        if aggregated:
            self._store_cached_search_hits(query, aggregated)
        return aggregated

    def _parse_ddg_html_results(
        self,
        raw_html: str,
        soup: Any,
        *,
        source_engine: str,
        limit: int,
    ) -> list[SearchHit]:
        title_re = re.compile(
            r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            re.DOTALL | re.IGNORECASE,
        )
        snippet_re = re.compile(
            r'class="result__snippet"[^>]*>(.*?)</a>',
            re.DOTALL | re.IGNORECASE,
        )

        results: list[SearchHit] = []
        snippets = [match.group(1) for match in snippet_re.finditer(raw_html)]
        for index, (href, title) in enumerate(title_re.findall(raw_html)[:limit], start=1):
            real_url = self._extract_ddg_url(href)
            clean_title = _normalize_text(re.sub(r"<[^>]+>", "", title))
            clean_snippet = ""
            if index - 1 < len(snippets):
                clean_snippet = _normalize_text(re.sub(r"<[^>]+>", "", snippets[index - 1]))
            if real_url and clean_title:
                results.append(
                    SearchHit(
                        title=clean_title,
                        url=real_url,
                        snippet=clean_snippet,
                        source_engine=source_engine,
                        position=index,
                    )
                )

        if results:
            return results

        anchors = soup.select("a.result__a") or soup.select("a[href*='uddg=']")
        snippet_nodes = soup.select(".result__snippet") or soup.select(".result-snippet")
        for index, anchor in enumerate(anchors[:limit], start=1):
            href = anchor.get("href") or ""
            real_url = self._extract_ddg_url(href)
            clean_title = _normalize_text(anchor.get_text(" ", strip=True))
            clean_snippet = ""
            if index - 1 < len(snippet_nodes):
                clean_snippet = _normalize_text(snippet_nodes[index - 1].get_text(" ", strip=True))
            if real_url and clean_title:
                results.append(
                    SearchHit(
                        title=clean_title,
                        url=real_url,
                        snippet=clean_snippet,
                        source_engine=source_engine,
                        position=index,
                    )
                )
        return results

    def _parse_ddg_lite_results(
        self,
        raw_html: str,
        soup: Any,
        *,
        source_engine: str,
        limit: int,
    ) -> list[SearchHit]:
        del raw_html
        results: list[SearchHit] = []
        anchors = soup.select("a[href*='uddg=']") or soup.select("a.result-link")
        for anchor in anchors:
            if len(results) >= limit:
                break
            href = anchor.get("href") or ""
            real_url = self._extract_ddg_url(href)
            clean_title = _normalize_text(anchor.get_text(" ", strip=True))
            if not real_url or not clean_title:
                continue
            if "duckduckgo.com" in _domain(real_url):
                continue
            snippet = ""
            row = anchor.find_parent("tr")
            if row is not None:
                sibling = row.find_next_sibling("tr")
                if sibling is not None:
                    snippet = _normalize_text(sibling.get_text(" ", strip=True), limit=360)
            results.append(
                SearchHit(
                    title=clean_title,
                    url=real_url,
                    snippet=snippet,
                    source_engine=source_engine,
                    position=len(results) + 1,
                )
            )
        return results

    def _parse_bing_html_results(
        self,
        raw_html: str,
        soup: Any,
        *,
        source_engine: str,
        limit: int,
    ) -> list[SearchHit]:
        del raw_html
        results: list[SearchHit] = []
        for item in soup.select("li.b_algo")[:limit]:
            anchor = item.select_one("h2 a")
            if anchor is None:
                continue
            real_url = _normalize_text(anchor.get("href") or "", limit=500)
            clean_title = _normalize_text(anchor.get_text(" ", strip=True))
            snippet_node = item.select_one(".b_caption p") or item.select_one(".b_snippet")
            snippet = _normalize_text(
                snippet_node.get_text(" ", strip=True) if snippet_node is not None else "",
                limit=360,
            )
            if real_url and clean_title:
                results.append(
                    SearchHit(
                        title=clean_title,
                        url=real_url,
                        snippet=snippet,
                        source_engine=source_engine,
                        position=len(results) + 1,
                    )
                )
        return results

    def _parse_bing_rss_results(
        self,
        raw_html: str,
        soup: Any,
        *,
        source_engine: str,
        limit: int,
    ) -> list[SearchHit]:
        del soup
        import xml.etree.ElementTree as ET

        try:
            root = ET.fromstring(raw_html)
        except ET.ParseError:
            return []

        results: list[SearchHit] = []
        for item in root.findall("./channel/item")[:limit]:
            title = _normalize_text(item.findtext("title") or "")
            url = _normalize_text(item.findtext("link") or "", limit=500)
            snippet = _normalize_text(item.findtext("description") or "", limit=360)
            if title and url:
                results.append(
                    SearchHit(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source_engine=source_engine,
                        position=len(results) + 1,
                    )
                )
        return results

    async def _fetch_pages(self, hits: list[SearchHit], *, deep: bool) -> list[SearchPage]:
        pages: list[SearchPage] = []
        timeout_val = 14.0 if deep else 8.0 # Component A3: Timeout hardening
        
        # 1. Fallback / Normal Mode: canonical network gateway
        tasks = [self._fetch_page(None, hit, timeout_val=timeout_val) for hit in hits]
        fetched = await asyncio.gather(*tasks, return_exceptions=True)

        for item in fetched:
            if isinstance(item, SearchPage):
                pages.append(item)

        # 2. Deep mode: browser-recover only pages that HTTP could not read
        # completely.  Recovery is parallel; a blocked source must not make
        # every other independent source wait behind its browser timeout.
        if deep:
            browser_hits = [
                hit
                for hit in hits
                if not any(page.url == hit.url and len(page.text) > 1000 for page in pages)
            ]
            browser_tasks = [self._fetch_page_with_browser(hit) for hit in browser_hits]
            if browser_tasks:
                browser_results = await asyncio.gather(*browser_tasks, return_exceptions=True)
                for item in browser_results:
                    if isinstance(item, SearchPage):
                        # Overwrite or append
                        pages = [p for p in pages if p.url != item.url]
                        pages.append(item)

        return pages

    async def _fetch_page(self, client: Any, hit: SearchHit, *, timeout_val: float = 8.0) -> SearchPage | None:
        try:
            response = await asyncio.to_thread(
                get_network_gateway().request,
                "GET",
                hit.url,
                headers=_HEADERS,
                timeout=timeout_val,
                source="research_pipeline.fetch_page",
                read_only=True,
            )
            status = int(response.get("status_code") or 0)
            if status >= 400 or not response.get("ok"):
                # Component A3: Error classification
                if status in (403, 404, 410, 451):
                    logger.debug("Page fetch permanent error (%d) for %s", status, hit.url)
                else:
                    logger.debug("Page fetch transient error (%d) for %s", status, hit.url)
                return None
        except Exception as exc:  # noqa: BLE001 - one page is optional
            # Same rule as the browser path below: fetching one source must
            # never be able to destroy the whole search. Whatever an HTTP
            # client, a codec, or a redirect chain decides to raise, the
            # answer is the same — skip this hit, keep the search.
            record_degradation('research_pipeline', exc)
            logger.debug("Page fetch failed for %s: %s", hit.url, exc)
            return None

        content_type = str(dict(response.get("headers") or {}).get("content-type", ""))
        body = bytes(response.get("content") or b"").decode("utf-8", errors="replace")
        # Component A3: Content-type handling
        if content_type and not any(t in content_type.lower() for t in ("html", "text/plain", "application/json")):
            return None

        text = _html_to_text(body)
        if len(text) < 200:
            return None

        title = _extract_title(body) or hit.title
        return SearchPage(
            url=hit.url,
            title=title,
            text=text[:60000],
            snippet=hit.snippet,
            source_engine=hit.source_engine,
            position=hit.position,
        )

    async def _fetch_page_with_browser(self, hit: SearchHit) -> SearchPage | None:
        try:
            from core.capabilities.phantom_browser import PhantomBrowser

            browser = PhantomBrowser(visible=False, principal="research_pipeline")
            try:
                await browser.ensure_ready()
                ok = await browser.browse(hit.url)
                if not ok:
                    return None
                text = await browser.read_content()
                cleaned = _normalize_text(text, limit=60000)
                if len(cleaned) < 200:
                    return None
                title = hit.title
                try:
                    if browser.page is not None:
                        title = _normalize_text(await browser.page.title(), limit=220) or hit.title
                except (RuntimeError, AttributeError, TypeError):
                    title = hit.title
                return SearchPage(
                    url=hit.url,
                    title=title,
                    text=cleaned,
                    snippet=hit.snippet,
                    source_engine=f"{hit.source_engine}|browser",
                    position=hit.position,
                )
            finally:
                await browser.close()
        except Exception as exc:  # noqa: BLE001 - see below
            # Deliberately broad: fetching ONE page is optional enrichment, and
            # the pipeline already knows how to synthesize from snippets when
            # no page is fetched. The old list — ImportError, ConnectionError,
            # OSError, RuntimeError, TimeoutError, AttributeError — does not
            # include playwright's own Error class, so a single navigation
            # failure escaped and destroyed the entire search.
            #
            # Live 2026-07-27: "Can you look up what the current weather is in
            # Lisbon right now?" The search itself SUCCEEDED — hits returned,
            # "Causal Link Recorded: web_search -> Success" — then one
            # page.goto raised, the whole tool result became ok=False, and the
            # user was told "the model lane was unavailable and the fallback
            # was rate-limited", which is not what happened at all.
            #
            # A partial result is the thing this pipeline exists to produce.
            record_degradation('research_pipeline', exc)
            logger.debug("Browser fetch failed for %s: %s", hit.url, exc)
            return None

    def _rerank_chunks(
        self,
        query: str,
        expanded_queries: Iterable[str],
        pages: list[SearchPage],
        hits: list[SearchHit],
    ) -> list[dict[str, Any]]:
        query_tokens = set(_tokenize(" ".join(expanded_queries)))
        quoted_phrases = _quoted_phrases(query)
        source_reading = query_requires_source_reading(query)
        page_by_url = {page.url: page for page in pages}
        ranked: list[dict[str, Any]] = []

        for hit in hits[:8]:
            page = page_by_url.get(hit.url)
            if page is None:
                pseudo_text = _normalize_text(f"{hit.title}. {hit.snippet}", limit=600)
                if pseudo_text:
                    ranked.append(
                        {
                            "title": hit.title,
                            "url": hit.url,
                            "text": pseudo_text,
                            "evidence_kind": "search_snippet",
                            "fetched": False,
                            "score": self._score_chunk(
                                query_tokens,
                                quoted_phrases,
                                pseudo_text,
                                rank=hit.position,
                                title=hit.title,
                                url=hit.url,
                                source_reading=source_reading,
                            ),
                            "source_engine": hit.source_engine,
                        }
                    )
                continue

            for chunk in self._chunk_page(page):
                ranked.append(
                        {
                            "title": page.title,
                            "url": page.url,
                            "text": chunk,
                            "evidence_kind": "article_body",
                            "fetched": True,
                            "fetched_at": page.fetched_at,
                            "document_chars": len(page.text),
                            "document_sha256": hashlib.sha256(
                                page.text.encode("utf-8", errors="replace")
                            ).hexdigest(),
                            "score": self._score_chunk(
                                query_tokens,
                                quoted_phrases,
                                chunk,
                                rank=page.position,
                                title=page.title,
                                url=page.url,
                                source_reading=source_reading,
                            ),
                            "source_engine": page.source_engine,
                        }
                )

        ranked.sort(key=lambda item: float(item["score"]), reverse=True)

        # Preserve source diversity before taking additional chunks from the
        # same page.  Without this, the top eight chunks could all come from one
        # article even after three pages were fetched, forcing a second search
        # and making a multi-source synthesis semantically false.
        diversified: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for item in ranked:
            url = str(item.get("url") or "")
            if url and url not in seen_urls:
                diversified.append(item)
                seen_urls.add(url)
            if len(diversified) >= 8:
                return diversified
        for item in ranked:
            if item in diversified:
                continue
            diversified.append(item)
            if len(diversified) >= 8:
                break
        return diversified

    def _chunk_page(self, page: SearchPage) -> list[str]:
        paragraphs = [
            _normalize_text(part)
            for part in re.split(r"\n{2,}", page.text)
            if len(_normalize_text(part)) >= 50
        ]
        if not paragraphs:
            paragraphs = [_normalize_text(page.text)]

        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            if len(current) + len(paragraph) + 1 <= 650:
                current = f"{current}\n{paragraph}".strip()
                continue
            if current:
                chunks.append(current)
            current = paragraph
        if current:
            chunks.append(current)
        return chunks[:6]

    def _rerank_hits(self, query: str, hits: Iterable[SearchHit]) -> list[SearchHit]:
        source_reading = query_requires_source_reading(query)
        ranked = sorted(
            list(hits),
            key=lambda hit: self._hit_relevance_score(
                query,
                hit.title,
                hit.url,
                rank=hit.position,
                source_reading=source_reading,
            ),
            reverse=True,
        )
        return ranked

    def _hit_relevance_score(
        self,
        query: str,
        title: str,
        url: str,
        *,
        rank: int,
        source_reading: bool,
    ) -> float:
        query_tokens = set(_tokenize(query))
        title_tokens = set(_tokenize(f"{title} {url}"))
        overlap = len(query_tokens & title_tokens) / max(1, len(query_tokens))
        normalized_target = _normalized_match_text(title, url)
        phrase_bonus = 0.0
        for phrase in _quoted_phrases(query):
            normalized_phrase = _normalized_match_text(phrase)
            if normalized_phrase and normalized_phrase in normalized_target:
                phrase_bonus += 1.2
        rank_bonus = max(0.0, 0.25 - ((max(rank, 1) - 1) * 0.03))
        document_bonus = 0.0
        if source_reading and re.search(r"(story|article|post|chapter|thread|page|document|report)", normalized_target):
            document_bonus += 0.2
        host_penalty = 0.0
        if any(term in str(url or "").lower() for term in _NOISY_RESULT_HOST_TERMS):
            host_penalty -= 0.45
        if str(url or "").lower().endswith(".pdf"):
            host_penalty -= 0.15
        return round(overlap + phrase_bonus + rank_bonus + document_bonus + host_penalty, 4)

    def _title_alignment_bonus(
        self,
        query_tokens: set[str],
        quoted_phrases: list[str],
        title: str,
        url: str,
        *,
        rank: int,
        source_reading: bool,
    ) -> float:
        title_tokens = set(_tokenize(f"{title} {url}"))
        overlap = len(query_tokens & title_tokens) / max(1, len(query_tokens))
        normalized_target = _normalized_match_text(title, url)
        phrase_bonus = 0.0
        for phrase in quoted_phrases:
            normalized_phrase = _normalized_match_text(phrase)
            if normalized_phrase and normalized_phrase in normalized_target:
                phrase_bonus += 1.0
        rank_bonus = max(0.0, 0.2 - ((max(rank, 1) - 1) * 0.025))
        document_bonus = 0.15 if source_reading and re.search(
            r"(story|article|post|chapter|thread|page|document|report)",
            normalized_target,
        ) else 0.0
        return overlap + phrase_bonus + rank_bonus + document_bonus

    def _score_chunk(
        self,
        query_tokens: set[str],
        quoted_phrases: list[str],
        text: str,
        *,
        rank: int,
        title: str = "",
        url: str = "",
        source_reading: bool = False,
    ) -> float:
        lowered = text.lower()
        tokens = set(_tokenize(lowered))
        overlap = len(query_tokens & tokens) / max(1, len(query_tokens))
        phrase_bonus = sum(0.2 for phrase in quoted_phrases if phrase.lower() in lowered)
        rank_bonus = max(0.0, 0.2 - ((max(rank, 1) - 1) * 0.02))
        currentness_bonus = 0.0
        if any(term in lowered for term in _CURRENTNESS_TERMS):
            currentness_bonus += 0.05
        if re.search(r"\b20\d{2}\b", lowered):
            currentness_bonus += 0.05
        title_bonus = self._title_alignment_bonus(
            query_tokens,
            quoted_phrases,
            title,
            url,
            rank=rank,
            source_reading=source_reading,
        ) * 0.35
        return round(overlap + phrase_bonus + rank_bonus + currentness_bonus + title_bonus, 4)

    async def _synthesize_answer(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        context: dict[str, Any],
        *,
        allow_model: bool = True,
    ) -> dict[str, Any]:
        top_chunks = chunks[:5]
        source_reading = query_requires_source_reading(query)
        citations = [
            {"title": item["title"], "url": item["url"]}
            for item in top_chunks
            if item.get("url")
        ]

        if not top_chunks:
            return {
                "answer": "",
                "facts": [],
                "confidence": 0.0,
                "citations": citations,
                "evidence": [],
            }

        prompt_lines = [
            "You are a research analyst. Synthesize a thorough, accurate answer using ONLY the evidence below.",
            "Cross-reference multiple sources. Note where sources agree and where they conflict.",
            "Return ONLY JSON with keys: answer, facts, confidence.",
            "- answer: A comprehensive, well-sourced response (2-4 paragraphs for complex topics, 1-2 for simple facts).",
            "- facts: An array of 3-8 concrete, verifiable statements extracted from the evidence.",
            "- confidence: Float 0.0-1.0 reflecting how well the evidence supports the answer.",
            f"Question: {query}",
            "Evidence:",
        ]
        if source_reading:
            prompt_lines.insert(
                1,
                "This query is asking about a specific document, story, page, or source. Summarize ONLY what the retrieved page evidence actually says, and include concrete details that would only be visible after reading it.",
            )
        # Use much more content per source for M5/64GB — 8000 chars for deep, 4000 for standard
        chars_per_source = 8000 if len(top_chunks) <= 3 else 4000
        for index, item in enumerate(top_chunks, start=1):
            prompt_lines.append(
                f"[{index}] {item['title']} | {item['url']}\n{item['text'][:chars_per_source]}"
            )

        # Deep mode gets more synthesis time for thorough analysis
        synthesis_timeout = 20.0 if len(top_chunks) >= 4 else 12.0
        llm_output = ""
        if allow_model:
            llm_output = await self._reason(
                "\n\n".join(prompt_lines),
                context=context,
                timeout_seconds=synthesis_timeout,
            )
        if llm_output:
            parsed = self._parse_synthesis_json(llm_output)
            if parsed is not None:
                return {
                    "answer": parsed["answer"],
                    "facts": parsed["facts"],
                    "confidence": parsed["confidence"],
                    "citations": citations,
                    "evidence": top_chunks,
                }

        answer, facts, confidence = self._deterministic_synthesis(query, top_chunks)
        return {
            "answer": answer,
            "facts": facts,
            "confidence": confidence,
            "citations": citations,
            "evidence": top_chunks,
        }

    def _parse_synthesis_json(self, raw: str) -> dict[str, Any] | None:
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start == -1 or end <= start:
                return None
            data = json.loads(raw[start:end])
            answer = _normalize_text(str(data.get("answer") or ""), limit=25000)
            facts = [
                _normalize_text(str(item or ""), limit=2000)
                for item in list(data.get("facts") or [])[:8]
                if _normalize_text(str(item or ""))
            ]
            confidence = float(data.get("confidence", 0.55) or 0.55)
            if not answer:
                return None
            return {
                "answer": answer,
                "facts": facts,
                "confidence": max(0.0, min(1.0, confidence)),
            }
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

    def _deterministic_synthesis(
        self,
        query: str,
        chunks: list[dict[str, Any]],
    ) -> tuple[str, list[str], float]:
        query_tokens = set(_tokenize(query))
        source_reading = query_requires_source_reading(query)
        quoted_phrases = _quoted_phrases(query)
        
        # Extract sentences with keyword matches
        scored_sentences = []
        for item in chunks[:4]:
            sentences = re.split(r"(?<=[.!?])\s+", item["text"])
            for idx, sentence in enumerate(sentences):
                clean = _normalize_text(sentence, limit=200)
                if not clean or len(clean) < 35:
                    continue
                    
                sentence_tokens = set(_tokenize(clean))
                overlap = len(query_tokens & sentence_tokens)
                if not source_reading and overlap <= 0:
                    continue
                score = float(item.get("score", 0.0) or 0.0)
                score += overlap * 0.08
                if idx == 0:
                    score += 0.08
                if source_reading:
                    score += 0.2
                lower = clean.lower()
                if any(phrase.lower() in lower for phrase in quoted_phrases):
                    score += 0.35
                scored_sentences.append((score, clean))
                    
        # Sort by overlap score and pick top sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        best_sentences = []
        seen = set()
        for _, sentence in scored_sentences:
            if sentence not in seen:
                best_sentences.append(sentence)
                seen.add(sentence)
            if len(best_sentences) >= 3:
                break
                
        # If we failed to extract keyword matches, fallback to lead snippets
        if not best_sentences:
            top = chunks[0]
            lead = _normalize_text(top["text"], limit=320)
            second = _normalize_text(chunks[1]["text"], limit=220) if len(chunks) > 1 else ""
            answer = lead
            if second and second not in answer:
                answer = f"{lead} {second}"
            facts: list[str] = []
            for item in chunks[:3]:
                sentences = re.split(r"(?<=[.!?])\s+", item["text"])
                for sentence in sentences:
                    clean = _normalize_text(sentence, limit=200)
                    if clean and clean not in facts and len(clean) >= 35:
                        facts.append(clean)
                    if len(facts) >= 4:
                        break
                if len(facts) >= 4:
                    break
        else:
            answer = " ".join(best_sentences)
            facts = best_sentences[:4]
            
        top_score = float(chunks[0].get("score", 0.0) or 0.0) if chunks else 0.0
        confidence = min(0.92, max(0.40, 0.42 + (len(chunks) * 0.05) + min(top_score, 1.5) * 0.12))
        
        # Component A4: Confidence marker
        if confidence < 0.5:
            answer = f"[Confidence: Low] {answer}"
            
        return answer, facts, round(confidence, 2)

    async def _retain_artifact(self, artifact: SearchArtifact, context: dict[str, Any]) -> None:
        self.artifact_store.append(artifact)

        metadata = {
            "source": "web_search",
            "intent_source": "autonomous_research",
            "provenance_source": "web_search",
            "artifact_id": artifact.artifact_id,
            "query": artifact.query,
            "current": artifact.current,
            "confidence": artifact.confidence,
            "confidence_tier": "provisional",
            "requires_reconciliation": True,
            "citations": artifact.citations[:3],
        }
        memory_text = self._build_memory_note(artifact)
        retained = False
        deferral_reason = ""

        # 1. Primary: memory facade (episodic/dual memory)
        memory_facade = context.get("memory_facade")
        try:
            if memory_facade is None:
                from core.container import ServiceContainer
                memory_facade = ServiceContainer.get("memory_facade", default=None)
            if memory_facade is not None and hasattr(memory_facade, "add_memory"):
                result = memory_facade.add_memory(memory_text, metadata=metadata)
                if hasattr(result, "__await__"):
                    result = await result
                if result:
                    retained = True
                else:
                    status = getattr(memory_facade, "_last_add_memory_status", None)
                    if isinstance(status, dict):
                        deferral_reason = str(status.get("reason") or "")
        except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
            record_degradation('research_pipeline', exc)
            logger.debug("Memory facade retention failed: %s", exc)

        # 2. Secondary: semantic/vector memory
        semantic_memory = context.get("semantic_memory")
        try:
            if semantic_memory is None:
                from core.container import ServiceContainer
                semantic_memory = ServiceContainer.get("vector_memory_engine", default=None)
            if semantic_memory is not None and hasattr(semantic_memory, "store"):
                result = semantic_memory.store(
                    content=memory_text,
                    memory_type="semantic",
                    source="web_search",
                    tags=["research", "web_learning", artifact.query[:50]],
                    metadata=metadata,
                )
                if hasattr(result, "__await__"):
                    await result
                retained = True
            elif semantic_memory is not None and hasattr(semantic_memory, "remember"):
                result = semantic_memory.remember(memory_text, metadata)
                if hasattr(result, "__await__"):
                    result = await result
                if result:
                    retained = True
        except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
            record_degradation('research_pipeline', exc)
            logger.debug("Vector memory retention failed: %s", exc)

        # 3b. Local knowledge corpus: verified retained knowledge becomes a
        # permanently searchable reference document (source=web_retained) —
        # the corpus accretes from lived research, not only dump snapshots.
        try:
            import asyncio as _asyncio

            from core.knowledge.local_corpus import get_local_corpus_store

            _corpus = get_local_corpus_store()

            def _corpus_writeback() -> None:
                # The populated-guard AND the write both touch sqlite — the
                # guard alone (a COUNT(*) full scan) ran ON the loop and was
                # a fingerprinted 5s stall class at corpus scale. Everything
                # sqlite happens on this worker thread.
                if _corpus.has_documents():
                    _corpus.add_retained_document(
                        artifact.query[:300],
                        memory_text,
                        artifact_id=artifact.artifact_id,
                    )

            await _asyncio.to_thread(_corpus_writeback)
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation(
                "research_pipeline", exc, severity="debug",
                action="corpus writeback skipped",
            )

        # 3. Tertiary: update belief system with facts
        try:
            from core.container import ServiceContainer
            belief_engine = ServiceContainer.get("belief_revision_engine", default=None)
            if belief_engine and hasattr(belief_engine, "add_belief"):
                for fact in artifact.facts[:5]:
                    try:
                        belief_engine.add_belief(
                            content=fact,
                            confidence=artifact.confidence,
                            domain="learned_from_web",
                            source=f"web_search:{artifact.query[:40]}",
                        )
                    except (RuntimeError, AttributeError, TypeError):
                        pass  # no-op: intentional
                retained = True
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation('research_pipeline', exc)
            logger.debug("Belief system retention failed: %s", exc)

        # 4. Feed to WorldState as a salient learning event
        try:
            from core.world_state import get_world_state
            ws = get_world_state()
            ws.record_event(
                f"Learned: {artifact.answer[:100]}",
                source="web_search",
                salience=0.4,
                ttl=7200,
            )
        except (ImportError, AttributeError, RuntimeError):
            pass  # no-op: intentional

        # 5. Satisfy curiosity drive (learning something satisfies curiosity)
        try:
            from core.container import ServiceContainer
            drive = ServiceContainer.get("drive_engine", default=None)
            if drive:
                try:
                    get_task_tracker().create_task(
                        drive.satisfy("curiosity", 20.0),
                        name="research_pipeline.drive_curiosity",
                    )
                except RuntimeError:
                    pass  # no-op: intentional
        except (ImportError, AttributeError, RuntimeError):
            pass  # no-op: intentional

        if not retained:
            # A deferral is a "not now", and research that already cost real
            # time and network calls must not evaporate because the runtime
            # happened to be unwilling at that instant. Hold it and retry;
            # a refusal on the merits is still dropped by the queue itself.
            held = False
            try:
                from core.memory.deferred_retention import (
                    get_deferred_retention_queue,
                )

                held = await get_deferred_retention_queue().enqueue_async(
                    memory_text,
                    metadata,
                    reason=deferral_reason or "memory_write_deferred",
                    origin="research_pipeline",
                )
            except (ImportError, RuntimeError, OSError, TypeError, ValueError) as exc:
                record_degradation(
                    "research_pipeline", exc, severity="warning",
                    action="deferred research artifact could not be held for retry",
                )
            if held:
                logger.info(
                    "Knowledge write deferred (%s); artifact held for retry.",
                    deferral_reason or "no reason given",
                )
            else:
                logger.warning(
                    "Knowledge persistence failed: no memory backend accepted "
                    "the artifact (%s).", deferral_reason or "no reason given",
                )

    def _build_memory_note(self, artifact: SearchArtifact) -> str:
        fact_lines = "\n".join(f"- {fact}" for fact in artifact.facts[:4])
        citation_lines = "\n".join(
            f"- {item.get('title', '')}: {item.get('url', '')}"
            for item in artifact.citations[:3]
        )
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(artifact.updated_at))
        return (
            f"[WebLearning {timestamp}] Query: {artifact.query}\n"
            f"Answer: {artifact.answer}\n"
            f"Facts:\n{fact_lines or '- None extracted'}\n"
            f"Sources:\n{citation_lines or '- No citations recorded'}"
        )

    def _result_to_artifact(self, result: dict[str, Any], *, freshness_seconds: int) -> SearchArtifact:
        query = str(result.get("query") or "").strip()
        citations = list(result.get("citations") or [])
        digest_source = query + "|" + "|".join(str(item.get("url") or "") for item in citations[:5])
        artifact_id = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:16]
        now = _now()
        return SearchArtifact(
            artifact_id=artifact_id,
            query=query,
            normalized_query=_normalize_query(query),
            answer=_normalize_text(str(result.get("answer") or ""), limit=25000),
            summary=_normalize_text(str(result.get("summary") or ""), limit=25000),
            facts=[
                _normalize_text(str(item or ""), limit=2000)
                for item in list(result.get("facts") or [])[:8]
            ],
            citations=[
                {
                    "title": _normalize_text(str(item.get("title") or ""), limit=500),
                    "url": _normalize_text(str(item.get("url") or ""), limit=1000),
                }
                for item in citations[:8]
            ],
            evidence=[
                {
                    "title": _normalize_text(str(item.get("title") or ""), limit=500),
                    "url": _normalize_text(str(item.get("url") or ""), limit=1000),
                    "text": _normalize_text(str(item.get("text") or ""), limit=15000),
                    "score": float(item.get("score", 0.0) or 0.0),
                }
                for item in list(result.get("chunks") or [])[:5]
            ],
            created_at=now,
            updated_at=now,
            freshness_seconds=freshness_seconds,
            confidence=float(result.get("confidence", 0.6) or 0.6),
            current=_query_is_current(query),
            source=_normalize_text(str(result.get("source") or ""), limit=500),
        )

    def _artifact_to_result(self, artifact: SearchArtifact, *, cached: bool) -> dict[str, Any]:
        hits = [
            SearchHit(
                title=_normalize_text(str(item.get("title") or ""), limit=220),
                url=_normalize_text(str(item.get("url") or ""), limit=500),
                snippet="",
                source_engine="retained_memory",
                position=index,
            )
            for index, item in enumerate(artifact.citations, start=1)
            if item.get("url")
        ]
        result = {
            "ok": True,
            "query": artifact.query,
            "results": [asdict(hit) for hit in hits],
            "answer": artifact.answer,
            "facts": artifact.facts,
            "confidence": artifact.confidence,
            "citations": artifact.citations,
            "summary": artifact.summary or artifact.answer,
            "source": artifact.source or (artifact.citations[0]["url"] if artifact.citations else ""),
            "mode": "cached",
            "count": len(hits),
            "chunks": artifact.evidence,
            "content": "\n\n".join(str(item.get("text") or "") for item in artifact.evidence[:3]),
            "cached": cached,
            "artifact_id": artifact.artifact_id,
        }
        result["result"] = result["answer"] or result["content"] or self._format_hits(hits)
        result["message"] = self._format_message(artifact.query, result)
        return result

    def _format_message(self, query: str, result: dict[str, Any]) -> str:
        answer = _normalize_text(str(result.get("answer") or result.get("summary") or ""), limit=500)
        citations = list(result.get("citations") or [])
        if not citations:
            return answer
        source_lines = "\n".join(
            f"- {item.get('title', '')}: {item.get('url', '')}"
            for item in citations[:3]
        )
        return f"I searched for '{query}'. {answer}\nSources:\n{source_lines}"

    def _format_hits(self, hits: Iterable[SearchHit]) -> str:
        return "\n\n".join(
            f"{index}. {hit.title}\n{hit.snippet}\n{hit.url}"
            for index, hit in enumerate(hits, start=1)
        )

    def _should_retain(
        self,
        query: str,
        *,
        deep: bool,
        retain: bool | None,
        context: dict[str, Any],
        result: dict[str, Any],
    ) -> bool:
        if retain is not None:
            return retain
        origin = str(
            context.get("intent_source")
            or context.get("origin")
            or context.get("request_origin")
            or ""
        ).lower()
        if origin in {
            "research_cycle",
            "curiosity_explorer",
            "autonomous_volition",
            "autonomous_thought",
            "impulse",
            "background",
            "world_monitor",
        }:
            return True
        if deep:
            return True
        if _query_is_current(query):
            return True
        return bool(result.get("facts"))

    async def _reason(
        self,
        prompt: str,
        *,
        context: dict[str, Any],
        timeout_seconds: float,
    ) -> str:
        router = context.get("llm_router")
        if router is None:
            try:
                from core.container import ServiceContainer

                router = ServiceContainer.get("llm_router", default=None)
            except (ImportError, AttributeError):
                router = None
        if router is not None and hasattr(router, "think"):
            try:
                result = await asyncio.wait_for(
                    router.think(
                        prompt,
                        priority=0.25,
                        is_background=str(context.get("origin", "")).lower() not in {"user", "voice", "admin"},
                    ),
                    timeout=timeout_seconds,
                )
                return _normalize_text(str(result or ""), limit=4000)
            except TypeError:
                try:
                    result = await asyncio.wait_for(router.think(prompt), timeout=timeout_seconds)
                    return _normalize_text(str(result or ""), limit=4000)
                except (TimeoutError, RuntimeError, AttributeError, TypeError):
                    pass  # no-op: intentional
            except (TimeoutError, RuntimeError, AttributeError):
                pass  # no-op: intentional

        brain = context.get("brain")
        if brain is not None and hasattr(brain, "think"):
            try:
                result = brain.think(prompt)
                if hasattr(result, "__await__"):
                    result = await asyncio.wait_for(result, timeout=timeout_seconds)
                if hasattr(result, "content"):
                    return _normalize_text(str(result.content or ""), limit=4000)
                if isinstance(result, dict):
                    return _normalize_text(str(result.get("content") or result.get("text") or ""), limit=4000)
                return _normalize_text(str(result or ""), limit=4000)
            except (TimeoutError, RuntimeError, AttributeError, TypeError):
                return ""
        return ""

    def _dedupe_hits(self, hits: Iterable[SearchHit]) -> list[SearchHit]:
        deduped: list[SearchHit] = []
        seen: set[str] = set()
        for hit in hits:
            normalized = hit.url.rstrip("/")
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(hit)
        return deduped

    def _load_cached_search_hits(self, query: str, *, limit: int) -> list[SearchHit]:
        normalized_query = _normalize_query(query)
        if not normalized_query:
            return []
        cached = _SEARCH_HIT_CACHE.get(normalized_query)
        if not cached:
            return []
        cached_at, hits = cached
        if (_now() - cached_at) > _SEARCH_HIT_CACHE_TTL_SECONDS:
            _SEARCH_HIT_CACHE.pop(normalized_query, None)
            return []
        return [
            SearchHit(
                title=hit.title,
                url=hit.url,
                snippet=hit.snippet,
                source_engine=hit.source_engine,
                position=index,
            )
            for index, hit in enumerate(hits[:limit], start=1)
        ]

    def _store_cached_search_hits(self, query: str, hits: Iterable[SearchHit]) -> None:
        normalized_query = _normalize_query(query)
        if not normalized_query:
            return
        deduped = self._dedupe_hits(hits)
        if not deduped:
            return
        _SEARCH_HIT_CACHE[normalized_query] = (
            _now(),
            [
                SearchHit(
                    title=hit.title,
                    url=hit.url,
                    snippet=hit.snippet,
                    source_engine=hit.source_engine,
                    position=index,
                )
                for index, hit in enumerate(deduped, start=1)
            ],
        )

    @staticmethod
    def _extract_ddg_url(href: str) -> str:
        if href.startswith("http"):
            return href
        match = re.search(r"uddg=([^&]+)", href)
        if match:
            from urllib.parse import unquote

            return unquote(match.group(1))
        if href.startswith("//"):
            return "https:" + href
        return href

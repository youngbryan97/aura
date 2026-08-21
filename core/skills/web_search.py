"""Enhanced web search and research skill for Aura."""


import logging
from typing import Any

from pydantic import BaseModel, Field

from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from core.search import ResearchSearchPipeline
from core.search.research_pipeline import freshness_window_for_query, query_requires_source_reading
from core.skills.base_skill import BaseSkill
from core.skills.deep_research import run_deep_research

logger = logging.getLogger("Skills.WebSearch")


class _DeepResearchBrainAdapter:
    """Compat adapter for deep_research's ``brain.generate() -> {'response': ...}`` contract."""

    def __init__(self, engine: Any):
        self.engine = engine

    async def generate(self, prompt: str, **kwargs) -> dict[str, str]:
        # The caller's priority is HONOURED, not discarded.
        #
        # This accepted **kwargs and threw them away, hardcoding
        # is_background=True. So a synthesis the person is waiting for was
        # admitted as background work, queued behind foreground headroom, and
        # came back instantly empty — and the deep-research retry that asks for
        # foreground could not take effect because its request never left this
        # method. Measured live:
        #   "Deep research gathered 5 source(s) over 1 quer(ies) in 9.4s but
        #    could not synthesize them (the model returned no text)"
        # ...twice, including the retry.
        #
        # Background stays the default: ordinary research really is background.
        foreground = bool(kwargs.get("foreground_request", False))
        raw = await self.engine.generate(
            prompt,
            origin=str(kwargs.get("origin") or ("user" if foreground else "system")),
            purpose="research",
            use_strategies=False,
            is_background=not foreground,
            foreground_request=foreground,
            priority=float(kwargs.get("priority", 1.0 if foreground else 0.5)),
        )
        if isinstance(raw, dict):
            text = raw.get("response") or raw.get("content") or raw.get("result") or ""
        else:
            text = str(raw or "")
        return {"response": str(text or "")}


class WebSearchInput(BaseModel):
    query: str = Field(..., description="The search query to look up on the web.")
    deep: bool = Field(False, description="If True, fetch and synthesize multiple result pages.")
    num_results: int = Field(5, ge=1, le=20, description="Number of search hits to return.")
    retain: bool | None = Field(
        None,
        description="Whether Aura should retain what she learned from this search.",
    )
    force_refresh: bool = Field(False, description="If True, bypass cache and force a new live search.")


def _comprehension_of(
    results: Any, *, query: str = "", answer: str = ""
) -> dict[str, Any]:
    """What the top result claims, judged. Empty when it cannot be read.

    A search that returns text and never asks what it means leaves a blob no
    later turn can say anything about.
    """
    rows = [row for row in (results or []) if isinstance(row, dict)]
    if not rows:
        return {}
    top = rows[0]
    try:
        from core.knowledge.source_comprehension import comprehend_source

        record = comprehend_source(
            url=str(top.get("url") or top.get("link") or ""),
            title=str(top.get("title") or query or ""),
            text=str(top.get("snippet") or top.get("content") or answer or ""),
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return {}
    return {"comprehension": record.to_dict()} if record.understood else {}


class EnhancedWebSearchSkill(BaseSkill):
    """Hybrid live web search with retrieval, synthesis, and retention."""

    name = "web_search"
    description = (
        "Search the internet for current information, research a topic across multiple pages, "
        "synthesize an evidence-grounded answer, and retain what was learned when appropriate."
    )
    input_model = WebSearchInput
    # Deep research is a multi-page pipeline, not a single fetch.
    #
    # 60s covered a plain search and starved the thing this skill exists for:
    # live 2026-07-28, "find 3 recent articles about orcas, read them, and
    # write a synthesis" died on "Task web_search timed out" at 60,046ms, and
    # the desktop task then reported "research returned 0 usable source(s)".
    # Every page that HAD been fetched was discarded with it.
    #
    # Query expansion, several page fetches, reranking and synthesis do not
    # fit in a minute on a local 32B. The budget matches the work; the deep
    # pipeline still bounds itself by MAX_RESEARCH_LOOPS, and every individual
    # fetch already fails fast and independently.
    timeout_seconds = 180.0
    metabolic_cost = 2

    def __init__(self):
        super().__init__()
        self.pipeline = ResearchSearchPipeline()
        self.browser = _DormantBrowser()

    def _normalize_deep_research_result(self, query: str, result: dict[str, Any]) -> dict[str, Any]:
        sources = list(result.get("sources") or [])
        citations = []
        evidence = []
        for item in sources[:8]:
            url = str(item.get("url") or item.get("uri") or "").strip()
            title = str(item.get("title") or item.get("name") or url or "").strip()
            if not url:
                continue
            citations.append({"title": title, "url": url})
            evidence.append(
                {
                    "title": title,
                    "url": url,
                    "text": str(item.get("text") or item.get("snippet") or "").strip(),
                    "score": float(item.get("score", 0.0) or 0.0),
                    "evidence_kind": str(item.get("evidence_kind") or ""),
                    "fetched": bool(item.get("fetched")),
                    "fetched_at": item.get("fetched_at"),
                    "document_chars": item.get("document_chars"),
                    "document_sha256": str(item.get("document_sha256") or ""),
                    "published_at": str(item.get("published_at") or ""),
                }
            )

        answer = str(result.get("answer") or "").strip()
        summary = answer or str(result.get("summary") or "").strip()
        normalized = {
            "ok": True,
            "query": query,
            "answer": answer,
            "summary": summary,
            "facts": list(result.get("facts") or []),
            "confidence": float(result.get("confidence", 0.82) or 0.82),
            "sources": citations,
            "citations": citations,
            "source": citations[0]["url"] if citations else "",
            "mode": "deep",
            "count": len(citations),
            "chunks": evidence,
            "content": answer,
        }
        normalized["result"] = normalized["answer"] or normalized["content"] or ""
        normalized["message"] = self.pipeline._format_message(query, normalized)
        return normalized

    @staticmethod
    def _finalize_result(query: str, result: dict[str, Any]) -> dict[str, Any]:
        """Return one evidence contract for every successful search lane.

        Live search, retained artifacts, snippet-only retrieval, and the local
        reference corpus all produce useful but materially different evidence.
        Capability verification consumes ``sources``; no branch may therefore
        return before its native evidence has been projected onto that field.
        The projection preserves provenance rather than upgrading an offline
        snapshot or search snippet into a page that was fetched and read.
        """
        finalized = dict(result or {})
        finalized.setdefault("query", query)
        finalized.setdefault(
            "summary",
            finalized.get("answer") or finalized.get("message") or "",
        )
        if not finalized.get("ok"):
            return finalized

        provenance = str(finalized.get("provenance") or "").strip()
        native_sources = (
            finalized.get("sources")
            or finalized.get("citations")
            or finalized.get("chunks")
            or finalized.get("results")
            or []
        )
        if isinstance(native_sources, dict):
            native_sources = [native_sources]
        elif not isinstance(native_sources, (list, tuple)):
            native_sources = []

        sources: list[dict[str, Any]] = []
        for item in native_sources:
            if not isinstance(item, dict):
                continue
            source = dict(item)
            if provenance == "local_corpus" or source.get("provenance") == "local_corpus":
                source["provenance"] = "local_corpus"
                source.setdefault("evidence_kind", "offline_reference_snapshot")
                source.setdefault("fetched", False)
            sources.append(source)

        source_ref = str(finalized.get("source") or "").strip()
        if not sources and source_ref:
            if source_ref.startswith(("http://", "https://")):
                sources.append({"url": source_ref, "provenance": provenance or "live_web"})
            else:
                sources.append({"source": source_ref, "provenance": provenance or "declared"})

        finalized["sources"] = sources
        finalized.setdefault("count", len(sources))
        if sources:
            criteria_results = finalized.get("criteria_results")
            if not isinstance(criteria_results, dict):
                criteria_results = {}
            criteria_results["sources gathered"] = True
            finalized["criteria_results"] = criteria_results
        return finalized

    async def execute(self, params: Any, context: dict[str, Any]) -> dict[str, Any]:
        # Who asked? Curiosity researching on its own is a feature and stays
        # one — it simply must not escalate onto the foreground lane, because
        # the person at the keyboard is the one actually waiting.
        _ctx = dict(context or {})
        _origin = str(
            _ctx.get("authority_origin") or _ctx.get("origin") or _ctx.get("source") or ""
        ).strip().lower()
        _requested_by_user = bool(
            _ctx.get("foreground_request")
            or _ctx.get("user_facing")
            or _origin in {
                "user", "desktop_ui", "voice", "chat", "web_interlocutor",
                "desktop_chat", "admin",
            }
        )
        if isinstance(params, dict):
            query = params.get("query") or params.get("q", "")
            deep = bool(params.get("deep", False))
            num_results = int(params.get("num_results", 5))
            retain = params.get("retain")
            force_refresh = bool(params.get("force_refresh", False))
        elif isinstance(params, WebSearchInput):
            query = params.query
            deep = params.deep
            num_results = params.num_results
            retain = params.retain
            force_refresh = params.force_refresh
        else:
            query = str(params)
            deep = False
            num_results = 5
            retain = None
            force_refresh = False

        query = str(query or "").strip()
        if not query:
            return {"ok": False, "error": "No search query provided."}

        source_reading = query_requires_source_reading(query)
        effective_deep = bool(deep or source_reading)

        # Ask the local corpus BEFORE the network when the question is not
        # time-sensitive.
        #
        # LIVE, 2026-08-10. Asked for a detail about Michael T. Wright's
        # Antikythera planetarium model, she correctly decided to look it up and
        # spent 23,145ms on a web search. The same fact is in the local corpus —
        # 7,189,653 Wikipedia pages — which answers that class of query in
        # 29-77ms. The corpus was already wired here, but only as a DEGRADED
        # fallback for when the web is unreachable, so the fast, private, offline
        # copy was consulted only after the slow path had already failed.
        #
        # For a dated-snapshot question that ordering is backwards on every
        # axis: latency, privacy (no egress at all), and offline capability. The
        # web stays first for anything the snapshot cannot know, which is what
        # freshness_window_for_query already decides for the research pipeline —
        # reused here rather than restated.
        if not force_refresh and not source_reading:
            local_first = self._local_corpus_first(query, num_results)
            if local_first is not None:
                return self._finalize_result(query, local_first)

        logger.info(
            "🔍 WebSearch: '%s' (deep=%s, effective_deep=%s, retain=%s, force_refresh=%s)",
            query[:80],
            deep,
            effective_deep,
            retain,
            force_refresh,
        )
        
        if deep and not source_reading:
            # v2.0: Deep Research LangGraph Pipeline implementation
            try:
                engine = (
                    ServiceContainer.get("cognitive_engine", default=None)
                    or ServiceContainer.get("brain", default=None)
                )
                if engine is None:
                    raise RuntimeError("No cognitive engine available for deep research")
                brain = _DeepResearchBrainAdapter(engine)
                
                # Adapting existing Search pipeline format to standard search_fn format
                async def _search_fn(q: str):
                    # The CALLER's count, not a constant.
                    #
                    # This was a flat 5, so desktop_task asking for 3 sources
                    # still had five fetched and read here — and reading is
                    # the entire cost of a research turn. Every count above
                    # this line now follows the request; this was the one
                    # place it stopped.
                    res = await self.pipeline.search(
                        q,
                        num_results=num_results,
                        deep=False,
                        force_refresh=force_refresh,
                    )
                    results = res.get("results", [])
                    evidence = res.get("chunks") or []
                    # format sources
                    content = res.get("answer") or "\n\n".join(
                        str(item.get("text") or "")
                        for item in evidence
                        if isinstance(item, dict)
                    ) or str([r.get("snippet", "") for r in results])
                    # LIVE, 2026-08-03: a search returned text and nothing
                    # asked what it meant, so the result was a blob a later
                    # turn could say nothing about. The comprehension travels
                    # with the result — what the top source claims, what KIND
                    # of source it is, and the rhetoric worth discounting.
                    return {
                        "ok": True,
                        "content": content,
                        # Fetched evidence is the source surface for a reading
                        # request. Search-result snippets remain a fallback,
                        # explicitly labelled upstream so they cannot later be
                        # mistaken for pages that were opened and read.
                        "sources": evidence or results,
                        **_comprehension_of(results, query=q, answer=content),
                    }
                
                # Curiosity may research all it likes; it just may not
                # take the foreground lane to do it. Only a person's request
                # earns that escalation.
                res = await run_deep_research(
                    query,
                    brain,
                    _search_fn,
                    requested_by_user=_requested_by_user,
                )
                answer = str(res.get("answer") or "").strip()
                if answer:
                    normalized = self._normalize_deep_research_result(query, res)
                    if self.pipeline._should_retain(
                        query,
                        deep=True,
                        retain=retain,
                        context=context or {},
                        result=normalized,
                    ):
                        artifact = self.pipeline._result_to_artifact(
                            normalized,
                            freshness_seconds=freshness_window_for_query(query),
                        )
                        await self.pipeline._retain_artifact(artifact, context or {})
                        normalized["retained"] = True
                        normalized["artifact_id"] = artifact.artifact_id
                    try:
                        from core.advanced_cognition import ExternalEvidenceDeliberator

                        normalized["deliberation_receipts"] = ExternalEvidenceDeliberator.deliberate_many(
                            normalized.get("chunks") or [],
                            source_type="web_search",
                            goal=query,
                        )
                    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                        record_degradation("web_search", exc, severity="warning", action="continued without deep evidence deliberation")
                    return self._finalize_result(query, normalized)
                # "Empty answer" used to be the whole story, which read as
                # "the research found nothing". Usually it found plenty and
                # could not synthesize it — on 2026-07-25, because background
                # inference was queued behind foreground headroom. Those are
                # different failures and only one of them is about the web.
                log = logger.warning if _requested_by_user else logger.info
                log(
                    "Deep Research produced no answer for '%s' (%s; %d source(s) "
                    "gathered); falling back to retrieval pipeline.",
                    query,
                    res.get("synthesis_detail") or res.get("synthesis_status") or "no detail",
                    len(res.get("sources") or []),
                )
            except (ImportError, AttributeError, RuntimeError) as e:
                record_degradation(
                    "web_search",
                    e,
                    severity="warning",
                    action="fell back to retrieval pipeline after deep research failed",
                    extra={"query": query[:240]},
                )
                logger.error("Deep Research failed, falling back to legacy: %s", e)

        # Legacy direct search — a consequential network action, wrapped in
        # a welfare transaction (begin → execute → complete with outcome) so
        # the consequence bus sees real egress effects, not a decorative
        # import (the previous unused ActionExecutor import was exactly that
        # and rightly died to lint).
        from core.being.welfare_transaction import WelfareTransaction

        _tx = WelfareTransaction.begin(
            domain="network_research",
            action=f"web_search:{query[:80]}",
        )
        pipeline_error = ""
        try:
            result = await self.pipeline.search(
                query,
                num_results=num_results,
                deep=effective_deep,
                retain=retain,
                context=context or {},
                force_refresh=force_refresh,
            )
        except (RuntimeError, OSError, ValueError, TypeError, AttributeError, ImportError) as exc:
            # A RAISING pipeline (missing backend, hard network failure) must
            # reach the local-corpus fallback exactly like a returned
            # failure — observed live: the exception path bypassed the
            # fallback and the curiosity loop logged web_search FAILED with
            # 6.5M offline documents sitting available.
            record_degradation(
                "web_search", exc, severity="warning",
                action="pipeline raised; degrading to local corpus",
            )
            offline = self._local_corpus_fallback(query, num_results)
            if offline is not None:
                pipeline_error = str(exc)[:200]
                offline["web_error"] = pipeline_error
                result = offline
            else:
                _tx.complete(outcome="failure", error=str(exc)[:200])
                raise
        _tx.complete(
            outcome="partial" if pipeline_error else ("success" if result.get("ok") else "failure"),
            error=pipeline_error or ("" if result.get("ok") else str(result.get("error") or "")[:200]),
        )
        if not result.get("ok") and force_refresh:
            logger.info(
                "WebSearch forced refresh failed for '%s'; retrying with retained-artifact fallback.",
                query[:80],
            )
            result = await self.pipeline.search(
                query,
                num_results=num_results,
                deep=effective_deep,
                retain=retain,
                context=context or {},
                force_refresh=False,
            )
        if not result.get("ok"):
            # Web unreachable/failed: answer from the local knowledge corpus
            # (6.5M offline reference docs) instead of returning empty-handed.
            # Provenance is explicit — a dated snapshot, never passed off as
            # live web results.
            offline = self._local_corpus_fallback(query, num_results)
            if offline is not None:
                offline["web_error"] = str(
                    result.get("error") or result.get("message") or "web search failed"
                )
                result = offline
        result = self._finalize_result(query, result)
        try:
            from core.advanced_cognition import ExternalEvidenceDeliberator

            artifacts = result.get("chunks") or result.get("results") or []
            if artifacts:
                result["deliberation_receipts"] = ExternalEvidenceDeliberator.deliberate_many(
                    artifacts,
                    source_type="web_search",
                    goal=query,
                )
            elif result.get("summary"):
                result["deliberation_receipts"] = [
                    ExternalEvidenceDeliberator()
                    .deliberate(
                        source_type="web_search",
                        source_ref=result.get("source") or query,
                        content=str(result.get("summary") or ""),
                        goal=query,
                        metadata=result,
                    )
                    .to_dict()
                ]
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("web_search", exc, severity="warning", action="continued without evidence deliberation receipts")
        return result

    @classmethod
    def _local_corpus_first(
        cls, query: str, num_results: int
    ) -> dict[str, Any] | None:
        """Answer from the offline corpus when the web cannot know better.

        Returns None — and the caller proceeds to the network unchanged —
        whenever the corpus is absent, has no match, or the question is
        time-sensitive. Provenance is stated explicitly: a snapshot answer must
        never be presentable as live web evidence, which is a mistake this
        runtime has already made out loud ("I checked live web evidence" over a
        result that was not live).
        """
        if cls._query_wants_current_information(query):
            return None
        answered = cls._local_corpus_fallback(query, num_results)
        if answered is None:
            return None
        answered["offline_fallback"] = False
        answered["offline_preferred"] = True
        answered["summary"] = (
            "Answered from the local offline reference corpus (dated snapshot, "
            "no network used). Ask again with force_refresh for live sources."
        )
        return answered

    @staticmethod
    def _query_wants_current_information(query: str) -> bool:
        """True when a dated snapshot is the wrong source for this question.

        Delegates to the research pipeline's own freshness policy so there is
        one definition of "this needs to be current" in the runtime rather than
        a second list that drifts from the first.
        """
        try:
            from core.search.research_pipeline import (
                _query_is_current,
                freshness_window_for_query,
            )
        except (ImportError, AttributeError):
            # Unknown freshness means the network is the safe answer.
            return True
        try:
            if bool(_query_is_current(query)):
                return True
            # A short retention window is the pipeline saying this decays fast.
            return int(freshness_window_for_query(query)) < 24 * 60 * 60
        except (RuntimeError, TypeError, ValueError):
            return True

    @staticmethod
    def _local_corpus_fallback(query: str, num_results: int) -> dict[str, Any] | None:
        """Degrade to the local knowledge corpus when the web is unreachable.

        Returns None when the corpus is absent/empty or has no match, so the
        caller keeps the original web failure result.
        """
        try:
            from core.knowledge.local_corpus import get_local_corpus_store

            store = get_local_corpus_store()
            # has_documents(), not document_count(): the guard only asks
            # "is there anything here". document_count() is SELECT COUNT(*)
            # over 7.19M rows in a 37GB table — measured at ~6s on this host,
            # which was the entire cost of a corpus consult that then took
            # 29-77ms to actually answer. The O(1) helper already existed and
            # says so in its own docstring; this path had never been switched.
            if not store.has_documents():
                return None
            hits = store.search(query, limit=max(1, min(int(num_results), 10)))
            if not hits:
                return None
            logger.info(
                "WebSearch degraded to local corpus for '%s' (%d offline hits)",
                query[:80],
                len(hits),
            )
            return {
                "ok": True,
                "provenance": "local_corpus",
                "offline_fallback": True,
                "results": [
                    {
                        "title": hit.title,
                        "snippet": hit.snippet,
                        "source": hit.source,
                        "provenance": "local_corpus",
                    }
                    for hit in hits
                ],
                "summary": (
                    "Web search was unavailable; answered from the local "
                    f"offline reference corpus ({len(hits)} matches, dated snapshot)."
                ),
            }
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation(
                "web_search",
                exc,
                severity="debug",
                action="local corpus fallback unavailable",
            )
            return None

    async def on_stop_async(self):
        """Lifecycle hook retained for skill manager shutdown symmetry."""
        return None


class _DormantBrowser:
    """Dormant browser adapter used until a governed browser session is opened."""

    is_active = False

    async def ensure_ready(self):
        return None

    async def browse(self, url: str):
        return False

    async def click(self, text_match: str = "", selector: str = "") -> bool:
        return False

    async def close(self):
        return None

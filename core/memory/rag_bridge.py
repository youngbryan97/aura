
"""core/memory/rag_bridge.py

The Invisible RAG Bridge. Runs parallel to the main cognitive pipeline
to fetch semantic context from the BlackHoleVault/MemoryFacade before inference.

Every retrieval is recorded in recall telemetry (hit rate, candidate and
kept counts, latency) so memory quality is a measured quantity, not a
vibe — this is what makes "memory expands effective model capacity" a
checkable claim.
"""
import asyncio
import logging
import time

from core.container import ServiceContainer
from core.memory.recall_telemetry import get_recall_telemetry
from core.memory.temporal_rag import TimeWeightedRetriever
from core.runtime.errors import record_degradation
from core.runtime.service_access import resolve_orchestrator

logger = logging.getLogger("Aura.RAGBridge")

temporal_retriever = TimeWeightedRetriever(decay_rate=0.012)

async def fetch_deep_context(
    user_query: str,
    threshold_words: int = 4,
    *,
    principal_id: str = "",
    principal_surface: str = "",
) -> str:
    """
    Silently pulls vectorized memories related to the query.
    Bypasses short, meaningless interactions (like "hey") to save compute.
    """
    telemetry = get_recall_telemetry()
    query_words = len(str(user_query or "").split())
    started = time.monotonic()

    def _record(candidates: int, kept: int, skipped_reason: str = "") -> None:
        telemetry.record(
            query_words=query_words,
            candidates=candidates,
            kept=kept,
            latency_ms=(time.monotonic() - started) * 1000.0,
            skipped_reason=skipped_reason,
        )

    if not user_query or query_words < threshold_words:
        _record(0, 0, skipped_reason="query_below_threshold")
        return ""

    # Pull the MemoryFacade (unified gateway)
    memory_facade = ServiceContainer.get("memory_facade", default=None)
    if not memory_facade:
        logger.debug("RAG Bridge: MemoryFacade not found.")
        _record(0, 0, skipped_reason="memory_facade_unavailable")
        return ""

    try:
        # 1. Get raw, flat vector results. MemoryFacade.search is ASYNC —
        # the old asyncio.to_thread(...) call handed back an unawaited
        # coroutine, so every fetch failed silently (July external review:
        # the bridge had zero working integrations). Await it properly;
        # a sync facade double still works through a thread.
        search = getattr(memory_facade, "search", None)
        if search is None:
            _record(0, 0, skipped_reason="memory_facade_has_no_search")
            return ""
        scope_kwargs = (
            {
                "principal_id": principal_id,
                "principal_surface": principal_surface,
            }
            if principal_id and principal_surface
            else {}
        )
        if asyncio.iscoroutinefunction(search):
            raw_results = await search(query=user_query, limit=10, **scope_kwargs)
        else:
            raw_results = await asyncio.to_thread(
                search,
                query=user_query,
                limit=10,
                **scope_kwargs,
            )

        if not raw_results:
            _record(0, 0)
            return ""

        # 2. [NEW] Apply Temporal Decay Math and format
        # This filters out stale memories and applies seasonal tags like [2 months ago]
        temporal_context = await temporal_retriever.rerank_and_format(raw_results, limit=4)

        # Pull any ecosystem context cached by the orchestrator
        ecosystem_context = ""
        try:
            orchestrator = resolve_orchestrator()
            if orchestrator and hasattr(orchestrator, "_current_ecosystem_context"):
                ecosystem_context = orchestrator._current_ecosystem_context
        except (ImportError, AttributeError, RuntimeError) as _e:
            record_degradation('rag_bridge', _e)
            logger.debug('Ignored Exception in rag_bridge.py: %s', _e)

        final_context = temporal_context
        if ecosystem_context:
            final_context = f"{ecosystem_context}\n{final_context}"

        kept = len([line for line in str(temporal_context or "").splitlines() if line.strip()])
        if final_context.strip():
            _record(len(raw_results), max(1, kept))
            return final_context
        _record(len(raw_results), 0)
        return ""

    except (ImportError, AttributeError, RuntimeError) as e:
        record_degradation('rag_bridge', e)
        logger.debug("Temporal RAG Bridge failed: %s", e)
        _record(0, 0, skipped_reason=f"error:{type(e).__name__}")
        return ""

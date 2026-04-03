"""core/memory/rag_bridge.py

The Invisible RAG Bridge. Runs parallel to the main cognitive pipeline
to fetch semantic context from the BlackHoleVault/MemoryFacade before inference.
"""
import asyncio
import logging
from typing import Optional
from core.container import ServiceContainer
from core.memory.temporal_rag import TimeWeightedRetriever

logger = logging.getLogger("Aura.RAGBridge")

temporal_retriever = TimeWeightedRetriever(decay_rate=0.012)

async def fetch_deep_context(user_query: str, threshold_words: int = 4) -> str:
    """
    Silently pulls vectorized memories related to the query.
    Bypasses short, meaningless interactions (like "hey") to save compute.
    """
    if not user_query or len(user_query.split()) < threshold_words:
        return ""

    # Pull the MemoryFacade (unified gateway)
    memory_facade = ServiceContainer.get("memory_facade", default=None)
    if not memory_facade:
        logger.debug("RAG Bridge: MemoryFacade not found.")
        return ""

    try:
        # 1. Get raw, flat vector results
        raw_results = await asyncio.to_thread(
            memory_facade.search, 
            query=user_query, 
            limit=10  # Pull a wider net initially
        )

        if not raw_results:
            return ""

        # 2. [NEW] Apply Temporal Decay Math and format
        # This filters out stale memories and applies seasonal tags like [2 months ago]
        temporal_context = await temporal_retriever.rerank_and_format(raw_results, limit=4)
        
        # Pull any ecosystem context cached by the orchestrator
        ecosystem_context = ""
        try:
            from core.container import ServiceContainer
            orchestrator = ServiceContainer.get("orchestrator", default=None)
            if orchestrator and hasattr(orchestrator, "_current_ecosystem_context"):
                ecosystem_context = orchestrator._current_ecosystem_context
        except Exception as _e:
            logger.debug('Ignored Exception in rag_bridge.py: %s', _e)

        final_context = temporal_context
        if ecosystem_context:
            final_context = f"{ecosystem_context}\n{final_context}"
            
        if final_context.strip():
            return final_context
        return ""
            
    except Exception as e:
        logger.debug("Temporal RAG Bridge failed: %s", e)
        return ""

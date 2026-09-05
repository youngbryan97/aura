"""core/memory/ingestion_loop.py
==============================
Background ingestion service that queries web actuators based on curiosity and active goals.
Chunks retrieved content and writes it into semantic memory.
"""

import asyncio
import inspect
import logging

from core.actuators.actuator_registry import get_actuator_registry
from core.container import ServiceContainer
from core.memory import embedding_model
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.IngestionLoop")


class IngestionLoop:
    """Background service that drives Aura's external information curiosity."""

    def __init__(self, interval_s: float = 300.0):
        self.interval_s = interval_s
        self.running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> bool:
        if self.running:
            return True
        self.running = True
        
        # Avoid blocking boot sequence, launch in background
        from core.utils.task_tracker import get_task_tracker
        self._task = get_task_tracker().create_task(
            self._run(),
            name="AuraIngestionLoop"
        )
        logger.info("IngestionLoop background service ONLINE.")
        return True

    async def stop(self) -> bool:
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError as _exc:
                logger.debug("Suppressed %s in core.memory.ingestion_loop: %s", type(_exc).__name__, _exc)
            self._task = None
        logger.info("IngestionLoop background service SHUTDOWN.")
        return True

    async def _run(self):
        while self.running:
            try:
                await asyncio.sleep(self.interval_s)
                await self._ingest_cycle()
            except asyncio.CancelledError:
                raise
            except (RuntimeError, OSError, AttributeError, TypeError, ValueError) as e:
                record_degradation("ingestion_loop", e, severity="warning", action="skipped failed ingestion cycle")
                logger.error("Error in ingestion cycle: %s", e)

    async def _ingest_cycle(self):
        # 1. Fetch curiosity metric from LiquidSubstrate
        substrate = ServiceContainer.get("liquid_substrate", default=None)
        curiosity = 0.5
        if substrate:
            if hasattr(substrate, "current"):
                curiosity = substrate.current.curiosity
            elif hasattr(substrate, "x") and hasattr(substrate, "idx_curiosity"):
                curiosity = substrate.x[substrate.idx_curiosity]

        # 2. Get active goals
        goal_engine = ServiceContainer.get("goal_engine", default=None)
        goals = []
        if goal_engine and hasattr(goal_engine, "get_goals"):
            try:
                all_goals = await asyncio.to_thread(goal_engine.get_goals)
                goals = [g for g in all_goals if g.get("status") == "in_progress"]
            except (RuntimeError, AttributeError, TypeError, ValueError) as ex:
                record_degradation("ingestion_loop", ex, severity="warning", action="continued without goal-driven ingestion query")
                logger.debug("Failed to fetch goals: %s", ex)

        # We trigger ingestion if curiosity is high (>= 0.6) or we have active goals
        if curiosity < 0.6 and not goals:
            return

        # Determine a search query based on active goals or a default curious topic
        query = "Aura artificial general intelligence breakthroughs"
        if goals:
            # Pick first goal's objective
            first_goal = goals[0]
            query = first_goal.get("objective") or query
        elif curiosity >= 0.8:
            query = "advanced neuro-symbolic cognitive architecture upgrades"

        logger.info("⚡ [INGESTION] Triggered by curiosity=%.2f. Target query: '%s'", curiosity, query)

        # 3. Call web search actuator
        registry = get_actuator_registry()
        if not registry.get_actuator("web_search") or not registry.get_actuator("web_fetch"):
            logger.warning("Web actuators not registered. Skipping ingestion step.")
            return

        search_res = await asyncio.to_thread(
            registry.execute_action,
            "web_search",
            {"query": query, "num_results": 3},
            context={
                "source": "ingestion_loop",
                "priority": 0.45 + 0.3 * min(1.0, max(0.0, float(curiosity))),
                "objective": query,
                "autonomous": True,
            },
        )

        if not search_res.success:
            logger.warning("Web search failed for ingestion: %s", search_res.message)
            return

        results = search_res.updates.get("search_results", {}).get("results", [])
        if not results:
            # Check other possible update structures
            results = search_res.updates.get("search_results", {}).get("hits", [])
        
        if not results:
            logger.info("No search results returned for query: %s", query)
            return

        # Limit to top 2 URLs to avoid spamming
        urls = []
        for res in results[:2]:
            url = None
            if isinstance(res, dict):
                url = res.get("url") or res.get("link")
            elif isinstance(res, str) and res.startswith("http"):
                url = res
            if url:
                urls.append(url)

        memory_facade = ServiceContainer.get("memory_facade", default=None)
        if not memory_facade:
            logger.warning("MemoryFacade not available to record ingestion chunks.")
            return

        for url in urls:
            logger.info("⚡ [INGESTION] Fetching target url: %s", url)
            fetch_res = await asyncio.to_thread(
                registry.execute_action,
                "web_fetch",
                {"url": url},
                context={
                    "source": "ingestion_loop",
                    "priority": 0.45 + 0.3 * min(1.0, max(0.0, float(curiosity))),
                    "objective": query,
                    "autonomous": True,
                },
            )
            
            if not fetch_res.success:
                continue

            # Extracts raw text content
            content = fetch_res.updates.get("fetch_results", {}).get("content", "")
            if not content:
                content = fetch_res.updates.get("fetch_results", {}).get("text", "")
            
            if not content or len(content.strip()) < 100:
                continue

            # 4. Chunk content and ingest
            # No explicit size: one page is normally one vector now. The old
            # 800/100 split shredded every page, and chunks[:5] then threw
            # away everything past ~4,000 words of it.
            chunks = self._chunk_text(content)
            for i, chunk in enumerate(chunks[:5]): # Ingest up to 5 chunks to save memory space
                maybe_write = memory_facade.add_memory(
                    text=f"Source URL: {url}\n\n{chunk}",
                    metadata={
                        "source": "web_ingestion",
                        "provenance_source": "web_ingestion",
                        "url": url,
                        "chunk_index": i,
                        "importance": float(0.4 + 0.4 * curiosity)
                    }
                )
                if inspect.isawaitable(maybe_write):
                    await maybe_write
            logger.info("Successfully ingested %d chunks from %s", min(5, len(chunks)), url)

    def _chunk_text(
        self,
        text: str,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> list[str]:
        """Split ingested text for embedding.

        With no explicit size, the split is DERIVED from the encoder's
        declared window (core/memory/embedding_model.py) — which now means a
        whole page is normally one chunk. The previous fixed 800/100 split
        was sized against a 256-token window and cut every ingested page into
        fragments whose facts could never co-occur in a single vector.

        An explicit ``chunk_size`` is still honoured: a caller that wants
        fine-grained passages has a real reason to, and curation is not
        embedding. It is clamped to what the encoder can actually read, so an
        explicit request can never reintroduce silent truncation.
        """
        if chunk_size is None:
            return embedding_model.chunk_for_embedding(text)

        ceiling = embedding_model.max_chunk_words()
        chunk_size = max(1, min(int(chunk_size), ceiling))
        overlap = 0 if overlap is None else max(0, int(overlap))
        if overlap >= chunk_size:
            overlap = chunk_size - 1

        words = text.split()
        chunks: list[str] = []
        step = chunk_size - overlap
        for i in range(0, len(words), step):
            chunks.append(" ".join(words[i:i + chunk_size]))
            if i + chunk_size >= len(words):
                break
        return chunks

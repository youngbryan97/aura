"""core/memory/knowledge_curator.py
=================================
Performs scheduled memory consolidation, deduplication, purging of low-relevance
outdated information, and promotion of frequently accessed memories to core knowledge.
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict

from core.container import ServiceContainer
from core.memory.semantic_defrag import SemanticDefragmenter
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.KnowledgeCurator")


class KnowledgeCurator:
    """Service to schedule, curate, and optimize long-term memory structures."""

    def __init__(self, interval_s: float = 600.0):
        self.interval_s = interval_s
        self.running = False
        self._task: asyncio.Task | None = None
        self._defragmenter = SemanticDefragmenter()

    async def start(self) -> bool:
        if self.running:
            return True
        self.running = True
        
        from core.utils.task_tracker import get_task_tracker
        self._task = get_task_tracker().create_task(
            self._run(),
            name="AuraKnowledgeCurator"
        )
        logger.info("KnowledgeCurator service ONLINE.")
        return True

    async def stop(self) -> bool:
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError as _exc:
                logger.debug("Suppressed %s in core.memory.knowledge_curator: %s", type(_exc).__name__, _exc)
            self._task = None
        logger.info("KnowledgeCurator service SHUTDOWN.")
        return True

    async def _run(self):
        while self.running:
            try:
                await asyncio.sleep(self.interval_s)
                await self.consolidate_and_curate()
            except asyncio.CancelledError:
                raise
            except (RuntimeError, OSError, AttributeError, TypeError, ValueError) as e:
                record_degradation("knowledge_curator", e, severity="warning", action="skipped failed curation cycle")
                logger.error("Error in knowledge curation cycle: %s", e)

    async def consolidate_and_curate(self) -> Dict[str, Any]:
        """Runs one full cycle of defragmentation, purging, and promoting memory."""
        logger.info("⚡ [CURATOR] Starting scheduled knowledge consolidation...")
        
        # 1. Run standard cluster defragmentation using SemanticDefragmenter
        defrag_stats = {}
        try:
            defrag_stats = await self._defragmenter.run_defrag_cycle()
            logger.info("⚡ [CURATOR] Defragmenter completed: %s", defrag_stats)
        except (RuntimeError, OSError, AttributeError, TypeError, ValueError) as e:
            record_degradation("knowledge_curator", e, severity="warning", action="continued curation without defrag results")
            logger.error("⚡ [CURATOR] Defragmenter failed: %s", e)

        # 2. Perform purging and promotion
        purged = 0
        purge_candidates = 0
        promoted = 0
        allow_delete = os.getenv("AURA_CURATOR_ALLOW_DELETE", "0").strip().lower() in {"1", "true", "yes", "on"}
        
        memory = ServiceContainer.get("vector_memory", default=None)
        if memory and not getattr(memory, "_fallback_mode", False):
            try:
                collection = getattr(memory, "_collection", memory)
                get = getattr(collection, "get", None)
                delete = getattr(collection, "delete", None)
                update = getattr(collection, "update", None)
                
                if get and callable(get):
                    results = get(include=["documents", "metadatas"], limit=100)
                    if isinstance(results, dict):
                        ids = results.get("ids", [])
                        metas = results.get("metadatas", [])
                        now = time.time()
                        to_delete = []
                        
                        for idx, memory_id in enumerate(ids):
                            metadata = metas[idx] if idx < len(metas) and isinstance(metas[idx], dict) else {}
                            timestamp = metadata.get("timestamp") or metadata.get("created") or now
                            age_s = now - timestamp
                            importance = metadata.get("importance") or 0.5
                            
                            # A: Outdated/Low Relevance Purge Criteria
                            # 3 days old and very low importance
                            if age_s > 259200 and importance < 0.15:
                                to_delete.append(memory_id)
                                purge_candidates += 1
                                continue
                                
                            # B: Promotion to Core Knowledge Criteria
                            # High importance or frequently accessed/rehearsed
                            is_consolidated = metadata.get("type") == "consolidated_concept"
                            if (importance >= 0.85 or is_consolidated) and metadata.get("category") != "core_knowledge":
                                if update and callable(update):
                                    metadata["category"] = "core_knowledge"
                                    metadata["promoted_at"] = now
                                    try:
                                        update(ids=[memory_id], metadatas=[metadata])
                                        promoted += 1
                                    except (RuntimeError, OSError, AttributeError, TypeError, ValueError) as ex:
                                        record_degradation("knowledge_curator", ex, severity="warning", action="kept memory unpromoted after update failure")
                                        logger.debug("Failed to promote memory %s: %s", memory_id, ex)
                        
                        if to_delete:
                            if allow_delete and delete and callable(delete):
                                try:
                                    delete(ids=to_delete)
                                    purged += len(to_delete)
                                    logger.info("⚡ [CURATOR] Purged %d low-relevance outdated memories.", len(to_delete))
                                except (RuntimeError, OSError, AttributeError, TypeError, ValueError) as ex:
                                    record_degradation("knowledge_curator", ex, severity="warning", action="kept stale memories after delete failure")
                                    logger.error("⚡ [CURATOR] Failed to delete purged memories: %s", ex)
                            elif update and callable(update):
                                try:
                                    purge_metadata = []
                                    for memory_id in to_delete:
                                        idx = ids.index(memory_id)
                                        metadata = dict(metas[idx] if idx < len(metas) and isinstance(metas[idx], dict) else {})
                                        metadata["candidate_for_purge"] = True
                                        metadata["purge_candidate_at"] = now
                                        purge_metadata.append(metadata)
                                    update(ids=to_delete, metadatas=purge_metadata)
                                    logger.info("⚡ [CURATOR] Marked %d stale memories as purge candidates.", len(to_delete))
                                except (RuntimeError, OSError, AttributeError, TypeError, ValueError) as ex:
                                    record_degradation("knowledge_curator", ex, severity="warning", action="kept stale memories after purge-candidate marking failure")
                                    logger.debug("Failed to mark purge candidates: %s", ex)
            except (RuntimeError, OSError, AttributeError, TypeError, ValueError) as e:
                record_degradation("knowledge_curator", e, severity="warning", action="left memory collection unchanged after curation failure")
                logger.error("⚡ [CURATOR] Purging/Promotion loop failed: %s", e)

        return {
            "defrag_stats": defrag_stats,
            "purged": purged,
            "purge_candidates": purge_candidates,
            "promoted": promoted
        }

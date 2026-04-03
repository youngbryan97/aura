"""core/managers/memory_manager.py

Unified facade for Aura's multi-layered memory systems.
Implements pruning, consolidation, and retrieval-confidence gating.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from core.container import ServiceContainer
from core.health.degraded_events import record_degraded_event
from core.utils.exceptions import capture_and_log

logger = logging.getLogger("Aura.MemoryManager")

class MemoryManager:
    """Unified access point for episodic, semantic, and vector memory."""

    def __init__(self, orchestrator=None, sqlite_memory=None, vector_memory=None):
        self.orch = orchestrator
        self.sqlite_memory = sqlite_memory
        self.vector_memory = vector_memory
        self.last_consolidation = time.time()
        self.consolidation_interval = 86400  # 24 hours
        self._mycelium = None
        self._pressure_threshold = 90.0

    def _is_pressure_high(self) -> bool:
        """v29 Hardening: Check for memory pressure."""
        try:
            import psutil
            return psutil.virtual_memory().percent > self._pressure_threshold
        except ImportError:
            return False

    # ── Mycelial Integration ──────────────────────────────
    def _get_mycelium(self):
        if self._mycelium is None:
            try:
                self._mycelium = ServiceContainer.get("mycelial_network", default=None)
            except Exception as e:
                capture_and_log(e, {'module': __name__})
        return self._mycelium

    def _pulse_hypha(self, source: str, target: str, success: bool = True):
        mycelium = self._get_mycelium()
        if mycelium:
            try:
                hypha = mycelium.get_hypha(source, target)
                if hypha:
                    hypha.pulse(success=success)
            except Exception as e:
                capture_and_log(e, {'module': __name__})

    async def _approve_memory_write(self, content: str, importance: float, tags: Optional[List[str]] = None) -> bool:
        constitutional_runtime_live = (
            ServiceContainer.has("executive_core")
            or ServiceContainer.has("aura_kernel")
            or ServiceContainer.has("kernel_interface")
            or bool(getattr(ServiceContainer, "_registration_locked", False))
        )
        try:
            from core.constitution import get_constitutional_core

            approved, reason = await get_constitutional_core(self.orch).approve_memory_write(
                "memory_manager_store",
                str(content or "")[:240],
                source="memory_manager",
                importance=max(0.0, min(1.0, float(importance or 0.0))),
                metadata={"tags": list(tags or [])[:10]},
            )
            if not approved:
                record_degraded_event(
                    "memory_manager",
                    "memory_write_blocked",
                    detail=str(content or "")[:160],
                    severity="warning",
                    classification="background_degraded",
                    context={"reason": reason},
                )
            return approved
        except Exception as exc:
            if constitutional_runtime_live:
                record_degraded_event(
                    "memory_manager",
                    "memory_write_gate_failed",
                    detail=str(content or "")[:160],
                    severity="warning",
                    classification="background_degraded",
                    context={"error": type(exc).__name__},
                    exc=exc,
                )
                return False
            logger.debug("MemoryManager constitutional gate unavailable: %s", exc)
            return True

    async def store(self, content: Any, importance: float = 0.5, tags: List[str] = None):
        """Stores content across appropriate memory layers."""
        # Heartbeat pulse — proves the memory subsystem is alive even if storage fails
        audit = ServiceContainer.get("subsystem_audit", default=None)
        if audit:
            audit.heartbeat("memory")
        try:
            # Phase 37 FIX: Ensure content is a string before storing
            content = str(content)
            if not await self._approve_memory_write(content, importance=importance, tags=tags):
                return
            
            # 1. Episodic (Short-term context)
            episodic = ServiceContainer.get("episodic_memory", default=None)
            if episodic:
                await episodic.add(content, importance=importance)
            
            # 2. Vector (Long-term semantic)
            vector = ServiceContainer.get("vector_memory", default=None)
            if vector and importance > 0.7:  # Only index high-importance items immediately
                await vector.index(content, metadata={"tags": tags or []})
            
            # Pulse mycelial root: memory → cognition
            self._pulse_hypha("memory", "cognition", success=True)
        except Exception as e:
            logger.error("Failed to store memory: %s", e)
            self._pulse_hypha("memory", "cognition", success=False)
            if audit:
                audit.report_failure("memory", str(e))

    async def retrieve(self, query: str, limit: int = 5, min_confidence: float = 0.6) -> List[Any]:
        """Retrieves and filters memories based on confidence/relevance."""
        results = []
        try:
            vector = ServiceContainer.get("vector_memory", default=None)
            if vector:
                # v48: VectorMemory.search is sync, wrap in to_thread
                raw_results = await asyncio.to_thread(vector.search, query, limit=limit)
                # Apply confidence gating
                results = [r for r in raw_results if r.get("score", 0) >= min_confidence]
            self._pulse_hypha("cognition", "memory", success=True)
        except Exception as e:
            logger.error("Failed to retrieve memory: %s", e)
            self._pulse_hypha("cognition", "memory", success=False)
        return results

    def search_similar(self, query: str, limit: int = 5, **kwargs) -> List[Dict]:
        """Sync delegation for legacy components (Theory of Mind, Context Manager)."""
        try:
            vector = ServiceContainer.get("vector_memory", default=None)
            if vector and hasattr(vector, 'search_similar'):
                return vector.search_similar(query, limit=limit, **kwargs)
        except Exception as e:
            logger.error("search_similar delegation failed: %s", e)
        return []

    async def run_maintenance(self):
        """Trigger pruning and consolidation if the interval has passed."""
        now = time.time()
        if now - self.last_consolidation > self.consolidation_interval:
            logger.info("🕒 Initiating scheduled memory consolidation...")
            await self.consolidate_memories()
            self.last_consolidation = now

    async def consolidate_memories(self):
        """Moves episodic memories into long-term storage and prunes low-importance data."""
        try:
            # Integration with ContextPruner if available
            pruner = ServiceContainer.get("context_pruner", default=None)
            if pruner:
                await pruner.prune_stale_context()
            
            # Summarize episodic bursts
            episodic = ServiceContainer.get("episodic_memory", default=None)
            if episodic:
                await episodic.consolidate()
        except Exception as e:
            logger.error("Memory consolidation failed: %s", e)

    async def log_event(self, event_type: str, content: Any, metadata: Dict[str, Any] = None):
        """Log a significant event to memory storage (used by orchestrator consolidation). - v29 Hardening"""
        if self._is_pressure_high():
            logger.warning("Memory pressure high: Dropping event '%s'.", event_type)
            return

        try:
            importance = 0.8 if event_type == "session_consolidation" else 0.5
            tags = [event_type]
            if metadata:
                tags.extend(metadata.get("tags", []))
            await self.store(content, importance=importance, tags=tags)
            logger.info("📝 Event logged: %s (%d chars)", event_type, len(str(content)[:200]))
        except Exception as e:
            logger.error("Failed to log event '%s': %s", event_type, e)

    def get_status(self) -> Dict[str, Any]:
        return {
            "last_consolidation": self.last_consolidation,
            "next_consolidation": self.last_consolidation + self.consolidation_interval,
            "status": "idle" if time.time() - self.last_consolidation < self.consolidation_interval else "maintenance_due"
        }

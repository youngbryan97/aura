"""core/memory/sovereign_pruner.py
Value-weighted memory consolidation for a sovereign identity.

Philosophy:
    Forget the raw experience. Keep what it made you.
    Protect memories that explain current values, even if those values
    have since evolved — they're the archaeological record of the self.
"""
from __future__ import annotations
import asyncio, json, logging, time, uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple
logger = logging.getLogger("Aura.SovereignPruner")

@dataclass
class MemoryRecord:
    id: str
    content: str
    timestamp: float
    source: str
    emotional_weight: float
    identity_relevance: float
    referenced_count: int = 0
    last_referenced: float = 0.0
    derived_insight: Optional[str] = None
    protected: bool = False

class SovereignPruner:
    def __init__(self, orchestrator=None, target_retention: float = 0.3):
        self.orchestrator = orchestrator
        self.target_retention = target_retention
        self._brain = None
        self._prune_lock = asyncio.Lock()
        self._last_prune_at = 0.0
        self._min_prune_interval_s = 20.0
        self._max_consolidations_per_pass = 4

    def _background_should_defer(self) -> bool:
        try:
            from core.container import ServiceContainer

            gate = ServiceContainer.get("inference_gate", default=None)
            if gate and hasattr(gate, "_background_local_deferral_reason"):
                return bool(gate._background_local_deferral_reason(origin="sovereign_pruner"))
        except Exception:
            return False
        return False

    async def prune(self, memories: List[MemoryRecord], current_values: Dict[str, float]) -> Tuple[List[MemoryRecord], List[str]]:
        if not memories:
            return memories, []
        if self._prune_lock.locked():
            return memories, ["Deferred prune: prior pass still running"]
        now = time.time()
        if (now - self._last_prune_at) < self._min_prune_interval_s:
            return memories, ["Deferred prune: cooldown active"]
        if self._background_should_defer():
            return memories, ["Deferred prune: background inference guard active"]
        
        async with self._prune_lock:
            self._last_prune_at = time.time()
            log = []
            scored = sorted([(m, self._score_memory(m, current_values)) for m in memories], key=lambda x: x[1], reverse=True)
            
            target_keep = max(1, int(len(scored) * self.target_retention))
            keep_ids = {mem.id for mem, _ in scored[:target_keep]}
            
            surviving = []
            consolidation_queue: List[MemoryRecord] = []

            for mem, score in scored:
                if mem.protected or mem.id in keep_ids:
                    surviving.append(mem)
                    continue
                
                if not mem.derived_insight:
                    consolidation_queue.append(mem)
                else:
                    mem.content = f"[CONSOLIDATED] {mem.derived_insight}"
                    surviving.append(mem)
                    log.append(f"Compressed: {mem.id[:8]}")

            pending = consolidation_queue[: self._max_consolidations_per_pass]
            deferred = consolidation_queue[self._max_consolidations_per_pass :]
            if deferred:
                surviving.extend(deferred)
                log.append(f"Deferred consolidation for {len(deferred)} memories")

            if pending:
                logger.info("💾 [PRUNER] Consolidating %d memory task(s) this pass (deferred=%d).", len(pending), len(deferred))
                results = await asyncio.gather(
                    *(self._consolidate(mem) for mem in pending),
                    return_exceptions=True,
                )
                
                for mem, result in zip(pending, results):
                    if isinstance(result, Exception):
                        logger.warning("⚠️ [PRUNER] Consolidation failed for %s: %s", mem.id[:8], result)
                        surviving.append(mem)
                        log.append(f"Failed: {mem.id[:8]}")
                    elif result:
                        mem.derived_insight = result
                        mem.content = f"[CONSOLIDATED] {result}"
                        surviving.append(mem)
                        log.append(f"Consolidated: {mem.id[:8]} → '{result[:60]}'")
                    else:
                        log.append(f"Pruned (no insight): {mem.id[:8]}")
            
            surviving = self._protect_contradictions(surviving, current_values)
            return surviving, log

    def _score_memory(self, mem: MemoryRecord, current_values: Dict[str, float]) -> float:
        age_days = (time.time() - mem.timestamp) / 86400
        recency = max(0.0, 1.0 - (age_days / 90))
        score = recency * 0.15 + mem.emotional_weight * 0.30 + mem.identity_relevance * 0.35
        score += min(1.0, mem.referenced_count / 10) * 0.15
        for value, importance in current_values.items():
            if value.lower() in mem.content.lower():
                score += importance * 0.05
        return min(1.0, score)

    def _protect_contradictions(self, memories: List[MemoryRecord], current_values: Dict[str, float]) -> List[MemoryRecord]:
        high_value_terms = {k for k, v in current_values.items() if v > 0.7}
        markers = ["i was wrong about", "changed my mind", "used to believe", "no longer think", "realized i was", "reconsidered"]
        for mem in memories:
            cl = mem.content.lower()
            for term in high_value_terms:
                if any(m in cl for m in markers):
                    mem.protected = True; break
        return memories

    async def _consolidate(self, mem: MemoryRecord) -> Optional[str]:
        brain = self._get_brain()
        if not brain: return None
        prompt = f"Distill this memory to its essential insight in one sentence. If it contributed nothing, say 'null'.\n\nMEMORY: {mem.content[:500]}\nSOURCE: {mem.source}\n\nInsight:"
        try:
            # Route memory consolidation through the 7B background lane so
            # housekeeping never steals the 32B conversation brain.
            if hasattr(brain, "think"):
                from core.brain.cognitive_engine import ThinkingMode

                thought = await asyncio.wait_for(
                    brain.think(
                        objective=prompt,
                        mode=ThinkingMode.FAST,
                        origin="sovereign_pruner",
                        is_background=True,
                        max_tokens=80,
                        temperature=0.3,
                    ),
                    timeout=5.0,
                )
                result = (getattr(thought, "content", "") or "").strip()
            else:
                result = (
                    await asyncio.wait_for(
                        brain.generate(
                            prompt,
                            temperature=0.3,
                            max_tokens=80,
                            origin="sovereign_pruner",
                            is_background=True,
                            prefer_tier="tertiary",
                        ),
                        timeout=5.0,
                    )
                ).strip()

            return None if not result or result.lower() == "null" else result
        except (asyncio.TimeoutError, Exception) as e:
            logger.debug("Consolidation for %s timed out or failed: %s", mem.id[:8], e)
            return None

    def _get_brain(self):
        if self.orchestrator: return getattr(self.orchestrator, "cognitive_engine", None)
        try:
            from core.container import ServiceContainer
            return ServiceContainer.get("cognitive_engine", default=None)
        except Exception: return None

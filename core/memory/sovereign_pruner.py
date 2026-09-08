"""core/memory/sovereign_pruner.py
Value-weighted memory consolidation for a sovereign identity.

Philosophy:
    Forget the raw experience. Keep what it made you.
    Protect memories that explain current values, even if those values
    have since evolved — they're the archaeological record of the self.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass

from core.memory.retention_policy import sovereign_pruner_target_retention
from core.runtime.errors import record_degradation
from core.runtime.runtime_settings import get_runtime_setting
from core.runtime.service_registry import get_runtime_service

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
    derived_insight: str | None = None
    protected: bool = False

class SovereignPruner:
    def __init__(self, orchestrator=None, target_retention: float | None = None):
        self.orchestrator = orchestrator
        self.target_retention = (
            sovereign_pruner_target_retention()
            if target_retention is None
            else max(0.0, min(0.98, float(target_retention)))
        )
        self._brain = None
        self._prune_lock = asyncio.Lock()
        self._last_prune_at = 0.0
        self._min_prune_interval_s = 20.0
        self._max_consolidations_per_pass = 4
        self._llm_consolidation_enabled = os.getenv(
            "AURA_PRUNER_LLM_CONSOLIDATION",
            "",
        ).strip().lower() in {"1", "true", "yes", "on"}

    def _background_should_defer(self) -> bool:
        try:
            gate = get_runtime_service("inference_gate", default=None)
            if gate and hasattr(gate, "_background_local_deferral_reason"):
                return bool(gate._background_local_deferral_reason(origin="sovereign_pruner"))
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("sovereign_pruner", exc)
            logger.debug("Inference deferral guard unavailable: %s", exc)
            return False
        return False

    async def prune(self, memories: list[MemoryRecord], current_values: dict[str, float]) -> tuple[list[MemoryRecord], list[str]]:
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
            consolidation_queue: list[MemoryRecord] = []

            for mem, _score in scored:
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
                
                for mem, result in zip(pending, results, strict=True):
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

    def _score_memory(self, mem: MemoryRecord, current_values: dict[str, float]) -> float:
        age_days = (time.time() - mem.timestamp) / 86400
        # Recency horizon = the user's memory.retention_days (default 365): a
        # memory's recency score decays to 0 across that span, so longer retention
        # keeps older memories competitive in the prune ranking. This is a ranking
        # weight only — nothing is hard-deleted purely by age. (docs/SETTINGS_WIRING_AUDIT.md)
        retention_days = max(7, int(get_runtime_setting("memory.retention_days", 365) or 365))
        recency = max(0.0, 1.0 - (age_days / retention_days))
        score = recency * 0.15 + mem.emotional_weight * 0.30 + mem.identity_relevance * 0.35
        score += min(1.0, mem.referenced_count / 10) * 0.15
        for value, importance in current_values.items():
            if value.lower() in mem.content.lower():
                score += importance * 0.05
        return min(1.0, score)

    def _protect_contradictions(self, memories: list[MemoryRecord], current_values: dict[str, float]) -> list[MemoryRecord]:
        high_value_terms = {k for k, v in current_values.items() if v > 0.7}
        markers = ["i was wrong about", "changed my mind", "used to believe", "no longer think", "realized i was", "reconsidered"]
        for mem in memories:
            cl = mem.content.lower()
            for term in high_value_terms:
                if term.lower() in cl and any(m in cl for m in markers):
                    mem.protected = True
                    break
        return memories

    async def _consolidate(self, mem: MemoryRecord) -> str | None:
        from core.runtime.backpressure import primary_inference_active

        if not bool(getattr(self, "_llm_consolidation_enabled", False)):
            return self._heuristic_insight(mem)
        if primary_inference_active():
            # Yield instead of competing with the user's turn OR the mind's own
            # cognition tick for the single 32B worker (and timing out).
            logger.debug("[PRUNER] Yielded consolidation of %s to the primary inference lane.", mem.id[:8])
            return self._heuristic_insight(mem)
        brain = self._get_brain()
        if not brain:
            return None
        prompt = f"Distill this memory to its essential insight in one sentence. If it contributed nothing, say 'null'.\n\nMEMORY: {mem.content[:500]}\nSOURCE: {mem.source}\n\nInsight:"
        try:
            # Route memory consolidation through the 7B background lane so
            # housekeeping never steals the cortex's conversation brain.
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

            from core.runtime.backpressure import clear_backpressure

            clear_backpressure("sovereign_pruner")
            return None if not result or result.lower() == "null" else result
        except TimeoutError as e:
            # A 5s-bounded background consolidation losing the model to the
            # foreground lane is routine yield, not a critical incident.
            from core.runtime.backpressure import record_expected_backpressure

            record_expected_backpressure(
                "sovereign_pruner",
                e,
                action="kept memory unconsolidated; retried next prune pass",
            )
            logger.debug("Consolidation for %s timed out: %s", mem.id[:8], e)
            return None
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation("sovereign_pruner", e)
            logger.debug("Consolidation for %s failed: %s", mem.id[:8], e)
            return self._heuristic_insight(mem)

    def _heuristic_insight(self, mem: MemoryRecord) -> str | None:
        """Cheap consolidation used by live runtime when model budget is reserved.

        This keeps memory metabolism active without letting housekeeping consume
        the foreground Cortex or wedge the event loop. It intentionally extracts
        the most identity/affect-relevant sentence instead of inventing a new
        interpretation.
        """

        text = re.sub(r"\s+", " ", str(mem.content or "")).strip()
        if not text:
            return None
        sentences = [
            part.strip(" -\t\r\n")
            for part in re.split(r"(?<=[.!?])\s+", text)
            if part.strip(" -\t\r\n")
        ]
        if not sentences:
            sentences = [text]
        markers = {
            "learned": 2.0,
            "realized": 2.0,
            "remember": 1.7,
            "prefer": 1.7,
            "choice": 1.7,
            "value": 1.6,
            "trust": 1.5,
            "goal": 1.4,
            "felt": 1.2,
            "failed": 1.0,
            "fixed": 1.0,
            "bryan": 0.7,
            "aura": 0.5,
        }

        def _score(sentence: str) -> tuple[float, int]:
            lower = sentence.lower()
            marker_score = sum(weight for marker, weight in markers.items() if marker in lower)
            length_bonus = min(len(sentence), 220) / 220.0
            source = str(getattr(mem, "source", "") or "")
            source_bonus = 0.25 if source and source.lower() in lower else 0.0
            affect_bonus = 0.35 * float(getattr(mem, "emotional_weight", 0.0) or 0.0)
            identity_bonus = 0.45 * float(getattr(mem, "identity_relevance", 0.0) or 0.0)
            return (marker_score + length_bonus + source_bonus + affect_bonus + identity_bonus, -len(sentence))

        selected = max(sentences[:8], key=_score)
        selected = selected[:240].rstrip()
        if not selected or selected.lower() in {"null", "none"}:
            return None
        return selected

    def _get_brain(self):
        if self.orchestrator:
            return getattr(self.orchestrator, "cognitive_engine", None)
        try:
            return get_runtime_service("cognitive_engine", default=None)
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("sovereign_pruner", exc)
            logger.debug("Cognitive engine lookup failed: %s", exc)
            return None

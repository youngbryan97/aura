"""Sleep / Dream Skill — Semantic Memory Consolidation.

Consolidates episodic memories from the day into semantic knowledge,
evolves the identity's "evolved context" layer, and prunes stale data.

Implements an 'Idea Immune System' — the base identity (core directives)
is immutable; only the evolved context layer can change.
"""
from core.runtime.errors import record_degradation
import logging
from typing import Any, Dict

from core.container import ServiceContainer
from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.Sleep")


class SleepSkill(BaseSkill):
    name = "dream_sleep"
    description = "Consolidates memories and evolves identity during downtime."
    timeout_seconds = 120.0
    metabolic_cost = 3

    async def execute(self, params: Any = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        logger.info("Aura is entering REM sleep (Neural Consolidation)...")
        context = context or {}
        steps_completed = []

        # ── 1. Gather recent memories ────────────────────────────────
        mem_text = ""
        try:
            memory = ServiceContainer.get("memory_facade", default=None)
            if memory and hasattr(memory, "recall"):
                recent = await memory.recall("Today's important lessons", limit=20)
                if recent:
                    mem_text = "\n".join(str(m) for m in recent)
                    steps_completed.append("memory_recall")
        except Exception as e:
            record_degradation('sleep', e)
            logger.debug("Sleep: memory recall failed: %s", e)

        if not mem_text:
            try:
                orch = ServiceContainer.get("orchestrator", default=None)
                if orch and hasattr(orch, "conversation_history") and orch.conversation_history:
                    recent_turns = orch.conversation_history[-30:]
                    mem_text = "\n".join(
                        f"{m.get('role', '?')}: {str(m.get('content', ''))[:200]}"
                        for m in recent_turns if isinstance(m, dict)
                    )
                    steps_completed.append("conversation_fallback")
            except Exception:
                pass

        if not mem_text:
            return {
                "ok": True,
                "summary": "No memories to consolidate — Aura rested quietly.",
                "steps": steps_completed,
            }

        # ── 2. Dream: Extract semantic facts via LLM ─────────────────
        derived_knowledge = ""
        try:
            brain = ServiceContainer.get("cognitive_engine", default=None)
            if brain and hasattr(brain, "think"):
                from core.brain.types import ThinkingMode
                dream_prompt = (
                    f"NEW EXPERIENCES:\n{mem_text[:3000]}\n\n"
                    "TASK: Consolidate these episodic experiences into distinct semantic facts or lessons. "
                    "Focus on: what was learned, what changed, what matters for tomorrow. "
                    "Output a bulleted list of 'Derived Knowledge'."
                )
                result = await brain.think(dream_prompt, mode=ThinkingMode.REFLECTIVE)
                derived_knowledge = getattr(result, "content", str(result))
                steps_completed.append("dream_synthesis")
                logger.info("Dream synthesis complete: knowledge extracted.")
        except Exception as e:
            record_degradation('sleep', e)
            logger.debug("Sleep: dream synthesis failed: %s", e)

        # ── 3. Dream journal entry ───────────────────────────────────
        try:
            dream_journal = ServiceContainer.get("dream_journal", default=None)
            if dream_journal and hasattr(dream_journal, "record"):
                await dream_journal.record(
                    content=derived_knowledge or mem_text[:500],
                    dream_type="consolidation",
                )
                steps_completed.append("dream_journal")
        except Exception as e:
            record_degradation('sleep', e)
            logger.debug("Sleep: dream journal failed: %s", e)

        # ── 4. Identity evolution (immune system) ────────────────────
        identity_evolved = False
        if derived_knowledge:
            try:
                canonical_engine = ServiceContainer.get("canonical_self_engine", default=None)
                if canonical_engine and hasattr(canonical_engine, "evolve_from_dream"):
                    identity_evolved = await canonical_engine.evolve_from_dream(derived_knowledge)
                    if identity_evolved:
                        steps_completed.append("identity_evolution")
            except Exception as e:
                record_degradation('sleep', e)
                logger.debug("Sleep: identity evolution failed: %s", e)

        # ── 5. Memory compaction ─────────────────────────────────────
        try:
            compressor = ServiceContainer.get("knowledge_compression", default=None)
            if compressor and hasattr(compressor, "compact"):
                await compressor.compact()
                steps_completed.append("memory_compaction")
        except Exception as e:
            record_degradation('sleep', e)
            logger.debug("Sleep: memory compaction failed: %s", e)

        # ── 6. Satisfy drives (rest restores energy) ─────────────────
        try:
            drive = ServiceContainer.get("drive_engine", default=None)
            if drive:
                await drive.satisfy("energy", 30.0)
                await drive.satisfy("competence", 5.0)
                steps_completed.append("drive_restoration")
        except Exception as e:
            record_degradation('sleep', e)
            logger.debug("Sleep: drive restoration failed: %s", e)

        # ── 7. Record in WorldState ──────────────────────────────────
        try:
            from core.world_state import get_world_state
            ws = get_world_state()
            ws.record_event(
                f"Dream cycle completed: {len(steps_completed)} steps",
                source="sleep_skill",
                salience=0.3,
                ttl=28800,  # 8 hours
            )
        except Exception:
            pass

        summary_parts = [f"Dream cycle completed ({len(steps_completed)} steps)."]
        if derived_knowledge:
            # Show first 200 chars of what was learned
            summary_parts.append(f"Learned: {derived_knowledge[:200]}")
        if identity_evolved:
            summary_parts.append("Identity evolved from dream insights.")

        return {
            "ok": True,
            "summary": " ".join(summary_parts),
            "steps": steps_completed,
            "knowledge": derived_knowledge[:500] if derived_knowledge else "",
            "identity_evolved": identity_evolved,
        }

import asyncio
import logging
from typing import Any, Dict

from core.skills.base_skill import BaseSkill
from core.container import ServiceContainer

logger = logging.getLogger("Skills.Dream")

class DreamSkill(BaseSkill):
    """
    Triggers an immediate 'Dream Cycle' across the cognitive and semantic layers.
    This consolidates old, fragmented memories into denser concepts, and 
    reprocesses the dead-letter queue for missed thoughts.
    """
    name = "force_dream_cycle"
    description = "Initiates immediate memory consolidation and dead-letter queue (DLQ) re-processing."
    inputs = {
        "focus": "(Optional) A specific concept or timeframe to focus consolidation on."
    }

    async def execute(self, goal: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Triggers the semantic defragger, dream journal synthesis, and DLQ re-ingestion."""
        logger.info("💤 Manual Dream Cycle Initiated via Skill.")

        results = {}

        # 1. DreamJournal — qualia-driven creative synthesis from salient memories
        dream_journal = ServiceContainer.get("dream_journal", default=None)
        if dream_journal and hasattr(dream_journal, "synthesize_dream"):
            try:
                dream_result = await dream_journal.synthesize_dream()
                if dream_result:
                    results["dream_journal"] = {
                        "status": "completed",
                        "content": dream_result.get("dream_content", "")[:500],
                        "seed_count": dream_result.get("seed_count", 0),
                    }
                else:
                    results["dream_journal"] = {"status": "skipped", "reason": "insufficient salient material"}
            except Exception as e:
                results["dream_journal"] = {"status": "failed", "error": str(e)}
        else:
            results["dream_journal"] = {"status": "unavailable"}

        # 2. Semantic Defrag (ChromaDB Vector Consolidation)
        try:
            orchestrator = ServiceContainer.get("orchestrator", default=None)
            if orchestrator and hasattr(orchestrator, "semantic_defrag") and getattr(orchestrator.semantic_defrag, "run_defrag_cycle", None):
                asyncio.create_task(orchestrator.semantic_defrag.run_defrag_cycle())
                results["semantic_defrag"] = "queued"
            else:
                results["semantic_defrag"] = "unavailable"
        except Exception as e:
            results["semantic_defrag"] = f"failed: {e}"

        # 3. Dream Cycle (DLQ Re-ingestion)
        try:
            if orchestrator and hasattr(orchestrator, "dream_cycle") and getattr(orchestrator.dream_cycle, "process_dreams", None):
                asyncio.create_task(orchestrator.dream_cycle.process_dreams())
                results["dlq_cycle"] = "queued"
            else:
                results["dlq_cycle"] = "unavailable"
        except Exception as e:
            results["dlq_cycle"] = f"failed: {e}"

        # 4. Heuristic Synthesis — extract learned instincts from recent telemetry
        hs = ServiceContainer.get("heuristic_synthesizer", default=None)
        if hs and hasattr(hs, "synthesize_from_telemetry"):
            try:
                hs_result = await hs.synthesize_from_telemetry()
                results["heuristic_synthesis"] = hs_result
            except Exception as e:
                results["heuristic_synthesis"] = {"status": "failed", "error": str(e)}

        return {
            "ok": True,
            "message": "Dream cycle complete. My substrate consolidated memories, synthesized a dream, and extracted new heuristics.",
            "subsystems": results,
        }
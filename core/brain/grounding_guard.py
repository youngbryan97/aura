import logging
import time
import asyncio
from collections import deque
from typing import Any, Dict, List, Optional
from core.config import config

logger = logging.getLogger("Cognition.GroundingGuard")

class GroundingGuard:
    """The 'Cynic' of the cognitive system.
    Validates LLM self-evaluations against physical reality (tool outputs).
    """
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.history = deque(maxlen=200)

    async def validate_eval(self, objective: str, eval_score: float, actual_result: Dict[str, Any]) -> float:
        """Adjusts the self-evaluation score based on evidence.
        Prevents 'Hallucination Loops' where the LLM thinks it succeeded but 
        the tool logs show a failure.
        """
        refined_score = eval_score
        
        # 1. Status Mismatch (The biggest hallucination signal)
        if eval_score > 0.7 and (actual_result.get("ok") is False or "error" in actual_result):
            logger.warning("🚨 HALLUCINATION DETECTED: LLM claims success (%s) but tool reported failure.", eval_score)
            refined_score = 0.2
        
        # 2. Evidence Requirement
        if eval_score > 0.5 and not actual_result.get("result") and not actual_result.get("ok"):
             logger.info("📉 Grounding: Zero evidence for mid-tier success. Penalizing.")
             refined_score -= 0.2

        # 3. Resource Awareness
        snapshot = self.orchestrator.metabolic_monitor.get_current_metabolism() if hasattr(self.orchestrator, 'metabolic_monitor') else None
        if snapshot and snapshot.health_score < 0.4 and eval_score > 0.8:
            logger.info("📉 Grounding: High score in low-health state. suspicious. Buffering.")
            refined_score *= 0.9

        self.history.append({
            "objective": objective,
            "original": eval_score,
            "grounded": refined_score,
            "mismatch": abs(eval_score - refined_score) > 0.1
        })
        
        return max(0.0, min(1.0, refined_score))

    def get_grounding_stats(self):
        return {
            "total_checks": len(self.history),
            "hallucinations_blocked": sum(1 for h in self.history if h["mismatch"])
        }
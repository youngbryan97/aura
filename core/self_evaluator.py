import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .agency_core import RobustOrchestrator

logger = logging.getLogger("Cognition.SelfEvaluator")

class SelfEvaluator:
    """Metacognitive module for post-action reflection.
    """
    def __init__(self, orchestrator: Optional["RobustOrchestrator"] = None):
        self.orchestrator = orchestrator
        self.grounding_guard = None
        self.last_grounding_assessment = None
        if orchestrator:
            from .brain.grounding_guard import GroundingGuard
            self.grounding_guard = GroundingGuard(orchestrator)
    
    async def evaluate_result(self, objective: str, result: Dict[str, Any]) -> float:
        """Judge the success of an action using grounding verification.
        Returns a score 0.0 - 1.0.
        """
        # Heuristic evaluation
        score = 0.5
        
        if result.get("status") == "success" or result.get("ok"):
            score += 0.3
            
        if result.get("error"):
            score -= 0.4
            
        
        logger.info("Self-Evaluation (Raw): %.2f for '%s'", score, objective)
        
        # Grounding Pass (Phase 22.1 Resilience)
        if self.grounding_guard:
            self.last_grounding_assessment = self.grounding_guard.assess(objective, score, result)
            score = await self.grounding_guard.validate_eval(objective, score, result)
            logger.info("Self-Evaluation (Grounded): %.2f", score)

        return max(0.0, min(1.0, score))

    def suggested_correction(self, score: float, context: Dict[str, Any]) -> str:
        assessment = self.last_grounding_assessment
        if assessment is not None and getattr(assessment, "needs_replan", False):
            intent = assessment.correction_intent or "observe"
            reason = assessment.explanation or assessment.failure_reason or "ungrounded outcome"
            return f"Grounding mismatch. Replan with {intent}: {reason}."
        if score < 0.4:
            return "Strategy failed. Requesting user guidance or switching to robust mode."
        elif score < 0.7:
             return "Success, but suboptimal. Considerations for refinement: Check latency."
        return "Optimal execution."



# Singleton instances are usually created during container registration now,
# but keeping this for legacy compatibility if needed.
self_evaluator = None 

def create_self_evaluator(orchestrator):
    return SelfEvaluator(orchestrator)

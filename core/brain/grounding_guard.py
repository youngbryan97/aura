import logging
import time
import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from core.config import config

logger = logging.getLogger("Cognition.GroundingGuard")


@dataclass
class GroundingAssessment:
    original_score: float
    grounded_score: float
    mismatch: bool
    evidence_present: bool
    failure_reason: str = ""
    correction_intent: str = ""
    correction_parameters: dict[str, Any] = field(default_factory=dict)
    explanation: str = ""

    @property
    def needs_replan(self) -> bool:
        return self.mismatch or bool(self.correction_intent)


class GroundingGuard:
    """The 'Cynic' of the cognitive system.
    Validates LLM self-evaluations against physical reality (tool outputs).
    """
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.history = deque(maxlen=200)

    def assess(self, objective: str, eval_score: float, actual_result: Dict[str, Any]) -> GroundingAssessment:
        refined_score = float(eval_score)
        result = actual_result if isinstance(actual_result, dict) else {"result": actual_result}
        ok = bool(result.get("ok") or result.get("status") == "success")
        error = str(result.get("error") or result.get("exception") or "")
        evidence_present = bool(
            result.get("result")
            or result.get("raw_result")
            or result.get("observation_after")
            or result.get("stdout")
            or result.get("data")
            or ok
        )

        failure_reason = ""
        correction_intent = ""
        correction_parameters: dict[str, Any] = {}
        explanation = ""

        if eval_score > 0.7 and (result.get("ok") is False or error):
            logger.warning("🚨 HALLUCINATION DETECTED: LLM claims success (%s) but tool reported failure.", eval_score)
            refined_score = min(refined_score, 0.2)
            failure_reason = error or "tool_reported_failure"
            correction_intent, correction_parameters = self._correction_for_error(failure_reason)
            explanation = "tool failure outranks self-evaluation"

        if eval_score > 0.5 and not evidence_present:
            logger.info("📉 Grounding: Zero evidence for mid-tier success. Penalizing and requesting observation.")
            refined_score = min(refined_score, eval_score - 0.2)
            correction_intent = correction_intent or "observe"
            explanation = explanation or "success claim lacked external evidence"

        snapshot = self.orchestrator.metabolic_monitor.get_current_metabolism() if hasattr(self.orchestrator, 'metabolic_monitor') else None
        if snapshot and snapshot.health_score < 0.4 and eval_score > 0.8:
            logger.info("📉 Grounding: High score in low-health state. suspicious. Buffering.")
            refined_score *= 0.9
            explanation = explanation or "high-confidence claim under low runtime health"

        refined_score = max(0.0, min(1.0, refined_score))
        mismatch = abs(eval_score - refined_score) > 0.1
        return GroundingAssessment(
            original_score=float(eval_score),
            grounded_score=refined_score,
            mismatch=mismatch,
            evidence_present=evidence_present,
            failure_reason=failure_reason,
            correction_intent=correction_intent,
            correction_parameters=correction_parameters,
            explanation=explanation,
        )

    async def validate_eval(self, objective: str, eval_score: float, actual_result: Dict[str, Any]) -> float:
        """Adjusts the self-evaluation score based on evidence.
        Prevents 'Hallucination Loops' where the LLM thinks it succeeded but
        the tool logs show a failure.
        """
        assessment = self.assess(objective, eval_score, actual_result)

        self.history.append({
            "objective": objective,
            "original": eval_score,
            "grounded": assessment.grounded_score,
            "mismatch": assessment.mismatch,
            "correction_intent": assessment.correction_intent,
            "failure_reason": assessment.failure_reason,
        })
        
        return assessment.grounded_score

    @staticmethod
    def _correction_for_error(error: str) -> tuple[str, dict[str, Any]]:
        text = str(error or "").lower()
        if "modal" in text or "prompt" in text or "confirmation" in text:
            return "resolve_modal", {"strategy": "safe_default"}
        if "unknown_intent" in text or "unsupported" in text or "precondition" in text:
            return "observe", {"reason": "action_semantics_invalid"}
        if "timeout" in text or "stalled" in text or "busy" in text:
            return "wait", {"reason": "runtime_pressure"}
        if "permission" in text or "authority" in text or "forbidden" in text:
            return "request_authority", {"reason": "governance_required"}
        return "observe", {"reason": "verify_state_before_replan"}

    def correction_action(self, objective: str, eval_score: float, actual_result: Dict[str, Any]) -> Dict[str, Any]:
        assessment = self.assess(objective, eval_score, actual_result)
        return {
            "needs_replan": assessment.needs_replan,
            "intent": assessment.correction_intent,
            "parameters": assessment.correction_parameters,
            "grounded_score": assessment.grounded_score,
            "reason": assessment.explanation or assessment.failure_reason,
        }

    def get_grounding_stats(self):
        return {
            "total_checks": len(self.history),
            "hallucinations_blocked": sum(1 for h in self.history if h["mismatch"]),
            "corrections_suggested": sum(1 for h in self.history if h.get("correction_intent")),
        }

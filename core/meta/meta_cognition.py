import logging
import time
from typing import Any

from core.runtime.service_registry import get_runtime_service

logger = logging.getLogger("Aura.MetaCognition")

class MetaCognition:
    """
    Orchestrates high-level self-reflection and recursive cognitive monitoring.
    Analyzes decision outcomes and identifies reasoning patterns.
    """

    def __init__(self):
        self.error_history: list[dict[str, Any]] = []
        self._last_review_at = 0.0
        logger.info("MetaCognition Loop initialized.")

    async def review_decision(self, decision: str, outcome: str, context: dict[str, Any] | None = None):
        """
        Record and analyze a cognitive decision.
        """
        entry = {
            "decision": decision,
            "outcome": outcome,
            "context": context or {},
            "timestamp": time.time()
        }
        
        if outcome == "failure":
            self.error_history.append(entry)
            logger.warning("Cognitive failure recorded: %s", decision)
            
            # If error threshold met, trigger a self-evolution reflex
            if len(self.error_history) > 5:
                await self._trigger_structural_review()
        else:
            logger.info("Cognitive success recorded: %s", decision)

    async def detect_patterns(self) -> dict[str, Any]:
        """
        Analyze error history for recurring failure modes.
        """
        if not self.error_history:
            return {"status": "stable", "pattern_count": 0}
            
        # Group by decision type or context fragments
        patterns = {}
        for err in self.error_history:
            d = err["decision"]
            patterns[d] = patterns.get(d, 0) + 1
            
        significant = {k: v for k, v in patterns.items() if v >= 2}
        return {
            "status": "degraded" if significant else "stable",
            "significant_patterns": significant,
            "pattern_count": len(significant)
        }

    async def _trigger_structural_review(self):
        """
        Calls the MetaEvolutionEngine to address recurring issues.
        """
        logger.info("🌀 Structural failure threshold met. Initiating Meta-Evolution reflex...")
        mee = get_runtime_service("meta_evolution", default=None)
        if mee:
            mee.queue_optimization(
                target_area="cognitive_patterns",
                context=f"Recurring failures detected: {self.error_history[-5:]}"
            )
            # Clear error history after queuing to avoid rapid re-triggering
            self.error_history = self.error_history[-2:]

    def get_health(self) -> dict[str, Any]:
        return {
            "errors": len(self.error_history),
            "last_review": self._last_review_at,
            "status": "active"
        }

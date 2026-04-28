from __future__ import annotations
from core.runtime.errors import record_degradation

import logging
import asyncio
from typing import Optional, TYPE_CHECKING
from core.kernel.bridge import Phase
from core.state.aura_state import AuraState

if TYPE_CHECKING:
    from core.kernel.aura_kernel import AuraKernel

logger = logging.getLogger("Aura.SelfReview")

class SelfReviewPhase(Phase):
    """
    [ASI Genesis] The Mirror Phase.
    Aura analyzes her own performance, technical debt, and code health.
    """

    def __init__(self, kernel: "AuraKernel"):
        super().__init__(kernel)
        self._last_review_ts = 0.0
        self._review_interval = 600.0 # 10 minutes

    async def execute(self, state: AuraState, objective: Optional[str] = None, **kwargs) -> AuraState:
        """
        Analyze current state for potential self-optimization.
        Does not block user interaction — performs meta-analysis on the side.
        """
        loop = asyncio.get_running_loop()
        now = loop.time()
        if now - self._last_review_ts < self._review_interval:
            return state

        # [TUNNELING] Analyze code debt & logic bottlenecks
        logger.info("🧠 [ASI] Initiating Recursive Self-Review...")
        
        # Guard against kernel not having loop_state() yet (early boot calls)
        loop_state_fn = getattr(self.kernel, "loop_state", None)
        if not callable(loop_state_fn):
            logger.debug("SelfReview: kernel.loop_state() not available yet. Skipping.")
            return state

        try:
            loop_state = loop_state_fn()
        except Exception as e:
            record_degradation('self_review', e)
            logger.warning("SelfReview: loop_state() raised: %s", e)
            return state

        phi = loop_state.get("phi", 0.0)
        entropy = loop_state.get("entropy", 0.0)

        # 2. If entropy is high, trigger optimization intent
        if entropy > 0.7 or phi < 0.2:
            logger.warning("📉 [ASI] High entropy/Low Phi detected. Proposing architectural refinement.")
            # Injecting intent for the AutonomousSelfModificationEngine
            state.cognition.pending_intents.append({
                "type": "architectural_review",
                "priority": "LOW",
                "context": {
                    "phi": phi,
                    "entropy": entropy,
                    "cause": "Recursive Self-Review trigger"
                }
            })

        self._last_review_ts = now
        return state

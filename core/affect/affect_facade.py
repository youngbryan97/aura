"""core/affect/affect_facade.py — Lightweight Coordinator Facade for Affect.

Provides a simplified interface to the underlying AffectEngineV2 (Damasio)
for use by the Orchestrator's boot sequence and coordinator layer.
"""
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("Aura.AffectFacade")


class AffectFacade:
    """Thin facade over AffectEngineV2 for orchestrator-level access.

    This exists so the boot sequence can register a synchronous entry-point
    before the full async Affect engine is ready.  Once the engine is live,
    all calls are transparently forwarded.
    """

    def __init__(self, orchestrator: Any = None):
        self.orchestrator = orchestrator
        self._engine = None

    # ── lazy resolution ────────────────────────────────────────────
    @property
    def engine(self):
        if self._engine is None:
            from core.container import ServiceContainer
            self._engine = ServiceContainer.get("affect_engine", default=None)
        return self._engine

    # ── public API ─────────────────────────────────────────────────
    def get_status(self) -> Dict[str, Any]:
        """Synchronous status snapshot."""
        if self.engine and hasattr(self.engine, "get_status"):
            return self.engine.get_status()
        return {"mood": "Initializing", "energy": 0, "curiosity": 50, "frustration": 0, "stability": 100, "valence": 0.0, "arousal": 0.0}

    async def get(self):
        """Async affect state (delegates to engine.get)."""
        if self.engine and hasattr(self.engine, "get"):
            return await self.engine.get()
        # Return a neutral baseline while the engine is still booting.
        from core.affect import AffectState, BASELINE_VALENCE, BASELINE_AROUSAL, BASELINE_ENGAGEMENT
        return AffectState(
            valence=BASELINE_VALENCE,
            arousal=BASELINE_AROUSAL,
            engagement=BASELINE_ENGAGEMENT,
            dominant_emotion="neutral",
        )

    async def react(self, trigger: str, context: Optional[Dict] = None):
        if self.engine and hasattr(self.engine, "react"):
            return await self.engine.react(trigger, context)
        logger.debug("AffectFacade: engine not ready, ignoring react(%s)", trigger)

    def get_context_injection(self) -> str:
        if self.engine and hasattr(self.engine, "get_context_injection"):
            return self.engine.get_context_injection()
        return "Mood: Neutral | Energy: 72bpm | Curiosity: 50%"

    def receive_qualia_echo(self, q_norm: float, pri: float, trend: float):
        """Compatibility bridge for loop monitor / qualia synthesizer callers."""
        if self.engine and hasattr(self.engine, "receive_qualia_echo"):
            return self.engine.receive_qualia_echo(q_norm=q_norm, pri=pri, trend=trend)
        logger.debug("AffectFacade: engine not ready, ignoring qualia echo.")

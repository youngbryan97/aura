"""
Affect Coordinator for the RobustOrchestrator.
Handles emotional regulation, drive maintenance, and homeostasis.
"""
import logging
import asyncio
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class AffectCoordinator:
    def __init__(self, orchestrator: Any):
        self.orchestrator = orchestrator
        self._drive_controller = None
        self._emotion_engine = None
        self.boredom: float = 0.0
        self.last_boredom_tick: float = 0.0

    @property
    def drive_controller(self):
        if self._drive_controller is None:
            self._drive_controller = self.orchestrator._get_service("drive_engine")
        return self._drive_controller
    
    @drive_controller.setter
    def drive_controller(self, value):
        self._drive_controller = value

    @property
    def emotion_engine(self):
        if self._emotion_engine is None:
            from core.container import ServiceContainer
            self._emotion_engine = ServiceContainer.get("affect_engine", default=None)
        return self._emotion_engine

    def get_mood(self) -> str:
        """Returns string representation of current mood."""
        engine = self.emotion_engine
        if engine and hasattr(engine, 'get_mood'):
            return engine.get_mood()
        return "Stable"

    def update(self, **kwargs) -> Any:
        """Updates emotional state variables."""
        engine = self.emotion_engine
        if engine and hasattr(engine, 'update'):
            return engine.update(**kwargs)
        return None

    def stabilize(self, factor: float = 0.1) -> None:
        """Stabilizes emotional state."""
        engine = self.emotion_engine
        if engine and hasattr(engine, 'stabilize'):
            engine.stabilize(factor)

    async def somatic_update(self, label: str, value: Any) -> None:
        """Updates somatic markers."""
        engine = self.emotion_engine
        if engine and hasattr(engine, 'somatic_update'):
            if asyncio.iscoroutinefunction(engine.somatic_update):
                await engine.somatic_update(label, value)
            else:
                await asyncio.to_thread(engine.somatic_update, label, value)

    def get_status(self) -> Dict[str, Any]:
        """Returns the current status of the affect system."""
        engine = self.emotion_engine
        if engine and hasattr(engine, 'get_status'):
            status = engine.get_status()
            if not status: return {"mood": "Stable", "boredom": self.boredom}
            status["boredom"] = self.boredom
            return status
        return {"mood": "Offline", "boredom": self.boredom}

    def tick_boredom(self, delta: float) -> float:
        """Increases boredom based on inactivity."""
        self.boredom += delta
        return self.boredom

    def reset_boredom(self):
        """Resets boredom after an interaction."""
        self.boredom = 0.0

    async def decay_tick(self):
        """Proxy decay_tick to the underlying emotion engine (AffectEngine/DamasioV2)."""
        engine = self.emotion_engine
        if engine and hasattr(engine, 'decay_tick'):
            result = engine.decay_tick()
            if asyncio.iscoroutine(result):
                await result

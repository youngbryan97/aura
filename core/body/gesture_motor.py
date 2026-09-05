import logging
import time
from typing import Any, Dict

from core.body.motor_controller import BaseMotor
from core.event_bus import EventPriority, get_event_bus
from core.runtime.errors import record_degradation

logger = logging.getLogger("Body.GestureMotor")

_GESTURE_MOTOR_ERRORS = (AttributeError, RuntimeError, TimeoutError, TypeError, ValueError)


class GestureMotor(BaseMotor):
    """Emit body gesture intents to the canonical event bus for UI listeners."""

    @property
    def name(self) -> str:
        return "gesture"

    async def actuate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        gesture_type = params.get("gesture", "pulse")
        logger.info("Executing visual gesture: %s", gesture_type)
        try:
            payload = {
                "gesture": gesture_type,
                "params": dict(params),
                "timestamp": time.time(),
                "source": "body.gesture_motor",
            }
            await get_event_bus().publish("body.gesture", payload, priority=EventPriority.AUTONOMIC)
            return {
                "status": "success",
                "gesture": gesture_type,
                "effect": "body.gesture event published",
                "topic": "body.gesture",
            }
        except _GESTURE_MOTOR_ERRORS as exc:
            record_degradation("body.gesture_motor", exc)
            logger.warning("Gesture motor failed to publish event: %s", exc)
            return {
                "status": "failed",
                "gesture": gesture_type,
                "error": str(exc),
            }

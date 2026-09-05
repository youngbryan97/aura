"""core/body/action_body.py
Coordinates actuator execution from intents to concrete motor controllers.
"""
import logging
from typing import Any, Dict, Optional

from core.body.motor_controller import get_motor_controller
from core.body.desktop_motor import DesktopMotor
from core.body.browser_motor import BrowserMotor
from core.body.file_motor import FileMotor
from core.body.terminal_motor import TerminalMotor
from core.body.voice_motor import VoiceMotor
from core.body.gesture_motor import GestureMotor
from core.runtime.errors import record_degradation

logger = logging.getLogger("Body.ActionBody")

_ACTION_BODY_ERRORS = (AttributeError, LookupError, OSError, RuntimeError, TimeoutError, TypeError, ValueError)


class ActionBody:
    """Action body mapping requested intents to concrete motor channels."""

    def __init__(self):
        self.controller = get_motor_controller()
        self._initialized = False

    def initialize_motors(self) -> None:
        if self._initialized:
            return
        
        self.controller.register(DesktopMotor())
        self.controller.register(BrowserMotor())
        self.controller.register(FileMotor())
        self.controller.register(TerminalMotor())
        self.controller.register(VoiceMotor())
        self.controller.register(GestureMotor())
        
        self._initialized = True
        logger.info("Actuator motor systems initialized.")

    async def execute_action(self, intent: Dict[str, Any], state: Any) -> Dict[str, Any]:
        """Routes action intent to the registered motor channel."""
        self.initialize_motors()
        
        channel = intent.get("channel")
        params = intent.get("params", {})

        if not channel:
            return {"status": "error", "message": "Missing actuator channel"}

        motor = self.controller.get_motor(channel)
        if not motor:
            return {"status": "error", "message": f"Actuator channel not found: {channel}"}

        logger.info("Actuating motor channel: %s", channel)
        try:
            receipt = await motor.actuate(params)
            receipt["channel"] = channel
            return receipt
        except _ACTION_BODY_ERRORS as e:
            record_degradation("body.action_body", e)
            logger.error("Failed to actuate channel %s: %s", channel, e)
            return {"status": "failed", "error": str(e), "channel": channel}


# Singleton Access
_action_body: Optional[ActionBody] = None


def get_action_body() -> ActionBody:
    global _action_body
    if _action_body is None:
        _action_body = ActionBody()
    return _action_body

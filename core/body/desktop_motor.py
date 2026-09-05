"""core/body/desktop_motor.py
Desktop motor controller executing UI interactions (focusing apps, resizing windows).
"""
import logging
import os
from subprocess import SubprocessError
from typing import Any, Dict

from core.body.motor_controller import BaseMotor
from core.runtime.errors import record_degradation
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Body.DesktopMotor")

_DESKTOP_MOTOR_ERRORS = (OSError, RuntimeError, SubprocessError, TimeoutError, TypeError, ValueError)


class DesktopMotor(BaseMotor):
    """Executes actions on the operating system desktop window manager."""

    @property
    def name(self) -> str:
        return "desktop"

    async def actuate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        action_type = params.get("type", "focus_app")
        app_name = params.get("target_app", "Terminal")

        if action_type == "focus_app":
            try:
                if os.path.exists("/usr/bin/osascript"):
                    script = f'tell application "{app_name}" to activate'
                    await get_subprocess_gateway().run_async(
                        ["/usr/bin/osascript", "-e", script],
                        check=True,
                        timeout=2.0,
                        source="body.desktop_motor",
                        accelerator_capability="none",
                    )
                    return {
                        "status": "success",
                        "action": "focus_app",
                        "app": app_name
                    }
            except _DESKTOP_MOTOR_ERRORS as e:
                record_degradation("body.desktop_motor", e)
                logger.debug("Failed to focus app via AppleScript: %s", e)

        return {
            "status": "not_executed",
            "action": action_type,
            "params": params,
            "reason": "desktop automation unavailable or blocked",
        }

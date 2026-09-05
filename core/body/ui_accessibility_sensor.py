"""core/body/ui_accessibility_sensor.py
UI accessibility tree and system window hierarchy sensor.
"""
import logging
import os
from subprocess import SubprocessError
from typing import Any, Dict

from core.body.sensor_registry import BaseSensor
from core.runtime.errors import record_degradation
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Body.UiAccessibilitySensor")

_UI_ACCESSIBILITY_SENSOR_ERRORS = (OSError, RuntimeError, SubprocessError, TimeoutError, TypeError, ValueError)


class UiAccessibilitySensor(BaseSensor):
    """Retrieves accessibility tree details and front window position info."""

    @property
    def name(self) -> str:
        return "ui_accessibility"

    async def read(self) -> Dict[str, Any]:
        try:
            if os.path.exists("/usr/bin/osascript"):
                # Use a lightweight AppleScript to get window titles of all running apps
                script = 'tell application "System Events" to get title of every window of (every process whose visible is true)'
                cmd = ["/usr/bin/osascript", "-e", script]
                res = await get_subprocess_gateway().run_async(
                    cmd,
                    read_only=True,
                    timeout=2.0,
                    source="body.ui_accessibility_sensor",
                    accelerator_capability="auto",
                )
                if res.returncode == 0:
                    return {
                        "available": True,
                        "windows": res.stdout.strip(),
                        "driver": "AppleScript System Events"
                    }
        except _UI_ACCESSIBILITY_SENSOR_ERRORS as e:
            record_degradation("body.ui_accessibility_sensor", e)
            logger.debug("Failed to query accessibility windows via AppleScript: %s", e)

        return {
            "available": False,
            "windows": "Aura Terminal Output",
            "driver": "unavailable_accessibility_tree",
        }

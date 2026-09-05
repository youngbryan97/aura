"""core/body/browser_state_sensor.py
Browser state and active URL query sensor.
"""
import logging
import os
from subprocess import SubprocessError
from typing import Any, Dict

from core.body.sensor_registry import BaseSensor
from core.runtime.errors import record_degradation
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Body.BrowserStateSensor")

_BROWSER_STATE_SENSOR_ERRORS = (OSError, RuntimeError, SubprocessError, TimeoutError, TypeError, ValueError)


class BrowserStateSensor(BaseSensor):
    """Tracks current active tab URL and page loading states."""

    @property
    def name(self) -> str:
        return "browser_state"

    async def read(self) -> Dict[str, Any]:
        """Query Chrome/Safari on macOS via AppleScript to get the current tab URL."""
        try:
            if os.path.exists("/usr/bin/osascript"):
                # Query Google Chrome if running
                script = 'tell application "Google Chrome" to get URL of active tab of first window'
                cmd = ["/usr/bin/osascript", "-e", script]
                res = await get_subprocess_gateway().run_async(
                    cmd,
                    read_only=True,
                    timeout=1.5,
                    source="body.browser_state_sensor",
                    accelerator_capability="auto",
                )
                if res.returncode == 0:
                    return {
                        "browser": "Google Chrome",
                        "active_url": res.stdout.strip(),
                        "status": "connected"
                    }
        except _BROWSER_STATE_SENSOR_ERRORS as e:
            record_degradation("body.browser_state_sensor", e)
            logger.debug("Failed to query browser via AppleScript: %s", e)

        return {
            "browser": "None",
            "active_url": "about:blank",
            "status": "inactive"
        }

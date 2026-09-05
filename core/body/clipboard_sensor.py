"""core/body/clipboard_sensor.py
Clipboard polling sensor reading macOS pbpaste.
"""
import logging
import os
from subprocess import SubprocessError
from typing import Any

from core.body.sensor_registry import BaseSensor
from core.runtime.errors import record_degradation
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Body.ClipboardSensor")

_CLIPBOARD_SENSOR_ERRORS = (OSError, RuntimeError, SubprocessError, TimeoutError, TypeError, ValueError)


class ClipboardSensor(BaseSensor):
    """Monitors clipboard content changes."""

    @property
    def name(self) -> str:
        return "clipboard"

    async def read(self) -> dict[str, Any]:
        try:
            if os.path.exists("/usr/bin/pbpaste"):
                res = await get_subprocess_gateway().run_async(
                    ["/usr/bin/pbpaste"],
                    read_only=True,
                    check=True,
                    timeout=1.0,
                    source="body.clipboard_sensor",
                    accelerator_capability="none",
                )
                content = res.stdout.strip()
                # Truncate large contents.
                display_content = content[:200] + "..." if len(content) > 200 else content
                return {
                    "has_content": len(content) > 0,
                    "content": display_content,
                    "length": len(content)
                }
        except _CLIPBOARD_SENSOR_ERRORS as e:
            record_degradation("body.clipboard_sensor", e)
            logger.debug("Failed to read clipboard via pbpaste: %s", e)

        return {
            "has_content": False,
            "content": "",
            "length": 0
        }

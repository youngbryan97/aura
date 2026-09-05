"""core/body/keyboard_mouse_state.py
Keyboard and mouse state tracking sensor.
"""
import time
from typing import Any

from core.body.sensor_registry import BaseSensor


class KeyboardMouseSensor(BaseSensor):
    """Tracks mouse location coordinates and user interaction signals."""

    @property
    def name(self) -> str:
        return "keyboard_mouse"

    async def read(self) -> dict[str, Any]:
        # Direct pointer telemetry is not polled unless an approved host adapter is present.
        return {
            "mouse_x": None,
            "mouse_y": None,
            "idle_duration_s": None,
            "user_active": None,
            "status": "adapter_unavailable",
            "timestamp": time.time()
        }

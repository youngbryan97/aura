"""core/body/microphone_sensor.py
Microphone input sensor measuring audio telemetry and wake words.
"""
from typing import Any, Dict

from core.body.sensor_registry import BaseSensor


class MicrophoneSensor(BaseSensor):
    """Monitors ambient decibel levels and audio availability."""

    @property
    def name(self) -> str:
        return "microphone"

    async def read(self) -> Dict[str, Any]:
        # Audio ingestion is owned by the governed voice/perception runtime.
        return {
            "status": "perception_runtime_owned",
            "db_level": None,
            "input_device": None,
            "wake_word_detected": False
        }

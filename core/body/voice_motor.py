"""core/body/voice_motor.py
Voice motor output channel executing speech synthesis.
"""
import logging
import os
from subprocess import SubprocessError
from typing import Any

from core.body.motor_controller import BaseMotor
from core.runtime.errors import record_degradation
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Body.VoiceMotor")

_VOICE_MOTOR_ERRORS = (OSError, RuntimeError, SubprocessError, TimeoutError, TypeError, ValueError)


class VoiceMotor(BaseMotor):
    """Somatic voice synthesizer executing vocal feedback."""

    @property
    def name(self) -> str:
        return "voice"

    async def actuate(self, params: dict[str, Any]) -> dict[str, Any]:
        text = params.get("text", "")
        if not text:
            return {"status": "ignored", "message": "Empty voice text"}

        try:
            if os.path.exists("/usr/bin/say"):
                await get_subprocess_gateway().run_async(
                    ["/usr/bin/say", text],
                    check=True,
                    timeout=5.0,
                    source="body.voice_motor",
                    accelerator_capability="none",
                )
                return {
                    "status": "success",
                    "spoken": text,
                    "engine": "macOS say"
                }
        except _VOICE_MOTOR_ERRORS as e:
            record_degradation("body.voice_motor", e)
            logger.debug("macOS say utility failed: %s", e)

        return {
            "status": "not_executed",
            "spoken": text,
            "engine": "unavailable_voice_engine",
        }

"""Mute, unmute, and stop speaking.

Three hardwired pathways matched "aura, mute", "unmute" and "shut up" at
priorities 9, 9 and 10 — above every other route — and dispatched them to skill
names that no module implemented. Saying "stop talking" reached the top of the
routing table and then nothing. The pathways predate any implementation; they
were never wired to one.

"Be quiet" has to work while she is mid-sentence, which is the one moment the
normal request path is busy, so each of these does its work by flipping the
engine's own flags rather than by asking the speech pipeline for a turn.
"""
from __future__ import annotations

import logging
from typing import Any

from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT
from infrastructure import BaseSkill

logger = logging.getLogger("Skills.VoiceControl")


def _voice_engine() -> Any | None:
    return ServiceContainer.get("voice_engine", default=None)


def _no_engine(action: str) -> dict[str, Any]:
    # Not an error the person needs to see twice: it means voice was never
    # started in this process, so there is nothing to silence.
    return {
        "ok": False,
        "status": "voice_engine_unavailable",
        "message": f"Voice is not running, so there is nothing to {action}.",
    }


class VoiceMuteSkill(BaseSkill):
    #: What a caller gets back. The shared part only: every skill
    #: here returns `ok`, and a schema claiming to be complete
    #: would be wrong for every one that adds a field.
    result_schema = THE_SHARED_RESULT

    name = "voice_mute"
    description = "Disable the microphone and speech output."
    effect_scope = "state_mutation"

    async def execute(self, goal: Any, context: dict[str, Any]) -> dict[str, Any]:
        engine = _voice_engine()
        if engine is None or not hasattr(engine, "mute"):
            return _no_engine("mute")
        try:
            engine.mute()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("skills.voice_control.mute", exc)
            return {"ok": False, "status": "mute_failed", "error": str(exc)}
        logger.info("🔇 Voice muted by request.")
        return {"ok": True, "status": "muted", "message": "Microphone and speech are off."}


class VoiceUnmuteSkill(BaseSkill):
    #: What a caller gets back. The shared part only: every skill here
    #: returns `ok`, and a schema claiming to be complete would be wrong
    #: for every one that adds a field.
    result_schema = THE_SHARED_RESULT

    name = "voice_unmute"
    description = "Re-enable the microphone and speech output."
    effect_scope = "state_mutation"

    async def execute(self, goal: Any, context: dict[str, Any]) -> dict[str, Any]:
        engine = _voice_engine()
        if engine is None or not hasattr(engine, "unmute"):
            return _no_engine("unmute")
        try:
            engine.unmute()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("skills.voice_control.unmute", exc)
            return {"ok": False, "status": "unmute_failed", "error": str(exc)}
        logger.info("🔊 Voice unmuted by request.")
        return {"ok": True, "status": "unmuted", "message": "Microphone and speech are on."}


class VoiceStopTtsSkill(BaseSkill):
    #: What a caller gets back. The shared part only: every skill
    #: here returns `ok`, and a schema claiming to be complete
    #: would be wrong for every one that adds a field.
    result_schema = THE_SHARED_RESULT

    name = "voice_stop_tts"
    description = "Stop speaking now, without disabling the microphone."
    effect_scope = "state_mutation"

    async def execute(self, goal: Any, context: dict[str, Any]) -> dict[str, Any]:
        engine = _voice_engine()
        if engine is None:
            return _no_engine("stop")
        # Deliberately not mute(): "shut up" is about the current utterance and
        # the speaker, and taking the microphone away with it would mean the
        # next thing the person says goes nowhere.
        try:
            engine.speaking_enabled = False
        except (AttributeError, TypeError) as exc:
            record_degradation("skills.voice_control.stop_tts", exc)
            return {"ok": False, "status": "stop_failed", "error": str(exc)}
        logger.info("🤐 Speech stopped by request; microphone left on.")
        return {
            "ok": True,
            "status": "speech_stopped",
            "message": "Stopped speaking. I am still listening.",
        }

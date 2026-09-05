"""core/voice/voice_session.py — Voice Command Session Manager
================================================================
Manages a single voice command session from wake word to completion.

Provides narration hooks so Aura speaks progress updates, handles
barge-in, and produces honest failure reports.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.service_registry import get_runtime_service, register_runtime_service
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Aura.VoiceSession")


class SessionState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    EXECUTING = "executing"
    REPORTING = "reporting"


@dataclass
class VoiceSessionLog:
    """A single voice session with timing and results."""
    session_id: int
    command: str = ""
    state: SessionState = SessionState.IDLE
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    narration: list[str] = field(default_factory=list)
    mission_id: str = ""
    success: bool = False
    error: str = ""

    def duration_s(self) -> float:
        end = self.completed_at or time.time()
        return round(end - self.started_at, 1)


class VoiceSessionManager:
    """Manages voice command sessions.

    Integrates with TTS for spoken narration so people watching
    understand what's happening:
    - "I'm opening Notes."
    - "PDF created, moving to folder."
    - "Bryan, export failed. I made the PDF another way."

    Error narration is HONEST. No smooth-over fallbacks.
    """

    def __init__(self) -> None:
        self._current_session: VoiceSessionLog | None = None
        self._session_history: list[VoiceSessionLog] = []
        self._max_history = 50
        self._session_counter = 0
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        register_runtime_service("voice_session", self, required=False, owner="core/voice/voice_session.py", registered_by="VoiceSessionManager.start")
        self._started = True
        logger.info("VoiceSessionManager ONLINE")

    def begin_session(self, command: str = "") -> VoiceSessionLog:
        """Start a new voice session."""
        self._session_counter += 1
        session = VoiceSessionLog(
            session_id=self._session_counter,
            command=command,
            state=SessionState.LISTENING,
        )
        self._current_session = session
        logger.info("Voice session #%d started", session.session_id)
        return session

    def set_command(self, command: str) -> None:
        """Set the transcribed command for the current session."""
        if self._current_session:
            self._current_session.command = command
            self._current_session.state = SessionState.PROCESSING

    async def narrate(self, message: str) -> None:
        """Speak a progress update.

        Also logs to the session narration log.
        """
        if self._current_session:
            self._current_session.narration.append(message)

        logger.info("🗣️ %s", message)

        # Try to speak via TTS
        try:
            tts = get_runtime_service("tts_engine", default=None)
            if tts and hasattr(tts, "speak"):
                await tts.speak(message)
            else:
                # Fallback: macOS `say` command
                proc = await get_subprocess_gateway().spawn_async(
                    ["say", "-v", "Samantha", message],
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                    source="voice_session.say_fallback",
                    accelerator_capability="none",
                )
                await asyncio.wait_for(proc.wait(), timeout=10.0)
        except (TimeoutError, ImportError, OSError, RuntimeError) as e:
            record_degradation("voice_session.tts", e)

    async def narrate_progress(self, step_description: str) -> None:
        """Narrate a task step starting."""
        await self.narrate(f"Working on: {step_description}")

    async def narrate_success(self, summary: str) -> None:
        """Narrate successful completion."""
        await self.narrate(f"Done. {summary}")

    async def narrate_failure(self, error: str, recovery_action: str = "") -> None:
        """Narrate a failure HONESTLY.

        No smooth-over. No canned response. She explains what went wrong
        and what she did about it.
        """
        if recovery_action:
            await self.narrate(f"{error} {recovery_action}")
        else:
            await self.narrate(f"{error}")

    def end_session(self, success: bool, error: str = "") -> None:
        """End the current voice session."""
        if self._current_session:
            self._current_session.state = SessionState.IDLE
            self._current_session.completed_at = time.time()
            self._current_session.success = success
            self._current_session.error = error

            self._session_history.append(self._current_session)
            if len(self._session_history) > self._max_history:
                self._session_history = self._session_history[-self._max_history:]

            logger.info(
                "Voice session #%d ended: %s (%.1fs)",
                self._current_session.session_id,
                "success" if success else f"failed: {error[:50]}",
                self._current_session.duration_s(),
            )
            self._current_session = None

    @property
    def is_active(self) -> bool:
        return self._current_session is not None

    def get_status(self) -> dict[str, Any]:
        return {
            "active": self.is_active,
            "session_count": self._session_counter,
            "current": {
                "id": self._current_session.session_id,
                "command": self._current_session.command[:60],
                "state": self._current_session.state.value,
                "duration_s": self._current_session.duration_s(),
            } if self._current_session else None,
            "recent": [
                {
                    "id": s.session_id,
                    "command": s.command[:40],
                    "success": s.success,
                    "duration_s": s.duration_s(),
                }
                for s in self._session_history[-5:]
            ],
        }


_instance: VoiceSessionManager | None = None


def get_voice_session_manager() -> VoiceSessionManager:
    global _instance
    if _instance is None:
        _instance = VoiceSessionManager()
    return _instance


__all__ = ["VoiceSessionManager", "VoiceSessionLog", "SessionState", "get_voice_session_manager"]

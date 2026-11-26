"""core/embodiment/voice_presence.py — Voice Presence.
"""
from __future__ import annotations

import asyncio
import asyncio.subprocess
import logging
import os
import shlex
import subprocess
import sys
from abc import ABC, abstractmethod
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.service_registry import get_runtime_service
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Aura.VoicePresence")


class TTSEngine(ABC):
    @abstractmethod
    async def speak(self, text: str) -> None:
        """Speak the given text. Subclasses must implement."""
        raise NotImplementedError

    async def stop(self) -> None:
        """Stop any ongoing speech."""
        logger.debug("TTSEngine.stop: No-op in base class.")

class MacOSTTS(TTSEngine):
    def __init__(self):
        self._speaking_proc: asyncio.subprocess.Process | None = None

    async def speak(self, text: str) -> None:
        # Check proc and returncode safely
        proc = self._speaking_proc
        if proc is not None:
            try:
                # v Zenith: Robust check for process life
                if proc.returncode is None:
                    proc.terminate()
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('voice_presence', e)
                logger.debug("MacOSTTS: Cleanup of previous process failed: %s", e)
            
        clean = self._clean(text)
        if not clean:
            return
        try:
            # v Zenith: Use subprocess.DEVNULL for redirection to avoid asyncio.subprocess issues
            self._speaking_proc = await get_subprocess_gateway().spawn_async(
                ["say", clean],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                source="tool_execution:voice_presence.macos_say",
                accelerator_capability="none",
            )
        except (subprocess.SubprocessError, OSError) as e:
            record_degradation('voice_presence', e)
            logger.debug("MacOSTTS: Speak exec failed: %s", e)

    async def stop(self) -> None:
        proc = self._speaking_proc
        if proc is not None:
            try:
                if proc.returncode is None:
                    proc.terminate()
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('voice_presence', e)
                logger.debug("MacOSTTS: Stop failed: %s", e)

    @staticmethod
    def _clean(text: str) -> str:
        import re
        text = re.sub(r"```[\s\S]*?```", "", text)
        text = re.sub(r"`[^`]+`", "", text)
        text = re.sub(r"https?://\S+", "", text)
        return text.strip()

class LinuxTTS(TTSEngine):
    async def speak(self, text: str) -> None:
        clean = MacOSTTS._clean(text)
        if not clean:
            return
        try:
            # Try espeak first, then festival
            for cmd in ["espeak", "festival --tts"]:
                try:
                    proc = await get_subprocess_gateway().spawn_shell_async(
                        f"{cmd} {shlex.quote(clean)}",
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        source="tool_execution:voice_presence.linux_tts",
                        accelerator_capability="auto",
                    )
                    await proc.wait()
                    return
                except (subprocess.SubprocessError, OSError):
                    continue
        except (subprocess.SubprocessError, OSError) as e:
            record_degradation('voice_presence', e)
            logger.error("LinuxTTS: Speak failed: %s", e)

class WindowsTTS(TTSEngine):
    async def speak(self, text: str) -> None:
        clean = MacOSTTS._clean(text)
        if not clean:
            return
        try:
            escaped = clean.replace("'", "''")
            ps_cmd = f"Add-Type -AssemblyName System.Speech; $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; $speak.Speak('{escaped}')"
            proc = await get_subprocess_gateway().spawn_async(
                ["powershell", "-Command", ps_cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                source="tool_execution:voice_presence.windows_tts",
                accelerator_capability="none",
            )
            await proc.wait()
        except (subprocess.SubprocessError, OSError) as e:
            record_degradation('voice_presence', e)
            logger.error("WindowsTTS: Speak failed: %s", e)

class DummyTTS(TTSEngine):
    """The platform has no TTS backend. Say so; do not pretend to speak.

    ``speak`` logged the text at info level and returned, exactly like every
    engine that had just made a sound. Nothing downstream — no caller, no
    health surface — could tell an utterance from a line in a log file. A
    voice that cannot make sound is a degradation, and reporting it as
    success is the same defect this codebase keeps finding elsewhere: the
    absence of an action recorded as the action.
    """

    def __init__(self) -> None:
        self._announced = False

    async def speak(self, text: str) -> None:
        if not self._announced:
            # Once per process. A degradation per utterance would bury the
            # condition it is trying to report.
            self._announced = True
            record_degradation(
                'voice_presence',
                RuntimeError(f"no TTS backend for platform {sys.platform!r}"),
                action="text is written to the log instead of spoken",
            )
        logger.info("[no TTS backend — text only]: %s", text)

class VoicePresence:
    def __init__(self, orchestrator: Any):
        self.orchestrator = orchestrator
        # Platform-aware TTS selection
        if sys.platform == "darwin":
            self._tts = MacOSTTS()
        elif sys.platform == "linux":
            self._tts = LinuxTTS()
        elif sys.platform == "win32":
            self._tts = WindowsTTS()
        else:
            logger.warning("VoicePresence: Unsupported platform '%s'. Using DummyTTS.", sys.platform)
            self._tts = DummyTTS()

        #: Whether this process can actually produce sound. Readable so a
        #: caller asking for speech can tell it will only get a log line.
        self.voice_output_available = not isinstance(self._tts, DummyTTS)

        self._running = False
        self._voice_enabled = os.environ.get("AURA_VOICE_SILENT", "0") != "1"
        if not self._voice_enabled:
            logger.info("VoicePresence: Muted (AURA_VOICE_SILENT=1)")

    async def start(self) -> None:
        self._running = True
        logger.info("VoicePresence started.")

    async def stop(self) -> None:
        self._running = False
        await self._tts.stop()

    async def speak(self, text: str) -> None:
        if not self._voice_enabled or not text:
            return
        await self._tts.speak(text)

    async def speak_response(self, response: str, phi: float = 0.0) -> None:
        if phi > 0.4 or "i feel" in response.lower():
            await self.speak(response)

async def maybe_speak_response(response: str, state: Any) -> None:
    try:
        voice = get_runtime_service("voice_presence", default=None)
        if voice:
            await voice.speak_response(response, phi=getattr(state, "phi", 0.0))
    except (ImportError, AttributeError, RuntimeError) as e:
        record_degradation('voice_presence', e)
        logger.debug("maybe_speak_response failed: %s", e)

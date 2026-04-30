"""core/embodiment/voice_presence.py — Voice Presence.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation


import asyncio
import asyncio.subprocess
import subprocess
import logging
import time
from typing import Any, Callable, Optional

import sys
import os

logger = logging.getLogger("Aura.VoicePresence")


class TTSEngine:
    async def speak(self, text: str) -> None:
        """Speak the given text. Subclasses must implement."""
        raise RuntimeError(f"{type(self).__name__}.speak must be implemented by a TTS engine")

    async def stop(self) -> None:
        """Stop any ongoing speech."""
        logger.debug("TTSEngine.stop: No-op in base class.")

class MacOSTTS(TTSEngine):
    def __init__(self):
        self._speaking_proc: Optional[asyncio.subprocess.Process] = None

    async def speak(self, text: str) -> None:
        # Check proc and returncode safely
        proc = self._speaking_proc
        if proc is not None:
            try:
                # v Zenith: Robust check for process life
                if proc.returncode is None:
                    proc.terminate()
            except Exception as e:
                record_degradation('voice_presence', e)
                logger.debug(f"MacOSTTS: Cleanup of previous process failed: {e}")
            
        clean = self._clean(text)
        if not clean: return
        try:
            # v Zenith: Use subprocess.DEVNULL for redirection to avoid asyncio.subprocess issues
            self._speaking_proc = await asyncio.create_subprocess_exec(
                "say", clean, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception as e:
            record_degradation('voice_presence', e)
            logger.debug("🔊 MacOSTTS: Speak exec failed: %s", e)

    async def stop(self) -> None:
        proc = self._speaking_proc
        if proc is not None:
            try:
                if proc.returncode is None:
                    proc.terminate()
            except Exception as e:
                record_degradation('voice_presence', e)
                logger.debug(f"MacOSTTS: Stop failed: {e}")

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
        if not clean: return
        try:
            # Try espeak first, then festival
            for cmd in ["espeak", "festival --tts"]:
                try:
                    proc = await asyncio.create_subprocess_shell(
                        f"{cmd} '{clean}'", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                    await proc.wait()
                    return
                except Exception: continue
        except Exception as e:
            record_degradation('voice_presence', e)
            logger.error("🔊 LinuxTTS: Speak failed: %s", e)

class WindowsTTS(TTSEngine):
    async def speak(self, text: str) -> None:
        clean = MacOSTTS._clean(text)
        if not clean: return
        try:
            ps_cmd = f"Add-Type -AssemblyName System.Speech; $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; $speak.Speak('{clean}')"
            proc = await asyncio.create_subprocess_exec(
                "powershell", "-Command", ps_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            await proc.wait()
        except Exception as e:
            record_degradation('voice_presence', e)
            logger.error("🔊 WindowsTTS: Speak failed: %s", e)

class DummyTTS(TTSEngine):
    async def speak(self, text: str) -> None:
        logger.info("🔊 [DUMMY VOICE]: %s", text)

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
            logger.warning("🔊 VoicePresence: Unsupported platform '%s'. Using DummyTTS.", sys.platform)
            self._tts = DummyTTS()
            
        self._running = False
        self._voice_enabled = True

    async def start(self) -> None:
        self._running = True
        logger.info("VoicePresence started.")

    async def stop(self) -> None:
        self._running = False
        await self._tts.stop()

    async def speak(self, text: str) -> None:
        if not self._voice_enabled or not text: return
        await self._tts.speak(text)

    async def speak_response(self, response: str, phi: float = 0.0) -> None:
        if phi > 0.4 or "i feel" in response.lower():
            await self.speak(response)

async def maybe_speak_response(response: str, state: Any) -> None:
    try:
        from core.container import ServiceContainer
        voice = ServiceContainer.get("voice_presence", default=None)
        if voice:
            await voice.speak_response(response, phi=getattr(state, "phi", 0.0))
    except Exception as e:
        record_degradation('voice_presence', e)
        logger.debug(f"maybe_speak_response failed: {e}")

import asyncio
import logging
import os
import subprocess
import sys
from typing import Any

from pydantic import BaseModel, Field

from core.runtime.errors import FallbackClassification, record_degradation
from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.Voice")


def _record_speak_degradation(
    error: BaseException,
    *,
    action: str,
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "speak",
        error,
        severity="warning",
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=False,
        extra=extra,
    )


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", name, raw, default)
        return default
    return max(minimum, min(maximum, value))


class SpeakInput(BaseModel):
    text: str = Field(..., description="The text to convert to speech.")
    voice: str | None = Field(None, description="Optional voice name (e.g., 'Samantha').")
    rate: int | None = Field(185, ge=80, le=320, description="Speech rate/speed.")


class SpeakSkill(BaseSkill):
    name = "speak"
    description = "Convert text to audible speech using local system voices."
    input_model = SpeakInput

    def __init__(self):
        super().__init__()
        self._voice_engine = None
        self._voice_engine_checked = False
        self._fallback_engine = None
        self._max_text_chars = _env_int(
            "AURA_SPEAK_MAX_CHARS",
            5000,
            minimum=1,
            maximum=20000,
        )
        self._say_timeout_seconds = _env_int(
            "AURA_SPEAK_TIMEOUT_SECONDS",
            45,
            minimum=1,
            maximum=300,
        )
        if sys.platform != "darwin":
            try:
                import pyttsx3

                self._fallback_engine = pyttsx3.init()
                self._fallback_engine.setProperty("rate", 175)
            except (ImportError, AttributeError, RuntimeError) as exc:
                _record_speak_degradation(
                    exc,
                    action="Disabled pyttsx3 fallback and kept speak skill available for other engines",
                    extra={"platform": sys.platform},
                )
                logger.warning("pyttsx3 init failed: %s", exc)

    def _get_engine(self):
        """Resolve voice engine from ServiceContainer once per skill instance."""
        if self._voice_engine_checked:
            return self._voice_engine
        self._voice_engine_checked = True
        try:
            from core.container import ServiceContainer

            self._voice_engine = ServiceContainer.get("voice_engine")
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_speak_degradation(
                exc,
                action="Skipped sovereign voice engine and continued to local speech fallbacks",
            )
            logger.error("Failed to resolve voice_engine: %s", exc)
        return self._voice_engine

    async def _speak_with_macos_say(self, text: str, voice: str, rate: int) -> None:
        process = await asyncio.create_subprocess_exec(
            "say",
            "-v",
            voice,
            "-r",
            str(rate),
            text,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            _stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=float(self._say_timeout_seconds),
            )
        except TimeoutError as exc:
            process.kill()
            await process.communicate()
            raise TimeoutError(
                f"macOS say timed out after {self._say_timeout_seconds}s"
            ) from exc
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"macOS say exited {process.returncode}: {detail}")

    async def _speak_with_pyttsx3(self, text: str) -> None:
        if self._fallback_engine is None:
            raise RuntimeError("pyttsx3 fallback engine unavailable")

        def _run() -> None:
            self._fallback_engine.say(text)
            self._fallback_engine.runAndWait()

        await asyncio.to_thread(_run)

    async def execute(self, params: SpeakInput, context: dict[str, Any]) -> dict[str, Any]:
        if isinstance(params, dict):
            try:
                params = SpeakInput(**params)
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                _record_speak_degradation(
                    exc,
                    action="Rejected invalid speak input before attempting audio output",
                )
                return {"ok": False, "error": f"Invalid input: {exc}"}

        text = params.text.strip()
        voice = params.voice or "Samantha"
        rate = params.rate or 185

        if not text:
            return {"ok": False, "error": "No text provided to speak."}
        if len(text) > self._max_text_chars:
            return {
                "ok": False,
                "error": f"Speech text exceeds {self._max_text_chars} characters.",
            }

        logger.info("🔊 Speaking: %s...", text[:50])
        attempts: list[dict[str, str]] = []

        # Strategy 1: Sovereign Voice Engine (Premium/Standard)
        engine = self._get_engine()
        if engine:
            try:
                await engine.synthesize_speech(text)
                return {
                    "ok": True,
                    "mode": "sovereign",
                    "message": "Spoken via Sovereign Voice Engine.",
                }
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                attempts.append({"mode": "sovereign", "error": str(exc)[:250]})
                _record_speak_degradation(
                    exc,
                    action="Fell back from sovereign voice engine to local speech options",
                    extra={"mode": "sovereign"},
                )
                logger.error("Sovereign synthesis failed: %s", exc)

        # Strategy 2: macOS 'say' (High Quality Fallback)
        if sys.platform == "darwin":
            try:
                await self._speak_with_macos_say(text, voice, rate)
                return {
                    "ok": True,
                    "mode": "macos_say",
                    "message": f"Spoken via macOS ({voice}).",
                }
            except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
                attempts.append({"mode": "macos_say", "error": str(exc)[:250]})
                _record_speak_degradation(
                    exc,
                    action="Fell back from macOS say to generic local speech engine",
                    extra={"mode": "macos_say", "voice": voice, "rate": str(rate)},
                )
                logger.error("macOS say failed: %s", exc)

        # Strategy 3: Local pyttsx3 (Generic Fallback)
        if self._fallback_engine:
            try:
                await self._speak_with_pyttsx3(text)
                return {"ok": True, "mode": "pyttsx3", "message": "Spoken via local engine."}
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                attempts.append({"mode": "pyttsx3", "error": str(exc)[:250]})
                _record_speak_degradation(
                    exc,
                    action="Reported speech failure after all configured voice engines failed",
                    extra={"mode": "pyttsx3"},
                )
                logger.error("pyttsx3 failed: %s", exc)

        return {
            "ok": False,
            "error": "No voice engine available.",
            "attempts": attempts,
        }

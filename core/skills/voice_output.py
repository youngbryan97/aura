"""core/skills/voice_output.py — Neural TTS Voice Synthesis
=============================================================
First-class BaseSkill that extends Aura's vocal capabilities with:
  - Multi-engine TTS (MLX, Piper, macOS say, pyttsx3)
  - Voice cloning preparation (XTTS-v2 integration path)
  - Emotion-modulated speech parameters
  - Audio file output for async playback
  - SSML support for fine-grained prosody control

Works alongside the existing speak.py skill but provides file-based
output and richer voice control for streaming/async playback.

This closes the "voice output" gap in the senses polish layer.
"""

import asyncio
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.config import config
from core.runtime.errors import FallbackClassification, record_degradation
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.skills.base_skill import BaseSkill
from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT

logger = logging.getLogger("Skills.VoiceOutput")

_VOICE_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    TimeoutError,
)


def _resolve_piper_command() -> str | None:
    """Resolve Piper from explicit config, PATH, or Aura's bundled venv."""
    candidates: list[str | None] = [
        os.environ.get("AURA_PIPER_BIN"),
        shutil.which("piper"),
        str(Path(sys.executable).with_name("piper")),
        str(Path(__file__).resolve().parents[2] / ".venv" / "bin" / "piper"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
    return None


def _record_voice_degradation(
    error: BaseException,
    *,
    action: str,
    severity: str = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "voice_output",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=False,
        extra=extra,
    )


class VoiceOutputInput(BaseModel):
    text: str = Field(..., description="Text to synthesize into speech.")
    voice: str | None = Field(
        None,
        description="Voice name or ID (e.g., 'Samantha', 'en_US-amy-medium').",
    )
    rate: float = Field(
        1.0,
        ge=0.5,
        le=3.0,
        description="Speech rate multiplier (1.0 = normal).",
    )
    pitch: float = Field(
        1.0,
        ge=0.5,
        le=2.0,
        description="Pitch multiplier (1.0 = normal).",
    )
    emotion: str | None = Field(
        None,
        description="Emotion to modulate voice with (e.g., 'calm', 'excited', 'serious').",
    )
    output_format: str = Field(
        "wav",
        description="Output audio format: 'wav', 'mp3', 'aiff'.",
    )
    play_audio: bool = Field(
        True,
        description="Whether to play the audio immediately after synthesis.",
    )
    save_to_file: bool = Field(
        True,
        description="Whether to save the audio to a file.",
    )


class VoiceOutputSkill(BaseSkill):
    #: What a caller gets back. The shared part only: every skill
    #: here returns `ok`, and a schema claiming to be complete
    #: would be wrong for every one that adds a field.
    result_schema = THE_SHARED_RESULT

    name = "voice_output"
    description = (
        "Synthesize text into speech audio files using multiple TTS engines. "
        "Supports voice selection, rate/pitch control, emotion modulation, "
        "and file output. Use for generating spoken responses, narration, "
        "and audio content."
    )
    input_model = VoiceOutputInput
    timeout_seconds = 60.0
    metabolic_cost = 2
    effect_scope = "sandboxed_compute"

    def __init__(self):
        super().__init__()
        self._piper_available = False
        self._piper_checked = False
        self._piper_command: str | None = None
        self._output_dir = Path(config.paths.data_dir) / "voice_output"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._max_text_chars = 10000

        # Emotion -> speech parameter mapping
        self._emotion_params: dict[str, dict[str, float]] = {
            "calm": {"rate": 0.9, "pitch": 0.95},
            "excited": {"rate": 1.2, "pitch": 1.1},
            "serious": {"rate": 0.85, "pitch": 0.9},
            "happy": {"rate": 1.1, "pitch": 1.05},
            "sad": {"rate": 0.8, "pitch": 0.85},
            "urgent": {"rate": 1.3, "pitch": 1.05},
            "thoughtful": {"rate": 0.85, "pitch": 0.95},
        }

    def _check_piper(self) -> bool:
        """Check if Piper TTS is available."""
        if self._piper_checked:
            return self._piper_available
        self._piper_checked = True
        command = _resolve_piper_command()
        if command is None:
            self._piper_available = False
            self._piper_command = None
            return self._piper_available
        try:
            result = get_subprocess_gateway().run(
                [command, "--help"],
                timeout=5,
                read_only=True,
                source="voice_output_piper_probe",
                accelerator_capability="none",
            )
            self._piper_available = result.returncode == 0
            self._piper_command = command if self._piper_available else None
        except _VOICE_RECOVERABLE_ERRORS:
            self._piper_available = False
            self._piper_command = None
        return self._piper_available

    def _apply_emotion(
        self, params: VoiceOutputInput
    ) -> tuple[float, float]:
        """Apply emotion modulation to rate and pitch."""
        rate = params.rate
        pitch = params.pitch

        if params.emotion:
            emotion_key = params.emotion.lower().strip()
            mods = self._emotion_params.get(emotion_key, {})
            rate *= mods.get("rate", 1.0)
            pitch *= mods.get("pitch", 1.0)

        return rate, pitch

    async def execute(
        self, params: VoiceOutputInput, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Synthesize speech from text."""
        if isinstance(params, dict):
            try:
                params = VoiceOutputInput(**params)
            except _VOICE_RECOVERABLE_ERRORS as exc:
                _record_voice_degradation(
                    exc,
                    action="rejected invalid voice output input",
                )
                return {"ok": False, "error": f"Invalid input: {exc}"}

        text = params.text.strip()
        if not text:
            return {"ok": False, "error": "No text provided."}

        if len(text) > self._max_text_chars:
            return {
                "ok": False,
                "error": f"Text exceeds {self._max_text_chars} character limit.",
            }

        rate, pitch = self._apply_emotion(params)
        attempts: list[dict[str, str]] = []

        # Strategy 1: Sovereign Voice Engine (MLX/XTTS)
        result = await self._try_sovereign_engine(text, params, rate, pitch)
        if result:
            if result["ok"]:
                return result
            attempts.append({"engine": "sovereign", "error": result.get("error", "unknown")})

        # Strategy 2: Piper TTS (high quality, fast)
        result = await self._try_piper(text, params, rate)
        if result:
            if result["ok"]:
                return result
            attempts.append({"engine": "piper", "error": result.get("error", "unknown")})

        # Strategy 3: macOS say (built-in, reliable)
        if sys.platform == "darwin":
            result = await self._try_macos_say(text, params, rate)
            if result:
                if result["ok"]:
                    return result
                attempts.append({"engine": "macos_say", "error": result.get("error", "unknown")})

        # Strategy 4: pyttsx3 (cross-platform fallback)
        result = await self._try_pyttsx3(text, params)
        if result:
            if result["ok"]:
                return result
            attempts.append({"engine": "pyttsx3", "error": result.get("error", "unknown")})

        return {
            "ok": False,
            "error": "All TTS engines unavailable.",
            "attempts": attempts,
        }

    async def _try_sovereign_engine(
        self,
        text: str,
        params: VoiceOutputInput,
        rate: float,
        pitch: float,
    ) -> dict[str, Any] | None:
        """Try the sovereign voice engine (MLX/XTTS)."""
        try:
            from core.container import ServiceContainer
            engine = ServiceContainer.get("voice_engine", default=None)
            if not engine:
                return None

            timestamp = int(time.time())
            output_path = self._output_dir / f"voice_{timestamp}.{params.output_format}"

            if hasattr(engine, "synthesize_to_file"):
                await engine.synthesize_to_file(
                    text,
                    str(output_path),
                    voice=params.voice,
                    rate=rate,
                    pitch=pitch,
                )
            elif hasattr(engine, "synthesize_speech"):
                await engine.synthesize_speech(text)
                return {
                    "ok": True,
                    "engine": "sovereign",
                    "played": True,
                    "summary": "Speech synthesized via sovereign engine (direct playback).",
                }
            else:
                return None

            if params.play_audio:
                await self._play_audio(output_path)

            return {
                "ok": True,
                "engine": "sovereign",
                "path": str(output_path),
                "played": params.play_audio,
                "url": f"/data/voice_output/{output_path.name}",
                "summary": "Speech synthesized via sovereign engine.",
            }

        except _VOICE_RECOVERABLE_ERRORS as exc:
            _record_voice_degradation(
                exc,
                action="fell back from sovereign voice engine",
                extra={"engine": "sovereign"},
            )
            return {"ok": False, "error": str(exc)}

    async def _try_piper(
        self,
        text: str,
        params: VoiceOutputInput,
        rate: float,
    ) -> dict[str, Any] | None:
        """Try Piper TTS (fast, high-quality local TTS)."""
        if not self._check_piper():
            return None

        timestamp = int(time.time())
        output_path = self._output_dir / f"voice_{timestamp}.wav"
        voice = params.voice or "en_US-amy-medium"

        try:
            # Piper reads from stdin, writes to file
            process = await get_subprocess_gateway().spawn_async(
                [
                    self._piper_command or _resolve_piper_command() or "piper",
                    "--model",
                    voice,
                    "--output_file",
                    str(output_path),
                    "--length_scale",
                    str(1.0 / rate),
                ],
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                source="tool_execution:voice_output.piper",
                accelerator_capability="none",
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=text.encode("utf-8")),
                timeout=30.0,
            )

            if process.returncode != 0:
                detail = stderr.decode("utf-8", errors="replace").strip()
                return {"ok": False, "error": f"Piper failed: {detail}"}

            if params.play_audio:
                await self._play_audio(output_path)

            return {
                "ok": True,
                "engine": "piper",
                "voice": voice,
                "path": str(output_path),
                "played": params.play_audio,
                "url": f"/data/voice_output/{output_path.name}",
                "summary": f"Speech synthesized via Piper TTS ({voice}).",
            }

        except _VOICE_RECOVERABLE_ERRORS as exc:
            _record_voice_degradation(
                exc,
                action="fell back from Piper TTS",
                extra={"engine": "piper", "voice": voice},
            )
            return {"ok": False, "error": str(exc)}

    async def _try_macos_say(
        self,
        text: str,
        params: VoiceOutputInput,
        rate: float,
    ) -> dict[str, Any] | None:
        """Try macOS 'say' command for speech synthesis."""
        voice = params.voice or "Samantha"
        # macOS say uses words-per-minute, not multiplier
        wpm = int(185 * rate)

        timestamp = int(time.time())
        output_path = self._output_dir / f"voice_{timestamp}.aiff"

        try:
            cmd = ["say", "-v", voice, "-r", str(wpm)]

            if params.save_to_file:
                cmd.extend(["-o", str(output_path)])

            cmd.append(text)

            process = await get_subprocess_gateway().spawn_async(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                source="tool_execution:voice_output.macos_say",
                accelerator_capability="auto",
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=45.0,
            )

            if process.returncode != 0:
                detail = stderr.decode("utf-8", errors="replace").strip()
                return {"ok": False, "error": f"macOS say failed: {detail}"}

            result: dict[str, Any] = {
                "ok": True,
                "engine": "macos_say",
                "voice": voice,
                "played": not params.save_to_file,
                "summary": f"Speech synthesized via macOS ({voice}).",
            }

            if params.save_to_file and output_path.exists():
                result["path"] = str(output_path)
                result["url"] = f"/data/voice_output/{output_path.name}"

                if params.play_audio:
                    await self._play_audio(output_path)
                    result["played"] = True

            return result

        except _VOICE_RECOVERABLE_ERRORS as exc:
            _record_voice_degradation(
                exc,
                action="fell back from macOS say",
                extra={"engine": "macos_say", "voice": voice},
            )
            return {"ok": False, "error": str(exc)}

    async def _try_pyttsx3(
        self,
        text: str,
        params: VoiceOutputInput,
    ) -> dict[str, Any] | None:
        """Try pyttsx3 as a cross-platform fallback."""
        try:
            import pyttsx3
        except ImportError:
            return None

        try:
            def _synthesize():
                engine = pyttsx3.init()
                engine.setProperty("rate", int(175 * params.rate))
                if params.voice:
                    voices = engine.getProperty("voices")
                    for v in voices:
                        if params.voice.lower() in v.name.lower():
                            engine.setProperty("voice", v.id)
                            break

                if params.save_to_file:
                    timestamp = int(time.time())
                    output_path = str(
                        self._output_dir / f"voice_{timestamp}.wav"
                    )
                    engine.save_to_file(text, output_path)
                    engine.runAndWait()
                    return output_path

                engine.say(text)
                engine.runAndWait()
                return None

            output_path = await asyncio.to_thread(_synthesize)

            result: dict[str, Any] = {
                "ok": True,
                "engine": "pyttsx3",
                "played": not params.save_to_file,
                "summary": "Speech synthesized via pyttsx3.",
            }

            if output_path:
                result["path"] = output_path
                result["url"] = f"/data/voice_output/{Path(output_path).name}"

            return result

        except _VOICE_RECOVERABLE_ERRORS as exc:
            _record_voice_degradation(
                exc,
                action="pyttsx3 synthesis failed",
                extra={"engine": "pyttsx3"},
            )
            return {"ok": False, "error": str(exc)}

    async def _play_audio(self, path: Path) -> None:
        """Play an audio file using the system player."""
        if not await asyncio.to_thread(path.exists):
            return

        try:
            if sys.platform == "darwin":
                process = await get_subprocess_gateway().spawn_async(
                    ["afplay", str(path)],
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    source="tool_execution:voice_output.afplay",
                    accelerator_capability="none",
                )
                await asyncio.wait_for(process.communicate(), timeout=60.0)
            else:
                # Linux: try aplay, then paplay
                for player in ("aplay", "paplay"):
                    try:
                        process = await get_subprocess_gateway().spawn_async(
                            [player, str(path)],
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            source="tool_execution:voice_output.linux_player",
                            accelerator_capability="none",
                        )
                        await asyncio.wait_for(
                            process.communicate(), timeout=60.0
                        )
                        break
                    except FileNotFoundError:
                        continue
        except _VOICE_RECOVERABLE_ERRORS as exc:
            logger.debug("Audio playback failed: %s", exc)

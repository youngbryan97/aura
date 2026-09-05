"""Process-scoped model residency for full-duplex voice.

A socket is a conversation transport, not a model lifetime. Keeping model
ownership on the socket made every reconnect tear down Whisper, Kokoro, their
compiled kernels, and their model-lane leases. This runtime owns those heavy
resources until explicit lane eviction or process shutdown while handing each
session an isolated incremental-transcription state machine.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from typing import Any

from core.runtime.lockdep import LockRank, checked_lock
from core.voice.duplex.config import DuplexConfig
from core.voice.duplex.streaming_asr import StreamingAsr, _WhisperBackend
from core.voice.duplex.tts_stream import StreamingTts

logger = logging.getLogger("Aura.Voice.ModelRuntime")


def _config_identity(config: DuplexConfig) -> str:
    payload = {
        "asr": asdict(config.asr),
        "tts": asdict(config.tts),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class VoiceModelRuntime:
    """Heavy ASR/TTS resources shared by all sockets in this process."""

    def __init__(
        self,
        config: DuplexConfig,
        *,
        model_lane_controller: Any = None,
    ) -> None:
        self._config = config
        self._identity = _config_identity(config)
        self._asr_backend = _WhisperBackend(
            config.asr,
            model_lane_controller=model_lane_controller,
        )
        self._tts = StreamingTts(
            config.tts,
            model_lane_controller=model_lane_controller,
        )
        self._closed = False
        self._lock = checked_lock("voice.model_runtime.state", rank=LockRank.LEAF)

    @property
    def config_identity(self) -> str:
        return self._identity

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def tts(self) -> StreamingTts:
        if self._closed:
            raise RuntimeError("voice_model_runtime_closed")
        return self._tts

    def new_asr(self) -> StreamingAsr:
        """Return isolated turn state over the shared immutable model backend."""
        if self._closed:
            raise RuntimeError("voice_model_runtime_closed")
        return StreamingAsr(
            self._config.asr,
            backend=self._asr_backend,
            owns_backend=False,
        )

    def status(self) -> dict[str, Any]:
        return {
            "schema": "aura.voice.process_model_runtime.v1",
            "config_identity": self._identity,
            "closed": self._closed,
            "asr": self._asr_backend.status(),
            "tts": self._tts.status(),
        }

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._asr_backend.shutdown()
        self._tts.shutdown()


_RUNTIME: VoiceModelRuntime | None = None
_RUNTIME_LOCK = checked_lock("voice.model_runtime.singleton", rank=LockRank.LEAF)


def get_voice_model_runtime(config: DuplexConfig | None = None) -> VoiceModelRuntime:
    """Return the process model owner; configuration is immutable after boot."""
    global _RUNTIME
    resolved = config or DuplexConfig.load()
    identity = _config_identity(resolved)
    with _RUNTIME_LOCK:
        if _RUNTIME is not None:
            if _RUNTIME.closed:
                _RUNTIME = None
            elif _RUNTIME.config_identity != identity:
                logger.warning(
                    "Voice model configuration changed after residency; retaining the "
                    "boot-bound model runtime until process restart"
                )
            if _RUNTIME is not None:
                return _RUNTIME

        runtime = VoiceModelRuntime(resolved)
        from core.runtime.runtime_hygiene import get_runtime_hygiene

        get_runtime_hygiene().register_shutdown_resource(
            runtime,
            kind="voice_model_runtime",
            name="duplex-voice-model-runtime",
            source="core.voice.duplex.model_runtime",
            closer=runtime.shutdown,
            timeout_s=3.0,
            required=True,
            blocking=True,
        )
        _RUNTIME = runtime
        return runtime


def reset_voice_model_runtime_for_tests() -> None:
    """Release the singleton between hermetic tests."""
    global _RUNTIME
    with _RUNTIME_LOCK:
        runtime, _RUNTIME = _RUNTIME, None
    if runtime is not None:
        runtime.shutdown()


def voice_model_runtime_status() -> dict[str, Any]:
    """Inspect residency without constructing or loading the runtime."""
    with _RUNTIME_LOCK:
        runtime = _RUNTIME
    if runtime is None:
        return {
            "schema": "aura.voice.process_model_runtime.v1",
            "present": False,
            "closed": False,
        }
    return {"present": True, **runtime.status()}


__all__ = [
    "VoiceModelRuntime",
    "get_voice_model_runtime",
    "reset_voice_model_runtime_for_tests",
    "voice_model_runtime_status",
]

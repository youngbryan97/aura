"""core/voice/duplex/tts_stream.py — Streaming synthesis with instant cancel.

Two properties matter more than anything else here.

**Time to first audio.** Perceived latency is when she *starts* speaking,
not when she finishes. So the first clause is cut short (clause_chunker),
synthesised alone, and pushed out while the rest is still being generated.
Measured on this host: Kokoro-82M runs 6–8.6x realtime, ~190 ms for a short
clause, which is the floor this lane can hit.

**Cancellation.** Barge-in is worthless if it takes a second to take
effect. Every synthesis job checks a cancellation token between chunks, and
the session flushes the client's playback buffer independently — so audio
stops even if a chunk is mid-flight through the ONNX graph.

Kokoro is ONNX on CPU, which matters on this host: the resident 32B holds
~20 GB of GPU memory, and a TTS engine competing for Metal would show up as
jitter in her actual thinking. This one does not touch the GPU at all.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.runtime.errors import record_degradation
from core.runtime.lockdep import LockRank, checked_lock
from core.utils.task_tracker import get_task_tracker
from core.voice.duplex.config import OUTPUT_RATE, TtsConfig
from core.voice.duplex.prosody import ProsodySpec

logger = logging.getLogger("Aura.Voice.Tts")


class CancellationToken:
    """One-shot cancel flag, checked between and inside synthesis stages."""

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def reset(self) -> None:
        self._event.clear()


@dataclass(slots=True)
class SynthesisResult:
    samples: np.ndarray          # float32 mono @ OUTPUT_RATE
    sample_rate: int
    text: str
    synth_ms: float
    engine: str

    @property
    def duration_s(self) -> float:
        return len(self.samples) / float(self.sample_rate or OUTPUT_RATE)


class _KokoroEngine:
    """Kokoro-82M via ONNX Runtime. Primary engine."""

    name = "kokoro"

    def __init__(self, config: TtsConfig) -> None:
        self._config = config
        self._kokoro: Any = None
        self._lock = checked_lock("voice.tts.__kokoro_engine", rank=LockRank.LEAF)
        self._rate = 24_000
        self._available = False
        self._voices: frozenset[str] = frozenset()

    def load(self) -> bool:
        model = Path(self._config.model_path)
        voices = Path(self._config.voices_path)
        if not model.is_file() or not voices.is_file():
            logger.warning(
                "Kokoro assets missing (model=%s voices=%s); run tools/fetch_voice_models.py",
                model.exists(),
                voices.exists(),
            )
            return False
        try:
            from kokoro_onnx import Kokoro

            self._kokoro = Kokoro(str(model), str(voices))
            try:
                self._voices = frozenset(self._kokoro.get_voices())
            except (AttributeError, RuntimeError, TypeError) as exc:
                logger.debug("Kokoro voice enumeration unavailable: %s", exc)
                self._voices = frozenset()
            self._available = True
            logger.info("Kokoro TTS loaded (%d voices)", len(self._voices))
            return True
        except (ImportError, OSError, RuntimeError, ValueError, AttributeError) as exc:
            record_degradation(
                "voice_duplex.tts",
                exc,
                action="Kokoro unavailable; falling back to Piper",
            )
            return False

    @property
    def available(self) -> bool:
        return self._available

    @property
    def voices(self) -> frozenset[str]:
        return self._voices

    def resolve_voice(self, requested: str) -> str:
        if not self._voices or requested in self._voices:
            return requested
        logger.warning("Voice %r not in Kokoro pack; using %r", requested, self._config.voice)
        return self._config.voice if self._config.voice in self._voices else sorted(self._voices)[0]

    def synthesize(self, text: str, spec: ProsodySpec) -> tuple[np.ndarray, int]:
        # The ONNX session is not re-entrant; concurrent create() calls
        # corrupt each other's output buffers.
        with self._lock:
            samples, rate = self._kokoro.create(
                text,
                voice=self.resolve_voice(spec.voice),
                speed=float(spec.speed),
                lang="en-us",
            )
        return np.asarray(samples, dtype=np.float32), int(rate)


class _PiperEngine:
    """Piper fallback. Robotic next to Kokoro but very fast and dependency-light."""

    name = "piper"

    def __init__(self, config: TtsConfig) -> None:
        self._config = config
        self._voice: Any = None
        self._lock = checked_lock("voice.tts.__piper_engine", rank=LockRank.LEAF)
        self._rate = 22_050
        self._available = False

    def load(self) -> bool:
        try:
            from piper import PiperVoice

            root = Path.home() / ".aura/live-source/data/voice_models/piper_voices"
            models = sorted(root.glob("*.onnx")) if root.is_dir() else []
            if not models:
                return False
            self._voice = PiperVoice.load(str(models[0]))
            self._rate = int(getattr(getattr(self._voice, "config", None), "sample_rate", 22_050))
            self._available = True
            logger.info("Piper TTS loaded: %s", models[0].name)
            return True
        except (ImportError, OSError, RuntimeError, ValueError, AttributeError, IndexError) as exc:
            record_degradation(
                "voice_duplex.tts",
                exc,
                action="Piper unavailable; falling back to system speech",
                severity="warning",
            )
            return False

    @property
    def available(self) -> bool:
        return self._available

    def synthesize(self, text: str, spec: ProsodySpec) -> tuple[np.ndarray, int]:
        with self._lock:
            chunks: list[np.ndarray] = []
            for audio in self._voice.synthesize(text):
                raw = getattr(audio, "audio_int16_bytes", None)
                if raw is None:
                    raw = bytes(audio)
                chunks.append(np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0)
        if not chunks:
            return np.zeros(0, dtype=np.float32), self._rate
        return np.concatenate(chunks), self._rate


class _ClonedVoiceEngine:
    """XTTS-v2 zero-shot cloning from a reference clip.

    Opt-in, and off by default, because the trade is real and steep. Kokoro
    reaches first audio in ~190 ms; XTTS is roughly realtime on this host, so
    a cloned voice costs one to three seconds of extra latency on every reply.
    That is the difference between a phone call and a walkie-talkie, which is
    precisely the quality the rest of this lane exists to protect.

    Enable it when the specific voice matters more than the responsiveness —
    and note it only speaks as well as the reference clip: 6-20 seconds of
    clean, single-speaker audio with no music or background noise.
    """

    name = "xtts_clone"

    def __init__(self, config: TtsConfig) -> None:
        self._config = config
        self._tts: Any = None
        self._lock = checked_lock("voice.tts.__cloned_voice_engine", rank=LockRank.LEAF)
        self._rate = 24_000
        self._available = False
        self._reference: Path | None = None

    def load(self) -> bool:
        from core.voice.duplex import coqui_compat

        reference = Path(self._config.clone_reference or "")
        if not reference.is_file():
            logger.info("Cloned voice requested but reference clip %s is missing", reference)
            return False

        if not coqui_compat.license_accepted():
            # XTTS-v2 is CPML-licensed. Accepting on the operator's behalf is
            # not this code's call, so fail closed with a clear reason.
            logger.warning(
                "Cloned voice disabled: XTTS-v2 is CPML-licensed and no acceptance "
                "flag is set (COQUI_TOS_AGREED / AURA_COQUI_CPML_ACCEPTED)."
            )
            return False

        if not coqui_compat.apply():
            return False

        try:
            from TTS.api import TTS  # noqa: N811 — upstream class name

            self._tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            self._reference = reference
            self._available = True
            logger.info("XTTS cloned voice loaded from %s", reference.name)
            return True
        except (ImportError, OSError, RuntimeError, ValueError, AttributeError, TypeError) as exc:
            record_degradation(
                "voice_duplex.tts",
                exc,
                action="cloned voice unavailable; using preset voices",
                severity="warning",
            )
            return False

    @property
    def available(self) -> bool:
        return self._available

    def synthesize(self, text: str, spec: ProsodySpec) -> tuple[np.ndarray, int]:
        with self._lock:
            wav = self._tts.tts(
                text=text,
                speaker_wav=str(self._reference),
                language="en",
                speed=float(spec.speed),
            )
        return np.asarray(wav, dtype=np.float32), self._rate


def _resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Rate-convert to the lane's output rate.

    Uses polyphase resampling when scipy is present because linear
    interpolation aliases audibly on speech; falls back to interpolation
    rather than failing, since a slightly harsh voice beats no voice.
    """
    if src_rate == dst_rate or samples.size == 0:
        return samples
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(src_rate), int(dst_rate))
        return resample_poly(samples, dst_rate // g, src_rate // g).astype(np.float32)
    except (ImportError, ValueError, RuntimeError) as exc:
        record_degradation(
            "voice_duplex.tts",
            exc,
            action="resampled with linear interpolation",
            severity="debug",
        )
        ratio = dst_rate / float(src_rate)
        n = int(round(samples.size * ratio))
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        idx = np.linspace(0, samples.size - 1, n, dtype=np.float64)
        return np.interp(idx, np.arange(samples.size), samples).astype(np.float32)


@dataclass(slots=True)
class _EngineState:
    clone: _ClonedVoiceEngine | None = None
    kokoro: _KokoroEngine | None = None
    piper: _PiperEngine | None = None
    loaded: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class StreamingTts:
    """Chunked synthesis with pipelining and cancellation."""

    def __init__(
        self,
        config: TtsConfig | None = None,
        *,
        model_lane_controller: Any = None,
    ) -> None:
        self._config = config or TtsConfig()
        self._state = _EngineState()
        self._model_lane_controller = model_lane_controller
        self._lane_lease: Any = None
        self._lifecycle_lock = checked_lock("voice.tts.lifecycle", rank=LockRank.LEAF)
        self._active_syntheses = 0
        self._accepting_synthesis = False
        self._closing = False
        self._load_retry_after = 0.0
        self._warm_task: asyncio.Task[bool] | None = None
        self._warmed = False
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, self._config.workers),
            thread_name_prefix="aura-tts",
        )

    def _acquire_model_lane(self) -> Any:
        with self._lifecycle_lock:
            if self._lane_lease is not None:
                return self._lane_lease
            if self._closing:
                raise RuntimeError("voice TTS is closing")

        from core.runtime.model_lane_control import (
            acquire_synchronous_in_process_model_lane,
        )

        clone_enabled = bool(
            self._config.prefer_clone and self._config.clone_reference
        )
        candidate = acquire_synchronous_in_process_model_lane(
            owner_id=f"voice-duplex-tts:{id(self)}",
            model_path=(
                "voice-duplex/tts/xtts-kokoro-piper"
                if clone_enabled
                else "voice-duplex/tts/kokoro-piper"
            ),
            purpose="serve",
            request_gb=3.0 if clone_enabled else 0.75,
            priority=30,
            preemptible=False,
            evict=self._evict_model_lane,
            compensate=self._compensate_model_lane,
            metadata={
                "engine": "voice_duplex",
                "model_role": "tts",
                "clone_enabled": clone_enabled,
                "lifecycle_state": "loading",
            },
            controller=self._model_lane_controller,
        )

        release_reason = ""
        with self._lifecycle_lock:
            if self._closing:
                release_reason = "voice_tts_closed_during_model_admission"
                authoritative = None
            elif self._lane_lease is None:
                self._lane_lease = candidate
                return candidate
            else:
                release_reason = "voice_tts_duplicate_model_admission"
                authoritative = self._lane_lease

        candidate.release(reason=release_reason)
        if authoritative is not None:
            return authoritative
        raise RuntimeError("voice TTS closed during model admission")

    def _detach_model_lane_locked(self) -> tuple[Any, bool]:
        self._accepting_synthesis = False
        self._state.clone = None
        self._state.kokoro = None
        self._state.piper = None
        self._state.loaded = False
        self._warmed = False
        lease, self._lane_lease = self._lane_lease, None
        return lease, self._active_syntheses == 0 and self._lane_lease is None

    def _release_model_lane_if_idle(self, *, reason: str) -> bool:
        with self._lifecycle_lock:
            if self._active_syntheses:
                return False
            lease, released = self._detach_model_lane_locked()
        if lease is not None:
            lease.release(reason=reason)
        return released

    async def _evict_model_lane(self, _owner: Any, reason: str) -> bool:
        released = await asyncio.to_thread(
            self._release_model_lane_if_idle,
            reason=f"voice_tts_lane_eviction:{reason}",
        )
        if not released:
            logger.warning("TTS model preemption refused during active synthesis: %s", reason)
        return released

    async def _compensate_model_lane(self, _owner: Any, reason: str) -> bool:
        logger.info("Restoring duplex TTS after failed model candidate: %s", reason)
        return await self.ensure_loaded()

    async def ensure_loaded(self) -> bool:
        """Load engines once, off the event loop."""
        async with self._state.lock:
            if self._state.loaded:
                return self.available
            if self._closing:
                return False
            if time.monotonic() < self._load_retry_after:
                return False
            loop = asyncio.get_running_loop()
            try:
                lease = await loop.run_in_executor(self._pool, self._acquire_model_lane)
            except (
                AttributeError,
                OSError,
                RuntimeError,
                TimeoutError,
                TypeError,
                ValueError,
            ) as exc:
                record_degradation(
                    "voice_duplex.tts",
                    exc,
                    action="neural TTS model admission failed; no unowned model was loaded",
                )
                self._state.loaded = False
                self._load_retry_after = time.monotonic() + 2.0
                return False

            # Cloning first only when explicitly preferred: it is the slowest
            # engine by a wide margin, so it must never be picked by accident.
            if self._config.prefer_clone and self._config.clone_reference:
                clone = _ClonedVoiceEngine(self._config)
                if await loop.run_in_executor(self._pool, clone.load):
                    self._state.clone = clone

            kokoro = _KokoroEngine(self._config)
            if await loop.run_in_executor(self._pool, kokoro.load):
                self._state.kokoro = kokoro

            if self._state.kokoro is None:
                piper = _PiperEngine(self._config)
                if await loop.run_in_executor(self._pool, piper.load):
                    self._state.piper = piper

            self._state.loaded = True

            neural_available = bool(
                self._state.clone or self._state.kokoro or self._state.piper
            )
            if neural_available:
                if not lease.set_preemptible(True):
                    with self._lifecycle_lock:
                        detached, _released = self._detach_model_lane_locked()
                    if detached is not None:
                        await asyncio.to_thread(
                            detached.release,
                            reason="voice_tts_model_activation_fence_lost",
                        )
                    self._load_retry_after = time.monotonic() + 2.0
                    logger.error("TTS model lane lost its activation fence")
                    return False
                with self._lifecycle_lock:
                    self._accepting_synthesis = True
                self._load_retry_after = 0.0
            else:
                with self._lifecycle_lock:
                    detached, _released = self._detach_model_lane_locked()
                if detached is not None:
                    await asyncio.to_thread(
                        detached.release,
                        reason="voice_tts_no_neural_engine_loaded",
                    )
                # The configured assets are absent or unusable. This is a
                # stable state until settings/files change, not a reason to
                # retry model admission on every spoken clause.
                self._state.loaded = True

            if not self.available:
                logger.error("No TTS engine available — the voice lane cannot speak")
            return self.available

    @property
    def available(self) -> bool:
        return bool(
            self._state.clone
            or self._state.kokoro
            or self._state.piper
        )

    def available_voices(self) -> list[str]:
        """Voices the loaded engine can actually produce."""
        kokoro = self._state.kokoro
        if kokoro is not None:
            return sorted(kokoro.voices)
        return []

    @property
    def engine_name(self) -> str:
        if self._state.clone:
            return "xtts_clone"
        if self._state.kokoro:
            return "kokoro"
        if self._state.piper:
            return "piper"
        return "none"

    def status(self) -> dict[str, Any]:
        with self._lifecycle_lock:
            return {
                "schema": "aura.voice.tts_model_runtime.v1",
                "engine": self.engine_name,
                "loaded": self._state.loaded,
                "warmed": self._warmed,
                "model_lane_owned": self._lane_lease is not None,
                "accepting_synthesis": self._accepting_synthesis,
                "active_syntheses": self._active_syntheses,
                "closing": self._closing,
            }

    async def warm_up(self, spec: ProsodySpec) -> bool:
        """Run one throwaway synthesis so the first real one is not the cold one.

        Measured cold-start on this host is ~635 ms versus ~190 ms warm —
        the difference between a natural opening and an awkward one.
        """
        if self._warmed and self.available:
            return True
        task = self._warm_task
        if task is None or task.done():
            task = get_task_tracker().create_task(
                self._run_warm_up(spec),
                name="VoiceTTS.warm_up",
            )
            self._warm_task = task
        try:
            return await asyncio.shield(task)
        finally:
            if task.done() and self._warm_task is task:
                self._warm_task = None

    async def _run_warm_up(self, spec: ProsodySpec) -> bool:
        if not await self.ensure_loaded():
            return False
        try:
            result = await self.synthesize("Okay.", spec, CancellationToken())
        except (RuntimeError, ValueError, OSError) as exc:
            await asyncio.to_thread(
                record_degradation,
                "voice_duplex.tts",
                exc,
                action="warmup synthesis failed; first utterance pays cold-start cost",
                severity="warning",
            )
            return False
        with self._lifecycle_lock:
            if self._closing:
                return False
            self._warmed = result is not None
            return self._warmed

    async def synthesize(
        self,
        text: str,
        spec: ProsodySpec,
        token: CancellationToken,
    ) -> SynthesisResult | None:
        """Synthesise one chunk. Returns None if cancelled or empty."""
        text = (text or "").strip()
        if not text or token.cancelled:
            return None
        # The synthesiser pronounces characters, not meaning: "$1.5B", "45%",
        # "2026-07-27" and "https://…" all reach the listener as noise unless
        # something turns them into the words a person would say. Kokoro takes
        # no SSML, so this is the only channel there is — and it is also where
        # a speaker's pauses get put, since punctuation is the only pacing
        # instrument the model exposes.
        try:
            from core.voice.duplex.spoken_form import prepare_for_speech

            spoken = prepare_for_speech(text)
            if spoken.strip():
                text = spoken
        except (RuntimeError, ValueError, TypeError, ImportError, AttributeError) as exc:
            record_degradation(
                "voice_duplex.tts",
                exc,
                severity="warning",
                action="synthesised the raw clause without spoken-form normalisation",
            )
        if not await self.ensure_loaded():
            return None

        started = time.perf_counter()
        loop = asyncio.get_running_loop()
        release_deferred_to_native_completion = False
        with self._lifecycle_lock:
            if not self._accepting_synthesis:
                return None
            self._active_syntheses += 1
            engines = (
                self._state.clone,
                self._state.kokoro,
                self._state.piper,
            )

        try:
            for engine in engines:
                if engine is None:
                    continue
                try:
                    synthesis = loop.run_in_executor(
                        self._pool, engine.synthesize, text, spec
                    )
                    try:
                        samples, rate = await asyncio.shield(synthesis)
                    except asyncio.CancelledError:
                        # A Python cancellation cannot stop a running native
                        # synthesis thread. Transfer the active-count release to
                        # its completion callback so repeated cancellation
                        # cannot release model ownership while native code is
                        # still using the weights.
                        release_deferred_to_native_completion = True
                        synthesis.add_done_callback(self._cancelled_synthesis_done)
                        raise
                except (RuntimeError, ValueError, OSError, AttributeError, MemoryError) as exc:
                    record_degradation(
                        "voice_duplex.tts",
                        exc,
                        action=f"{engine.name} synthesis failed; trying next engine",
                        severity="warning",
                    )
                    continue

                if token.cancelled:
                    # Interrupted while the graph was running. Discard rather
                    # than play stale audio over the user.
                    return None

                samples = _resample(samples, rate, OUTPUT_RATE)
                if spec.gain != 1.0:
                    samples = samples * float(spec.gain)
                if spec.trailing_pause_ms > 0:
                    pad = int(OUTPUT_RATE * spec.trailing_pause_ms / 1000.0)
                    if pad > 0:
                        samples = np.concatenate(
                            (samples, np.zeros(pad, dtype=np.float32))
                        )

                return SynthesisResult(
                    samples=samples.astype(np.float32, copy=False),
                    sample_rate=OUTPUT_RATE,
                    text=text,
                    synth_ms=(time.perf_counter() - started) * 1000.0,
                    engine=engine.name,
                )

            return None
        finally:
            if not release_deferred_to_native_completion:
                self._finish_synthesis()

    def _finish_synthesis(self) -> None:
        detached = None
        with self._lifecycle_lock:
            self._active_syntheses = max(0, self._active_syntheses - 1)
            if self._closing and self._active_syntheses == 0:
                detached, _released = self._detach_model_lane_locked()
        if detached is not None:
            detached.release(reason="voice_tts_shutdown")

    def _cancelled_synthesis_done(self, synthesis: asyncio.Future[Any]) -> None:
        try:
            synthesis.result()
        except (
            asyncio.CancelledError,
            AttributeError,
            MemoryError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            logger.debug("Cancelled native TTS synthesis finished with: %s", exc)
        finally:
            self._finish_synthesis()

    async def stream(
        self,
        chunks: AsyncIterator[str],
        spec: ProsodySpec,
        token: CancellationToken,
    ) -> AsyncIterator[SynthesisResult]:
        """Synthesise an async stream of text chunks, one chunk ahead.

        Pipelining matters: synthesising chunk N+1 while chunk N is playing
        keeps the audio gapless. Without it there is an audible seam at every
        clause boundary, which is exactly the artefact that makes streaming
        TTS sound synthetic.
        """
        pending: asyncio.Task[SynthesisResult | None] | None = None

        async def _synth(text: str) -> SynthesisResult | None:
            return await self.synthesize(text, spec, token)

        try:
            async for chunk in chunks:
                if token.cancelled:
                    break
                text = (chunk or "").strip()
                if not text:
                    continue
                task = get_task_tracker().create_task(
                    _synth(text),
                    name="VoiceTTSStream.synthesize_chunk",
                )
                if pending is not None:
                    result = await pending
                    if token.cancelled:
                        task.cancel()
                        break
                    if result is not None:
                        yield result
                pending = task

            if pending is not None and not token.cancelled:
                result = await pending
                pending = None
                if result is not None and not token.cancelled:
                    yield result
        finally:
            if pending is not None and not pending.done():
                pending.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await pending

    def shutdown(self) -> None:
        detached = None
        with self._lifecycle_lock:
            self._closing = True
            self._accepting_synthesis = False
            if self._active_syntheses == 0:
                detached, _released = self._detach_model_lane_locked()
        if detached is not None:
            detached.release(reason="voice_tts_shutdown")
        self._pool.shutdown(wait=False, cancel_futures=True)

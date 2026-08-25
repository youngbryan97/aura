"""core/voice/duplex/streaming_asr.py — Incremental transcription.

Whisper is not a streaming model: it decodes a window and can revise any
word in it. Showing raw output live produces text that visibly rewrites
itself, and endpointing on it is worse than useless.

The fix is LocalAgreement-2 (Liu et al.): decode the growing buffer
repeatedly and treat as *stable* only the word prefix on which the last two
independent decodes agree. Whisper is free to revise the tail; the prefix
does not move. Live captions render the stable prefix solidly and the tail
faintly, and endpointing reasons only over the stable part.

Measured on this host: partial decode (small.en) ~72 ms, final decode
(large-v3-turbo) ~195 ms.
"""
from __future__ import annotations

import asyncio
import gc
import importlib

# `import importlib` does NOT bind importlib.util — the submodule must be
# imported explicitly. Every find_spec() call below sat inside an
# `except AttributeError`, so on any interpreter where nothing else had
# imported importlib.util first, backend detection silently returned "not
# installed" and the whole lane degraded without a word. Import it.
import importlib.util
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.runtime.errors import record_degradation
from core.runtime.lockdep import LockRank, checked_async_lock, checked_lock
from core.runtime.third_party_imports import (
    import_attribute_serialized,
    import_module_serialized,
)
from core.voice.duplex.config import CAPTURE_RATE, AsrConfig

logger = logging.getLogger("Aura.Voice.Asr")

# mlx-whisper exposes one process-global ModelHolder. Without fencing and
# restoring it, alternating a small partial model and a large final model
# reloads weights on every transition. The lock also prevents independent
# voice sessions from swapping the holder while another decode is in flight.
_MLX_HOLDER_LOCK = checked_lock("voice.asr.mlx_holder", rank=LockRank.LEAF)

# Whisper's canonical outputs for "the mic was on but nobody spoke". It
# produces these confidently on silence, so they must never become a turn.
_HALLUCINATION_PATTERNS = (
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "subscribe to",
    "you you you",
    "[blank_audio]",
    "[ silence ]",
    "(silence)",
    "♪",
)


def _normalise_words(text: str) -> list[str]:
    """Word list for prefix comparison.

    Case and punctuation are stripped for *matching* only — Whisper's
    capitalisation and commas legitimately change as context grows, and
    treating that as disagreement would keep the stable prefix empty.
    """
    return [w for w in re.split(r"\s+", text.strip()) if w]


def _match_key(word: str) -> str:
    return re.sub(r"[^\w']", "", word).lower()


def _common_prefix_len(a: list[str], b: list[str]) -> int:
    n = 0
    # strict=False is the point: the two hypotheses are deliberately
    # different lengths, and the shorter one bounds the agreed prefix.
    for x, y in zip(a, b, strict=False):
        if _match_key(x) != _match_key(y):
            break
        n += 1
    return n


def looks_hallucinated(text: str) -> bool:
    """True when the text is Whisper's silence-filler rather than speech."""
    low = text.strip().lower()
    if not low:
        return True
    if len(low) <= 2 and low not in ("hi", "no", "ok", "ye"):
        return True
    return any(p in low for p in _HALLUCINATION_PATTERNS)


@dataclass(slots=True)
class Transcript:
    """One incremental result."""

    stable: str = ""
    tentative: str = ""
    is_final: bool = False
    decode_ms: float = 0.0
    audio_s: float = 0.0

    @property
    def full(self) -> str:
        return " ".join(p for p in (self.stable.strip(), self.tentative.strip()) if p)


def _is_parakeet_repo(repo: str) -> bool:
    return "parakeet" in str(repo or "").lower()


def _parakeet_available() -> bool:
    try:
        return importlib.util.find_spec("parakeet_mlx") is not None
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        logger.debug("parakeet-mlx unavailable: %s", exc)
        return False


class _WhisperBackend:
    """Parakeet if installed, mlx-whisper next, faster-whisper last.

    Model handles are cached per repo id because construction cost is
    dominated by weight load and Metal kernel compilation (measured 13–35 s
    cold), which must never happen inside a live turn.
    """

    def __init__(
        self,
        config: AsrConfig,
        *,
        model_lane_controller: Any = None,
    ) -> None:
        self._config = config
        self._impl = ""
        self._mlx: Any = None
        self._mlx_holder: Any = None
        self._mlx_models: dict[str, Any] = {}
        self._cache: dict[str, Any] = {}
        self._cache_lock = checked_lock("voice.asr.cache", rank=LockRank.LEAF)
        self._usage_lock = checked_lock("voice.asr.usage", rank=LockRank.LEAF)
        self._warm_lock = checked_lock("voice.asr.warm", rank=LockRank.LEAF)
        self._warmed_repos: set[str] = set()
        self._lane_lock = checked_lock("voice.asr.lane", rank=LockRank.LEAF)
        self._lane_lease: Any = None
        self._model_lane_controller = model_lane_controller
        self._parakeet: Any = None
        self._parakeet_models: dict[str, Any] = {}
        # Parakeet first: measured 166 ms on 12.4 s of speech against
        # small.en's 193 ms and large-v3-turbo's 317 ms (M5 Pro, 12 Aug 2026,
        # real speech with a known transcript). One decode is cheaper than the
        # incumbent PARTIAL model, which is what lets one model serve both
        # stages. Whisper remains the fallback chain, unchanged.
        if _parakeet_available():
            self._impl = "parakeet"
        else:
            try:
                mlx_spec = importlib.util.find_spec("mlx_whisper")
            except (ImportError, AttributeError, RuntimeError, ValueError):
                mlx_spec = None
            if mlx_spec is not None:
                self._impl = "mlx"

    @property
    def available(self) -> bool:
        return bool(self._impl) or self._faster_whisper_available()

    def status(self) -> dict[str, Any]:
        """Non-loading lifecycle receipt for health and operator surfaces."""
        return {
            "schema": "aura.voice.asr_model_runtime.v1",
            "backend": self._impl or "unavailable",
            "native_module_loaded": (
                self._parakeet is not None
                or self._mlx is not None
                or self._impl == "faster"
            ),
            "model_lane_owned": self._lane_lease is not None,
            "retained_models": sorted({*self._mlx_models, *self._parakeet_models}),
            "warmed_models": sorted(self._warmed_repos),
        }

    @staticmethod
    def _faster_whisper_available() -> bool:
        try:
            return importlib.util.find_spec("faster_whisper") is not None
        except (ImportError, OSError, RuntimeError, ValueError) as _exc:
            logger.debug("faster-whisper unavailable: %s", _exc)
            return False

    def _ensure_impl_loaded(self) -> None:
        """Import native ASR code only on the off-event-loop decode thread."""
        if self._impl == "parakeet" and self._parakeet is not None:
            return
        if self._impl == "parakeet":
            try:
                self._parakeet = importlib.import_module("parakeet_mlx")
                return
            except (ImportError, OSError, RuntimeError, AttributeError) as exc:
                record_degradation(
                    "voice_duplex.asr",
                    exc,
                    action="parakeet-mlx unavailable; falling back to whisper",
                    severity="warning",
                )
                self._parakeet = None
                self._impl = "mlx" if importlib.util.find_spec("mlx_whisper") else ""
        if self._impl == "mlx" and self._mlx is not None:
            return
        if self._impl == "mlx":
            try:
                self._mlx = import_module_serialized("mlx_whisper")
                transcribe_module = import_module_serialized("mlx_whisper.transcribe")
                self._mlx_holder = getattr(transcribe_module, "ModelHolder", None)
                if self._mlx_holder is None:
                    logger.warning(
                        "mlx-whisper model holder unavailable; model alternation may reload"
                    )
                return
            except (ImportError, OSError, RuntimeError, AttributeError) as exc:
                record_degradation(
                    "voice_duplex.asr",
                    exc,
                    action="mlx-whisper unavailable; trying faster-whisper",
                    severity="warning",
                )
                self._mlx = None
                self._mlx_holder = None
                self._impl = ""
        if not self._impl and self._faster_whisper_available():
            self._impl = "faster"

    def _faster_model(self, repo: str) -> Any:
        with self._cache_lock:
            key = f"fw::{repo}"
            model = self._cache.get(key)
            if model is None:
                whisper_model_cls = import_attribute_serialized(
                    "faster_whisper",
                    "WhisperModel",
                )

                # Repo ids carry an mlx-community prefix that faster-whisper
                # does not understand; fall back to a size it does.
                size = "small.en" if "small" in repo else "large-v3"
                model = whisper_model_cls(size, device="cpu", compute_type="int8")
                self._cache[key] = model
            return model

    @staticmethod
    def _footprint_gb(repo: str) -> float:
        lowered = str(repo or "").lower()
        if "parakeet" in lowered:
            # 0.6B in bf16 plus mel/activation headroom; 2.3 GB on disk.
            return 1.5
        if "large" in lowered:
            return 4.0
        if "medium" in lowered:
            return 2.0
        if "small" in lowered:
            return 1.0
        if "tiny" in lowered:
            return 0.25
        return 0.5

    def _acquire_model_lane(self) -> tuple[Any, bool]:
        with self._lane_lock:
            if self._lane_lease is not None:
                return self._lane_lease, False
            from core.runtime.model_lane_control import (
                acquire_synchronous_in_process_model_lane,
            )

            models = (self._config.partial_model, self._config.final_model)
            request_gb = sum(self._footprint_gb(repo) for repo in dict.fromkeys(models))
            lease = acquire_synchronous_in_process_model_lane(
                owner_id=f"voice-duplex-asr:{id(self)}",
                model_path="voice-duplex/asr/" + "+".join(dict.fromkeys(models)),
                purpose="serve",
                request_gb=request_gb,
                priority=30,
                preemptible=False,
                evict=self._evict_model_lane,
                compensate=self._compensate_model_lane,
                metadata={
                    "engine": "voice_duplex",
                    "model_role": "asr",
                    "backend": self._impl or "unavailable",
                    "lifecycle_state": "loading",
                },
                controller=self._model_lane_controller,
            )
            self._lane_lease = lease
            return lease, True

    def _release_model_lane_locked(self, *, reason: str) -> bool:
        with _MLX_HOLDER_LOCK:
            holder = self._mlx_holder
            retained = tuple(self._mlx_models.values())
            if (
                holder is not None
                and getattr(holder, "model", None) is not None
                and any(getattr(holder, "model", None) is model for model in retained)
            ):
                holder.model = None
                holder.model_path = None
            self._mlx_models.clear()
            # Parakeet handles are plain MLX modules with no global holder,
            # but they hold weights just the same — an eviction that released
            # the lane while leaving these resident would hand back memory it
            # was still using.
            retained = retained + tuple(self._parakeet_models.values())
            self._parakeet_models.clear()
            self._warmed_repos.clear()
        if retained:
            gc.collect()
            try:
                mlx_core = importlib.import_module("mlx.core")
                clear_cache = getattr(mlx_core, "clear_cache", None)
                if callable(clear_cache):
                    clear_cache()
            except (ImportError, OSError, RuntimeError, AttributeError) as exc:
                logger.debug("MLX cache release after ASR eviction was unavailable: %s", exc)
        with self._cache_lock:
            self._cache.clear()
        with self._lane_lock:
            lease, self._lane_lease = self._lane_lease, None
        if lease is not None:
            lease.release(reason=reason)
        return not self._cache and self._lane_lease is None

    def _release_model_lane_if_idle(self, *, reason: str) -> bool:
        if not self._usage_lock.acquire(blocking=False):
            return False
        try:
            return self._release_model_lane_locked(reason=reason)
        finally:
            self._usage_lock.release()

    async def _evict_model_lane(self, _owner: Any, reason: str) -> bool:
        released = await asyncio.to_thread(
            self._release_model_lane_if_idle,
            reason=f"voice_asr_lane_eviction:{reason}",
        )
        if not released:
            logger.warning("ASR model preemption refused during active decode: %s", reason)
        return released

    async def _compensate_model_lane(self, _owner: Any, reason: str) -> bool:
        logger.info("Restoring duplex ASR after failed model candidate: %s", reason)
        await asyncio.to_thread(self.warm, self._config.partial_model)
        await asyncio.to_thread(self.warm, self._config.final_model)
        return self._lane_lease is not None

    def transcribe(self, audio: np.ndarray, repo: str) -> str:
        """Blocking decode. Always called on a worker thread."""
        with self._usage_lock:
            self._ensure_impl_loaded()
            if not self._impl:
                raise RuntimeError("no voice ASR backend is installed")
            lease, acquired = self._acquire_model_lane()
            try:
                if self._impl == "parakeet":
                    text = self._transcribe_parakeet(audio, repo)
                elif self._impl == "mlx":
                    result = self._transcribe_mlx(audio, repo)
                    text = str(result.get("text", "") or "")
                else:
                    model = self._faster_model(repo)
                    segments, _info = model.transcribe(
                        audio, beam_size=1, language=self._config.language
                    )
                    text = "".join(seg.text for seg in segments)
                if not lease.set_preemptible(True):
                    raise RuntimeError("voice ASR model-lane activation fence was lost")
                return text
            except (
                AttributeError,
                ImportError,
                MemoryError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ):
                if acquired:
                    self._release_model_lane_locked(reason="voice_asr_model_load_failed")
                raise

    def _transcribe_parakeet(self, audio: np.ndarray, repo: str) -> str:
        """Decode with Parakeet TDT.

        Note the API shape: ``BaseParakeet.transcribe`` takes a FILE PATH, so
        it is not usable here — Aura holds a live capture buffer, never a
        file. The array path is get_logmel -> generate, which is what the
        streaming decoder uses internally too.

        Model handles are cached per repo for the same reason Whisper's are:
        weight load plus Metal kernel compilation must never land inside a
        live turn.
        """
        import mlx.core as mx
        from parakeet_mlx import from_pretrained
        from parakeet_mlx.audio import get_logmel

        with self._cache_lock:
            model = self._parakeet_models.get(repo)
            if model is None:
                model = from_pretrained(repo)
                self._parakeet_models[repo] = model

        mel = get_logmel(mx.array(np.asarray(audio, dtype=np.float32)), model.preprocessor_config)
        results = model.generate(mel)
        return str(results[0].text) if results else ""

    def _transcribe_mlx(self, audio: np.ndarray, repo: str) -> dict[str, Any]:
        holder = self._mlx_holder
        if holder is None:
            return self._mlx.transcribe(
                audio,
                path_or_hf_repo=repo,
                language=self._config.language,
                fp16=True,
                condition_on_previous_text=False,
            )

        with _MLX_HOLDER_LOCK:
            cached = self._mlx_models.get(repo)
            if cached is not None:
                holder.model = cached
                holder.model_path = repo
            result = self._mlx.transcribe(
                audio,
                path_or_hf_repo=repo,
                language=self._config.language,
                fp16=True,
                condition_on_previous_text=False,
            )
            loaded = getattr(holder, "model", None)
            if loaded is not None:
                self._mlx_models[repo] = loaded
            return result

    def warm(self, repo: str) -> bool:
        """Force weight load + kernel compile outside the latency path."""
        with self._warm_lock:
            if repo in self._warmed_repos:
                return True
            silence = np.zeros(CAPTURE_RATE, dtype=np.float32)
            try:
                self.transcribe(silence, repo)
            except (
                AttributeError,
                ImportError,
                MemoryError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                record_degradation(
                    "voice_duplex.asr",
                    exc,
                    action=f"warmup decode failed for {repo}; first live decode will pay the cost",
                    severity="warning",
                )
                return False
            self._warmed_repos.add(repo)
            return True

    def shutdown(self) -> None:
        with self._usage_lock:
            self._release_model_lane_locked(reason="voice_asr_shutdown")


class StreamingAsr:
    """Growing-buffer incremental decoder with a stable prefix.

    Decoding runs on a thread executor: mlx-whisper releases the GIL during
    Metal work, but the Python-side pre/post-processing does not, and
    blocking the event loop here would stall audio intake for the whole
    session.
    """

    def __init__(
        self,
        config: AsrConfig | None = None,
        *,
        model_lane_controller: Any = None,
        backend: _WhisperBackend | None = None,
        owns_backend: bool = True,
    ) -> None:
        self._config = config or AsrConfig()
        self._backend = backend or _WhisperBackend(
            self._config, model_lane_controller=model_lane_controller
        )
        self._owns_backend = bool(owns_backend or backend is None)
        self._prev_words: list[str] = []
        self._stable_words: list[str] = []
        self._tentative_words: list[str] = []
        self._last_partial_at = 0.0
        self._decode_lock = checked_async_lock("voice.asr.decode", rank=LockRank.LEAF)
        self._warmed = False

    @property
    def available(self) -> bool:
        return self._backend.available

    async def warm_up(self) -> bool:
        """Pay the cold-start cost before the user says anything."""
        if self._warmed:
            return True
        loop = asyncio.get_running_loop()
        results: list[bool] = []
        for repo in (self._config.partial_model, self._config.final_model):
            results.append(
                bool(await loop.run_in_executor(None, self._backend.warm, repo))
            )
        self._warmed = all(results)
        if self._warmed:
            logger.info(
                "ASR warm: partial=%s final=%s",
                self._config.partial_model,
                self._config.final_model,
            )
        return self._warmed

    def reset(self) -> None:
        self._prev_words = []
        self._stable_words = []
        self._tentative_words = []
        self._last_partial_at = 0.0

    def shutdown(self) -> None:
        if self._owns_backend:
            self._backend.shutdown()

    def due_for_partial(self, now: float, audio_s: float) -> bool:
        """Rate-limit partials; decoding faster than this buys nothing."""
        if audio_s * 1000.0 < self._config.min_decode_ms:
            return False
        return (now - self._last_partial_at) * 1000.0 >= self._config.partial_interval_ms

    async def partial(self, audio: np.ndarray) -> Transcript | None:
        """Decode the buffer and fold the result into the stable prefix."""
        if not self.available or audio.size == 0:
            return None
        if self._decode_lock.locked():
            # A decode is already in flight. Skipping is correct: the next
            # one sees a longer buffer and supersedes this one anyway.
            return None
        async with self._decode_lock:
            self._last_partial_at = time.monotonic()
            started = time.perf_counter()
            try:
                loop = asyncio.get_running_loop()
                text = await loop.run_in_executor(
                    None, self._backend.transcribe, audio, self._config.partial_model
                )
            except (RuntimeError, ValueError, OSError, AttributeError, MemoryError) as exc:
                record_degradation(
                    "voice_duplex.asr",
                    exc,
                    action="dropped one partial decode; endpointing continues on VAD alone",
                    severity="warning",
                )
                return None
            decode_ms = (time.perf_counter() - started) * 1000.0

        words = _normalise_words(text)
        # LocalAgreement-2: stability is agreement between consecutive
        # independent decodes, never a single decode's own confidence.
        agreed = _common_prefix_len(self._prev_words, words)
        if agreed > len(self._stable_words):
            self._stable_words = words[:agreed]
        self._prev_words = words
        self._tentative_words = words[len(self._stable_words):]

        return Transcript(
            stable=" ".join(self._stable_words),
            tentative=" ".join(self._tentative_words),
            is_final=False,
            decode_ms=decode_ms,
            audio_s=audio.size / float(CAPTURE_RATE),
        )

    async def probe(self, audio: np.ndarray) -> str:
        """Transcribe a side-channel sample without changing turn state.

        Overlap arbitration needs to distinguish a short acknowledgement from
        a short objection. Reusing :meth:`partial` for that corrupts
        LocalAgreement-2 because the overlap becomes the previous hypothesis
        for the next real utterance. This method shares the model lane and
        decode lock but deliberately leaves every agreement buffer and
        scheduling timestamp untouched.
        """
        if not self.available or audio.size == 0:
            return ""
        try:
            async with self._decode_lock:
                loop = asyncio.get_running_loop()
                text = await loop.run_in_executor(
                    None,
                    self._backend.transcribe,
                    audio,
                    self._config.partial_model,
                )
        except (RuntimeError, ValueError, OSError, AttributeError, MemoryError) as exc:
            record_degradation(
                "voice_duplex.asr",
                exc,
                action="kept the timing-based overlap verdict after probe failure",
                severity="debug",
            )
            return ""

        cleaned = text.strip()
        return "" if looks_hallucinated(cleaned) else cleaned

    async def finalize(self, audio: np.ndarray) -> Transcript:
        """One accurate decode of the complete utterance.

        This is the text her mind reasons over, so it runs on the large
        model even though it costs ~195 ms — accuracy here is worth more
        than the latency, and the filler lane covers the gap.
        """
        if audio.size == 0:
            return Transcript(is_final=True)
        started = time.perf_counter()
        text = ""
        try:
            async with self._decode_lock:
                loop = asyncio.get_running_loop()
                text = await loop.run_in_executor(
                    None, self._backend.transcribe, audio, self._config.final_model
                )
        except (RuntimeError, ValueError, OSError, AttributeError, MemoryError) as exc:
            record_degradation(
                "voice_duplex.asr",
                exc,
                action="final decode failed; falling back to the stable partial prefix",
            )
            # The stable prefix is real transcribed speech, not a guess, so
            # using it is honest — but it may be missing the tail.
            text = " ".join(self._stable_words)

        cleaned = text.strip()
        if looks_hallucinated(cleaned):
            logger.info("Discarded hallucinated final transcript: %r", cleaned[:60])
            cleaned = ""

        return Transcript(
            stable=cleaned,
            tentative="",
            is_final=True,
            decode_ms=(time.perf_counter() - started) * 1000.0,
            audio_s=audio.size / float(CAPTURE_RATE),
        )

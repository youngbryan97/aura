"""core/voice/duplex/session.py — The duplex conversation state machine.

This is where the pieces become a conversation. The shape that matters is
that there are two clocks running at once:

  * a **reflex clock** on every 32 ms audio frame — VAD, barge-in detection,
    backchannel placement, endpointing. Nothing here waits on a model, so it
    stays responsive while her mind is busy.
  * a **cognition clock** per turn — transcription, the governed turn, and
    synthesis, all of which take from hundreds of milliseconds to seconds.

Half-duplex designs collapse these into one loop, which is exactly why they
cannot say "mhm" while you talk or stop when you cut in. Keeping them apart
is the whole architecture.

Audio intake never awaits cognition: :meth:`feed_audio` does bounded work
and hands longer jobs to tasks. If that invariant breaks, the microphone
stutters whenever she thinks.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
from core.voice.duplex import protocol
from core.voice.duplex.addressivity import AddressivityGate
from core.voice.duplex.audio import (
    FrameSplitter,
    UtteranceBuffer,
    float32_to_pcm16,
    pcm16_to_float32,
)
from core.voice.duplex.backchannel import BackchannelReflex
from core.voice.duplex.clause_chunker import StreamingChunker
from core.voice.duplex.config import (
    CAPTURE_RATE,
    OUTPUT_RATE,
    VAD_FRAME_SAMPLES,
    DuplexConfig,
)
from core.voice.duplex.echo_guard import EchoGuard
from core.voice.duplex.endpointing import Completeness, Endpointer
from core.voice.duplex.fillers import FillerReflex, ThinkingCause
from core.voice.duplex.governed_stream import stream_governed_reply
from core.voice.duplex.mind_bridge import MindBridge, SpokenRecord, StreamingTurn
from core.voice.duplex.overlap import OverlapArbiter, OverlapVerdict
from core.voice.duplex.paralinguistics import (
    DeliveryReading,
    SpeakerBaseline,
    convergence_factors,
)
from core.voice.duplex.paralinguistics import (
    analyze as analyze_delivery,
)
from core.voice.duplex.paralinguistics import (
    interpret as interpret_delivery,
)
from core.voice.duplex.prosody import (
    ProsodyCompiler,
    carry_breakthrough,
    live_affect,
    live_speech_profile,
)
from core.voice.duplex.streaming_asr import StreamingAsr, looks_hallucinated
from core.voice.duplex.style import StyleController
from core.voice.duplex.tts_stream import CancellationToken, StreamingTts
from core.voice.duplex.vad_gate import SpeechEvent, VadGate

#: Exception class names that mean "the peer's transport is gone".
#:
#: Matched by NAME rather than imported, because the concrete classes live in
#: the ASGI server and web-framework packages (starlette, uvicorn, websockets)
#: and core.voice must not depend on the transport it is served over.
_PEER_DISCONNECT_NAMES = frozenset(
    {
        "ClientDisconnected",
        "ConnectionClosed",
        "ConnectionClosedError",
        "ConnectionClosedOK",
        "WebSocketDisconnect",
    }
)


def _is_peer_disconnect(exc: BaseException | None) -> bool:
    """True when this exception means the person's transport went away.

    Walks the cause/context chain: the disconnect usually arrives wrapped, as
    ConnectionClosedOK -> ClientDisconnected -> WebSocketDisconnect, and only
    the outermost type is visible to an except clause.
    """

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ in _PEER_DISCONNECT_NAMES:
            return True
        message = str(current)
        if 'once a close message has been sent' in message:
            return True
        current = current.__cause__ or current.__context__
    return False

logger = logging.getLogger("Aura.Voice.Session")

MAX_AUDIO_MESSAGE_BYTES = 128 * 1024
MAX_TYPED_MESSAGE_BYTES = 64 * 1024
TASK_QUIESCENCE_TIMEOUT_S = 3.0
OVERLAP_PROBE_TIMEOUT_S = 1.0

# Callbacks the transport supplies. Returning awaitables lets the session
# apply backpressure if the socket is slow.
JsonSender = Callable[[dict[str, Any]], Awaitable[None]]
BinarySender = Callable[[bytes], Awaitable[None]]


class SessionState(Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    LISTENING = "listening"
    USER_SPEAKING = "user_speaking"
    THINKING = "thinking"
    SPEAKING = "speaking"
    CLOSED = "closed"


@dataclass(slots=True)
class TurnMetrics:
    """Latency ledger for one turn. Reported so claims stay measurable."""

    speech_end_at: float = 0.0
    final_transcript_at: float = 0.0
    reply_ready_at: float = 0.0
    first_audio_at: float = 0.0
    asr_ms: float = 0.0
    asr_speculative_ms: float = 0.0  # decode that ran inside the endpoint wait
    cognition_ms: float = 0.0
    tts_first_chunk_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        ttfa = (
            (self.first_audio_at - self.speech_end_at) * 1000.0
            if self.first_audio_at and self.speech_end_at
            else 0.0
        )
        return {
            "asr_ms": round(self.asr_ms, 1),
            "asr_speculative_ms": round(self.asr_speculative_ms, 1),
            "cognition_ms": round(self.cognition_ms, 1),
            "tts_first_chunk_ms": round(self.tts_first_chunk_ms, 1),
            "time_to_first_audio_ms": round(ttfa, 1),
        }


@dataclass(slots=True)
class _SpeakingTrack:
    """Bookkeeping for the utterance currently being played.

    ``chunks`` records the text and duration of every chunk handed to the
    client, so a barge-in can be mapped back to the exact words the user
    actually heard rather than the whole reply she intended.
    """

    utterance_id: int = 0
    intended: str = ""
    chunks: list[tuple[str, float]] = field(default_factory=list)
    sent_duration_s: float = 0.0
    started_at: float = 0.0
    token: CancellationToken = field(default_factory=CancellationToken)

    def spoken_prefix(self, played_s: float) -> str:
        """The text corresponding to ``played_s`` seconds of playback."""
        if played_s <= 0:
            return ""
        out: list[str] = []
        remaining = played_s
        for text, duration in self.chunks:
            if remaining <= 0:
                break
            if remaining >= duration:
                out.append(text)
                remaining -= duration
                continue
            # Partially played chunk: approximate by word fraction. Speech
            # is not uniform in time, but for the purpose of "what did they
            # hear" a proportional cut is far closer than all-or-nothing.
            words = text.split()
            if words:
                take = max(1, int(len(words) * (remaining / max(duration, 1e-6))))
                out.append(" ".join(words[:take]))
            remaining = 0
        return " ".join(out).strip()


class DuplexVoiceSession:
    """One live full-duplex voice conversation."""

    def __init__(
        self,
        *,
        session_id: str,
        send_json: JsonSender,
        send_binary: BinarySender,
        config: DuplexConfig | None = None,
        mind: MindBridge | None = None,
        asr: StreamingAsr | None = None,
        tts: StreamingTts | None = None,
    ) -> None:
        self._id = session_id

        async def guarded_json(payload: dict[str, Any]) -> None:
            if self._closed:
                return
            try:
                await send_json(payload)
            except BaseException as exc:  # noqa: BLE001 - classified below
                if not _is_peer_disconnect(exc):
                    raise
                # The person ended voice mode. Their transport going away is the
                # normal end of a session, not a fault: `_closed` only tracks
                # OUR close, so a client-initiated close left this flag False and
                # the next send raised into a done-callback as an unhandled
                # error. Measured live, after "Can you hear what I'm saying?":
                #   Cannot call "send" once a close message has been sent
                #   ... WebSocketDisconnect ... Exception in callback
                # and she answered "My response was cut short."
                self._closed = True

        async def guarded_binary(payload: bytes) -> None:
            if self._closed:
                return
            try:
                await send_binary(payload)
            except BaseException as exc:  # noqa: BLE001 - classified below
                if not _is_peer_disconnect(exc):
                    raise
                self._closed = True

        self._send_json = guarded_json
        self._send_binary = guarded_binary
        self._config = config or DuplexConfig()

        self._vad = VadGate(self._config.vad)
        self._owns_asr = asr is None
        self._asr = asr or StreamingAsr(self._config.asr)
        self._endpointer = Endpointer(self._config.endpoint)
        self._backchannel = BackchannelReflex(self._config.backchannel)
        self._filler = FillerReflex()
        self._echo = EchoGuard()
        self._style = StyleController()
        self._overlap = OverlapArbiter()
        self._overlap_audio: list[np.ndarray] = []
        self._overlap_epoch = 0
        self._voice_override = ""
        self._owns_tts = tts is None
        self._tts = tts or StreamingTts(self._config.tts)
        self._prosody = ProsodyCompiler(
            base_voice=self._config.tts.voice,
            base_speed=self._config.tts.speed,
        )
        self._mind = mind or MindBridge(
            session_id=session_id,
            cognition_timeout_s=self._config.cognition_timeout_s,
            spoken_reply_words=self._config.spoken_reply_words,
        )

        self._splitter = FrameSplitter(VAD_FRAME_SAMPLES)
        self._utterance = UtteranceBuffer(
            self._config.asr.max_utterance_s, CAPTURE_RATE
        )

        self._state = SessionState.IDLE
        self._muted = False
        self._closed = False

        self._turn_task: asyncio.Task[None] | None = None
        self._partial_task: asyncio.Task[None] | None = None
        self._filler_task: asyncio.Task[None] | None = None
        self._filler_stop: asyncio.Event | None = None
        self._side_tasks: set[asyncio.Task[Any]] = set()

        self._speaking: _SpeakingTrack | None = None
        self._utterance_counter = 0
        self._utterance_generation = 0
        self._barge_run_ms = 0.0
        self._client_played_s = 0.0
        self._client_playback_utterance_id = 0
        self._client_playback_drained = False
        self._client_playback_overflow_samples = 0
        self._metrics = TurnMetrics()
        self._capture_available = True
        self._device_generation = 0
        self._device_state = "unreported"
        self._device_reason = ""
        self._device_observed_at = time.monotonic()
        self._stable_text = ""
        # A final decode started during the endpoint's silence wait, valid
        # only until the user speaks again.
        self._speculative: Any = None
        self._speculative_task: asyncio.Task[None] | None = None
        # How the user has been sounding, and how this turn deviates from it.
        self._speaker_baseline = SpeakerBaseline()
        self._delivery = DeliveryReading()

        # Ambient listening. The gate decides whether an utterance on an open
        # microphone was meant for her; it is bypassed entirely whenever the
        # user has opened the floor themselves, because a deliberate act of
        # address does not need to be second-guessed by a heuristic.
        self._ambient_gate = (
            AddressivityGate(
                names=self._config.ambient.names,
                open_floor_s=self._config.ambient.open_floor_s,
                min_cold_open_words=self._config.ambient.min_cold_open_words,
            )
            if self._config.ambient.enabled
            else None
        )
        self._floor_explicitly_open = not self._config.ambient.enabled
        self._last_reply_ended_at = 0.0

    # ── lifecycle ────────────────────────────────────────────────────────

    @property
    def state(self) -> SessionState:
        return self._state

    async def start(self) -> None:
        """Warm every model before the first word so nobody pays cold start.

        Measured cold costs on this host: Kokoro 635 ms vs 190 ms warm, and
        Whisper 13–35 s for weight load plus Metal kernel compilation. Paying
        that inside a live turn would be indistinguishable from a hang.
        """
        await self._set_state(SessionState.PREPARING)
        spec = self._prosody_spec()

        results = await asyncio.gather(
            self._tts.warm_up(spec),
            self._asr.warm_up(),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                record_degradation(
                    "voice_duplex.session",
                    result,
                    action="continued with a cold model; the first turn will be slower",
                    severity="warning",
                )

        tts_ready = (
            not isinstance(results[0], BaseException)
            and results[0] is not False
            and self._tts.engine_name != "none"
        )
        asr_ready = (
            not isinstance(results[1], BaseException)
            and results[1] is not False
            and self._asr.available
        )
        if not tts_ready or not asr_ready:
            await self._send_json(
                {
                    "type": protocol.EVT_ERROR,
                    "status": "voice_models_not_ready",
                    "message": "Voice models are not ready yet. Reconnecting will retry warmup.",
                    "asr_ready": asr_ready,
                    "tts_ready": tts_ready,
                }
            )
            raise RuntimeError(
                "voice_models_not_ready:"
                f"asr={str(asr_ready).lower()}:tts={str(tts_ready).lower()}"
            )

        await self._mind.start_activity_watch(self._filler.observe_activity)
        await self._mind.publish("session_started", {"engine": self._tts.engine_name})
        await self._send_json(
            {
                "type": protocol.EVT_READY,
                "tts_engine": self._tts.engine_name,
                "vad_backend": self._vad.backend_name,
                "asr_available": self._asr.available,
                "sample_rate_in": CAPTURE_RATE,
                "sample_rate_out": OUTPUT_RATE,
            }
        )
        await self._set_state(SessionState.LISTENING)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._cancel_playback()
        self._stop_fillers()
        owned_tasks = {
            task
            for task in (
                self._turn_task,
                self._partial_task,
                self._speculative_task,
                *tuple(self._side_tasks),
            )
            if task is not None
        }
        await self._cancel_and_quiesce_tasks(
            owned_tasks,
            reason="session_close",
        )
        with contextlib.suppress(asyncio.CancelledError, RuntimeError):
            await self._mind.stop_activity_watch()
        if self._owns_asr:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    asyncio.to_thread(self._asr.shutdown),
                    timeout=TASK_QUIESCENCE_TIMEOUT_S,
                )
        if self._owns_tts:
            self._tts.shutdown()
        self._state = SessionState.CLOSED
        await self._mind.publish("session_ended", {})

    # ── audio intake (the reflex clock) ──────────────────────────────────

    async def feed_audio(self, data: bytes) -> None:
        """Handle one arbitrary-sized PCM16 buffer from the client.

        Must stay bounded. Anything that could take longer than a frame
        period is dispatched to a task instead of awaited here.
        """
        if self._closed or self._muted or not self._capture_available:
            return
        if len(data) > MAX_AUDIO_MESSAGE_BYTES:
            raise ValueError(
                f"voice audio message exceeds {MAX_AUDIO_MESSAGE_BYTES} bytes"
            )
        try:
            samples = pcm16_to_float32(data)
        except (ValueError, TypeError) as exc:
            record_degradation(
                "voice_duplex.session",
                exc,
                action="dropped a malformed audio buffer",
                severity="warning",
            )
            return

        for frame in self._splitter.push(samples):
            await self._process_frame(frame)

    async def _process_frame(self, frame: np.ndarray) -> None:
        try:
            vf = self._vad.process(frame)
        except (ValueError, RuntimeError) as exc:
            record_degradation(
                "voice_duplex.session", exc, action="skipped a VAD frame", severity="debug"
            )
            return

        # While she is speaking, audio feeds barge-in detection and nothing
        # else. This exclusivity is load-bearing: the ordinary onset rule
        # fires after 2 frames above 0.55, while barge-in deliberately
        # demands 4 frames above 0.72. Run both and onset always wins the
        # race, silently converting every interruption into "user started a
        # new turn" — her audio keeps playing and nothing is ever cut off.
        if self._state is SessionState.SPEAKING:
            # Retain the frames anyway, so the words that did the
            # interrupting survive the time it takes to classify them.
            self._utterance.observe_silence(frame)
            await self._handle_overlap(frame, vf)
            return

        if vf.event is SpeechEvent.ONSET:
            await self._begin_user_turn()
        elif not self._vad.in_speech:
            self._utterance.observe_silence(frame)
            return

        if self._vad.in_speech:
            self._utterance.append(frame)

        if vf.event is SpeechEvent.PAUSE:
            await self._on_pause(vf.silence_ms, vf.speech_ms)
        elif vf.event in (SpeechEvent.CONTINUING, SpeechEvent.RESUMED):
            if vf.event is SpeechEvent.RESUMED:
                # They were not finished after all, so any speculative
                # transcript describes an unfinished sentence. Drop it.
                self._discard_speculation()
            self._maybe_schedule_partial()

    async def _begin_user_turn(
        self,
        *,
        captured_audio: np.ndarray | None = None,
        reason: str = "superseded_by_new_speech",
    ) -> None:
        """The user started talking."""
        # Cancel a pending turn: if they started again before she answered,
        # the newer utterance supersedes the older one.
        await self._cancel_active_turn(
            reason=reason,
            return_to_listening=False,
        )

        if captured_audio is None:
            self._utterance.begin()
        else:
            # Overlap arbitration waits for evidence before taking the floor.
            # Seed the new turn with the entire bounded capture, not merely
            # the rolling 320 ms preroll, or the first word is often lost.
            self._utterance.clear()
            self._utterance.begin()
            self._utterance.append(captured_audio)
        self._asr.reset()
        self._discard_speculation()
        self._stable_text = ""
        # A partial decode started for the previous utterance can still be
        # in flight. Bump the generation so its result is discarded instead
        # of overwriting this utterance's transcript with the last one's.
        self._utterance_generation += 1
        self._backchannel.on_user_turn_start()
        await self._set_state(SessionState.USER_SPEAKING)

    async def _on_pause(self, silence_ms: float, speech_ms: float) -> None:
        """A gap inside or at the end of the user's turn."""
        decision = self._endpointer.evaluate(
            transcript=self._stable_text,
            silence_ms=silence_ms,
            speech_ms=speech_ms,
            min_utterance_ms=self._config.vad.min_utterance_ms,
            terminality=self._read_terminality(),
        )

        if decision.should_end:
            await self._end_user_turn(decision.reason)
            return

        # The endpointer is about to spend hundreds of milliseconds waiting
        # to see whether they resume. Spend that time decoding instead.
        self._maybe_speculate_final(decision, silence_ms)

        # Not an endpoint — so this gap is a prosodic boundary, which is
        # exactly where acknowledgement belongs.
        bc = self._backchannel.consider(
            silence_ms=silence_ms,
            speech_ms=speech_ms,
            aura_is_speaking=self._state is SessionState.SPEAKING,
            substrate=self._substrate_snapshot(),
        )
        if bc.should_emit:
            self._spawn(self._speak_aside(bc.text, protocol.AudioOpcode.BACKCHANNEL, bc.gain))
            await self._send_json(
                {"type": protocol.EVT_BACKCHANNEL, "text": bc.text, "register": bc.register}
            )
            await self._mind.publish("backchannel", {"text": bc.text, "register": bc.register})

        self._maybe_schedule_partial()

    def _evaluate_addressivity(self, transcript: str, final: Any) -> Any:
        """Decide whether to answer this utterance.

        Fails *open* if the gate itself breaks — the opposite of the gate's own
        policy, and deliberately so. A crash in a heuristic must not make her
        deaf; an unanswered user is a worse outcome than an occasional
        unwanted answer, once the alternative is "the feature silently stopped
        working".
        """
        try:
            from core.voice.duplex.addressivity import AddressContext, AddressVerdict

            if self._ambient_gate is None:
                return AddressVerdict(True, "gate_disabled", reasons=("ambient gating is off",))

            since = None
            if self._last_reply_ended_at:
                since = max(0.0, time.monotonic() - self._last_reply_ended_at)

            # Only trust a loudness reading once the baseline has settled;
            # before that, "quieter than usual" has no "usual" to mean.
            loudness_z = (
                float(self._delivery.energy_z)
                if self._speaker_baseline.ready
                else None
            )

            context = AddressContext(
                since_last_reply_s=since,
                # The recogniser does not report a per-utterance confidence,
                # so this stays None and the rung that would use it is skipped
                # rather than fed a fabricated 1.0.
                asr_confidence=None,
                loudness_z=loudness_z,
                competing_speech=bool(self._overlap.active),
                duration_s=self._utterance.sample_count / float(CAPTURE_RATE),
                floor_explicitly_open=self._floor_explicitly_open,
            )
            return self._ambient_gate.evaluate(transcript, context)
        except (RuntimeError, ValueError, TypeError, AttributeError, ImportError) as exc:
            record_degradation(
                "voice_duplex.session",
                exc,
                action="answered without the addressivity check rather than going deaf",
                severity="warning",
            )
            from core.voice.duplex.addressivity import AddressVerdict

            return AddressVerdict(True, "gate_failed", reasons=("the addressivity gate errored",))

    def _read_terminality(self) -> Any:
        """The pitch contour at the end of what has been said so far.

        Runs on every pause, which sounds expensive and is not: it is a
        polyfit over the last three quarters of a second of audio that is
        already in memory. It is also the difference between waiting out a
        breath and talking over it, which is the single loudest complaint
        about every voice assistant that ships.
        """
        try:
            from core.voice.duplex.acoustic_endpoint import read_terminality

            audio = self._utterance.audio()
            if audio is None or getattr(audio, "size", 0) == 0:
                return None
            return read_terminality(audio, CAPTURE_RATE)
        except (RuntimeError, ValueError, TypeError, AttributeError, ImportError) as exc:
            record_degradation(
                "voice_duplex.session",
                exc,
                action="endpointed on the transcript alone, without the pitch contour",
                severity="debug",
            )
            return None

    def _maybe_speculate_final(self, decision: Any, silence_ms: float) -> None:
        """Start the final decode during the endpoint's wait, not after it.

        The endpointer deliberately waits 340–1500 ms to see whether the user
        resumes. Serially, the ~320 ms final decode then happens *after* that
        wait, so the two costs add. Overlapping them means that when the
        endpoint confirms, the transcript her mind needs already exists.

        Safe because a decode is a pure function of the audio: a wasted one
        costs CPU and nothing else. It is thrown away the instant speech
        resumes, since it would then describe an unfinished sentence.
        """
        cfg = self._config.asr
        if not cfg.speculative_final or self._speculative is not None:
            return
        if self._speculative_task is not None and not self._speculative_task.done():
            return
        if silence_ms < cfg.speculate_after_ms:
            return
        # A trailing "um" means they are audibly still composing; speculating
        # there would burn a decode on a sentence that is about to grow.
        if getattr(decision, "completeness", None) is Completeness.THINKING:
            return
        if self._utterance.sample_count == 0:
            return
        self._speculative_task = self._spawn(self._run_speculative_final())

    async def _run_speculative_final(self) -> None:
        audio = self._utterance.audio()
        if audio.size == 0:
            return
        generation = self._utterance_generation
        result = await self._asr.finalize(audio)
        # Two ways this can be stale: a whole new utterance began, or they
        # resumed mid-pause and _discard_speculation cleared the slot.
        if generation != self._utterance_generation:
            return
        if self._vad.in_speech and self._vad.last_probability >= self._config.vad.speech_threshold:
            return
        self._speculative = result

    def _read_delivery(self, audio: np.ndarray, transcript: str) -> None:
        """Measure how this turn was said, relative to how they usually sound.

        Runs after the transcript exists because speaking rate needs a word
        count. Costs 3–5 ms of numpy on audio already in memory.
        """
        try:
            signature = analyze_delivery(
                audio, CAPTURE_RATE, word_count=len(transcript.split())
            )
            # Interpret against the baseline *before* folding this turn into
            # it — otherwise every utterance partly normalises itself away.
            self._delivery = interpret_delivery(signature, self._speaker_baseline)
            self._speaker_baseline.observe(signature)
            if self._delivery.notable:
                logger.info("Delivery: %s", ", ".join(self._delivery.descriptors))
        except (ValueError, TypeError, FloatingPointError, ZeroDivisionError) as exc:
            record_degradation(
                "voice_duplex.paralinguistics",
                exc,
                action="spoke without paralinguistic context for this turn",
                severity="debug",
            )
            self._delivery = DeliveryReading()

    def _discard_speculation(self) -> None:
        self._speculative = None
        task = self._speculative_task
        self._speculative_task = None
        if task is not None and not task.done():
            task.cancel()

    def _maybe_schedule_partial(self) -> None:
        """Kick off an incremental decode if one is due and none is running."""
        if self._partial_task is not None and not self._partial_task.done():
            return
        audio_s = self._utterance.sample_count / float(CAPTURE_RATE)
        if not self._asr.due_for_partial(time.monotonic(), audio_s):
            return
        self._partial_task = self._spawn(self._run_partial())

    async def _run_partial(self) -> None:
        audio = self._utterance.audio()
        if audio.size == 0:
            return
        generation = self._utterance_generation
        result = await self._asr.partial(audio)
        if result is None:
            return
        if generation != self._utterance_generation:
            # The utterance was reset while this decode ran. Its text
            # describes speech that is no longer the current turn.
            return
        self._stable_text = result.stable
        await self._send_json(
            {
                "type": protocol.EVT_PARTIAL,
                "stable": result.stable,
                "tentative": result.tentative,
                "decode_ms": round(result.decode_ms, 1),
            }
        )

    # ── barge-in ─────────────────────────────────────────────────────────

    async def _handle_overlap(self, frame: np.ndarray, vf: Any) -> None:
        """User speech while she is speaking. Duck first, decide second.

        The two cases — "mhm" and "no, stop" — are acoustically identical at
        onset, so any classifier forced to decide immediately is reliably
        wrong. Ducking is the correct response to *both*, is instant, and is
        reversible, which lets the irreversible call wait for real evidence.
        """
        cfg = self._config.barge_in
        track = self._speaking
        if not cfg.enabled or track is None:
            return

        # Grace window: without it, the tail of the user's own question or a
        # scrap of residual echo kills the reply the instant it begins.
        if (time.monotonic() - track.started_at) * 1000.0 < cfg.grace_ms:
            return

        is_speech = vf.probability >= cfg.threshold
        if not self._overlap.active:
            if not is_speech:
                return
            self._overlap.begin()
            self._overlap_audio = []
            self._overlap_epoch += 1

        self._overlap_audio.append(frame)
        verdict = self._overlap.observe(
            frame_ms=VAD_FRAME_SAMPLES / CAPTURE_RATE * 1000.0,
            is_speech=is_speech,
            energy=float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))),
        )

        if self._overlap.should_duck():
            await self._send_json(
                {
                    "type": protocol.EVT_DUCK,
                    "gain": self._overlap.duck_gain,
                    "ramp_ms": 60,
                }
            )

        if verdict is OverlapVerdict.PENDING:
            return

        if verdict is OverlapVerdict.BARGE_IN:
            overlap_audio = (
                np.concatenate(self._overlap_audio)
                if self._overlap_audio
                else np.zeros(0, dtype=np.float32)
            )
            self._overlap.reset()
            self._overlap_audio = []
            await self._begin_user_turn(
                captured_audio=overlap_audio,
                reason="user_barge_in",
            )
            return

        # Timing says backchannel, but a short "no" has the same acoustic
        # shape. Keep playback ducked until a side-effect-free decode resolves
        # the ambiguity.
        overlap_audio = (
            np.concatenate(self._overlap_audio) if self._overlap_audio else None
        )
        overlap_epoch = self._overlap_epoch
        self._overlap.reset()
        self._overlap_audio = []
        if overlap_audio is not None and overlap_audio.size:
            self._spawn(
                self._verify_backchannel(
                    overlap_audio,
                    track,
                    overlap_epoch,
                )
            )
        else:
            await self._resume_after_backchannel(track, "", overlap_epoch)

    async def _verify_backchannel(
        self,
        audio: np.ndarray,
        track: _SpeakingTrack,
        overlap_epoch: int,
    ) -> None:
        """Confirm a backchannel verdict against what was actually said."""
        try:
            text = await asyncio.wait_for(
                self._asr.probe(audio),
                timeout=OVERLAP_PROBE_TIMEOUT_S,
            )
        except (RuntimeError, ValueError, OSError, AttributeError) as exc:
            record_degradation(
                "voice_duplex.overlap",
                exc,
                action="kept the timing-based backchannel verdict",
                severity="debug",
            )
            text = ""
        if (
            self._speaking is not track
            or self._closed
            or overlap_epoch != self._overlap_epoch
        ):
            return

        from core.voice.duplex.overlap import looks_like_backchannel

        if not text or looks_like_backchannel(text):
            await self._resume_after_backchannel(track, text, overlap_epoch)
            return

        logger.info("Verified barge-in: overlap was %r", text[:50])
        await self._begin_user_turn(
            captured_audio=audio,
            reason="verified_barge_in",
        )
        # The short utterance already ended while arbitration was pending, so
        # no later VAD pause will close it for us.
        await self._end_user_turn("verified_barge_in")

    async def _resume_after_backchannel(
        self,
        track: _SpeakingTrack,
        transcript: str,
        overlap_epoch: int,
    ) -> None:
        if (
            self._speaking is not track
            or self._closed
            or overlap_epoch != self._overlap_epoch
        ):
            return
        await self._send_json(
            {"type": protocol.EVT_DUCK, "gain": 1.0, "ramp_ms": 140}
        )
        await self._mind.publish(
            "user_backchannel",
            {"transcript": transcript[:80]},
        )
        logger.info("User backchannel over her speech — continuing")

    async def _interrupt(self, *, reason: str, played_s: float | None = None) -> None:
        """Stop speaking now and record what was actually heard."""
        track = self._speaking
        if track is None:
            return

        self._barge_run_ms = 0.0
        self._overlap_epoch += 1
        self._overlap.reset()
        self._overlap_audio = []
        track.token.cancel()

        # The client owns the playback buffer, so its reported position is
        # the only real answer to "what did they hear". Wall-clock is the
        # last resort and is always an over-estimate, because the server
        # finishes sending long before the client finishes playing.
        if played_s is None:
            if self._client_played_s > 0:
                played_s = self._client_played_s
            else:
                elapsed = time.monotonic() - track.started_at
                played_s = min(elapsed, track.sent_duration_s)
        played_s = max(0.0, min(played_s, track.sent_duration_s))

        spoken = track.spoken_prefix(played_s)
        record = SpokenRecord(
            intended=track.intended,
            spoken=spoken,
            interrupted=True,
            started_at=track.started_at,
            ended_at=time.monotonic(),
        )
        self._mind.record_spoken(record)
        self._speaking = None

        # Tell the client to drop everything buffered — otherwise it keeps
        # playing audio the server has already stopped generating.
        await self._send_json({"type": protocol.EVT_FLUSH, "utterance_id": track.utterance_id})
        await self._send_json(
            {
                "type": protocol.EVT_INTERRUPTED,
                "reason": reason,
                "spoken": spoken,
                "unheard": record.unheard,
            }
        )
        await self._mind.publish(
            "interrupted",
            {"reason": reason, "spoken_chars": len(spoken), "unheard_chars": len(record.unheard)},
        )
        logger.info("Barge-in (%s): heard %d chars, dropped %d", reason, len(spoken), len(record.unheard))

        # Leave the session listening. A real barge-in continues into
        # _begin_user_turn, which opens the utterance; an explicit "stop"
        # command has no follow-on speech and correctly ends here.
        await self._set_state(SessionState.LISTENING)

    async def _cancel_and_quiesce_tasks(
        self,
        tasks: set[asyncio.Task[Any]],
        *,
        reason: str,
    ) -> bool:
        current = asyncio.current_task()
        pending = {
            task
            for task in tasks
            if task is not current and not task.done()
        }
        for task in pending:
            task.cancel()
        if not pending:
            return True
        _done, still_pending = await asyncio.wait(
            pending,
            timeout=TASK_QUIESCENCE_TIMEOUT_S,
        )
        if still_pending:
            record_degradation(
                "voice_duplex.session",
                TimeoutError(
                    f"{len(still_pending)} voice tasks did not quiesce after {reason}"
                ),
                action="left model ownership active until the worker operations exit",
                severity="warning",
            )
        return not still_pending

    async def _cancel_active_turn(
        self,
        *,
        reason: str,
        return_to_listening: bool,
    ) -> None:
        if self._speaking is not None:
            await self._interrupt(reason=reason)
        self._stop_fillers()
        tasks = {
            task
            for task in (
                self._turn_task,
                self._partial_task,
                self._speculative_task,
            )
            if task is not None
        }
        self._discard_speculation()
        quiesced = await self._cancel_and_quiesce_tasks(tasks, reason=reason)
        if not quiesced and not self._closed:
            raise RuntimeError(f"prior_voice_turn_failed_to_quiesce:{reason}")
        self._turn_task = None
        self._partial_task = None
        self._speculative_task = None
        if return_to_listening and not self._closed:
            await self._set_state(SessionState.LISTENING)

    # ── turn handling (the cognition clock) ──────────────────────────────

    async def _end_user_turn(self, reason: str) -> None:
        self._vad.close_utterance()
        audio = self._utterance.audio()
        self._utterance.clear()
        self._backchannel.on_user_turn_end()

        if audio.size == 0:
            await self._set_state(SessionState.LISTENING)
            return

        self._metrics = TurnMetrics(speech_end_at=time.monotonic())
        await self._set_state(SessionState.THINKING)
        self._turn_task = self._spawn(self._run_turn(audio, reason))

    async def _run_turn(self, audio: np.ndarray, reason: str) -> None:
        try:
            speculative = self._speculative
            self._speculative = None
            if speculative is not None:
                # Already decoded during the endpoint's wait — the turn's
                # transcript costs nothing here.
                final = speculative
                self._metrics.asr_ms = 0.0
                self._metrics.asr_speculative_ms = speculative.decode_ms
            else:
                final = await self._asr.finalize(audio)
                self._metrics.asr_ms = final.decode_ms
            self._metrics.final_transcript_at = time.monotonic()

            transcript = final.stable.strip()
            if not transcript or looks_hallucinated(transcript):
                # Silence or ASR noise. Say nothing — inventing a turn here
                # is how a voice agent starts talking to an empty room.
                await self._send_json(
                    {"type": protocol.EVT_FINAL, "text": "", "discarded": True}
                )
                await self._set_state(SessionState.LISTENING)
                return

            # Last line against her own voice coming back through the
            # speakers. If this "turn" is mostly her own recent words, it is
            # echo — answering it would have her talking to herself.
            verdict = self._echo.evaluate(transcript)
            if verdict.is_echo:
                await self._send_json(
                    {
                        "type": protocol.EVT_FINAL,
                        "text": "",
                        "discarded": True,
                        "reason": "echo_rejected",
                        "overlap": round(verdict.overlap, 3),
                    }
                )
                await self._mind.publish(
                    "echo_rejected", {"overlap": verdict.overlap, "text": transcript[:120]}
                )
                await self._set_state(SessionState.LISTENING)
                return

            # Delivery is measured before the addressivity check, not after,
            # because how near and how loudly something was said is evidence
            # about who it was said to.
            self._read_delivery(audio, transcript)

            # Was that meant for her? On an open microphone this is the
            # question that decides whether she is present in the room or
            # merely switched on in it. Note the ordering: the transcript is
            # sent either way. She heard it, the user can see that she heard
            # it, and the only thing in question is whether to answer.
            address = self._evaluate_addressivity(transcript, final)
            await self._send_json(
                {
                    "type": protocol.EVT_FINAL,
                    "text": transcript,
                    "endpoint_reason": reason,
                    "decode_ms": round(final.decode_ms, 1),
                    "addressed": address.addressed,
                    "address_rung": address.rung,
                    "address_why": list(address.reasons or address.vetoes),
                }
            )
            if not address.addressed:
                logger.info("not answering: %s", address.narrative())
                await self._mind.publish(
                    "not_addressed",
                    {"transcript": transcript[:160], **address.as_dict()},
                )
                await self._set_state(SessionState.LISTENING)
                return

            # Delivery requests take effect on this very reply, not the next
            # one — that immediacy is most of what makes it feel responsive.
            style_change = self._style.observe(transcript)
            if style_change:
                await self._send_json(
                    {"type": protocol.EVT_STYLE, "change": style_change}
                )
                await self._mind.publish("style_changed", {"change": style_change})

            self._mind.notify_user_spoke()
            await self._mind.publish(
                "user_turn",
                {
                    "transcript": transcript,
                    "reason": reason,
                    "delivery": list(self._delivery.descriptors),
                },
            )

            # Fillers start now and run until the first real audio goes out.
            self._start_fillers()

            cognition_started = time.perf_counter()

            # The reply is spoken as it forms. Until this was wired, the first
            # syllable waited for the last token, which made time-to-first-
            # audio proportional to total reply length — and the word cap that
            # compensated for it was why spoken answers were shallower than
            # the same question typed.
            turn = await self._mind.respond_streaming(
                transcript, delivery_context=self._delivery.as_context()
            )
            if turn is None:
                await self._set_state(SessionState.LISTENING)
                return

            # Cognition now runs in its own task so its reply can be read
            # while it forms — which means cancelling *this* task no longer
            # stops it by itself. Superseding a turn has to actually stop the
            # previous one: a governed turn nobody is listening to still holds
            # the 32B, and the whole point of a newer utterance is that the
            # older answer is no longer wanted.
            async with self._cognition_bound_to_this_turn(turn):
                streamed = await self._speak_streamed_reply(turn)
                reply = await turn.final()
                self._metrics.cognition_ms = (
                    time.perf_counter() - cognition_started
                ) * 1000.0
                self._metrics.reply_ready_at = time.monotonic()

                if streamed:
                    return

                if not reply:
                    self._stop_fillers()
                    await self._speak_cognition_failure()
                    await self._set_state(SessionState.LISTENING)
                    return

                await self._speak_reply(reply)
        except asyncio.CancelledError:
            # Superseded by a newer utterance; not an error.
            raise
        except (RuntimeError, ValueError, AttributeError, TypeError, OSError) as exc:
            record_degradation(
                "voice_duplex.session",
                exc,
                action="voice turn aborted; returned the session to listening",
            )
            await self._send_json(
                {"type": protocol.EVT_ERROR, "message": "The voice turn failed before a reply formed."}
            )
            await self._set_state(SessionState.LISTENING)
        finally:
            self._stop_fillers()

    def _start_fillers(self) -> None:
        """Start exactly one filler loop for the current cognition turn."""
        self._stop_fillers()
        self._filler.begin_turn()
        stop = asyncio.Event()
        self._filler_stop = stop
        self._filler_task = self._spawn(self._run_fillers(time.monotonic(), stop))

    async def _run_fillers(
        self,
        started_at: float,
        stop: asyncio.Event,
    ) -> None:
        """Emit thinking sounds while cognition runs."""
        cfg = self._config.filler
        if not cfg.enabled:
            return
        try:
            while not stop.is_set():
                await asyncio.sleep(0.1)
                if stop.is_set():
                    break
                elapsed = (time.monotonic() - started_at) * 1000.0
                utterance = self._filler.due(
                    elapsed,
                    first=cfg.first_delay_ms,
                    second=cfg.second_delay_ms,
                    third=cfg.third_delay_ms,
                )
                if utterance is None:
                    continue
                await self._send_json(
                    {
                        "type": protocol.EVT_FILLER,
                        "text": utterance.text,
                        "tier": utterance.tier,
                        "cause": utterance.cause.value,
                    }
                )
                await self._speak_aside(
                    utterance.text, protocol.AudioOpcode.FILLER, utterance.gain
                )
                await self._mind.publish(
                    "filler", {"text": utterance.text, "cause": utterance.cause.value}
                )
        except asyncio.CancelledError:
            raise

    def _stop_fillers(self) -> None:
        stop = self._filler_stop
        self._filler_stop = None
        if stop is not None:
            stop.set()
        task = self._filler_task
        self._filler_task = None
        if task is not None and not task.done():
            task.cancel()

    # ── speech output ────────────────────────────────────────────────────

    async def _speak_reply(self, reply: str) -> None:
        await self._send_json({"type": protocol.EVT_REPLY, "text": reply})
        await self._speak_text(reply, cause=None)

    @contextlib.asynccontextmanager
    async def _cognition_bound_to_this_turn(self, turn: StreamingTurn) -> Any:
        """Tie a streaming cognition task to the lifetime of the turn task.

        ``respond_streaming`` runs cognition in its own task, which is what
        lets the reply be read while it is still forming. Two guarantees the
        single-task version gave for free have to be restored by hand:

        **Cancelling the turn stops the thinking.** A governed turn nobody is
        listening to still holds the 32B, and a newer utterance means the
        older answer is not wanted.

        **The turn is not "finished" until the thinking has actually
        stopped.** This is the subtle one, and it is why the teardown is
        awaited rather than fire-and-forget. ``_cancel_active_turn`` decides a
        replacement is safe to start by waiting for the turn task to
        quiesce; if the turn task returns while its cognition is still
        running, that check passes on a lie and two governed turns run at
        once. Awaiting here means a cognition that will not stop keeps the
        turn task alive, the quiescence check times out, and the session
        fails closed.
        """
        try:
            yield turn
        finally:
            if not turn.task.done():
                turn.task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await turn.task

    async def _speak_streamed_reply(self, turn: StreamingTurn) -> bool:
        """Speak a governed reply while it is still being produced.

        Returns True if the turn was delivered here. False means the stream
        carried nothing — every turn that did not take the streaming cognition
        path — and the caller should speak the finished reply the ordinary way.

        Three things have to be true at once, and the order matters:

        1. **Nothing ungoverned is spoken.** Each clause passes the
           clause-local gates in ``govern_clause`` before a single sample of
           it is synthesised.
        2. **The finished reply is still the authority.** Streamed text is
           pre-stabilisation. When the turn completes, what was said out loud
           is reconciled against what the turn stands behind.
        3. **A disagreement is spoken, not swallowed.** If governance revised
           text that had already been heard, she says so. The alternative —
           continuing as though the listener heard the corrected version — is
           the failure this whole lane exists to avoid.
        """
        from core.conversation.reply_stream import reconcile

        spoken_via_stream = ""

        def _still_speaking() -> bool:
            track = self._speaking
            return not (track is not None and track.token.cancelled)

        # stream_governed_reply pushes governed clauses; the synthesiser pulls.
        # A small bounded queue bridges the two, and its bound is what keeps
        # back-pressure intact: the synthesiser's rate paces the release, so a
        # fast model can never build a backlog of audio nobody has heard yet.
        bridge: asyncio.Queue[str] = asyncio.Queue(maxsize=4)

        async def _hand_to_synthesis(clause: str) -> None:
            await bridge.put(clause)

        async def _produce() -> None:
            nonlocal spoken_via_stream
            outcome = await stream_governed_reply(
                turn.channel.drain(timeout_s=self._config.cognition_timeout_s),
                first_max_chars=self._config.tts.first_chunk_max_chars,
                max_chars=self._config.tts.chunk_max_chars,
                speak=_hand_to_synthesis,
                on_first_chunk=self._stop_fillers,
                should_continue=_still_speaking,
            )
            spoken_via_stream = outcome.spoken

        producer = self._spawn(_produce())

        async def _next_clause() -> str | None:
            """The next governed clause, or None once there will not be one.

            Deliberately not a sentinel value on the queue. A sentinel has to
            be enqueued by the producer's ``finally``, and the producer can be
            cancelled while blocked on a *full* queue — at which point the
            sentinel cannot be enqueued either, and the consumer waits for a
            message that will never come. Asking the producer task whether it
            is finished has no such hole: a task that is done is done however
            it ended.
            """
            while True:
                try:
                    return await asyncio.wait_for(bridge.get(), timeout=0.2)
                except TimeoutError:
                    if producer.done():
                        # Drain anything the producer managed to enqueue
                        # between the last get and finishing.
                        if not bridge.empty():
                            return bridge.get_nowait()
                        return None
                    continue

        try:
            # Peek the first clause before committing to an utterance. Most
            # turns do not stream at all — every path through cognition that
            # returns its reply in one piece — and starting a speaking track
            # for a stream that never produces anything would report a
            # synthesis failure for a turn that was simply not streamed.
            first = await _next_clause()
            if first is None:
                return False

            async def _pieces() -> AsyncIterator[str]:
                yield first
                while True:
                    piece = await _next_clause()
                    if piece is None:
                        return
                    yield piece

            delivered = await self._deliver_utterance(_pieces(), cause=None)
            # Captured before the finally, because a barge-in clears
            # ``_speaking`` and both of these decide whether a correction is
            # owed at all.
            last = self._mind.last_spoken
            interrupted = bool(last and last.interrupted)
            delivery_complete = bool(last and last.delivery_complete)
        finally:
            if not producer.done():
                producer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await producer

        visible_reply = (
            str(last.spoken or "")
            if interrupted
            else (delivered or spoken_via_stream)
        )
        await self._send_json(
            {
                "type": protocol.EVT_REPLY,
                "text": visible_reply,
                "interrupted": interrupted,
            }
        )

        final = await turn.final()

        if interrupted or not delivery_complete:
            # Either the listener cut her off, or synthesis stopped short.
            # Both mean the reply is incomplete for a reason that has nothing
            # to do with governance, and both already record the heard prefix
            # and hand the unheard tail to the next turn. A governance
            # correction on top would apologise for a revision to text they
            # never reached.
            return True

        # Reconcile the *governed clause text*, not the delivered text.
        #
        # They are not the same string and the difference is not cosmetic: the
        # synthesiser cannot pronounce characters, so everything downstream of
        # governance passes through `spoken_form.prepare_for_speech`, which
        # turns "45%" into "forty-five percent" and "2026-07-27" into a date a
        # person would say. Comparing that against the raw final reply reports
        # a divergence for every answer containing a number, a percentage, a
        # date, a currency amount or a URL — which is to say, constantly, and
        # each one would have her interrupt herself to "correct" a
        # pronunciation.
        #
        # What is being checked here is whether *governance* changed its mind,
        # so both sides have to be in the representation governance produced.
        verdict = reconcile(spoken_via_stream, final or "")

        if verdict.consistent and verdict.remainder:
            # Governance kept everything already said and had more to add.
            await self._speak_text(verdict.remainder, cause=None)
            return True

        if verdict.diverged:
            record_degradation(
                "voice_duplex.session",
                RuntimeError("governed reply diverged from the text already spoken"),
                action="told the user her own governance revised what they heard",
                severity="warning",
            )
            await self._mind.publish(
                "streamed_reply_diverged",
                {"spoken_chars": len(verdict.spoken), "final_chars": len(verdict.final)},
            )
            # Her next turn is told what happened, so the correction is in her
            # own words rather than a fixed apology string from this file.
            self._mind.note_correction(verdict.correction_context)
            await self._speak_correction(verdict)
            return True

        return True

    async def _speak_cognition_failure(self) -> None:
        """Say that the turn produced no answer, and say what actually stopped it.

        This is the one failure she genuinely cannot narrate herself: the
        thing that would do the narrating is what failed. So the words are
        here — but the *content* is not fixed. The reason comes from the
        degradation this turn actually recorded, so "my reasoning lane timed
        out after two minutes" and "governance refused the reply" are
        different sentences, because they are different situations and a
        listener can act on the difference.
        """
        reason = ""
        try:
            from core.runtime.errors import recent_degradations

            records = recent_degradations(
                limit=12,
                subsystem_prefixes=("voice_duplex.", "chat", "cognitive"),
            )
            for record in reversed(records):
                detail = str(record.get("action") or record.get("error") or "").strip()
                if detail:
                    reason = detail.rstrip(".")
                    break
        except (RuntimeError, AttributeError, TypeError, ValueError, ImportError) as exc:
            record_degradation(
                "voice_duplex.session",
                exc,
                action="reported the cognition failure without naming its cause",
                severity="debug",
            )

        if reason:
            spoken = (
                f"I didn't get an answer out of that one — {reason}. "
                "I'd rather tell you that than make something up."
            )
        else:
            spoken = (
                "I didn't get an answer out of that one, and I can't see why "
                "from here. I'd rather tell you that than make something up."
            )
        await self._speak_text(spoken, cause=ThinkingCause.UNCERTAINTY)

    async def _speak_correction(self, verdict: Any) -> None:
        """Say that the answer was revised after part of it was already heard.

        This is deliberately the one place in the streamed path that speaks
        text this module composed. It runs only when cognition has already
        finished and disagreed with itself, so there is no turn left to ask;
        the alternative is silence, which would leave the listener holding a
        sentence Aura does not stand behind.
        """
        final = str(getattr(verdict, "final", "") or "").strip()
        text = (
            "Hold on — let me correct that. " + final
            if final
            else (
                "Hold on — I need to take that back. What I just said did not "
                "survive my own checks, and I do not have a replacement answer yet."
            )
        )
        # Sent as a reply, not just spoken. The correction has to reach the
        # transcript and the chat thread as its own turn, or the visible
        # history keeps the retracted answer and disagrees with both what was
        # heard and what she remembers — which is the failure this is here to
        # prevent, reproduced one layer up.
        await self._send_json({"type": protocol.EVT_REPLY, "text": text})
        await self._speak_text(text, cause=ThinkingCause.UNCERTAINTY)

    async def _speak_text(self, text: str, *, cause: ThinkingCause | None) -> None:
        """Chunk, synthesise and stream one utterance whose text is known."""
        chunker = StreamingChunker(
            first_max_chars=self._config.tts.first_chunk_max_chars,
            max_chars=self._config.tts.chunk_max_chars,
        )
        pieces = chunker.push(text) + chunker.flush()

        async def _static_pieces() -> AsyncIterator[str]:
            for piece in pieces:
                yield piece

        await self._deliver_utterance(
            _static_pieces(),
            cause=cause,
            expected_pieces=len(pieces),
            intended=text,
        )

    async def _deliver_utterance(
        self,
        pieces: AsyncIterator[str],
        *,
        cause: ThinkingCause | None,
        expected_pieces: int | None = None,
        intended: str | None = None,
    ) -> str:
        """Synthesise and stream an utterance whose text may still be arriving.

        Taking an async source rather than a finished string is the whole
        difference between speaking after thinking and speaking while
        thinking. For a known reply the source is a list; for a governed
        stream it is clauses released as the model produces them, and this
        method cannot tell the difference — which is the point, because
        everything downstream (echo registration, playback accounting, the
        record of what was actually heard) then works identically for both.

        ``expected_pieces`` is how completeness is judged when it is knowable.
        A stream does not know its own length in advance, so it passes None
        and completeness is decided by whether the source ended on its own
        terms; see ``_speak_streamed_reply``.

        Returns the text actually delivered.
        """
        spec = self._prosody_spec()
        self._utterance_counter += 1
        track = _SpeakingTrack(
            utterance_id=self._utterance_counter,
            intended="",
            started_at=time.monotonic(),
        )
        self._speaking = track
        self._client_played_s = 0.0
        self._client_playback_utterance_id = track.utterance_id
        self._client_playback_drained = False
        self._client_playback_overflow_samples = 0
        self._overlap.reset()
        self._overlap_audio = []
        await self._set_state(SessionState.SPEAKING)

        intended_parts: list[str] = []

        async def _iter_chunks() -> AsyncIterator[str]:
            async for piece in pieces:
                if track.token.cancelled:
                    return
                intended_parts.append(piece)
                yield piece

        seq = 0
        first_audio = True
        synthesis_failed = False
        try:
            async for result in self._tts.stream(_iter_chunks(), spec, track.token):
                if track.token.cancelled or self._speaking is not track:
                    break
                if first_audio:
                    self._stop_fillers()
                    self._metrics.first_audio_at = time.monotonic()
                    self._metrics.tts_first_chunk_ms = result.synth_ms
                    first_audio = False

                track.chunks.append((result.text, result.duration_s))
                track.sent_duration_s += result.duration_s
                # Register the words so that if they come back through the
                # microphone a moment later, the echo guard recognises them.
                self._echo.note_spoken(result.text)

                await self._send_json(
                    {
                        "type": protocol.EVT_SPEAKING_CHUNK,
                        "text": result.text,
                        "seq": seq,
                        "utterance_id": track.utterance_id,
                    }
                )
                await self._send_binary(
                    protocol.encode_audio(
                        float32_to_pcm16(result.samples),
                        opcode=protocol.AudioOpcode.SPEECH,
                        seq=seq,
                        utterance_id=track.utterance_id,
                    )
                )
                seq += 1
        except asyncio.CancelledError:
            raise
        except (RuntimeError, ValueError, OSError, AttributeError) as exc:
            synthesis_failed = True
            record_degradation(
                "voice_duplex.session",
                exc,
                action="stopped synthesis for this utterance",
            )

        # What she meant to say. When the text was known up front it is the
        # whole of it, even if synthesis died before a single piece was
        # pulled — otherwise a total synthesis failure would record "nothing
        # was intended", and the unheard tail (which is what stops her
        # referring later to words nobody heard) would come out empty.
        # A stream has no such foreknowledge, so there it is what was released.
        intended_text = (
            str(intended)
            if intended is not None
            else " ".join(part.strip() for part in intended_parts if part.strip())
        )
        track.intended = intended_text

        if self._speaking is track and not track.token.cancelled:
            delivered_text = " ".join(chunk_text for chunk_text, _ in track.chunks)
            delivery_complete = bool(track.chunks) and not synthesis_failed and (
                expected_pieces is None or len(track.chunks) == expected_pieces
            )
            if not track.chunks:
                self._mind.record_spoken(
                    SpokenRecord(
                        intended=intended_text,
                        spoken="",
                        interrupted=False,
                        delivery_complete=False,
                        started_at=track.started_at,
                        ended_at=time.monotonic(),
                    )
                )
                self._speaking = None
                await self._send_json(
                    {
                        "type": protocol.EVT_ERROR,
                        "message": "Speech synthesis failed before audio was delivered.",
                    }
                )
                await self._mind.publish(
                    "speech_delivery_failed",
                    {"intended_chars": len(intended_text), "delivered_chars": 0},
                )
                await self._set_state(SessionState.LISTENING)
                return ""
            await self._send_binary(
                protocol.encode_audio(
                    b"",
                    opcode=protocol.AudioOpcode.SPEECH,
                    seq=seq,
                    utterance_id=track.utterance_id,
                    last=True,
                )
            )

            # Sending is not speaking. Kokoro synthesises 6–8x faster than
            # realtime, so a 14 s reply is fully transmitted in ~2 s while
            # the client is only two seconds into playing it. Returning to
            # LISTENING here would disarm barge-in for the remaining twelve
            # seconds — the user would talk over her and nothing would stop.
            # So the turn stays open until the audio has actually been heard.
            await self._await_playback_drain(track)
            if track.token.cancelled or self._speaking is not track:
                return delivered_text

            if self._client_playback_overflow_samples:
                delivery_complete = False
                record_degradation(
                    "voice_duplex.playback",
                    RuntimeError(
                        "client playback overflowed by "
                        f"{self._client_playback_overflow_samples} samples"
                    ),
                    action=(
                        "recorded speech delivery as incomplete instead of claiming "
                        "the dropped audio was heard"
                    ),
                    severity="warning",
                )

            self._mind.record_spoken(
                SpokenRecord(
                    intended=intended_text,
                    spoken=delivered_text,
                    interrupted=False,
                    delivery_complete=delivery_complete,
                    started_at=track.started_at,
                    ended_at=time.monotonic(),
                )
            )
            self._speaking = None
            # The floor is open from here. Inside the ambient window, a bare
            # reply needs no name — which is the whole reason a conversation
            # with her feels like one rather than like a series of commands.
            self._last_reply_ended_at = time.monotonic()
            await self._send_json({"type": protocol.EVT_METRICS, **self._metrics.as_dict()})
            await self._mind.publish(
                "spoke",
                {
                    "chars": len(delivered_text),
                    "delivery_complete": delivery_complete,
                    **self._metrics.as_dict(),
                },
            )
            await self._set_state(SessionState.LISTENING)
            return delivered_text
        return " ".join(chunk_text for chunk_text, _ in track.chunks)

    async def _await_playback_drain(self, track: _SpeakingTrack) -> None:
        """Stay in SPEAKING until the client has actually played the audio.

        The client's reported position is authoritative. The wall-clock
        deadline is a fallback for a client that stops reporting: without it
        a dropped report would strand the session in SPEAKING forever, which
        is a worse failure than ending the turn slightly early.
        """
        total = track.sent_duration_s
        if total <= 0:
            return
        deadline = time.monotonic() + total + 3.0
        while not track.token.cancelled and self._speaking is track:
            if (
                self._client_playback_utterance_id == track.utterance_id
                and self._client_playback_drained
            ):
                return
            if self._client_played_s >= total - 0.05:
                return
            if time.monotonic() >= deadline:
                logger.debug(
                    "Playback drain fell back to wall clock (played %.2fs of %.2fs)",
                    self._client_played_s,
                    total,
                )
                return
            await asyncio.sleep(0.05)

    async def _speak_aside(self, text: str, opcode: protocol.AudioOpcode, gain: float) -> None:
        """Synthesise a backchannel or filler.

        Asides deliberately do not touch ``_speaking``: they are not turns,
        they must not be interruptible as if they were, and they must not
        overwrite the record of what she actually said.
        """
        spec = self._prosody_spec()
        # An aside is not a line she arrives at, so it carries no bend.
        spec = spec.scaled(gain=spec.gain * gain).without_bend()
        token = CancellationToken()
        try:
            result = await self._tts.synthesize(text, spec, token)
        except (RuntimeError, ValueError, OSError, AttributeError) as exc:
            record_degradation(
                "voice_duplex.session",
                exc,
                action=f"skipped a {opcode.name.lower()} utterance",
                severity="warning",
            )
            return
        if result is None or self._closed:
            return
        await self._send_binary(
            protocol.encode_audio(
                float32_to_pcm16(result.samples),
                opcode=opcode,
                seq=0,
                utterance_id=0,
                last=True,
            )
        )

    def _cancel_playback(self) -> None:
        track = self._speaking
        if track is not None:
            track.token.cancel()
        self._speaking = None

    # ── client commands ──────────────────────────────────────────────────

    async def handle_command(self, message: dict[str, Any]) -> None:
        command = str(message.get("command") or message.get("type") or "")

        if command == protocol.CMD_STOP:
            await self._cancel_active_turn(
                reason="user_stop",
                return_to_listening=True,
            )
        elif command == protocol.CMD_MUTE:
            self._muted = True
            await self._send_json({"type": protocol.EVT_STATE, "state": self._state.value, "muted": True})
        elif command == protocol.CMD_UNMUTE:
            self._muted = False
            self._splitter.reset()
            self._vad.reset()
            await self._send_json({"type": protocol.EVT_STATE, "state": self._state.value, "muted": False})
        elif command == protocol.CMD_DEVICE_STATE:
            await self._apply_device_state(message)
        elif command == protocol.CMD_BARGE_IN:
            reported_utterance = self._bounded_nonnegative_int(
                message.get("utterance_id"),
                maximum=0xFFFFFFFF,
            )
            if (
                self._speaking is not None
                and reported_utterance
                and reported_utterance != self._speaking.utterance_id
            ):
                logger.debug(
                    "Ignored stale barge-in for utterance %s (current %s)",
                    reported_utterance,
                    self._speaking.utterance_id,
                )
                return
            played_ms = self._bounded_nonnegative_float(
                message.get("played_ms"),
                maximum=3_600_000.0,
            )
            await self._interrupt(reason="client_barge_in", played_s=played_ms / 1000.0)
            await self._cancel_active_turn(
                reason="client_barge_in",
                return_to_listening=True,
            )
        elif command == protocol.CMD_PLAYBACK:
            reported_utterance = self._bounded_nonnegative_int(
                message.get("utterance_id"),
                maximum=0xFFFFFFFF,
            )
            track = self._speaking
            if track is None:
                return
            if reported_utterance and reported_utterance != track.utterance_id:
                logger.debug(
                    "Ignored stale playback receipt for utterance %s (current %s)",
                    reported_utterance,
                    track.utterance_id,
                )
                return
            self._client_playback_utterance_id = track.utterance_id
            self._client_played_s = max(
                self._client_played_s,
                self._bounded_nonnegative_float(
                    message.get("played_ms"),
                    maximum=3_600_000.0,
                )
                / 1000.0,
            )
            self._client_playback_drained = bool(message.get("drained"))
            overflow_samples = self._bounded_nonnegative_int(
                message.get("overflow_samples"),
                maximum=100_000_000,
            )
            self._client_playback_overflow_samples += overflow_samples
            if overflow_samples:
                # Once a ring-buffer segment is dropped, later audio is no
                # longer a contiguous prefix of the intended utterance. Stop
                # at the last measured frame; continuing would make both the
                # audible sentence and the spoken-memory ledger unknowable.
                await self._interrupt(
                    reason="client_playback_overflow",
                    played_s=self._client_played_s,
                )
        elif command == protocol.CMD_SET_FLOOR:
            # Focused voice mode and push-to-talk are the user saying "this is
            # all for you". The addressivity gate is for a microphone that
            # happens to be on; it must not second-guess a deliberate act.
            #
            # When ambient gating is off entirely, the floor is permanently
            # open and the client cannot close it — otherwise a stale message
            # could switch on a gate the user never asked for.
            if self._ambient_gate is not None:
                self._floor_explicitly_open = bool(message.get("open"))
                logger.info(
                    "floor %s", "opened deliberately" if self._floor_explicitly_open else "released to ambient"
                )
        elif command == protocol.CMD_LIST_VOICES:
            await self._send_json(
                {"type": protocol.EVT_VOICES, "voices": self._tts.available_voices(),
                 "current": self._voice_override or self._config.tts.voice}
            )
        elif command == protocol.CMD_SET_VOICE:
            requested = str(message.get("voice") or "").strip()
            if requested and requested in self._tts.available_voices():
                self._voice_override = requested
                await self._send_json(
                    {"type": protocol.EVT_VOICES, "voices": self._tts.available_voices(),
                     "current": requested}
                )
                await self._mind.publish("voice_changed", {"voice": requested})
        elif command == protocol.CMD_TEXT:
            text = str(message.get("text") or "").strip()
            if text:
                if len(text.encode("utf-8", errors="replace")) > MAX_TYPED_MESSAGE_BYTES:
                    raise ValueError(
                        f"voice typed message exceeds {MAX_TYPED_MESSAGE_BYTES} bytes"
                    )
                await self._cancel_active_turn(
                    reason="superseded_by_typed_turn",
                    return_to_listening=False,
                )
                self._metrics = TurnMetrics(speech_end_at=time.monotonic())
                await self._set_state(SessionState.THINKING)
                self._turn_task = self._spawn(self._run_typed_turn(text))

    async def _apply_device_state(self, message: dict[str, Any]) -> None:
        """Commit one monotonic browser capture-generation receipt."""
        generation = self._bounded_nonnegative_int(
            message.get("generation"),
            maximum=0x7FFFFFFF,
        )
        if generation < self._device_generation:
            return
        state = str(message.get("state") or "").strip().lower()
        if state not in {"active", "recovering", "muted", "stopped", "unavailable"}:
            raise ValueError(f"unsupported voice device state: {state or 'empty'}")
        reason = str(message.get("reason") or "")[:160]
        discontinuity = state in {"recovering", "stopped", "unavailable"}
        if discontinuity:
            self._capture_available = False
            self._splitter.reset()
            self._vad.reset()
            self._utterance.clear()
            self._stable_text = ""
            self._discard_speculation()
            if self._state is SessionState.USER_SPEAKING:
                await self._set_state(SessionState.LISTENING)
        elif state == "active":
            self._capture_available = True

        self._device_generation = generation
        self._device_state = state
        self._device_reason = reason
        self._device_observed_at = time.monotonic()
        receipt = {
            "type": protocol.EVT_DEVICE,
            "generation": generation,
            "state": state,
            "reason": reason,
            "capture_available": self._capture_available,
        }
        await self._send_json(receipt)
        await self._mind.publish("voice_device_state", dict(receipt))

    async def _run_typed_turn(self, text: str) -> None:
        """A typed message while in voice mode still gets a spoken answer."""
        try:
            await self._send_json({"type": protocol.EVT_FINAL, "text": text, "typed": True})
            self._start_fillers()
            turn = await self._mind.respond_streaming(text)
            if turn is None:
                await self._set_state(SessionState.LISTENING)
                return
            async with self._cognition_bound_to_this_turn(turn):
                if await self._speak_streamed_reply(turn):
                    return
                reply = await turn.final()
                if reply:
                    await self._speak_reply(reply)
                else:
                    await self._speak_cognition_failure()
                    await self._set_state(SessionState.LISTENING)
        except asyncio.CancelledError:
            raise
        except (RuntimeError, ValueError, AttributeError, TypeError, OSError) as exc:
            record_degradation(
                "voice_duplex.session", exc, action="typed voice turn failed"
            )
            await self._set_state(SessionState.LISTENING)
        finally:
            self._stop_fillers()

    # ── helpers ──────────────────────────────────────────────────────────

    def _prosody_spec(self) -> Any:
        """Her compiled prosody, plus any delivery the user asked for.

        Substrate first, user override second: if she is low-energy the
        voice is still slower, but "talk faster" moves it from wherever it
        currently is rather than snapping to a fixed rate.
        """
        spec = self._prosody.compile(live_speech_profile())
        # A feeling past her ordinary range is carried by the voice as well as
        # the words. See `carry_breakthrough`.
        spec = carry_breakthrough(spec, live_affect())

        # Move partway toward how the user is speaking. Full mirroring is
        # mimicry; none at all is the flat-affect problem this exists to fix.
        conv_speed, conv_gain = convergence_factors(self._delivery)
        if conv_speed != 1.0 or conv_gain != 1.0:
            spec = spec.scaled(
                speed=max(0.7, min(1.4, spec.speed * conv_speed)),
                gain=max(0.25, min(1.3, spec.gain * conv_gain)),
            )

        adjustment = self._style.adjustment
        if adjustment.active:
            spec = spec.scaled(
                speed=max(0.7, min(1.4, spec.speed * (1.0 + adjustment.rate_delta))),
                gain=max(0.25, min(1.3, spec.gain * (1.0 + adjustment.gain_delta))),
            )
        if self._voice_override:
            spec.voice = self._voice_override
        return spec

    async def _set_state(self, state: SessionState) -> None:
        if state is self._state:
            return
        self._state = state
        await self._send_json({"type": protocol.EVT_STATE, "state": state.value})

    def _substrate_snapshot(self) -> dict[str, float]:
        """Current affect, for choosing a backchannel register."""
        try:
            profile = live_speech_profile()
            if profile is None:
                return {}
            return {
                "valence": float(getattr(profile, "warmth", 0.5)) * 2.0 - 1.0,
                "arousal": float(getattr(profile, "energy", 0.5)),
                "curiosity": float(getattr(profile, "playfulness", 0.3)),
            }
        except (AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "voice_duplex.session",
                exc,
                action="used a neutral backchannel register",
                severity="debug",
            )
            return {}

    @staticmethod
    def _bounded_nonnegative_float(value: Any, *, maximum: float) -> float:
        try:
            parsed = float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(parsed):
            return 0.0
        return max(0.0, min(parsed, maximum))

    @staticmethod
    def _bounded_nonnegative_int(value: Any, *, maximum: int) -> int:
        try:
            parsed = int(value or 0)
        except (TypeError, ValueError, OverflowError):
            return 0
        return max(0, min(parsed, int(maximum)))

    def _task_done(self, task: asyncio.Task[Any]) -> None:
        self._side_tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except BaseException as exc:  # noqa: BLE001 - done-callback boundary
            if _is_peer_disconnect(exc):
                # Ending voice mode is a thing people do. It is not a failure,
                # and reporting it as one produced a full traceback in the
                # neural feed every single time.
                self._closed = True
                return
            if isinstance(exc, asyncio.CancelledError):
                return
            if not isinstance(
                exc, (RuntimeError, ValueError, AttributeError, TypeError, OSError)
            ):
                raise
            record_degradation(
                "voice_duplex.session.background_task",
                exc,
                action="recorded an unexpected voice background-task failure",
                severity="warning",
            )

    def _spawn(self, coro: Any) -> asyncio.Task[Any]:
        """Track background tasks so close() can cancel every one."""
        task = get_task_tracker().create_task(
            coro,
            name=f"VoiceSession:{self._id}:side-task",
        )
        self._side_tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def status(self) -> dict[str, Any]:
        return {
            "session_id": self._id,
            "state": self._state.value,
            "muted": self._muted,
            "tts_engine": self._tts.engine_name,
            "vad_backend": self._vad.backend_name,
            "asr_available": self._asr.available,
            "metrics": self._metrics.as_dict(),
            "device": {
                "generation": self._device_generation,
                "state": self._device_state,
                "reason": self._device_reason,
                "capture_available": self._capture_available,
                "observed_age_s": round(
                    max(0.0, time.monotonic() - self._device_observed_at),
                    3,
                ),
            },
        }


SessionEvent = protocol  # re-exported for callers that want the event names

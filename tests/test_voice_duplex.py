"""Contract tests for the full-duplex voice lane.

These pin the behaviours that the end-to-end harness caught regressions in.
Three of them correspond to bugs that shipped-looking code actually had, and
each one silently defeated a headline feature rather than raising:

  * a trailing-word check that let Whisper's speculative full stop cut the
    user off mid-sentence,
  * a barge-in rule that could never fire because the ordinary onset rule
    always won the race,
  * a turn that closed when audio was *sent* rather than *heard*, disarming
    barge-in for most of the time the user was actually listening.

No model weights, no audio device, no network: everything here is either
pure logic or driven through injected fakes, so it runs in the offline suite.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from core.voice.duplex.audio import (
    FrameSplitter,
    UtteranceBuffer,
    float32_to_pcm16,
    pcm16_to_float32,
)
from core.voice.duplex.backchannel import BackchannelReflex
from core.voice.duplex.clause_chunker import StreamingChunker, first_chunk, split_for_speech
from core.voice.duplex.config import (
    VAD_FRAME_SAMPLES,
    AsrConfig,
    BackchannelConfig,
    DuplexConfig,
)
from core.voice.duplex.echo_guard import EchoGuard
from core.voice.duplex.endpointing import Completeness, Endpointer, classify
from core.voice.duplex.fillers import FillerReflex, ThinkingCause
from core.voice.duplex.mind_bridge import MindBridge, SpokenRecord
from core.voice.duplex.model_runtime import VoiceModelRuntime
from core.voice.duplex.prosody import ProsodySpec
from core.voice.duplex.protocol import AudioOpcode, decode_audio, encode_audio
from core.voice.duplex.session import (
    MAX_AUDIO_MESSAGE_BYTES,
    DuplexVoiceSession,
    _SpeakingTrack,
)
from core.voice.duplex.streaming_asr import StreamingAsr, looks_hallucinated
from core.voice.duplex.style import StyleController
from core.voice.duplex.tts_stream import (
    CancellationToken,
    StreamingTts,
)
from tests.chat_lane_support import patch_chat_lane

# ── endpointing ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        ("What time is it?", Completeness.COMPLETE),
        ("Hello.", Completeness.COMPLETE),
        ("Tell me about the way you handle interruptions", Completeness.COMPLETE),
        ("yeah", Completeness.COMPLETE),
        ("I was thinking we could maybe", Completeness.INCOMPLETE),
        ("I went to the", Completeness.INCOMPLETE),
        ("I think that's right but", Completeness.INCOMPLETE),
        ("so the thing is, um", Completeness.THINKING),
        ("wait...", Completeness.THINKING),
        ("the cat", Completeness.NEUTRAL),
    ],
)
def test_completeness_classification(text, expected):
    assert classify(text) is expected


def test_whisper_period_does_not_override_a_dangling_word():
    """The regression that cut users off mid-sentence.

    Whisper's full stops are a language-model prior, not an acoustic
    reading; it writes "So I was thinking that maybe." for someone who is
    plainly still talking. Trusting that period ends the turn early, which
    is the single worst failure this module can have.
    """
    assert classify("So I was thinking that maybe.") is Completeness.INCOMPLETE
    # A question mark is trustworthy and must still win.
    assert classify("What time is it?") is Completeness.COMPLETE


def test_silence_budget_adapts_to_completeness():
    ep = Endpointer()
    finished = ep.evaluate(
        transcript="What time is it?", silence_ms=400, speech_ms=2000, min_utterance_ms=220
    )
    midthought = ep.evaluate(
        transcript="I was thinking we could maybe",
        silence_ms=400,
        speech_ms=2000,
        min_utterance_ms=220,
    )
    assert finished.should_end is True
    # Same 400 ms pause, opposite decision — this is the whole point.
    assert midthought.should_end is False
    assert midthought.required_silence_ms > finished.required_silence_ms


def test_short_noise_burst_is_not_a_turn():
    ep = Endpointer()
    decision = ep.evaluate(
        transcript="hm", silence_ms=5000, speech_ms=80, min_utterance_ms=220
    )
    assert decision.should_end is False
    assert decision.reason == "below_min_utterance"


def test_turn_always_ends_eventually():
    ep = Endpointer()
    decision = ep.evaluate(
        transcript="and then I", silence_ms=5000, speech_ms=3000, min_utterance_ms=220
    )
    assert decision.should_end is True
    assert decision.reason == "max_silence"


# ── barge-in accounting ──────────────────────────────────────────────────


def test_spoken_prefix_reflects_only_what_played():
    """Her memory must hold what was heard, not what was sent."""
    track = _SpeakingTrack(intended="A B C. D E F. G H I.")
    track.chunks = [("A B C.", 1.0), ("D E F.", 1.0), ("G H I.", 1.0)]
    track.sent_duration_s = 3.0

    assert track.spoken_prefix(0.0) == ""
    assert track.spoken_prefix(1.0) == "A B C."
    assert track.spoken_prefix(2.0) == "A B C. D E F."
    assert track.spoken_prefix(3.0) == "A B C. D E F. G H I."

    # Mid-chunk interruption keeps a proportional share of the words.
    partial = track.spoken_prefix(1.5)
    assert partial.startswith("A B C.")
    assert "G H I." not in partial


def test_interruption_hands_the_unheard_tail_to_the_next_turn():
    record = SpokenRecord(
        intended="Yeah, I think so. The tricky part is the rest of this.",
        spoken="Yeah, I think so.",
        interrupted=True,
    )
    assert record.unheard == "The tricky part is the rest of this."

    bridge = MindBridge(session_id="t")
    bridge.record_spoken(record)
    effective = bridge._compose_effective_message("wait, what?")

    # The engine is told what the user did and did not hear...
    assert "did not hear" in effective
    assert "The tricky part" in effective
    # ...and the user's own words survive verbatim at the end.
    assert effective.endswith("wait, what?")


def test_uninterrupted_turn_adds_no_interruption_note():
    bridge = MindBridge(session_id="t")
    bridge.record_spoken(SpokenRecord(intended="All of it.", spoken="All of it.", interrupted=False))
    effective = bridge._compose_effective_message("next question")

    assert "did not hear" not in effective
    assert effective.endswith("next question")


def test_every_voice_turn_carries_the_spoken_length_directive():
    """Reply length *is* time-to-first-audio on this path.

    The governed turn returns one finished string, so nothing can be spoken
    until the last token is decoded. Dropping this directive silently costs
    seconds per reply, so it is pinned.
    """
    bridge = MindBridge(session_id="t", spoken_reply_words=45)
    effective = bridge._compose_effective_message("what do you think?")

    assert "spoken turn" in effective
    assert "45 words" in effective
    assert "No markdown" in effective
    # The user's own words still arrive last and verbatim.
    assert effective.endswith("what do you think?")


# ── echo rejection ───────────────────────────────────────────────────────


def test_echo_guard_rejects_her_own_words_returning():
    guard = EchoGuard()
    guard.note_spoken("The tricky part is that interruption handling has to edit what I said")
    verdict = guard.evaluate("the tricky part is that interruption handling has to edit")
    assert verdict.is_echo is True


def test_echo_guard_lets_a_real_interruption_through():
    guard = EchoGuard()
    guard.note_spoken("The tricky part is that interruption handling has to edit what I said")
    assert guard.evaluate("wait stop that is not what I meant").is_echo is False
    # Short interjections are exactly what a real barge-in looks like.
    assert guard.evaluate("no").is_echo is False


def test_echo_guard_is_inert_before_she_speaks():
    assert EchoGuard().evaluate("hello can you hear me").is_echo is False


# ── style control ────────────────────────────────────────────────────────


def test_delivery_requests_change_prosody():
    style = StyleController()
    assert style.observe("can you talk a bit slower")
    assert style.adjustment.rate_delta < 0
    assert style.observe("you're too loud")
    assert style.adjustment.gain_delta < 0


def test_topic_mention_of_speed_is_not_a_delivery_request():
    """"We should speed up the release" must not retune her voice."""
    style = StyleController()
    assert style.observe("we should speed up the release") == ""
    assert style.adjustment.active is False


def test_style_adjustments_stay_in_a_speakable_range():
    style = StyleController()
    for _ in range(20):
        style.observe("talk much faster")
    assert style.adjustment.rate_delta <= 0.28


# ── backchannels ─────────────────────────────────────────────────────────


def test_backchannel_needs_a_prosodic_boundary():
    reflex = BackchannelReflex(BackchannelConfig(fire_probability=1.0))
    reflex.on_user_turn_start(now=0.0)
    common = dict(speech_ms=6000.0, aura_is_speaking=False, now=100.0)

    # Too short to be a boundary — this is just inter-word silence.
    assert reflex.consider(silence_ms=50.0, **common).should_emit is False
    # Long enough to be an endpoint, not a boundary.
    assert reflex.consider(silence_ms=900.0, **common).should_emit is False
    # In the window.
    assert reflex.consider(silence_ms=250.0, **common).should_emit is True


def test_backchannel_never_talks_over_her_own_speech():
    reflex = BackchannelReflex(BackchannelConfig(fire_probability=1.0))
    reflex.on_user_turn_start(now=0.0)
    decision = reflex.consider(
        silence_ms=250.0, speech_ms=6000.0, aura_is_speaking=True, now=100.0
    )
    assert decision.should_emit is False
    assert decision.reason == "aura_speaking"


def test_backchannel_requires_the_user_to_have_held_the_floor():
    reflex = BackchannelReflex(BackchannelConfig(fire_probability=1.0))
    reflex.on_user_turn_start(now=0.0)
    decision = reflex.consider(
        silence_ms=250.0, speech_ms=500.0, aura_is_speaking=False, now=100.0
    )
    assert decision.should_emit is False


# ── fillers ──────────────────────────────────────────────────────────────


def test_filler_tiers_escalate_and_never_repeat_a_tier():
    reflex = FillerReflex()
    reflex.begin_turn()
    bounds = dict(first=380.0, second=1900.0, third=6500.0)

    assert reflex.due(100.0, **bounds) is None
    tier1 = reflex.due(400.0, **bounds)
    assert tier1 is not None and tier1.tier == 1
    assert reflex.due(500.0, **bounds) is None  # tier 1 already spent
    tier2 = reflex.due(2000.0, **bounds)
    assert tier2 is not None and tier2.tier == 2


def test_filler_words_follow_the_real_activity():
    """"Let me look that up" should mean a search is genuinely running."""
    reflex = FillerReflex()
    reflex.begin_turn()
    reflex.observe_activity("sovereign_browser")
    assert reflex.cause is ThinkingCause.WEB_SEARCH
    filler = reflex.due(2000.0, first=380.0, second=1900.0, third=6500.0)
    assert filler is not None
    assert filler.cause is ThinkingCause.WEB_SEARCH


# ── chunking ─────────────────────────────────────────────────────────────


def test_abbreviations_and_decimals_do_not_split_sentences():
    assert split_for_speech("Dr. Chen said it was 3.5 seconds. Then he left.", max_chars=200) == [
        "Dr. Chen said it was 3.5 seconds. Then he left."
    ]


def test_first_chunk_is_short_but_not_a_fragment():
    head, rest = first_chunk(
        "Yeah, I think that's basically right, though there's a wrinkle in it.", max_chars=48
    )
    assert len(head.split()) >= 2
    assert rest
    assert head.endswith(",") or head.endswith(".")


def test_streaming_chunker_preserves_word_boundaries():
    """Regression: a stripped remainder fused onto the next token.

    "median." + "Most" became "median.Most", which the TTS then pronounces
    as one mangled word.
    """
    chunker = StreamingChunker(first_max_chars=40, max_chars=90)
    out: list[str] = []
    for token in ["Okay, so ", "the answer is yes. ", "But it depends on the tail latency. ", "Most people say median."]:
        out += chunker.push(token)
    out += chunker.flush()
    joined = " ".join(out)
    assert "median.Most" not in joined
    assert ".M" not in joined.replace(". M", "")
    assert all(chunk.strip() for chunk in out)


def test_streaming_chunker_rejects_a_nonprogressing_splitter(monkeypatch):
    """A splitter defect must fail the turn instead of pinning the event loop."""
    from core.voice.duplex import clause_chunker

    monkeypatch.setattr(
        clause_chunker,
        "first_chunk",
        lambda text, *, max_chars: ("synthetic head", text),
    )
    chunker = StreamingChunker(first_max_chars=8, max_chars=8)

    with pytest.raises(RuntimeError, match="forward progress"):
        chunker.push("a buffer longer than the configured speech budget")


# ── audio plumbing ───────────────────────────────────────────────────────


def test_pcm_roundtrip_preserves_signal():
    original = np.linspace(-0.9, 0.9, 480, dtype=np.float32)
    restored = pcm16_to_float32(float32_to_pcm16(original))
    assert restored.shape == original.shape
    assert np.max(np.abs(restored - original)) < 1e-3


def test_torn_frame_does_not_raise():
    """An odd trailing byte is a torn socket read, not a reason to die."""
    assert pcm16_to_float32(b"\x01\x02\x03").size == 1


def test_frame_splitter_emits_exact_frames_and_keeps_remainder():
    splitter = FrameSplitter(VAD_FRAME_SAMPLES)
    assert splitter.push(np.zeros(VAD_FRAME_SAMPLES - 10, dtype=np.float32)) == []
    frames = splitter.push(np.zeros(20, dtype=np.float32))
    assert len(frames) == 1
    assert frames[0].size == VAD_FRAME_SAMPLES
    assert splitter.pending_samples == 10


def test_utterance_buffer_keeps_preroll_so_the_first_word_survives():
    buffer = UtteranceBuffer(max_seconds=10, sample_rate=16000, preroll_ms=100)
    for _ in range(10):
        buffer.observe_silence(np.ones(160, dtype=np.float32))
    buffer.begin()
    # VAD always confirms speech a frame or two late; without preroll the
    # leading consonant is gone and Whisper mis-hears the first word.
    assert buffer.sample_count > 0


# ── protocol ─────────────────────────────────────────────────────────────


def test_audio_frame_roundtrip():
    payload = b"\x01\x02\x03\x04"
    frame = encode_audio(payload, opcode=AudioOpcode.BACKCHANNEL, seq=9, utterance_id=1234, last=True)
    opcode, last, seq, utterance, body = decode_audio(frame)
    assert opcode is AudioOpcode.BACKCHANNEL
    assert (last, seq, utterance, body) == (True, 9, 1234, payload)


def test_truncated_frame_is_rejected():
    with pytest.raises(ValueError):
        decode_audio(b"\x01\x02")


# ── ASR guards ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text", ["", "  ", "Thank you for watching!", "please subscribe", "♪", "[BLANK_AUDIO]"]
)
def test_silence_hallucinations_are_discarded(text):
    """Whisper emits these confidently on silence. Answering one has her
    talking to an empty room."""
    assert looks_hallucinated(text) is True


@pytest.mark.parametrize("text", ["hey can you hear me", "yes", "no", "stop"])
def test_real_speech_is_not_discarded(text):
    assert looks_hallucinated(text) is False


# ── mind bridge ──────────────────────────────────────────────────────────


def test_activity_watch_unsubscribes_on_stop(monkeypatch):
    class FakeBus:
        def __init__(self) -> None:
            self.queue: asyncio.Queue[object] = asyncio.Queue()
            self.subscribed = asyncio.Event()
            self.unsubscribed = asyncio.Event()

        async def subscribe(self, topic: str):
            assert topic == "telemetry"
            self.subscribed.set()
            return self.queue

        async def unsubscribe(self, topic: str, queue):
            assert topic == "telemetry"
            assert queue is self.queue
            self.unsubscribed.set()

    async def exercise() -> None:
        from core import event_bus

        bus = FakeBus()
        monkeypatch.setattr(event_bus, "get_event_bus", lambda: bus)
        bridge = MindBridge(session_id="activity-lifecycle")

        await bridge.start_activity_watch(lambda _: None)
        await asyncio.wait_for(bus.subscribed.wait(), timeout=1.0)
        await bridge.stop_activity_watch()

        assert bridge._activity_stop.is_set()
        assert bridge._activity_task is None
        assert bus.unsubscribed.is_set()

    asyncio.run(exercise())


def test_restarting_fillers_stops_the_previous_turn_loop():
    async def send_json(_payload):
        return None

    async def send_binary(_payload):
        return None

    async def exercise() -> None:
        session = DuplexVoiceSession(
            session_id="filler-lifecycle",
            send_json=send_json,
            send_binary=send_binary,
        )

        session._start_fillers()
        first_task = session._filler_task
        first_stop = session._filler_stop
        assert first_task is not None
        assert first_stop is not None

        session._start_fillers()
        await asyncio.sleep(0)
        assert first_stop.is_set()
        assert first_task.done()

        second_task = session._filler_task
        second_stop = session._filler_stop
        session._stop_fillers()
        await asyncio.sleep(0)
        assert second_stop is not None and second_stop.is_set()
        assert second_task is not None and second_task.done()
        assert session._filler_task is None

    asyncio.run(exercise())


def test_governed_voice_turn_reuses_complete_chat_handler(monkeypatch):
    async def exercise() -> None:
        from fastapi.responses import JSONResponse

        from interface.routes import chat

        observed: dict[str, object] = {}

        patch_chat_lane(monkeypatch, "validate_runtime_security_request",
            lambda request: observed.update(security_path=request.url.path),
        )
        patch_chat_lane(monkeypatch, "_require_internal",
            lambda request: observed.update(internal_path=request.url.path),
        )
        patch_chat_lane(monkeypatch, "_check_rate_limit",
            lambda request: observed.update(rate_path=request.url.path),
        )

        async def fake_api_chat(*, body, request, _, __):
            observed["message"] = body.message
            observed["session_id"] = body.session_id
            observed["surface"] = request.headers["x-aura-response-surface"]
            observed["context"] = chat._INTERNAL_SURFACE_CONTEXT.get()
            observed["device_token"] = request.headers["x-aura-device-token"]
            return JSONResponse({"response": "governed voice reply", "status": "ok"})

        monkeypatch.setattr(chat, "api_chat", fake_api_chat)
        reply = await chat.run_governed_voice_chat_turn(
            "hello",
            surface_context="[spoken turn]",
            session_id="voice-governed",
            timeout_s=2.0,
            source_headers=((b"x-aura-device-token", b"adt1.bound"),),
            client_host="10.0.0.8",
        )

        assert reply == "governed voice reply"
        assert observed == {
            "message": "hello",
            "session_id": "voice-governed",
            "surface": "voice",
            "context": "[spoken turn]",
            "device_token": "adt1.bound",
            "security_path": "/api/chat",
            "internal_path": "/api/chat",
            "rate_path": "/api/chat",
        }

    asyncio.run(exercise())


def test_governed_voice_turn_preserves_remote_owner_token(monkeypatch):
    async def exercise() -> None:
        from fastapi.responses import JSONResponse

        from interface import auth
        from interface.routes import chat

        observed: dict[str, str] = {}

        monkeypatch.setattr(auth.config, "api_token", "owner-secret")
        monkeypatch.setattr(auth.config.security, "internal_only_mode", False)

        async def fake_api_chat(*, body, request, _, __):
            observed["api_token"] = request.headers["x-api-token"]
            observed["surface"] = request.headers["x-aura-response-surface"]
            return JSONResponse({"response": "owner reply", "status": "ok"})

        monkeypatch.setattr(chat, "api_chat", fake_api_chat)
        reply = await chat.run_governed_voice_chat_turn(
            "hello",
            surface_context="[spoken turn]",
            session_id="voice-owner",
            timeout_s=2.0,
            source_headers=((b"x-api-token", b"owner-secret"),),
            client_host="203.0.113.8",
        )

        assert reply == "owner reply"
        assert observed == {
            "api_token": "owner-secret",
            "surface": "voice",
        }

    asyncio.run(exercise())


def test_governed_voice_handoff_preserves_only_authenticated_principal():
    from interface.routes.voice_duplex import _governed_chat_source_headers

    hostile_headers = (
        (b"authorization", b"Bearer attacker"),
        (b"cookie", b"aura_device_session=adt1.attacker"),
        (b"x-api-token", b"attacker"),
        (b"x-aura-device-token", b"adt1.attacker"),
        (b"user-agent", b"AuraVoiceTest"),
    )

    owner_headers = _governed_chat_source_headers(
        hostile_headers,
        owner_token="owner-secret",
        device_token=None,
    )
    assert owner_headers == (
        (b"user-agent", b"AuraVoiceTest"),
        (b"x-api-token", b"owner-secret"),
    )

    device_headers = _governed_chat_source_headers(
        hostile_headers,
        owner_token=None,
        device_token="adt1.verified",
    )
    assert device_headers == (
        (b"user-agent", b"AuraVoiceTest"),
        (b"x-aura-device-token", b"adt1.verified"),
    )

    local_headers = _governed_chat_source_headers(
        hostile_headers,
        owner_token=None,
        device_token=None,
        local_owner=True,
    )
    assert local_headers == ((b"user-agent", b"AuraVoiceTest"),)

    with pytest.raises(ValueError, match="exactly one authenticated identity"):
        _governed_chat_source_headers(
            (),
            owner_token="owner-secret",
            device_token="adt1.verified",
        )


def test_repeated_typed_turn_cancels_and_quiesces_previous_turn():
    async def send_json(_payload):
        return None

    async def send_binary(_payload):
        return None

    async def exercise() -> None:
        first_started = asyncio.Event()
        first_cancelled = asyncio.Event()
        second_started = asyncio.Event()
        calls = 0

        async def responder(
            transcript,
            *,
            effective_message,
            session_id,
            timeout_s,
        ):
            nonlocal calls
            calls += 1
            if calls == 1:
                first_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    first_cancelled.set()
                    raise
            else:
                second_started.set()
            return None

        session = DuplexVoiceSession(
            session_id="typed-supersession",
            send_json=send_json,
            send_binary=send_binary,
            mind=MindBridge(session_id="typed-supersession", responder=responder),
        )
        await session.handle_command({"command": "text", "text": "first"})
        await asyncio.wait_for(first_started.wait(), timeout=1.0)
        first_task = session._turn_task

        await session.handle_command({"command": "text", "text": "second"})
        await asyncio.wait_for(first_cancelled.wait(), timeout=1.0)
        # Wait for the replacement turn to actually enter cognition rather
        # than assuming one loop pass is enough to get there. Cognition runs
        # in its own task now (that is what lets a reply be spoken while it is
        # still forming), so "the turn was spawned" and "the turn is thinking"
        # are separated by a scheduling hop. Waiting on the event asserts the
        # stronger property the test is actually about.
        await asyncio.wait_for(second_started.wait(), timeout=1.0)

        assert first_task is not None and first_task.done()
        assert session._turn_task is not first_task
        assert calls == 2
        await session.close()
        assert all(task.done() for task in tuple(session._side_tasks))

    asyncio.run(exercise())


def test_replacement_turn_fails_closed_when_prior_task_will_not_quiesce(
    monkeypatch,
):
    from core.voice.duplex import session as session_module

    async def send_json(_payload):
        return None

    async def send_binary(_payload):
        return None

    async def exercise() -> None:
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        calls = 0

        async def responder(
            transcript,
            *,
            effective_message,
            session_id,
            timeout_s,
        ):
            nonlocal calls
            calls += 1
            if calls == 1:
                first_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    await release_first.wait()
            return None

        monkeypatch.setattr(session_module, "TASK_QUIESCENCE_TIMEOUT_S", 0.01)
        session = DuplexVoiceSession(
            session_id="typed-nonquiescent",
            send_json=send_json,
            send_binary=send_binary,
            mind=MindBridge(session_id="typed-nonquiescent", responder=responder),
        )
        await session.handle_command({"command": "text", "text": "first"})
        await asyncio.wait_for(first_started.wait(), timeout=1.0)
        first_task = session._turn_task

        with pytest.raises(
            RuntimeError,
            match="prior_voice_turn_failed_to_quiesce",
        ):
            await session.handle_command({"command": "text", "text": "second"})

        assert session._turn_task is first_task
        assert calls == 1
        release_first.set()
        assert first_task is not None
        # The turn ends cancelled, and that is the correct outcome: it *was*
        # cancelled. A responder that swallows its CancelledError no longer
        # launders the turn into a normal completion — cognition runs in its
        # own task, so the turn task's own cancellation stands regardless of
        # what the responder does with its. What matters here is that the task
        # finishes once released rather than leaking.
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(first_task, timeout=1.0)
        assert first_task.done()
        await session.close()

    asyncio.run(exercise())


def test_stop_cancels_cognition_and_close_prevents_late_transport_sends():
    async def exercise() -> None:
        sends: list[dict[str, object]] = []
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def send_json(payload):
            sends.append(dict(payload))

        async def send_binary(_payload):
            return None

        async def responder(
            transcript,
            *,
            effective_message,
            session_id,
            timeout_s,
        ):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        session = DuplexVoiceSession(
            session_id="stop-cognition",
            send_json=send_json,
            send_binary=send_binary,
            mind=MindBridge(session_id="stop-cognition", responder=responder),
        )
        await session.handle_command({"command": "text", "text": "long turn"})
        await asyncio.wait_for(started.wait(), timeout=1.0)
        await session.handle_command({"command": "stop"})
        await asyncio.wait_for(cancelled.wait(), timeout=1.0)
        count_at_close = len(sends)
        await session.close()
        await asyncio.sleep(0.05)

        assert session.state.value == "closed"
        assert len(sends) == count_at_close

    asyncio.run(exercise())


def test_zero_audio_is_never_recorded_as_fully_spoken():
    class SilentTts:
        async def stream(self, _chunks, _spec, _token):
            if False:
                yield None

        def shutdown(self):
            return None

    async def exercise() -> None:
        events: list[dict[str, object]] = []

        async def send_json(payload):
            events.append(dict(payload))

        async def send_binary(_payload):
            return None

        mind = MindBridge(session_id="delivery-truth")
        session = DuplexVoiceSession(
            session_id="delivery-truth",
            send_json=send_json,
            send_binary=send_binary,
            mind=mind,
        )
        session._tts = SilentTts()

        await session._speak_text(
            "This sentence was never synthesized.",
            cause=None,
        )
        record = mind.last_spoken

        assert record is not None
        assert record.spoken == ""
        assert record.delivery_complete is False
        assert record.unheard == "This sentence was never synthesized."
        assert any(event.get("type") == "voice.error" for event in events)
        await session.close()

    asyncio.run(exercise())


def test_oversized_audio_is_rejected_before_frame_iteration():
    async def send_json(_payload):
        return None

    async def send_binary(_payload):
        return None

    async def exercise() -> None:
        session = DuplexVoiceSession(
            session_id="oversized-audio",
            send_json=send_json,
            send_binary=send_binary,
        )
        with pytest.raises(ValueError, match="audio message exceeds"):
            await session.feed_audio(b"\0" * (MAX_AUDIO_MESSAGE_BYTES + 1))
        await session.close()

    asyncio.run(exercise())


def test_voice_session_admission_is_globally_and_per_principal_bounded(monkeypatch):
    from interface.routes import voice_duplex

    with voice_duplex._SESSION_RESERVATION_LOCK:
        voice_duplex._SESSION_RESERVATIONS.clear()
    monkeypatch.setattr(voice_duplex, "MAX_VOICE_SESSIONS", 2)
    monkeypatch.setattr(voice_duplex, "MAX_VOICE_SESSIONS_PER_PRINCIPAL", 1)

    assert voice_duplex._reserve_voice_session("a", "owner:a") is True
    assert voice_duplex._reserve_voice_session("b", "owner:a") is False
    assert voice_duplex._reserve_voice_session("c", "owner:b") is True
    assert voice_duplex._reserve_voice_session("d", "owner:c") is False

    voice_duplex._release_voice_session("a")
    voice_duplex._release_voice_session("c")


def test_authentication_rejects_non_object_json(monkeypatch):
    async def exercise() -> None:
        from interface import auth, server
        from interface.routes import voice_duplex

        class FakeSocket:
            client = SimpleNamespace(host="203.0.113.5")
            scope = {"headers": ()}

            def __init__(self):
                self.closed: list[tuple[int, str]] = []

            async def accept(self):
                return None

            async def receive_text(self):
                return "[]"

            async def close(self, *, code, reason):
                self.closed.append((code, reason))

        socket = FakeSocket()
        monkeypatch.setattr(auth, "device_for_request", lambda _request: None)
        monkeypatch.setattr(
            auth,
            "request_has_allowed_local_browser_origin",
            lambda _request: False,
        )
        monkeypatch.setattr(server.config, "api_token", "expected")

        await voice_duplex.voice_duplex_endpoint(socket)
        assert socket.closed == [(4001, "Invalid Auth Payload")]

    asyncio.run(exercise())


def test_silent_paired_revocation_closes_outgoing_lane(monkeypatch):
    async def exercise() -> None:
        from interface import auth, server
        from interface.routes import voice_duplex

        device = SimpleNamespace(device_id="paired-1")
        verify_calls = 0

        def verify(_token):
            nonlocal verify_calls
            verify_calls += 1
            return device if verify_calls == 1 else None

        class FakeSession:
            def __init__(self, **_kwargs):
                return None

            async def start(self):
                return None

            async def close(self):
                return None

            def status(self):
                return {"state": "listening"}

        class FakeSocket:
            client = SimpleNamespace(host="203.0.113.5")
            scope = {"headers": ()}

            def __init__(self):
                self.closed = asyncio.Event()
                self.text: list[str] = []

            async def accept(self):
                return None

            async def receive_text(self):
                return json.dumps({"type": "auth", "token": "adt1.paired"})

            async def receive(self):
                await self.closed.wait()
                return {"type": "websocket.disconnect"}

            async def send_text(self, value):
                self.text.append(value)

            async def send_bytes(self, _value):
                return None

            async def close(self, *, code, reason):
                self.close_code = code
                self.close_reason = reason
                self.closed.set()

        socket = FakeSocket()
        monkeypatch.setattr(auth, "device_for_request", lambda _request: None)
        monkeypatch.setattr(
            auth,
            "request_has_allowed_local_browser_origin",
            lambda _request: False,
        )
        monkeypatch.setattr(server, "_verify_ws_device_token", verify)
        monkeypatch.setattr(server, "_live_device_scopes", lambda _device_id: {"voice"})
        monkeypatch.setattr(voice_duplex, "DuplexVoiceSession", FakeSession)
        monkeypatch.setattr(server.config, "api_token", "expected")

        await asyncio.wait_for(
            voice_duplex.voice_duplex_endpoint(socket),
            timeout=2.0,
        )
        assert socket.close_code == 4003
        assert socket.close_reason == "Voice authorization revoked"
        assert any("paired_device_session_revoked" in text for text in socket.text)
        assert voice_duplex._SESSION_RESERVATIONS == {}

    asyncio.run(exercise())


def test_remote_owner_token_rotation_revokes_open_voice_session(monkeypatch):
    async def exercise() -> None:
        from interface import auth, server
        from interface.routes import voice_duplex

        class FakeSession:
            def __init__(self, **_kwargs):
                return None

            async def start(self):
                server.config.api_token = "rotated"

            async def close(self):
                return None

            def status(self):
                return {"state": "listening"}

        class FakeSocket:
            client = SimpleNamespace(host="203.0.113.6")
            scope = {"headers": ()}

            def __init__(self):
                self.closed = asyncio.Event()
                self.text: list[str] = []

            async def accept(self):
                return None

            async def receive_text(self):
                return json.dumps({"type": "auth", "token": "expected"})

            async def receive(self):
                await self.closed.wait()
                return {"type": "websocket.disconnect"}

            async def send_text(self, value):
                self.text.append(value)

            async def send_bytes(self, _value):
                return None

            async def close(self, *, code, reason):
                self.close_code = code
                self.close_reason = reason
                self.closed.set()

        socket = FakeSocket()
        monkeypatch.setattr(auth, "device_for_request", lambda _request: None)
        monkeypatch.setattr(
            auth,
            "request_has_allowed_local_browser_origin",
            lambda _request: False,
        )
        monkeypatch.setattr(server, "_verify_ws_device_token", lambda _token: None)
        monkeypatch.setattr(server, "_live_device_scopes", lambda _device_id: set())
        monkeypatch.setattr(voice_duplex, "DuplexVoiceSession", FakeSession)
        monkeypatch.setattr(server.config, "api_token", "expected")

        await asyncio.wait_for(
            voice_duplex.voice_duplex_endpoint(socket),
            timeout=2.0,
        )
        assert socket.close_code == 4003
        assert socket.close_reason == "Voice authorization revoked"
        assert any("owner_session_revoked" in text for text in socket.text)
        assert voice_duplex._SESSION_RESERVATIONS == {}

    asyncio.run(exercise())


def test_duplex_asr_model_load_holds_fenced_owner(monkeypatch):
    from core.runtime import model_lane_control
    from core.voice.duplex.streaming_asr import _WhisperBackend

    events: list[object] = []

    class Lease:
        def set_preemptible(self, value):
            events.append(("preemptible", value))
            return True

        def release(self, *, reason):
            events.append(("release", reason))
            return True

    def acquire(**kwargs):
        events.append(("acquire", kwargs))
        return Lease()

    class FakeMlx:
        @staticmethod
        def transcribe(_audio, **_kwargs):
            return {"text": "hello"}

    monkeypatch.setattr(
        model_lane_control,
        "acquire_synchronous_in_process_model_lane",
        acquire,
    )
    backend = _WhisperBackend(
        AsrConfig(partial_model="small", final_model="large"),
    )
    backend._impl = "mlx"
    backend._mlx = FakeMlx()

    assert backend.transcribe(np.zeros(16, dtype=np.float32), "small") == "hello"
    backend.shutdown()

    assert events[0][0] == "acquire"
    assert events[1] == ("preemptible", True)
    assert events[-1] == ("release", "voice_asr_shutdown")


def test_mlx_partial_and_final_models_remain_resident_without_alternating_reload(
    monkeypatch,
):
    from core.runtime import model_lane_control
    from core.voice.duplex.streaming_asr import _WhisperBackend

    loads: list[str] = []

    class Lease:
        @staticmethod
        def set_preemptible(_value):
            return True

        @staticmethod
        def release(*, reason):
            assert reason == "voice_asr_shutdown"
            return True

    class Holder:
        model = None
        model_path = None

    class FakeMlx:
        @staticmethod
        def transcribe(_audio, *, path_or_hf_repo, **_kwargs):
            if Holder.model is None or Holder.model_path != path_or_hf_repo:
                loads.append(path_or_hf_repo)
                Holder.model = object()
                Holder.model_path = path_or_hf_repo
            return {"text": path_or_hf_repo}

    monkeypatch.setattr(
        model_lane_control,
        "acquire_synchronous_in_process_model_lane",
        lambda **_kwargs: Lease(),
    )
    backend = _WhisperBackend(
        AsrConfig(partial_model="small", final_model="large"),
    )
    backend._impl = "mlx"
    backend._mlx = FakeMlx()
    backend._mlx_holder = Holder

    audio = np.zeros(16, dtype=np.float32)
    assert backend.transcribe(audio, "small") == "small"
    assert backend.transcribe(audio, "large") == "large"
    assert backend.transcribe(audio, "small") == "small"
    assert loads == ["small", "large"]

    backend.shutdown()
    assert Holder.model is None
    assert Holder.model_path is None


def test_asr_runtime_construction_does_not_import_native_models(monkeypatch):
    from core.voice.duplex import streaming_asr

    imported: list[str] = []

    def fail_import(name):
        imported.append(name)
        raise AssertionError(f"native import occurred during construction: {name}")

    monkeypatch.setattr(
        streaming_asr.importlib.util,
        "find_spec",
        lambda name: object() if name == "mlx_whisper" else None,
    )
    monkeypatch.setattr(streaming_asr.importlib, "import_module", fail_import)
    backend = streaming_asr._WhisperBackend(AsrConfig())

    assert backend.status()["backend"] == "mlx"
    assert backend.status()["native_module_loaded"] is False
    assert imported == []


def test_process_voice_runtime_shares_models_but_not_transcript_state():
    runtime = VoiceModelRuntime(
        DuplexConfig(asr=AsrConfig(partial_model="small", final_model="large"))
    )
    first = runtime.new_asr()
    second = runtime.new_asr()

    assert first._backend is second._backend
    first._prev_words = ["one"]
    first._stable_words = ["one"]
    assert second._prev_words == []
    assert second._stable_words == []
    assert runtime.tts is runtime.tts
    status = runtime.status()
    assert status["closed"] is False
    assert status["asr"]["retained_models"] == []
    assert status["tts"]["warmed"] is False

    runtime.shutdown()
    assert runtime.status()["closed"] is True


def test_session_does_not_shutdown_injected_process_models():
    async def exercise() -> None:
        calls: list[str] = []

        class SharedAsr:
            available = True

            async def warm_up(self):
                return None

            def shutdown(self):
                calls.append("asr_shutdown")

        class SharedTts:
            engine_name = "shared"

            async def warm_up(self, _spec):
                return None

            def shutdown(self):
                calls.append("tts_shutdown")

        class FakeMind:
            async def stop_activity_watch(self):
                return None

            async def publish(self, _event, _payload):
                return None

        async def send_json(_payload):
            return None

        async def send_binary(_payload):
            return None

        session = DuplexVoiceSession(
            session_id="shared-models",
            send_json=send_json,
            send_binary=send_binary,
            mind=FakeMind(),
            asr=SharedAsr(),
            tts=SharedTts(),
        )
        await session.close()
        assert calls == []

    asyncio.run(exercise())


def test_voice_ready_is_withheld_until_both_process_models_are_warm():
    async def exercise() -> None:
        events: list[dict[str, object]] = []

        class ColdAsr:
            available = True

            async def warm_up(self):
                return False

        class ReadyTts:
            engine_name = "kokoro"

            async def warm_up(self, _spec):
                return True

        class FakeMind:
            async def start_activity_watch(self, _callback):
                raise AssertionError("mind activity must not start before model readiness")

            async def stop_activity_watch(self):
                return None

            async def publish(self, _event, _payload):
                return None

        async def send_json(payload):
            events.append(dict(payload))

        async def send_binary(_payload):
            return None

        session = DuplexVoiceSession(
            session_id="cold-models",
            send_json=send_json,
            send_binary=send_binary,
            mind=FakeMind(),
            asr=ColdAsr(),
            tts=ReadyTts(),
        )
        with pytest.raises(RuntimeError, match="voice_models_not_ready"):
            await session.start()
        assert any(
            event.get("status") == "voice_models_not_ready" for event in events
        )
        assert not any(event.get("type") == "voice.ready" for event in events)
        assert not any(event.get("state") == "listening" for event in events)
        await session.close()

    asyncio.run(exercise())


def test_tts_warmup_is_singleflight_across_sessions(monkeypatch):
    async def exercise() -> None:
        from core.runtime.lockdep import assert_no_locks_held

        tts = StreamingTts()
        tts._state.kokoro = object()
        calls = 0

        async def loaded():
            assert_no_locks_held("TTS warmup model load", strict=True)
            return True

        async def synthesize(_text, _spec, _token):
            nonlocal calls
            assert_no_locks_held("TTS warmup synthesis", strict=True)
            calls += 1
            await asyncio.sleep(0)
            return object()

        monkeypatch.setattr(tts, "ensure_loaded", loaded)
        monkeypatch.setattr(tts, "synthesize", synthesize)
        spec = ProsodySpec(voice="af_heart")
        await asyncio.gather(tts.warm_up(spec), tts.warm_up(spec))
        assert calls == 1
        tts.shutdown()

    asyncio.run(exercise())


def test_tts_model_lane_effects_run_outside_the_lifecycle_lock(monkeypatch):
    from core.runtime import model_lane_control
    from core.runtime.lockdep import assert_no_locks_held

    released: list[str] = []

    class Lease:
        def release(self, *, reason):
            assert_no_locks_held("TTS model lane release", strict=True)
            released.append(reason)
            return True

    def acquire(**_kwargs):
        assert_no_locks_held("TTS model lane acquire", strict=True)
        return Lease()

    monkeypatch.setattr(
        model_lane_control,
        "acquire_synchronous_in_process_model_lane",
        acquire,
    )
    tts = StreamingTts()

    assert isinstance(tts._acquire_model_lane(), Lease)
    tts.shutdown()

    assert released == ["voice_tts_shutdown"]


def test_tts_releases_a_model_lane_that_arrives_after_shutdown(monkeypatch):
    from core.runtime import model_lane_control

    admission_started = threading.Event()
    finish_admission = threading.Event()
    released: list[str] = []
    failures: list[BaseException] = []

    class Lease:
        def release(self, *, reason):
            released.append(reason)
            return True

    def acquire(**_kwargs):
        admission_started.set()
        assert finish_admission.wait(timeout=2.0)
        return Lease()

    monkeypatch.setattr(
        model_lane_control,
        "acquire_synchronous_in_process_model_lane",
        acquire,
    )
    tts = StreamingTts()

    def admit() -> None:
        try:
            tts._acquire_model_lane()
        except BaseException as exc:  # noqa: BLE001 - crossing a test thread
            failures.append(exc)

    thread = threading.Thread(target=admit, name="late-tts-admission")
    thread.start()
    assert admission_started.wait(timeout=1.0)
    tts.shutdown()
    finish_admission.set()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)
    assert released == ["voice_tts_closed_during_model_admission"]
    assert tts._lane_lease is None


def test_cancelled_native_tts_retains_model_owner_until_worker_returns():
    async def exercise() -> None:
        native_started = threading.Event()
        release_native = threading.Event()
        released: list[str] = []

        class Lease:
            def release(self, *, reason):
                released.append(reason)
                return True

        class BlockingEngine:
            name = "blocking-test"

            @staticmethod
            def synthesize(_text, _spec):
                native_started.set()
                assert release_native.wait(timeout=2.0)
                return np.zeros(24, dtype=np.float32), 24_000

        tts = StreamingTts()
        tts._state.loaded = True
        tts._state.piper = BlockingEngine()
        tts._lane_lease = Lease()
        tts._accepting_synthesis = True

        task = asyncio.create_task(
            tts.synthesize(
                "hello",
                ProsodySpec(voice="test"),
                CancellationToken(),
            )
        )
        assert await asyncio.to_thread(native_started.wait, 1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        tts.shutdown()
        assert released == []
        assert tts._active_syntheses == 1

        release_native.set()
        for _ in range(100):
            if released:
                break
            await asyncio.sleep(0.01)
        assert released == ["voice_tts_shutdown"]
        assert tts._active_syntheses == 0

    asyncio.run(exercise())


# ── coqui compatibility shim ─────────────────────────────────────────────


def test_shim_restores_the_symbol_coqui_needs():
    """coqui-TTS imports a helper transformers 5.x removed.

    Pinning transformers back would satisfy the TTS package at the cost of
    the version mlx-lm and the resident 32B are built against — risking the
    mind to gain a voice option. The shim reinstates the one symbol instead.
    """
    from core.voice.duplex import coqui_compat

    assert coqui_compat.apply() is True
    import transformers.pytorch_utils as pytorch_utils

    assert hasattr(pytorch_utils, "isin_mps_friendly")


def test_shim_is_idempotent_and_does_not_clobber_a_real_symbol():
    import transformers.pytorch_utils as pytorch_utils

    from core.voice.duplex import coqui_compat

    coqui_compat.apply()
    sentinel = pytorch_utils.isin_mps_friendly
    coqui_compat._applied = False  # force a second pass
    coqui_compat.apply()
    assert pytorch_utils.isin_mps_friendly is sentinel


def test_shim_matches_torch_isin_semantics():
    import torch

    from core.voice.duplex.coqui_compat import _isin_mps_friendly

    elements = torch.tensor([1, 2, 3, 4, 5])
    test = torch.tensor([2, 4])
    assert torch.equal(_isin_mps_friendly(elements, test), torch.isin(elements, test))


def test_cloned_voice_refuses_without_licence_acceptance(monkeypatch, tmp_path):
    """XTTS-v2 is CPML-licensed; accepting on the operator's behalf is not
    this code's call, so it must fail closed with a reason."""
    from core.voice.duplex.config import TtsConfig
    from core.voice.duplex.tts_stream import _ClonedVoiceEngine

    for name in ("AURA_COQUI_CPML_ACCEPTED", "AURA_COQUI_COMMERCIAL_LICENSED", "COQUI_TOS_AGREED"):
        monkeypatch.delenv(name, raising=False)

    clip = tmp_path / "ref.wav"
    clip.write_bytes(b"RIFF")
    config = TtsConfig()
    config.clone_reference = str(clip)

    assert _ClonedVoiceEngine(config).load() is False


def test_responder_failure_returns_none_rather_than_inventing_a_reply():
    async def broken(transcript, *, effective_message, session_id, timeout_s):
        raise RuntimeError("cognition lane down")

    bridge = MindBridge(session_id="t", responder=broken)
    assert asyncio.run(bridge.respond("hello")) is None


def test_empty_transcript_never_reaches_cognition():
    calls: list[str] = []

    async def responder(transcript, *, effective_message, session_id, timeout_s):
        calls.append(transcript)
        return "hi"

    bridge = MindBridge(session_id="t", responder=responder)
    assert asyncio.run(bridge.respond("   ")) is None
    assert calls == []


# ── overlap: "mhm" is not an interruption ────────────────────────────────


def _drive_overlap(arbiter, *, speech_ms, then_silence_ms, energy=0.1):
    """Feed frames until a verdict settles."""
    from core.voice.duplex.overlap import OverlapVerdict as V

    frame = 32.0
    arbiter.begin()
    verdict = V.PENDING
    for _ in range(int(speech_ms / frame)):
        verdict = arbiter.observe(frame_ms=frame, is_speech=True, energy=energy)
        if verdict is not V.PENDING:
            return verdict
    for _ in range(int(then_silence_ms / frame)):
        verdict = arbiter.observe(frame_ms=frame, is_speech=False, energy=0.0)
        if verdict is not V.PENDING:
            return verdict
    return verdict


def test_short_acknowledgement_does_not_stop_her():
    """The defect this module exists to fix.

    Saying "mhm" while she talks previously killed her mid-sentence, which
    punishes the user for being a good listener.
    """
    from core.voice.duplex.overlap import OverlapArbiter, OverlapVerdict

    verdict = _drive_overlap(OverlapArbiter(), speech_ms=220, then_silence_ms=400)
    assert verdict is OverlapVerdict.BACKCHANNEL


def test_sustained_speech_takes_the_floor():
    from core.voice.duplex.overlap import OverlapArbiter, OverlapVerdict

    verdict = _drive_overlap(OverlapArbiter(), speech_ms=1200, then_silence_ms=0)
    assert verdict is OverlapVerdict.BARGE_IN


def test_ducking_happens_before_any_verdict():
    """Volume must drop while the decision is still pending — that is what
    makes the response instant without making it irreversible."""
    from core.voice.duplex.overlap import OverlapArbiter, OverlapVerdict

    arbiter = OverlapArbiter()
    arbiter.begin()
    ducked_at = None
    for i in range(20):
        verdict = arbiter.observe(frame_ms=32.0, is_speech=True, energy=0.1)
        if arbiter.should_duck() and ducked_at is None:
            ducked_at = (i + 1) * 32.0
            assert verdict is OverlapVerdict.PENDING
    assert ducked_at is not None and ducked_at <= 200.0


def test_duck_fires_only_once():
    from core.voice.duplex.overlap import OverlapArbiter

    arbiter = OverlapArbiter()
    arbiter.begin()
    ducks = 0
    for _ in range(30):
        arbiter.observe(frame_ms=32.0, is_speech=True, energy=0.1)
        if arbiter.should_duck():
            ducks += 1
    assert ducks == 1


@pytest.mark.parametrize("text", ["mhm", "yeah", "right", "okay", "haha", "yeah yeah"])
def test_acknowledgement_tokens_recognised(text):
    from core.voice.duplex.overlap import looks_like_backchannel

    assert looks_like_backchannel(text) is True


@pytest.mark.parametrize(
    "text", ["yeah but no", "wait that's wrong", "no I meant the other one", "stop talking"]
)
def test_real_objections_are_not_acknowledgement(text):
    from core.voice.duplex.overlap import looks_like_backchannel

    assert looks_like_backchannel(text) is False


def test_transcript_overrides_a_timing_misread():
    """A short sharp objection ("no—") has backchannel *timing*. The words
    are the tiebreaker."""
    from core.voice.duplex.overlap import OverlapArbiter, OverlapVerdict

    arbiter = OverlapArbiter()
    _drive_overlap(arbiter, speech_ms=220, then_silence_ms=400)
    assert arbiter.resolve("no wait that's wrong") is OverlapVerdict.BARGE_IN
    arbiter2 = OverlapArbiter()
    _drive_overlap(arbiter2, speech_ms=220, then_silence_ms=400)
    assert arbiter2.resolve("mhm") is OverlapVerdict.BACKCHANNEL


@pytest.mark.parametrize("text", ["no", "wait", "stop", "why"])
def test_single_word_objections_take_the_floor(text):
    from core.voice.duplex.overlap import OverlapArbiter, OverlapVerdict

    arbiter = OverlapArbiter()
    _drive_overlap(arbiter, speech_ms=220, then_silence_ms=400)
    assert arbiter.resolve(text) is OverlapVerdict.BARGE_IN


def test_overlap_probe_does_not_mutate_local_agreement(monkeypatch):
    async def exercise() -> None:
        asr = StreamingAsr(AsrConfig())
        asr._prev_words = ["existing"]
        asr._stable_words = ["existing"]
        asr._tentative_words = ["tail"]
        asr._last_partial_at = 123.0
        monkeypatch.setattr(
            type(asr._backend),
            "available",
            property(lambda _self: True),
        )
        monkeypatch.setattr(
            asr._backend,
            "transcribe",
            lambda _audio, _repo: "no",
        )

        assert await asr.probe(np.ones(1600, dtype=np.float32)) == "no"
        assert asr._prev_words == ["existing"]
        assert asr._stable_words == ["existing"]
        assert asr._tentative_words == ["tail"]
        assert asr._last_partial_at == 123.0

    asyncio.run(exercise())


def test_verified_short_objection_preserves_audio_and_closes_the_turn():
    class FakeAsr:
        def __init__(self) -> None:
            self.reset_calls = 0

        async def probe(self, _audio):
            return "no"

        def reset(self):
            self.reset_calls += 1

        def shutdown(self):
            return None

    async def send_json(_payload):
        return None

    async def send_binary(_payload):
        return None

    async def exercise() -> None:
        observed: dict[str, object] = {}
        session = DuplexVoiceSession(
            session_id="verified-overlap",
            send_json=send_json,
            send_binary=send_binary,
        )
        session._asr = FakeAsr()

        async def capture_turn(audio, reason):
            observed["audio"] = audio.copy()
            observed["reason"] = reason
            await session._set_state(type(session._state).LISTENING)

        session._run_turn = capture_turn
        session._speaking = _SpeakingTrack(
            utterance_id=1,
            intended="original answer",
            started_at=0.0,
        )
        session._state = type(session._state).SPEAKING
        captured = np.ones(3200, dtype=np.float32)
        track = session._speaking

        await session._verify_backchannel(captured, track, session._overlap_epoch)
        assert session._turn_task is not None
        await session._turn_task
        assert np.array_equal(observed["audio"], captured)
        assert observed["reason"] == "verified_barge_in"
        assert session._asr.reset_calls == 1
        assert session.state.value == "listening"
        await session.close()

    asyncio.run(exercise())


def test_stale_overlap_verifier_cannot_restore_playback():
    async def exercise() -> None:
        events: list[dict[str, object]] = []

        async def send_json(payload):
            events.append(dict(payload))

        async def send_binary(_payload):
            return None

        session = DuplexVoiceSession(
            session_id="stale-overlap",
            send_json=send_json,
            send_binary=send_binary,
        )
        track = _SpeakingTrack(utterance_id=1, intended="answer")
        session._speaking = track
        session._overlap_epoch = 2

        await session._resume_after_backchannel(track, "mhm", 1)
        assert not any(event.get("type") == "voice.duck" for event in events)
        await session.close()

    asyncio.run(exercise())


def test_playback_receipts_are_monotonic_and_bound_to_the_utterance():
    async def exercise() -> None:
        events: list[dict[str, object]] = []

        async def send_json(payload):
            events.append(dict(payload))

        async def send_binary(_payload):
            return None

        session = DuplexVoiceSession(
            session_id="playback-receipts",
            send_json=send_json,
            send_binary=send_binary,
        )
        session._speaking = _SpeakingTrack(utterance_id=7, intended="answer")

        await session.handle_command(
            {"command": "playback", "utterance_id": 6, "played_ms": 900}
        )
        assert session._client_played_s == 0.0

        await session.handle_command(
            {"command": "playback", "utterance_id": 7, "played_ms": 500}
        )
        await session.handle_command(
            {"command": "playback", "utterance_id": 7, "played_ms": 200}
        )
        assert session._client_played_s == 0.5

        # A buffer overflow destroys prefix continuity, so the session stops
        # at the last measured sample instead of claiming later audio played.
        await session.handle_command(
            {
                "command": "playback",
                "utterance_id": 7,
                "played_ms": 500,
                "drained": True,
                "overflow_samples": 320,
            }
        )
        assert session._client_playback_utterance_id == 7
        assert session._client_playback_drained is True
        assert session._client_playback_overflow_samples == 320
        assert session._speaking is None
        assert any(event.get("type") == "voice.interrupted" for event in events)
        await session.close()

    asyncio.run(exercise())


def test_stale_barge_in_cannot_interrupt_a_newer_utterance():
    async def exercise() -> None:
        events: list[dict[str, object]] = []

        async def send_json(payload):
            events.append(dict(payload))

        async def send_binary(_payload):
            return None

        session = DuplexVoiceSession(
            session_id="stale-client-barge",
            send_json=send_json,
            send_binary=send_binary,
        )
        track = _SpeakingTrack(
            utterance_id=9,
            intended="new answer",
            started_at=1.0,
        )
        session._speaking = track
        session._state = type(session._state).SPEAKING

        await session.handle_command(
            {"command": "barge_in", "utterance_id": 8, "played_ms": 100}
        )
        assert session._speaking is track
        assert not any(event.get("type") == "voice.interrupted" for event in events)
        await session.close()

    asyncio.run(exercise())


# ── paralinguistics ──────────────────────────────────────────────────────


def test_pitch_tracking_recovers_a_known_tone():
    from core.voice.duplex.paralinguistics import estimate_f0

    t = np.arange(16000 * 0.5) / 16000
    tone = (0.3 * np.sin(2 * np.pi * 150.0 * t)).astype(np.float32)
    voiced = estimate_f0(tone, 16000)
    voiced = voiced[~np.isnan(voiced)]
    assert voiced.size > 5
    assert abs(float(np.median(voiced)) - 150.0) < 8.0


def test_delivery_stays_quiet_without_a_baseline():
    """Reporting a mood from an absolute number is invention. With fewer
    than three samples there is no baseline, so there is nothing to say."""
    from core.voice.duplex.paralinguistics import SpeakerBaseline, analyze, interpret

    audio = (0.1 * np.sin(2 * np.pi * 140 * np.arange(16000) / 16000)).astype(np.float32)
    reading = interpret(analyze(audio, 16000, word_count=4), SpeakerBaseline())
    assert reading.as_context() == ""


def test_imperceptible_change_is_not_reported():
    """A tiny baseline variance makes an inaudible difference score many
    sigma; without a perceptibility floor she announces "quieter than usual"
    on a turn that sounded identical."""
    from core.voice.duplex.paralinguistics import SpeakerBaseline, VoiceSignature

    baseline = SpeakerBaseline()
    for value in (0.100, 0.101, 0.099, 0.100):
        sig = VoiceSignature(energy_rms=value, duration_s=2.0, voiced_ratio=0.6)
        baseline.observe(sig)
    nearly_identical = VoiceSignature(energy_rms=0.102, duration_s=2.0, voiced_ratio=0.6)
    assert baseline.energy_z(nearly_identical) == 0.0


def test_large_change_from_constant_baseline_is_visible():
    from core.voice.duplex.paralinguistics import SpeakerBaseline, VoiceSignature

    baseline = SpeakerBaseline()
    for _ in range(4):
        baseline.observe(
            VoiceSignature(energy_rms=0.1, duration_s=2.0, voiced_ratio=0.6)
        )
    louder = VoiceSignature(energy_rms=0.15, duration_s=2.0, voiced_ratio=0.6)
    assert baseline.energy_z(louder) > 1.15


def test_convergence_is_partial_not_mimicry():
    """Full mirroring reads as mockery; none at all is the flat-affect
    problem. Bounded partial movement is the point."""
    from core.voice.duplex.paralinguistics import DeliveryReading, convergence_factors

    speed, gain = convergence_factors(DeliveryReading(rate_z=5.0, energy_z=5.0))
    assert 1.0 < speed <= 1.15
    assert 1.0 < gain <= 1.2
    slow_speed, _ = convergence_factors(DeliveryReading(rate_z=-5.0, energy_z=-5.0))
    assert 0.85 <= slow_speed < 1.0


def test_neutral_delivery_leaves_her_voice_alone():
    from core.voice.duplex.paralinguistics import DeliveryReading, convergence_factors

    assert convergence_factors(DeliveryReading()) == (1.0, 1.0)


# ── adaptive reply length ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "question,ceiling",
    [
        ("is the build green?", 25),
        ("what time is it", 25),
        ("how many are left", 25),
    ],
)
def test_closed_questions_get_short_answers(question, ceiling):
    bridge = MindBridge(session_id="t", spoken_reply_words=45)
    words, offer = bridge._reply_budget_for(question)
    assert words <= ceiling
    assert offer is False


def test_explanatory_questions_get_room_and_an_offer():
    bridge = MindBridge(session_id="t", spoken_reply_words=45)
    words, offer = bridge._reply_budget_for("explain the tradeoff between the two")
    assert words >= 80
    assert offer is True


def test_conversational_default_is_unchanged():
    bridge = MindBridge(session_id="t", spoken_reply_words=45)
    words, offer = bridge._reply_budget_for("so I was thinking about the queue again")
    assert words == 45
    assert offer is False


# ── predictive fillers ───────────────────────────────────────────────────


def test_known_slow_cause_announces_itself_immediately():
    """"Let me look that up" at 300ms beats "uh…" then the same sentence at
    1.9s — knowing *why* she is slow is better than knowing *that* she is."""
    reflex = FillerReflex()
    reflex.begin_turn()
    reflex.observe_activity("sovereign_browser")

    filler = reflex.due(50.0, first=380.0, second=1900.0, third=6500.0)
    assert filler is not None
    assert filler.tier == 2
    assert filler.cause is ThinkingCause.WEB_SEARCH
    # And she must not then say "uh…" after already explaining herself.
    assert reflex.due(500.0, first=380.0, second=1900.0, third=6500.0) is None


def test_unknown_cause_still_waits_its_turn():
    reflex = FillerReflex()
    reflex.begin_turn()
    assert reflex.due(50.0, first=380.0, second=1900.0, third=6500.0) is None
    assert reflex.due(400.0, first=380.0, second=1900.0, third=6500.0) is not None


def test_voice_cognition_has_no_ungoverned_streaming_bypass():
    import inspect

    from core.voice.duplex.config import DuplexConfig

    assert "stream_reply" not in DuplexConfig.__dataclass_fields__
    assert not hasattr(MindBridge, "stream_response")
    assert "think_stream" not in inspect.getsource(MindBridge)

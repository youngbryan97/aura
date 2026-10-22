"""Voice presentation measurements expire when their source changes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from core.organism.nothing_measured import NothingMeasured

SOURCES = (
    "interface/static/aura.js",
    "core/voice/duplex/config.py", "core/voice/duplex/tts_stream.py",
    "core/voice/duplex/pocket_speech.py", "core/voice/duplex/session.py",
    "core/voice/duplex/original_speech.py", "core/voice/duplex/treatment.py",
    "tools/voice/original_speech_worker.py",
    "interface/static/voice_mode.js", "interface/static/voice_mode.css",
    "interface/static/voice-capture-processor.js", "interface/static/voice-playback-processor.js",
)
CHECKS = frozenset({
    "ready audio does not wait for another clause", "both playback lanes flushed",
    "local interrupt before server command", "late audio rejected", "next turn plays",
    "physical microphone stopped", "390px width", "keyboard containment", "focus restored",
    "local model synthesis", "no routine metrics notices",
    "settings refresh preserves focused voice",
    "voice preview is playback",
})
STREAM_SOURCES = (
    'core/brain/cognitive_engine_quick_reply.py', 'core/brain/request_contract.py',
    'core/brain/inference_gate_turn_setup.py', 'core/brain/llm_health_router_endpoint_call.py',
    'core/brain/llm/mlx_client.py', 'core/brain/llm/mlx_worker.py',
    'core/brain/llm/reply_previews.py', 'core/brain/llm/mlx_worker_surface_quality.py',
    'core/conversation/reply_stream.py', 'core/cognition/evidence_relevance.py',
    'core/voice/duplex/clause_chunker.py', 'core/voice/duplex/governed_stream.py',
    'core/voice/duplex/session.py',
    'tests/test_governed_reply_previews.py', 'tests/test_voice_sentence_onset.py',
    'tests/test_evidence_documents_are_not_encoded_again.py',
    'tests/test_voice_cognition_is_timed_before_playback.py',
    'tests/test_server_conversation_lane.py',
)
STREAM_CHECKS = frozenset({
    'checked opener released before cognition finishes', 'sentence boundary lookahead',
    'owning request channel captured', 'cancelled and foreign previews rejected',
    'history vectors bound to encoder', 'cognition measured independently of playback',
    'ordinary state contracts retain conversation resume',
})


def measured_voice(root: Path | None = None) -> bool:
    return _measured_receipt(root, 'voice-polish-*.json', 'aura.voice.presentation_measurement.v1',
                             SOURCES, CHECKS)


def measured_voice_streaming(root: Path | None = None) -> bool:
    return _measured_receipt(root, 'voice-streaming-*.json', 'aura.voice.streaming_measurement.v1',
                             STREAM_SOURCES, STREAM_CHECKS)


def _measured_receipt(root, pattern, schema, sources, checks) -> bool:
    root = root or Path(__file__).resolve().parents[2]
    for path in sorted((root / 'docs/evidence').glob(pattern), reverse=True):
        try:
            receipt = json.loads(path.read_text())
            if not isinstance(receipt, dict) or receipt.get('schema') != schema:
                continue
            if any(receipt.get("sources", {}).get(p) != hashlib.sha256((root / p).read_bytes()).hexdigest()
                   for p in sources):
                continue
        except (OSError, ValueError):
            continue
        return receipt.get("ok") is True and checks <= set(receipt.get("checks", []))
    raise NothingMeasured(f'No {schema} measurement matches the current source.')


def install_voice_presentation_claims(suite) -> None:
    from core.organism.model_validation import (
        Claim,
        Evidence,
        Observation,
        ValidationTest,
        boolean_score,
    )

    statement = "The measured voice client interrupts locally, rejects late audio, and releases the microphone when muted."
    suite.add_test(ValidationTest(
        name="voice_presentation_measured", description=statement, required_capability="",
        observation=Observation(name="voice_presentation", value=True,
                                source="tests/test_voice_ready_audio_does_not_wait.py; docs/evidence/voice-polish-*.json"),
        predict=lambda _m: measured_voice(),
        score=lambda p, o: boolean_score(bool(p), expected=bool(o.value), subject="voice presentation"),
        owner="interface/static/voice_mode.js",
    ))
    suite.add_claim(Claim(
        statement=statement, test="voice_presentation_measured", owner="interface/static/voice_mode.js",
        asserted_in=__file__, evidence=Evidence.MEASURED_SYNTHETIC,
        evidence_note="Chromium with a simulated socket and microphone, plus local model synthesis. This does not establish conversational parity with ChatGPT Voice.",
    ))
    statement = 'The isolated voice transport releases a checked opening sentence while cognition continues and keeps previews within their owning request.'
    suite.add_test(ValidationTest(
        name='voice_streaming_measured', description=statement, required_capability='',
        observation=Observation(name='voice_streaming', value=True,
                                source='tests/test_voice_sentence_onset.py; tests/test_governed_reply_previews.py'),
        predict=lambda _m: measured_voice_streaming(),
        score=lambda p, o: boolean_score(bool(p), expected=bool(o.value), subject='voice streaming'),
        owner='core/brain/llm/reply_previews.py',
    ))
    suite.add_claim(Claim(
        statement=statement, test='voice_streaming_measured', owner='core/brain/llm/reply_previews.py',
        asserted_in=__file__, evidence=Evidence.MEASURED_SYNTHETIC,
        evidence_note='Injected model frames and speech callbacks. Model inference, microphone capture, synthesis and OS playback were not exercised. Live conversation latency remains unmeasured for this change.',
    ))

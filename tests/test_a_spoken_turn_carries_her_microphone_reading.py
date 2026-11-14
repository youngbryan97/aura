"""A turn said aloud carries the reading of her microphone, as a camera look carries the camera's.

LIVE 13 November 2026 he asked aloud "can you hear me?" and she answered "my audio input is currently inactive" and
"your text is landing directly with me": nothing in the turn said the words had come through her microphone.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.conversation import turn_evidence_custody as custody
from core.senses.turn_evidence import TurnSensoryEvidence, build_microphone_turn_evidence
from core.voice.duplex import live_context
from interface.routes import chat_preflight
from interface.routes.chat_common import _INTERNAL_SURFACE_HEARD_AT


def _request(host: str, surface: str = "") -> SimpleNamespace:
    headers = {"X-Aura-Surface": surface} if surface else {}
    return SimpleNamespace(headers=headers, client=SimpleNamespace(host=host))


def test_the_reading_is_the_microphone_channel_with_the_words_heard():
    reading = TurnSensoryEvidence.from_value(build_microphone_turn_evidence("can you hear me?", source="test",
                                                                             heard_at=100.0))
    assert reading is not None
    assert (reading.channel, reading.ok, reading.observed_at) == ("microphone", True, 100.0)
    assert "can you hear me?" in reading.observation
    assert "microphone" in reading.observation


def test_read_before_the_words_are_known_it_names_none():
    reading = TurnSensoryEvidence.from_value(build_microphone_turn_evidence(source="test"))
    assert reading is not None and reading.request == ""
    assert "this turn's words" in reading.observation


@pytest.fixture
def recorded(monkeypatch):
    seen: list = []
    grounding: list = []
    monkeypatch.setattr(custody, "record_turn_sensory_evidence", lambda value: seen.append(value) or True)
    monkeypatch.setattr(custody, "record_turn_grounding", lambda value: grounding.append(value) or True)
    return seen, grounding


def test_a_governed_voice_turn_records_the_reading(recorded):
    seen, grounding = recorded
    token = _INTERNAL_SURFACE_HEARD_AT.set(1234.5)
    try:
        reading = chat_preflight._record_the_microphone_reading("are you there?", None)
    finally:
        _INTERNAL_SURFACE_HEARD_AT.reset(token)
    assert reading is not None and reading["channel"] == "microphone"
    assert seen == [reading]
    assert "channel: microphone" in grounding[0] and "are you there?" in grounding[0]


def test_a_typed_turn_records_nothing(recorded):
    seen, grounding = recorded
    assert chat_preflight._record_the_microphone_reading("are you there?", _request("127.0.0.1")) is None
    assert seen == [] and grounding == []


def test_the_wake_lane_post_over_loopback_is_believed(recorded):
    seen, _ = recorded
    assert chat_preflight._record_the_microphone_reading("hello?", _request("127.0.0.1", "voice")) is not None
    assert len(seen) == 1


def test_a_voice_header_from_another_host_is_not(recorded):
    seen, _ = recorded
    assert chat_preflight._record_the_microphone_reading("hello?", _request("192.168.1.20", "voice")) is None
    assert seen == []


def test_her_voice_live_context_reads_it_and_a_typed_turn_while_waking_does_not(monkeypatch):
    monkeypatch.setattr(live_context, "still_waking", lambda: "")
    heard = live_context.the_turns_evidence(heard=True)
    typed = live_context.the_turns_evidence()
    assert "channel: microphone" in heard
    assert "channel: microphone" not in typed

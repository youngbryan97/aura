"""Was that meant for her? Her model's reading decides; the floor and the room's acoustics stay what they were.

Until 13 November 2026 these tests pinned a ladder of phrase lists, and the ladder answered 46 of 70 declared
utterances that were not said to her (tools/voice/measure_address_margin.py). The judgment is now a reading of which
talk the words continue (core/voice/duplex/address_reading.py); what is tested here is how the gate acts on a reading,
and what it does when there is none. How well the reading itself separates situations is measured on her cortex and
recorded in artifacts/voice/address_margin.json, not asserted from phrases.
"""
from __future__ import annotations

import asyncio

import pytest

from core.voice.duplex.address_reading import AddressReading
from core.voice.duplex.addressivity import AddressContext, AddressivityGate

MARGIN = 2.0


@pytest.fixture()
def gate() -> AddressivityGate:
    return AddressivityGate(names=("aura",))


def read(hers: float, alone: float, room: float | None = None, **context) -> AddressContext:
    context.setdefault("duration_s", 2.0)
    return AddressContext(reading=AddressReading(hers=hers, alone=alone, room=room, tokens=6), margin=MARGIN,
                          **context)


def test_words_her_conversation_predicts_best_are_answered(gate):
    verdict = gate.evaluate("the blue one", read(hers=-1.0, alone=-4.5, since_last_reply_s=4.0))
    assert verdict.addressed and verdict.rung == "read"
    assert verdict.reading["lead"] == pytest.approx(3.5)


def test_words_the_room_predicts_as_well_are_not(gate):
    verdict = gate.evaluate("can you see how it thickens", read(hers=-2.0, alone=-5.0, room=-1.5))
    assert not verdict.addressed and verdict.rung == "read_elsewhere"
    assert "the room's talk" in verdict.vetoes[0]


def test_the_lead_is_held_to_the_margin_not_to_zero(gate):
    assert not gate.evaluate("hmm that's weird", read(hers=-2.0, alone=-3.5)).addressed  # lead 1.5 < 2.0


def test_quiet_speech_outside_an_exchange_is_not_taken_as_aimed_here(gate):
    quiet = read(hers=-1.0, alone=-5.0, loudness_z=-2.5)
    verdict = gate.evaluate("what's the weather tomorrow", quiet)
    assert not verdict.addressed and verdict.rung == "read_but_not_aimed_here"


def test_quiet_speech_inside_an_open_exchange_is_still_hers(gate):
    assert gate.evaluate("yeah do that", read(hers=-1.0, alone=-5.0, loudness_z=-2.5, since_last_reply_s=5.0)).addressed


def test_a_known_low_confidence_vetoes_and_an_unknown_one_does_not(gate):
    assert not gate.evaluate("what time is it", read(hers=-1.0, alone=-5.0, asr_confidence=0.3)).addressed
    assert gate.evaluate("what time is it", read(hers=-1.0, alone=-5.0, asr_confidence=None)).addressed


def test_opening_the_floor_deliberately_outranks_everything(gate):
    context = read(hers=-9.0, alone=-1.0, room=-0.5, loudness_z=-4.0, competing_speech=True,
                   floor_explicitly_open=True)
    verdict = gate.evaluate("the traffic was unbelievable", context)
    assert verdict.addressed and verdict.rung == "explicit"


def test_without_a_reading_her_name_or_an_open_exchange_is_still_answered(gate):
    assert gate.evaluate("Aura, are you there?", AddressContext()).rung == "named_unread"
    assert gate.evaluate("okay go ahead", AddressContext(since_last_reply_s=5.0)).rung == "open_floor_unread"


def test_without_a_reading_nothing_else_is(gate):
    assert not gate.evaluate("okay go ahead", AddressContext(since_last_reply_s=5.0, competing_speech=True)).addressed
    assert not gate.evaluate("what's on tv tonight", AddressContext()).addressed
    # Her name as part of another word is not her name.
    assert not gate.evaluate("the aurora was out last night", AddressContext()).addressed


def test_a_reading_without_a_margin_is_not_judged_against_a_made_up_one(gate):
    context = AddressContext(reading=AddressReading(hers=-1.0, alone=-6.0), margin=None)
    assert gate.evaluate("what's the time", context).rung == "unread"


def test_every_verdict_carries_its_reasons(gate):
    for verdict in (gate.evaluate("the blue one", read(hers=-1.0, alone=-4.5)),
                    gate.evaluate("call now", read(hers=-3.0, alone=-4.0, room=-1.0)),
                    gate.evaluate("hello", AddressContext())):
        assert verdict.reasons or verdict.vetoes
        assert verdict.narrative()


def test_empty_input_is_not_addressed(gate):
    assert not gate.evaluate("", AddressContext()).addressed
    assert not gate.evaluate("   ", read(hers=0.0, alone=-9.0)).addressed


# ── the session ──────────────────────────────────────────────────────────


async def _send(_payload):
    return None


class _Reader:
    """Stands in for her model: a fixed reading, and a record of what was overheard."""

    def __init__(self, reading: AddressReading | None, margin: float | None = MARGIN) -> None:
        self._reading = reading
        self.margin = None if margin is None else type("Margin", (), {"margin": margin})()
        self.room: list[str] = []
        self.asked: list[str] = []

    async def read(self, transcript: str):
        self.asked.append(transcript)
        return self._reading

    def overheard(self, transcript: str) -> None:
        self.room.append(transcript)


def test_the_session_judges_by_her_reading_and_keeps_what_was_not_hers_as_the_rooms_talk():
    from core.voice.duplex.session import DuplexVoiceSession

    async def exercise() -> None:
        elsewhere = _Reader(AddressReading(hers=-3.0, alone=-3.5, room=-1.0))
        session = DuplexVoiceSession(session_id="room", send_json=_send, send_binary=_send, address_reader=elsewhere)
        verdict = await session._evaluate_addressivity("call now and get your first month free", None)
        assert not verdict.addressed and verdict.rung == "read_elsewhere"
        assert elsewhere.room == ["call now and get your first month free"]
        await session.close()

        hers = _Reader(AddressReading(hers=-1.0, alone=-5.0))
        session = DuplexVoiceSession(session_id="hers", send_json=_send, send_binary=_send, address_reader=hers)
        verdict = await session._evaluate_addressivity("hello? is anyone there?", None)
        assert verdict.addressed and verdict.rung == "read"
        assert hers.room == []
        await session.close()

    asyncio.run(exercise())


def test_with_no_margin_measured_the_session_does_not_ask_her_model():
    from core.voice.duplex.session import DuplexVoiceSession

    async def exercise() -> None:
        reader = _Reader(AddressReading(hers=-1.0, alone=-5.0), margin=None)
        session = DuplexVoiceSession(session_id="unmeasured", send_json=_send, send_binary=_send,
                                     address_reader=reader)
        verdict = await session._evaluate_addressivity("Aura, are you there?", None)
        assert verdict.addressed and verdict.rung == "named_unread"
        assert reader.asked == []
        await session.close()

    asyncio.run(exercise())


def test_the_session_stands_the_gate_down_when_the_floor_is_opened():
    """Focused voice mode is consent, and consent outranks the judgment: the user opened it precisely to talk."""
    from core.voice.duplex.session import DuplexVoiceSession

    async def exercise() -> None:
        reader = _Reader(AddressReading(hers=-9.0, alone=-1.0))
        session = DuplexVoiceSession(session_id="floor", send_json=_send, send_binary=_send, address_reader=reader)
        assert session._ambient_gate is not None
        assert session._floor_explicitly_open is False

        await session.handle_command({"command": "set_floor", "open": True})
        assert session._floor_explicitly_open is True
        verdict = await session._evaluate_addressivity("the traffic was unbelievable", None)
        assert verdict.addressed and verdict.rung == "explicit"
        assert reader.asked == []  # nothing to judge, so her model is not asked

        await session.handle_command({"command": "set_floor", "open": False})
        assert session._floor_explicitly_open is False
        assert not (await session._evaluate_addressivity("the traffic was unbelievable", None)).addressed

        await session.close()

    asyncio.run(exercise())


def test_a_disabled_gate_cannot_be_switched_back_on_by_a_message():
    """With ambient gating off, the floor is permanently open; a stale or hostile message must not add a gate."""
    from core.voice.duplex.config import DuplexConfig
    from core.voice.duplex.session import DuplexVoiceSession

    async def exercise() -> None:
        config = DuplexConfig()
        config.ambient.enabled = False
        session = DuplexVoiceSession(session_id="no-gate", send_json=_send, send_binary=_send, config=config)
        assert session._ambient_gate is None
        assert session._floor_explicitly_open is True

        await session.handle_command({"command": "set_floor", "open": False})
        assert session._floor_explicitly_open is True
        assert (await session._evaluate_addressivity("anything at all", None)).addressed

        await session.close()

    asyncio.run(exercise())

"""A selected voice plays a sample without inventing a conversation turn."""
from unittest.mock import AsyncMock

import pytest

from core.voice.duplex.mind_bridge import MindBridge
from core.voice.duplex.protocol import AudioOpcode
from core.voice.duplex.prosody import ProsodySpec
from core.voice.duplex.session import DuplexVoiceSession


@pytest.mark.asyncio
async def test_voice_preview_never_enters_cognition_or_the_transcript(monkeypatch):
    responder = AsyncMock(side_effect=AssertionError("preview entered cognition"))
    events = []

    async def send_json(payload):
        events.append(payload)

    session = DuplexVoiceSession(
        session_id="preview-test", send_json=send_json, send_binary=AsyncMock(),
        mind=MindBridge(session_id="preview-test", responder=responder),
    )
    monkeypatch.setattr(session._tts, "available_voices", lambda: ["alba"])
    play = AsyncMock()
    monkeypatch.setattr(session, "_speak_aside", play)
    try:
        await session.handle_command({"command": "set_voice", "voice": "alba", "preview": True})
        await session._turn_task
        play.assert_awaited_once_with(
            "Okay. Here's how I sound now.", AudioOpcode.BACKCHANNEL, 1.0, nominal=True,
        )
        responder.assert_not_awaited()
        assert not any(event["type"] in {"voice.final", "voice.reply"} for event in events)
    finally:
        session._tts.shutdown()


@pytest.mark.asyncio
async def test_voice_audition_does_not_inherit_the_previous_conversations_delivery(monkeypatch):
    session = DuplexVoiceSession(
        session_id="nominal-preview", send_json=AsyncMock(), send_binary=AsyncMock(),
        mind=MindBridge(session_id="nominal-preview", responder=AsyncMock()),
    )
    session._voice_override = 'aura_kiwi'
    session._config.tts.speed = 1.0
    monkeypatch.setattr(session, '_prosody_spec', lambda: ProsodySpec(
        voice='another_voice', speed=1.15, gain=.8, onset_bend_cents=100, glide_ms=200,
    ))
    synthesize = AsyncMock(return_value=None)
    monkeypatch.setattr(session._tts, 'synthesize', synthesize)
    try:
        await session._speak_aside('A voice sample.', AudioOpcode.BACKCHANNEL, 1.0, nominal=True)
        spec = synthesize.await_args.args[1]
        assert (spec.voice, spec.speed, spec.gain) == ('aura_kiwi', 1.0, 1.0)
        assert spec.onset_bend_cents == 0 and spec.glide_ms == 0
    finally:
        session._tts.shutdown()

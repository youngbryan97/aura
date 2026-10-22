"""A selected voice plays a sample without inventing a conversation turn."""
from unittest.mock import AsyncMock

import pytest

from core.voice.duplex.mind_bridge import MindBridge
from core.voice.duplex.protocol import AudioOpcode
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
            "Okay. Here's how I sound now.", AudioOpcode.BACKCHANNEL, 1.0,
        )
        responder.assert_not_awaited()
        assert not any(event["type"] in {"voice.final", "voice.reply"} for event in events)
    finally:
        session._tts.shutdown()

"""A typed reply is shown in the window that asked for it as it is written, and nowhere else."""
from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from core.conversation.reply_stream import (
    ReplyStreamChannel,
    active_reply_stream,
    bind_reply_stream,
)
from interface.routes import chat_reply_as_it_forms as forms

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sent(monkeypatch):
    out: list[dict] = []
    monkeypatch.setattr(forms, "_send", lambda **facts: out.append(facts))
    monkeypatch.setattr(forms, "SENT_EVERY_S", 0.0)
    return out


def _asked_from(monkeypatch, key: str | None):
    from core.runtime import chat_delivery_progress

    monkeypatch.setattr(chat_delivery_progress, "current_chat_delivery_identity", lambda: {"idempotency_key": key} if key else None)


def test_her_words_reach_the_window_that_asked_as_they_are_written(monkeypatch, sent):
    _asked_from(monkeypatch, "k1")

    async def turn():
        with forms.reply_as_it_forms():
            forms.tell_the_window(doing="Listening")
            channel = active_reply_stream()
            channel.publish_threadsafe("Yes. ")
            await asyncio.to_thread(channel.publish_threadsafe, "She/her.")
            await asyncio.sleep(0.01)
        assert active_reply_stream() is None

    asyncio.run(turn())
    assert sent == [{"key": "k1", "doing": "Listening"}, {"key": "k1", "text": "Yes. "}, {"key": "k1", "text": "Yes. She/her."}, {"key": "k1", "done": True}]
    forms.tell_the_window(doing="Thinking")
    assert len(sent) == 4, "outside a turn there is no window to tell"


def test_a_turn_asked_from_no_window_or_already_spoken_is_left_as_it_was(monkeypatch, sent):
    _asked_from(monkeypatch, None)

    async def turn():
        with forms.reply_as_it_forms():
            assert active_reply_stream() is None
        _asked_from(monkeypatch, "k2")
        spoken = ReplyStreamChannel()
        with bind_reply_stream(spoken), forms.reply_as_it_forms():
            assert active_reply_stream() is spoken
            forms.tell_the_window(doing="Listening")

    asyncio.run(turn())
    assert sent == []


def test_nothing_waits_in_a_stream_nobody_takes_from(monkeypatch, sent):
    _asked_from(monkeypatch, "k3")
    monkeypatch.setattr(forms, "SENT_EVERY_S", 3600.0)

    async def turn():
        with forms.reply_as_it_forms():
            channel = active_reply_stream()
            for n in range(400):
                channel.publish(f"w{n} ")
            assert channel.backlog() == 0

    asyncio.run(turn())
    assert sent[-2]["text"].endswith("w399 ") and sent[-1] == {"key": "k3", "done": True}, "the last of it is sent however often it is sent"


def test_every_kind_of_request_has_something_she_is_said_to_be_doing():
    from core.understanding.effects import BY_NAME

    assert set(forms.DOING) >= set(BY_NAME)


def test_the_window_shows_only_the_turn_it_asked_and_puts_the_whole_reply_in_its_place():
    script = (ROOT / "interface/static/aura.js").read_text()
    assert "data.key !== item.idempotencyKey" in script and "type === 'aura_turn'" in script and "formed.remove()" in script
    chat = (ROOT / "interface/routes/chat.py").read_text()
    assert "with reply_as_it_forms():" in chat
    if shutil.which("node"):
        assert subprocess.run(["node", "--check", str(ROOT / "interface/static/aura.js")], check=False, capture_output=True).returncode == 0

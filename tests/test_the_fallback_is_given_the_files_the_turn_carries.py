"""A turn her cortex cannot serve still carries the files the person attached.

LIVE 8 November 2026: Bryan attached the paper about her and asked what she made of it. The preflight read the file
into the turn, her cortex could not serve the turn, and the smaller model that answered instead was handed his message
alone. It said, "I don't see the paper attached in this message", about a file the turn held. Every path that answers
a turn now hands its model the same files, fitted to that model's room.
"""

from __future__ import annotations

import asyncio

from core.conversation import attachments
from interface.routes import chat as chat_module

BLOCK = "[The person attached files to this message.]\n\n=== ATTACHED FILE: ISC_Paper.pdf ===\nAbstract ...\n=== END ISC_Paper.pdf ==="


class _Router:
    def __init__(self):
        self.calls = []

    async def think(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return "I have the paper here."


def _stub_block(monkeypatch, rooms=None):
    def context_block(ids, *, query="", room):
        if rooms is not None:
            rooms.append(room)
        return (BLOCK, [("ISC_Paper.pdf", "Abstract ...")]) if ids else ("", [])

    monkeypatch.setattr(attachments, "context_block", context_block)


def test_the_message_is_preceded_by_the_files_this_turn_carries(monkeypatch):
    rooms = []
    _stub_block(monkeypatch, rooms)

    async def turn():
        attachments.carry_this_turn(["53891a2918469d41"])
        return attachments.with_this_turns_files("What do you make of this?")

    asked = asyncio.run(turn())
    assert asked == f"{BLOCK}\nUser message: What do you make of this?"
    assert rooms == [attachments.FALLBACK_ROOM]


def test_a_turn_with_no_files_is_asked_as_it_was(monkeypatch):
    _stub_block(monkeypatch)

    async def turn():
        return attachments.with_this_turns_files("you ok?")

    assert asyncio.run(turn()) == "you ok?"


def test_the_smaller_model_is_handed_the_files(monkeypatch):
    _stub_block(monkeypatch)
    router = _Router()

    async def no_readings(_text):
        return []

    async def no_history(_text):
        return []

    monkeypatch.setattr(chat_module, "_readings_for", no_readings)
    monkeypatch.setattr("interface.routes.chat_foreground_lane._fallback_conversation_messages", no_history)
    monkeypatch.setattr("core.brain.llm_health_router.get_llm_router", lambda: router)

    async def turn():
        attachments.carry_this_turn(["53891a2918469d41"])
        return await chat_module._answer_from_fallback_ladder(
            "What do you make of this? This is the paper I wrote about you", reason="cortex unavailable"
        )

    asyncio.run(turn())
    asked = router.calls[0][1]["messages"][-1]
    assert asked["role"] == "user"
    assert asked["content"].startswith(BLOCK)
    assert asked["content"].endswith("User message: What do you make of this? This is the paper I wrote about you")


def test_the_files_do_not_outlive_their_turn():
    async def another_turn():
        return attachments.this_turn()

    assert asyncio.run(another_turn()) == ()

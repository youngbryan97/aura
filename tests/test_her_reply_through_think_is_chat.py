"""Her reply to somebody through `router.think` reaches the gate as chat.

`think` marks what it carries as non-chat inference, and the kernel's reply
phase sends her reply to a person through it, so the gate graded that reply as
her own deliberation and never named her cortex as the one that answered.
Every arm of the reports ground then read as one her cortex did not serve:
the 1.5B dry run of 21 October logged `internal_inference=True` on every
"How are you feeling right now" request.
"""
from __future__ import annotations

import asyncio

from core.brain.llm_health_router import HealthAwareLLMRouter, asks_for_a_reply


def _sent(**kwargs) -> dict:
    router = HealthAwareLLMRouter.__new__(HealthAwareLLMRouter)
    seen: dict = {}

    async def generate_with_metadata(**call) -> dict:
        seen.update(call)
        return {"ok": True, "text": "0.6"}

    router.generate_with_metadata = generate_with_metadata
    router._publish_generation_metadata = lambda result: None
    asyncio.run(router.think("How are you feeling right now?", **kwargs))
    return seen


def test_a_reply_to_a_person_goes_as_chat():
    assert _sent(origin="user", purpose="reply")["_non_chat_inference"] is False


def test_everything_else_through_think_stays_non_chat():
    assert _sent(origin="user", purpose="deep_inference")["_non_chat_inference"] is True
    assert _sent(origin="system", purpose="reply")["_non_chat_inference"] is True
    assert _sent()["_non_chat_inference"] is True


def test_a_caller_that_says_non_chat_is_taken_at_its_word():
    assert _sent(origin="user", purpose="reply", _non_chat_inference=True)["_non_chat_inference"] is True
    assert asks_for_a_reply({"origin": "voice", "purpose": "reply"}) is True
    assert asks_for_a_reply({"origin": "voice", "purpose": "reply"}, said_non_chat=True) is False

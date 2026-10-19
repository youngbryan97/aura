"""Her reply in a conversation on a page is asked of her model with the conversation as its turns, what she was asked
and what the page says as evidence, and the time the conversation keeps; nothing addressed to her."""
from __future__ import annotations

import asyncio

import pytest

from core.perception.a_conversation_on_a_page import Line

pytestmark = pytest.mark.unit


def test_her_model_is_given_the_conversation_the_evidence_and_the_time():
    from core.skills.her_reply_in_a_conversation import her_reply
    from core.utils.injected_blocks import RUNTIME_EVIDENCE_ROLE

    asked = {}

    class _Router:
        async def generate_with_metadata(self, prompt, system_prompt, timeout, **kwargs):  # noqa: ASYNC109 - the router's own signature
            asked.update(kwargs, prompt=prompt, system_prompt=system_prompt)
            return {"ok": True, "text": " not much, you? "}

    lines = [Line("them", "hey, how's it going"), Line("me", "good, you?"), Line("them", "where are you writing from?")]
    said = asyncio.run(her_reply(lines, request="Play Human or Not.", page_text=["Time left: 1:42"], router=_Router(),
                                 within_s=12.0))
    assert said == "not much, you?"
    assert asked["answer_within_s"] == 12.0 and asked["system_prompt"] is None and asked["prompt"] == ""
    roles = [m["role"] for m in asked["messages"]]
    assert roles == [RUNTIME_EVIDENCE_ROLE, "user", "assistant", "user"]
    assert "Play Human or Not." in asked["messages"][0]["content"] and "Time left" in asked["messages"][0]["content"]

"""Her reply to somebody through `router.think` reaches the gate as a reply.

`think` marks what it carries as non-chat inference, which keeps it off the
tool hand-off, and the router turned that into `internal_inference` for the
gate. The kernel's reply phase sends her reply to a person through `think`, so
the gate graded it as her own deliberation and never named her cortex as the
one that answered: the 1.5B dry run of 21 October logged
`internal_inference=True` on every "How are you feeling right now" request,
and every arm of the reports ground read as unserved.
"""
from __future__ import annotations

import asyncio

from core.brain.llm_health_router import HealthAwareLLMRouter, asks_for_a_reply


def _dispatched(**kwargs) -> dict:
    router = HealthAwareLLMRouter.__new__(HealthAwareLLMRouter)
    seen: dict = {}

    async def core(*args, **call) -> dict:
        seen.update(call)
        return {"ok": True, "text": "0.6"}

    router._generate_core = core
    router._publish_generation_metadata = lambda result: None
    router._background_suppression_result = lambda **_: None
    router._is_background_request = lambda **_: False
    router._maybe_route_expert_adapter = lambda *a, **k: None
    asyncio.run(router.think("How are you feeling right now?", **kwargs))
    return seen


def test_a_reply_to_a_person_is_not_marked_internal():
    assert not _dispatched(origin="user", purpose="reply").get("internal_inference")


def test_everything_else_through_think_stays_internal():
    assert _dispatched(origin="user", purpose="deep_inference").get("internal_inference") is True
    assert _dispatched(origin="system", purpose="reply").get("internal_inference") is True


def test_a_caller_that_says_internal_is_taken_at_its_word():
    assert _dispatched(origin="user", purpose="reply", internal_inference=True).get("internal_inference") is True
    assert asks_for_a_reply({"origin": "voice", "purpose": "reply"}) is True
    assert asks_for_a_reply({"origin": "voice", "purpose": "reply", "internal_inference": True}) is False

"""When the cortex cannot serve, answer with the smaller model.

The design already said so — _foreground_timeout_for_lane's own comment reads
"Give the ladder the time" — but nothing ever asked the ladder. The Brainstem
is loaded, has weights, and is not the lane owner, and it was invoked zero
times on every cold start. Meanwhile the first message after each launch was
answered with a sentence about lane readiness.
"""

from __future__ import annotations

import asyncio

from interface.routes import chat as chat_module


def _run(coro):
    return asyncio.run(coro)


class _Router:
    def __init__(self, answer="Yes — I'm here.", raises=None):
        self.answer = answer
        self.raises = raises
        self.calls = []

    async def think(self, text, prefer_tier=None, allow_cloud_fallback=True, **kwargs):
        self.calls.append((text, prefer_tier, allow_cloud_fallback, kwargs))
        if self.raises:
            raise self.raises
        return self.answer


def test_the_ladder_answers_instead_of_describing_the_lane(monkeypatch) -> None:
    router = _Router()
    monkeypatch.setattr("core.brain.llm_health_router.get_llm_router", lambda: router)

    reply = _run(chat_module._answer_from_fallback_ladder("are you there?", reason="lane_warming"))

    assert "Yes — I'm here." in reply


def test_it_uses_the_small_local_tier_and_never_the_cloud(monkeypatch) -> None:
    router = _Router()
    monkeypatch.setattr("core.brain.llm_health_router.get_llm_router", lambda: router)

    _run(chat_module._answer_from_fallback_ladder("hi there", reason="lane_warming"))

    _text, tier, cloud, kwargs = router.calls[0]
    assert tier == "tertiary"
    assert cloud is False
    # Naming the endpoint is what gets a background-only tier past the
    # foreground selector; a tier alone considered nothing at all.
    assert kwargs.get("prefer_endpoint") == "Reflex"


def test_the_answer_says_which_model_produced_it(monkeypatch) -> None:
    """A 9B answer served as though it were the 32B trades one lie for a worse one."""
    router = _Router()
    monkeypatch.setattr("core.brain.llm_health_router.get_llm_router", lambda: router)

    reply = _run(chat_module._answer_from_fallback_ladder("hi", reason="lane_warming"))

    assert "smaller model" in reply


def test_a_failing_ladder_yields_nothing_rather_than_a_bad_answer(monkeypatch) -> None:
    router = _Router(raises=RuntimeError("brainstem down"))
    monkeypatch.setattr("core.brain.llm_health_router.get_llm_router", lambda: router)

    assert _run(chat_module._answer_from_fallback_ladder("hi", reason="x")) == ""


def test_an_empty_answer_is_not_served(monkeypatch) -> None:
    router = _Router(answer="   ")
    monkeypatch.setattr("core.brain.llm_health_router.get_llm_router", lambda: router)

    assert _run(chat_module._answer_from_fallback_ladder("hi", reason="x")) == ""


def test_an_empty_question_never_reaches_the_router(monkeypatch) -> None:
    router = _Router()
    monkeypatch.setattr("core.brain.llm_health_router.get_llm_router", lambda: router)

    assert _run(chat_module._answer_from_fallback_ladder("   ", reason="x")) == ""
    assert router.calls == []


def test_a_missing_router_is_safe(monkeypatch) -> None:
    monkeypatch.setattr("core.brain.llm_health_router.get_llm_router", lambda: None)

    assert _run(chat_module._answer_from_fallback_ladder("hi", reason="x")) == ""


# ── the small model echoes its own scaffolding ───────────────────────────────
#
# LIVE 2026-08-17, first working ladder answer, verbatim to the user:
#     <answer>Yes, I'm not available.</answer>
# The tags are instructions to the model, not part of what it said, and
# shipping them makes a working fallback look broken.

def test_answer_tags_are_stripped(monkeypatch) -> None:
    router = _Router(answer="<answer>Yes, I'm here.</answer>")
    monkeypatch.setattr("core.brain.llm_health_router.get_llm_router", lambda: router)

    reply = _run(chat_module._answer_from_fallback_ladder("hi", reason="x"))

    assert "<answer>" not in reply
    assert "Yes, I'm here." in reply


def test_chat_template_markers_are_stripped(monkeypatch) -> None:
    router = _Router(answer="<|im_start|>Here you go<|im_end|>")
    monkeypatch.setattr("core.brain.llm_health_router.get_llm_router", lambda: router)

    reply = _run(chat_module._answer_from_fallback_ladder("hi", reason="x"))

    assert "<|" not in reply
    assert "Here you go" in reply


def test_a_reply_that_was_only_scaffolding_is_not_served(monkeypatch) -> None:
    router = _Router(answer="<answer></answer>")
    monkeypatch.setattr("core.brain.llm_health_router.get_llm_router", lambda: router)

    assert _run(chat_module._answer_from_fallback_ladder("hi", reason="x")) == ""


def test_ordinary_prose_is_untouched(monkeypatch) -> None:
    router = _Router(answer="I'm here, still waking up.")
    monkeypatch.setattr("core.brain.llm_health_router.get_llm_router", lambda: router)

    reply = _run(chat_module._answer_from_fallback_ladder("hi", reason="x"))

    assert reply.startswith("I'm here, still waking up.")

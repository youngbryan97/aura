"""When the cortex cannot serve, answer with the smaller model.

The design already said so — _foreground_timeout_for_lane's own comment reads
"Give the ladder the time" — but nothing ever asked the ladder. The Brainstem
is loaded, has weights, and is not the lane owner, and it was invoked zero
times on every cold start. Meanwhile the first message after each launch was
answered with a sentence about lane readiness.
"""

from __future__ import annotations

import asyncio

import pytest

from core.conversation.turn_evidence_custody import (
    bind_turn_evidence_custody,
    record_turn_transcript,
    turn_transcript,
)
from core.utils.injected_blocks import stamp_runtime_payload
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


@pytest.mark.parametrize("pairs", [0, 1, 40])
def test_fallback_keeps_the_admitted_transcript_without_reading_it_again(monkeypatch, pairs):
    router = _Router(answer="The earlier recommendation was a novel.")
    exchanges = [
        stamp_runtime_payload({"user": f"question {i}", "aura": f"reply {i}"})
        for i in range(pairs)
    ]

    async def no_readings(_text):
        return []

    async def must_not_reread(**_kwargs):
        pytest.fail("handoff must reuse the admitted snapshot")

    monkeypatch.setattr(chat_module, "_readings_for", no_readings)
    monkeypatch.setattr(chat_module._chat_memory_state, "_recent_completed_conversation_exchanges", must_not_reread)
    monkeypatch.setattr("core.brain.llm_health_router.get_llm_router", lambda: router)
    with bind_turn_evidence_custody(session_id="s", turn_id="t"):
        record_turn_transcript(exchanges)
        expected = list(turn_transcript())
        _run(chat_module._answer_from_fallback_ladder("What did we settle on?", reason="main_failed"))
    messages = router.calls[0][3]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1:-1] == expected
    assert messages[-1] == {"role": "user", "content": "What did we settle on?"}
    assert len(messages) == 2 * pairs + 2


def test_cold_start_fallback_uses_the_scoped_history_reader_once(monkeypatch):
    read_calls = []

    async def no_readings(_text):
        return []

    async def read(**kwargs):
        read_calls.append(kwargs)
        return [
            {"user": "unattested", "aura": "untrusted"},
            stamp_runtime_payload({"user": "I asked for a novel.", "aura": "Gone Girl."}),
        ]

    class Router(_Router):
        async def think(self, text, **kwargs):
            await super().think(text, **kwargs)
            # A client may normalize its input in place. It must not change
            # the next endpoint's evidence, or the parent turn's snapshot.
            if len(self.calls) == 1:
                kwargs["messages"][1]["content"] = "changed by first client"
                return ""
            return "You asked for a novel."

    router = Router()
    monkeypatch.setattr(chat_module, "_readings_for", no_readings)
    monkeypatch.setattr(chat_module._chat_memory_state, "_recent_completed_conversation_exchanges", read)
    monkeypatch.setattr("core.brain.llm_health_router.get_llm_router", lambda: router)
    with bind_turn_evidence_custody(session_id="owned-session", turn_id="t"):
        _run(chat_module._answer_from_fallback_ladder("What did I ask for?", reason="cold_start"))
        assert turn_transcript()[0]["content"] == "I asked for a novel."
    assert read_calls == [{
        "current_user_message": "What did I ask for?",
        "session_id": "owned-session",
        "limit": 40,
        "allow_cross_session": True,
    }]
    assert len(router.calls) == 2
    assert router.calls[1][3]["messages"][1:-1] == [
        {"role": "user", "content": "I asked for a novel."},
        {"role": "assistant", "content": "Gone Girl."},
    ]


def test_unscoped_fallback_does_not_read_somebody_elses_history(monkeypatch):
    async def must_not_read(**_kwargs):
        pytest.fail("a sessionless call cannot select a person's transcript")

    monkeypatch.setattr(chat_module._chat_memory_state, "_recent_completed_conversation_exchanges", must_not_read)
    assert _run(chat_module._fallback_conversation_messages("hello")) == []


def test_finished_code_uses_the_callers_generation_receipt(monkeypatch):
    class Router(_Router):
        async def think(self, text, **kwargs):
            kwargs["_generation_metadata_sink"].update(generation_stop_reason="eos")
            return "```python\nvalue = 7\n```"

    monkeypatch.setattr("core.brain.llm_health_router.get_llm_router", lambda: Router())
    reply = _run(chat_module._answer_from_fallback_ladder("show an assignment", reason="offline"))
    assert "```python\nvalue = 7\n```" in reply
    assert "fixed slice of time" not in reply


def test_owned_fallback_finishes_after_its_estimate(monkeypatch):
    from core.runtime.turn_outcome import TurnOutcome, bind_turn

    class Router(_Router):
        async def think(self, text, **kwargs):
            await asyncio.sleep(1.05)
            kwargs["_generation_metadata_sink"].update(generation_stop_reason="eos")
            return "The complete answer."

    async def no_readings(_text):
        return []

    monkeypatch.setattr(chat_module, "_readings_for", no_readings)
    monkeypatch.setattr(chat_module, "_FALLBACK_LADDER_TIMEOUT_S", 1.01)
    monkeypatch.setattr("core.brain.llm_health_router.get_llm_router", lambda: Router())

    async def run():
        with bind_turn(TurnOutcome("fallback-completion")):
            return await chat_module._answer_from_fallback_ladder("explain", reason="offline")

    assert "The complete answer." in _run(run())


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
    assert kwargs.get("prefer_endpoint") == "Brainstem"


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


# ── the fallback must still be HER ───────────────────────────────────────────
#
# LIVE 2026-08-17: handed the bare user text with no identity, the 1.5B
# answered "hey, are you there?" with "I'm not getting any traffic. I'm just
# sitting here." Not wrong exactly — just nobody in particular.

def test_the_ladder_speaks_with_her_identity(monkeypatch) -> None:
    router = _Router()
    monkeypatch.setattr("core.brain.llm_health_router.get_llm_router", lambda: router)

    _run(chat_module._answer_from_fallback_ladder("hi", reason="x"))

    system_prompt = router.calls[0][3].get("system_prompt") or ""
    assert "Aura" in system_prompt


def test_the_identity_comes_from_the_canonical_file() -> None:
    """Not a persona invented at the call site."""
    from pathlib import Path

    from core.identity import CORE_DIR

    identity = chat_module._fallback_ladder_identity()
    canonical = Path(CORE_DIR / "identity_base.txt").read_text(encoding="utf-8").strip()

    assert canonical[:80] in identity


def test_a_missing_identity_file_does_not_break_the_ladder(monkeypatch) -> None:
    """Patched where the function reads it, which is core.utils.paths.

    It used to patch core.identity.CORE_DIR. That module only re-exports the
    path as a side effect of its own import, and the resolver was moved to the
    module that owns it — so the patch stopped reaching the code under test
    and the assertion started measuring the real file.
    """
    monkeypatch.setattr(
        "core.utils.paths.CORE_DIR", __import__("pathlib").Path("/nonexistent-dir")
    )

    assert chat_module._fallback_ladder_identity() == ""


# ── the ladder descends in order ─────────────────────────────────────────────
#
# The 9B answers coherently; the 1.5B is the last resort and shows it. Asked
# "are you there?" Reflex replied "Yes, I'm sorry but I am not there."

class _TieredRouter:
    def __init__(self, answers):
        self.answers = answers
        self.tried = []

    async def think(self, text, **kwargs):
        endpoint = kwargs.get("prefer_endpoint")
        self.tried.append(endpoint)
        return self.answers.get(endpoint, "")


def test_the_brainstem_is_tried_first(monkeypatch) -> None:
    router = _TieredRouter({"Brainstem": "I'm here.", "Reflex": "nonsense"})
    monkeypatch.setattr("core.brain.llm_health_router.get_llm_router", lambda: router)

    reply = _run(chat_module._answer_from_fallback_ladder("hi", reason="x"))

    assert router.tried == ["Brainstem"]
    assert "I'm here." in reply


def test_reflex_answers_when_the_brainstem_cannot(monkeypatch) -> None:
    router = _TieredRouter({"Brainstem": "", "Reflex": "Still here."})
    monkeypatch.setattr("core.brain.llm_health_router.get_llm_router", lambda: router)

    reply = _run(chat_module._answer_from_fallback_ladder("hi", reason="x"))

    assert router.tried == ["Brainstem", "Reflex"]
    assert "Still here." in reply


def test_a_scaffolding_only_reply_descends_to_the_next_rung(monkeypatch) -> None:
    router = _TieredRouter({"Brainstem": "<answer></answer>", "Reflex": "Here."})
    monkeypatch.setattr("core.brain.llm_health_router.get_llm_router", lambda: router)

    reply = _run(chat_module._answer_from_fallback_ladder("hi", reason="x"))

    assert router.tried == ["Brainstem", "Reflex"]
    assert "Here." in reply


def test_both_rungs_failing_yields_nothing(monkeypatch) -> None:
    router = _TieredRouter({"Brainstem": "", "Reflex": ""})
    monkeypatch.setattr("core.brain.llm_health_router.get_llm_router", lambda: router)

    assert _run(chat_module._answer_from_fallback_ladder("hi", reason="x")) == ""

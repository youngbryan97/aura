"""A remark of hers while watching is her own turn: it answers nobody, so it is sized and framed as hers.

LIVE 20 October 2026 (rehearsal on a six-second film): each remark was handed the person's request ("Watch this with
me: ... before it starts, tell me ..., think out loud while it plays, and tell me what you thought after") as the
message just said to her, and answered all three parts again; the request's shape set a floor of 896 tokens over the
320 the remark asked for, the clock raised it to 1,682, and the remark took seven minutes.
"""

from __future__ import annotations

import pytest

from core.utils.injected_blocks import RUNTIME_EVIDENCE_ROLE, stamp_runtime_payload

REQUEST = "Watch this with me: http://127.0.0.1:8765/watching/a_page_that_plays.html - tell me what you thought after."
HER_ANSWER = "Before it starts: a test pattern with words over it. I think it is about colour as infrastructure."


async def _what_her_model_is_handed(monkeypatch, context: dict) -> dict:
    from core.brain import cognitive_engine as cognitive_engine_module
    from core.brain.cognitive_engine import CognitiveEngine
    from core.brain.types import ThinkingMode
    from core.phases import kept_procedures_in_a_turn

    handed: dict = {}

    class Router:
        async def think(self, **kwargs):
            handed.update(kwargs)
            return "The bars hold still while the words change."

        def get_last_generation_metadata(self):
            return {"surface_control_receipt": {"generation_stop_reason": "eos", "surface_quality_gate_reasons": []}}

    class Container:
        def get(self, name, default=None):
            return Router() if name == "llm_router" else default

    async def no_kept_answer(*_args, **_kwargs):
        return None

    monkeypatch.setattr(cognitive_engine_module, "get_container", lambda: Container())
    monkeypatch.setattr(kept_procedures_in_a_turn, "kept_answer_for", no_kept_answer)
    await CognitiveEngine()._direct_desktop_quick_reply(REQUEST, ThinkingMode.FAST, "desktop_watching_along", context,
                                                        timeout_s=60.0)
    return handed


@pytest.mark.asyncio
async def test_a_remark_ends_on_what_she_senses_and_keeps_the_room_it_was_given(monkeypatch):
    handed = await _what_her_model_is_handed(monkeypatch, {
        "desktop_quick_reply_contract": True, "visible_user_message": "", "max_tokens": 320, "her_own_turn": True,
        "allow_tools": False, "recent_completed_exchanges": [stamp_runtime_payload({"user": REQUEST, "aura": HER_ANSWER})],
        "turn_sensory_evidence": {"channel": "screen", "ok": True, "request": REQUEST, "observed_at": 1.0,
                                  "observation": "playing, at 0:04 of 0:06. seen: colour bars; heard: hue",
                                  "source": "watching_along", "scope": "what plays", "coverage": "partial"},
    })
    messages = handed["messages"]
    assert messages[-1]["role"] == RUNTIME_EVIDENCE_ROLE and "colour bars" in messages[-1]["content"]
    assert [m["content"] for m in messages if m["role"] == "user"] == [REQUEST]  # once, as what was said before
    assert handed["max_tokens"] == 320 and handed["user_surface_completion_floor"] == 320
    assert handed["her_own_turn"] is True
    # Her reasoning may take at most the answer's own room, so the whole of a turn of hers is twice it: LIVE 2026-10-20
    # the worker added its measured reserve of about 900 tokens, and a remark asked for 320 came back at 950.
    assert handed["hard_output_token_ceiling"] == 640
    # Owed to nobody, it is not continued past its end as an unfinished answer is (LIVE 2026-10-20, her answer before
    # the film ran a second generation of 921 tokens as a continuation of the first).
    assert handed["semantic_completion_contract"] is False
    # What the gate is handed passes its request contract: LIVE 2026-10-20 a floor of 0 was rejected as malformed,
    # the gate failed the turn closed, and her answer fell to the full pipeline with an 88,103-character prompt.
    from core.brain.request_contract import validate_request_context

    assert validate_request_context({k: v for k, v in handed.items() if k != "messages"}).rejected == {}


@pytest.mark.asyncio
async def test_a_turn_somebody_is_waiting_for_still_ends_on_what_they_said(monkeypatch):
    handed = await _what_her_model_is_handed(monkeypatch, {
        "desktop_quick_reply_contract": True, "visible_user_message": REQUEST, "max_tokens": 320})
    assert handed["messages"][-1] == {"role": "user", "content": REQUEST}
    assert handed["max_tokens"] >= 512 and handed["her_own_turn"] is False and handed["hard_output_token_ceiling"] is None


def test_the_ceiling_a_turn_of_hers_carries_reaches_the_worker():
    """LIVE 2026-10-20: the gate passed on only an output contract's ceiling, and a remark carrying 640 decoded 1,154."""
    from core.brain.inference_gate_turn_setup import _SetsTheTurnUp

    handed: dict = {}
    _SetsTheTurnUp._generate_with_metadata_sink_turn_executes_something(
        {"hard_output_token_ceiling": 640, "brief": ""}, 320, 320, handed, None, False, None, None)
    assert handed["hard_output_token_ceiling"] == 640
    nothing: dict = {}
    _SetsTheTurnUp._generate_with_metadata_sink_turn_executes_something({"brief": ""}, None, 768, nothing, None, False, None, None)
    assert "hard_output_token_ceiling" not in nothing

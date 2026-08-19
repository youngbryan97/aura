"""A move she reasoned out was thrown away for not being an answer.

LIVE 2026-08-19. Deciding the next move on a 2048 board, she answered
"left" — correct, and exactly what was asked for. The user-surface quality
gate rejected it as arithmetic_answer_missing, retried, was given "right",
rejected that too, exhausted its retries, and returned nothing. The pursuit
then reported that she had named no available move.

The question it was graded against was the deliberation's own prompt, which
mentions a 128 tile. Nothing was answering a person, so there was no reply
for the reply contract to hold.

The conflation was one clause: a call counts as user-facing when its origin
says so OR it asked for the primary tier. Wanting the good model is not the
same as producing the visible answer. Internal reasoning keeps the primary
tier and the foreground lane; it stops inheriting the contract that says its
output is the reply.
"""
from __future__ import annotations

import inspect

from core.brain import inference_gate as gate
from core.brain import llm_health_router as router_module


def _generate_source() -> str:
    return inspect.getsource(gate.InferenceGate)


def test_an_internal_generation_can_declare_it_is_not_the_reply():
    source = _generate_source()
    assert "internal_inference" in source, "there is no way to say a generation is not the reply"


def test_the_reply_contract_is_not_applied_to_internal_reasoning():
    """The user-surface contract must be skipped when nothing is being answered."""
    source = _generate_source()
    marker = "and not internal_inference"
    assert marker in source
    where = source.index(marker)
    window = source[max(0, where - 400) : where]
    assert "_is_user_facing" in window
    assert 'requested_tier == "primary"' in window


def test_asking_for_the_good_model_still_gets_it():
    """Internal reasoning keeps the primary tier — only the reply contract goes."""
    source = _generate_source()
    where = source.index("and not internal_inference")
    condition = source[max(0, where - 400) : where + 60]
    # The tier is part of the same condition, so declaring internal inference
    # changes the contract applied and nothing about routing.
    assert 'requested_tier == "primary"' in condition


def test_the_router_carries_the_declaration_instead_of_swallowing_it():
    source = inspect.getsource(router_module)
    where = source.index('non_chat_inference = bool(kwargs.pop("_non_chat_inference", False))')
    following = source[where : where + 700]
    assert 'kwargs["internal_inference"] = True' in following, (
        "the router consumed the signal and the gate could never see it"
    )


def test_the_deliberation_declares_itself():
    """Her act-time reasoning is internal by construction."""
    from core.agency import her_reasoning

    source = inspect.getsource(her_reasoning)
    assert "purpose=\"action_deliberation\"" in source


class _Amplified:
    def __init__(self, answer):
        self.answer = answer


def test_nothing_coming_back_is_reported_as_nothing_coming_back(monkeypatch):
    """An empty amplified answer is not a reply that named nothing.

    The amplifier absorbs a generation failure and returns an unverified
    answer with no candidates in it. Measured live while the resident worker
    was still starting, a pursuit reported "she named no available move" six
    times over — she had not been asked.
    """
    import asyncio

    from core.agency import her_reasoning as hr

    async def empty_amplify(objective, produce, **kw):
        return _Amplified("   ")

    import core.brain.reasoning_amplifier_v2 as amp

    monkeypatch.setattr(amp, "amplify_turn", empty_amplify)
    think = hr.her_reasoning(generate=lambda prompt, temp: None)

    async def run():
        await think("choose a move", ["Goal: x"])

    try:
        asyncio.run(run())
    except RuntimeError as exc:
        assert "produced nothing" in str(exc)
    else:
        raise AssertionError("an empty answer was passed on as if it were one")


def test_a_real_answer_is_returned_unchanged(monkeypatch):
    import asyncio

    from core.agency import her_reasoning as hr
    import core.brain.reasoning_amplifier_v2 as amp

    async def good_amplify(objective, produce, **kw):
        return _Amplified("left, because the corner holds")

    monkeypatch.setattr(amp, "amplify_turn", good_amplify)
    think = hr.her_reasoning(generate=lambda prompt, temp: None)
    assert asyncio.run(think("choose a move", [])) == "left, because the corner holds"

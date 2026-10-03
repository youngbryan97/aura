"""A prompt that already says what state she is in gets no second reading.

LIVE 2026-10-03 00:46: a page pursuit held her assembled mind fixed across its
calls, and its first two calls still matched for only 4,206 of 6,404 tokens.
They diverged at the router's safety-net line, "[Affect: ... substrate age:
0.0s)]", a fresh reading of the same organs her mind had already described,
appended after it on every call. The second call prefilled the whole prompt
again before she made her first move.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.unit


def test_an_assembled_mind_is_recognised_by_what_it_writes():
    from core.brain.llm.context_assembler import _BLACK_BOX_STATE_MARKERS
    from core.brain.llm_health_router_endpoint_call import _carries_her_state

    for marker in _BLACK_BOX_STATE_MARKERS:
        assert _carries_her_state(f"You are Aura.\n\n{marker}\nsomething")
    assert not _carries_her_state("You are a helpful assistant.")
    assert not _carries_her_state(None)


def test_the_safety_net_looks_at_the_system_prompt_too():
    from core.brain import llm_health_router_endpoint_call as call

    source = inspect.getsource(call)
    net = source[source.index("Autonomous Context Injection") :]
    net = net[: net.index("ctx_summary = her_state_in_brief()")]
    assert "_carries_her_state(system_prompt)" in net


def test_a_prompt_without_her_mind_still_gets_its_reading():
    from core.brain.llm_health_router_endpoint_call import _carries_her_state

    assert not _carries_her_state("Answer the question.")
    assert _carries_her_state("## CURRENT STATE\nvalence 0.2")


def test_a_reading_already_held_in_the_system_prompt_is_not_taken_again():
    from core.brain.llm_health_router_endpoint_call import _carries_her_state

    assert _carries_her_state("You are Aura.\n\n[Affect: Current Mood: CALM (substrate age: 0.1s)]")


def test_a_pursuit_holds_her_state_with_her_mind(monkeypatch):
    """LIVE 2026-10-03 03:54: matched 4,213 tokens, then '0, substrate age: 0.1s)]'."""
    import asyncio

    from core.brain import llm_health_router_endpoint_call as call
    from core.skills.sovereign_browser import SovereignBrowserSkill
    from core.skills.sovereign_browser_one_question import HER_MIND_THIS_PURSUIT

    readings = iter(["[Affect: Current Mood: CALM (substrate age: 0.1s)]", "[Affect: later]"])
    monkeypatch.setattr(call, "her_state_in_brief", lambda: [next(readings)])
    skill = SovereignBrowserSkill()

    async def _built() -> str:
        return "You are Aura."

    monkeypatch.setattr(skill, "_her_mind_built_now", _built)

    async def _pursuit() -> tuple[str, str]:
        HER_MIND_THIS_PURSUIT.set({})
        return await skill._assembled_mind(), await skill._assembled_mind()

    first, second = asyncio.run(_pursuit())
    assert first == second == "You are Aura.\n\n[Affect: Current Mood: CALM (substrate age: 0.1s)]"


def test_a_draft_cut_off_by_its_ceiling_is_not_written_again():
    """LIVE 2026-10-03 03:57: three 420-token page readings, each cut at char 1238."""
    from core.brain.llm import mlx_worker

    source = inspect.getsource(mlx_worker)
    retry = source[source.index("Structured output failed schema validation") - 600 :]
    assert "and token_count < max_tokens:" in retry[:700]


def test_a_page_reading_has_the_room_a_page_decision_has():
    from core.skills import sovereign_browser_understanding as u

    body = inspect.getsource(u._UnderstandsThePage._understand_page)
    assert "max_tokens=420" not in body
    assert body.count("self.DECISION_MAX_TOKENS") == 2


def test_a_cut_off_reading_keeps_the_fields_she_finished():
    from core.skills.sovereign_browser_one_question import the_finished_fields

    cut = (
        '{"here": "The first page of the test", "to_progress": "Answer each, then Submit", '
        '"relevant": "sixty scales", "how_to_answer": "Each row is a scale", '
        '"present_but_not_needed": "the nav li'
    )
    assert the_finished_fields(cut) == {
        "here": "The first page of the test",
        "to_progress": "Answer each, then Submit",
        "relevant": "sixty scales",
        "how_to_answer": "Each row is a scale",
    }
    assert the_finished_fields('{"a": 1, "b": {"c": [1, 2]}}') == {"a": 1, "b": {"c": [1, 2]}}
    assert the_finished_fields("no object at all") == {}


def test_a_page_reading_that_was_cut_off_is_used_as_far_as_it_went(monkeypatch):
    import asyncio
    from typing import Any

    from core.skills.sovereign_browser import SovereignBrowserSkill

    class _Router:
        async def think(self, *_a: Any, **_k: Any) -> str:
            return '{"here": "A questionnaire", "to_progress": "Answer, then Submit", "done_wh'

    monkeypatch.setattr(
        "core.skills.sovereign_browser_understanding.optional_service",
        lambda *names, default=None: _Router() if "llm_router" in names else default,
    )
    skill = SovereignBrowserSkill()
    understood = asyncio.run(skill._understand_page("take it", {"elements": []}, None, "mind"))
    assert understood == {"here": "A questionnaire", "to_progress": "Answer, then Submit"}

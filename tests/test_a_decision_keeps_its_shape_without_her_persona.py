"""A decision is structured work; the persona in front of it is a bonus.

LIVE 2026-09-28 18:26, with the receipt that says so: "Decision by the mechanics
lane on bare_generate: unparsable_decision", on the index page, before a single
click — so no test was taken and nothing was narrated. The branch that carries
the schema, the shape the decoder holds and the lane requirement was gated on
her assembled mind being present, and her assembled mind was empty on every
call: it asked the container for `aura_state`, which nothing registers.
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest

from core.skills import sovereign_browser_understanding as u
from core.skills.sovereign_browser import SovereignBrowserSkill as S

pytestmark = pytest.mark.unit


def test_the_shaped_branch_does_not_depend_on_a_persona():
    body = inspect.getsource(u._UnderstandsThePage._decide_next_actions)
    assert "if callable(think):" in body
    assert "and (mind or asks_about_her)" not in body


def test_a_bare_call_is_the_last_resort_and_not_the_first(monkeypatch):
    skill = S()
    bare: list[str] = []
    shaped: list[dict[str, Any]] = []

    async def _think(prompt: str, **kwargs: Any) -> str:
        shaped.append(kwargs)
        return '{"actions": [{"index": 0, "type": "click"}], "why": "x", "done": false}'

    async def _generate(prompt: str, **_kw: Any) -> str:
        bare.append(prompt)
        return "prose"

    class _Router:
        think = staticmethod(_think)
        generate = staticmethod(_generate)

    monkeypatch.setattr(
        "core.skills.sovereign_browser_understanding.optional_service",
        lambda *names, default=None: _Router() if "llm_router" in names else default,
    )

    async def _no_mind() -> str:
        return ""

    async def _no_fast_lane(*_a: Any, **_k: Any) -> str | None:
        return None

    monkeypatch.setattr(skill, "_assembled_mind", _no_mind)
    monkeypatch.setattr(skill, "_decide_on_the_fast_lane", _no_fast_lane)
    decision = asyncio.run(
        skill._decide_next_actions(
            "press next",
            {"elements": [{"role": "button", "name": "Next", "selector": "#n"}]},
            [],
        )
    )
    assert not bare, "a decision fell to a shapeless call with no persona present"
    assert shaped and shaped[0].get("schema"), "the schema must travel with it"
    assert not decision.get("error")


def test_her_state_is_read_from_the_repository_that_holds_it():
    """`aura_state` is registered by nothing and returned None every time."""
    body = inspect.getsource(u._UnderstandsThePage._her_mind_built_now)
    assert "state_repository" in body
    assert "_current" in body


def test_an_absent_state_is_still_recorded():
    body = inspect.getsource(u._UnderstandsThePage._her_mind_built_now)
    assert "no_current_state" in body
    assert "record_degradation" in body


def test_every_page_call_says_the_turn_is_waiting_on_it():
    """Without saying so it is background, deferred, and comes back empty.

    LIVE 2026-09-28 18:29: "Decision by the mechanics lane on fast_lane:
    empty_decision" on the index page, nothing clicked. The claim is checked
    against whether a user turn is actually in flight, so it is not a claim
    anything can make.
    """
    source = inspect.getsource(u)
    for method in (
        u._UnderstandsThePage._understand_page,
        u._UnderstandsThePage._decide_on_the_fast_lane,
        u._UnderstandsThePage._asked_of_her,
    ):
        body = inspect.getsource(method)
        assert "serves_current_turn=True" in body, (
            f"{method.__name__} asks for work the turn is waiting on without saying so"
        )
    # A page decision asks through those two, which say it for it.
    decide = inspect.getsource(u._UnderstandsThePage._decide_next_actions)
    assert "self._asked_of_her(prompt, mind, shaped=True)" in decide
    assert "self._decide_on_the_fast_lane(" in decide
    assert source.count("serves_current_turn=True") >= 3

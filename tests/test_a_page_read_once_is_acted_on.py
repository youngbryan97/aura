"""A page she has just read is acted on from that reading, without a second reading of it to decide.

LIVE 8 November 2026, dress rehearsal 8: asked to pick a video from a channel,
the pick took 2 min 45 s. On each of the two pages, her reading of it was
followed by a decision that prefilled the same page again (1,322 and 3,762
tokens) to say what the reading had already worked out.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from core.skills.sovereign_browser import SovereignBrowserSkill

PAGE = {
    "url": "https://videos.test/results?q=a+channel",
    "title": "a channel - search",
    "elements": [
        {"index": 0, "role": "link", "name": "A jet boat with a pulse jet 22:40", "selector": "#v0"},
        {"index": 1, "role": "link", "name": "A coil gun from a camera 12:10", "selector": "#v1"},
        {"index": 2, "role": "button", "name": "Search", "selector": "#search"},
    ],
}
READ = {"here": "results for the channel", "to_progress": "open the video I want", "done_when": "it plays"}


class _Router:
    def __init__(self, answer: dict[str, Any] | str):
        self.answer = answer
        self.asked: list[dict[str, Any]] = []

    async def think(self, prompt: str, **kwargs: Any) -> str:
        self.asked.append({"prompt": prompt, **kwargs})
        return self.answer if isinstance(self.answer, str) else json.dumps(self.answer)


def _skill(monkeypatch, answer):
    router = _Router(answer)
    monkeypatch.setattr(
        "core.skills.sovereign_browser_understanding.optional_service",
        lambda *names, default=None: router if "llm_router" in names else default,
    )
    return SovereignBrowserSkill(), router


def _read(skill, acts_from="what she brings"):
    return asyncio.run(skill._understand_page("pick a video", PAGE, None, "mind", "", acts_from))


def test_what_a_reading_chose_is_taken_once_and_kept_out_of_the_reading(monkeypatch):
    skill, router = _skill(monkeypatch, {**READ, "actions": [{"index": 0, "type": "click"}],
                                         "why": "The jet boat has the best arc.", "expect": "it opens", "done": False})
    understood = _read(skill)
    assert understood == READ
    asked = router.asked[0]
    assert "what she brings" in asked["prompt"] and '"actions"' in asked["prompt"]
    assert asked["schema"] is skill._READING_THAT_ACTS_SCHEMA
    assert asked["max_tokens"] == skill.DECISION_MAX_TOKENS + skill.ACTING_ON_A_READING_TAKES
    decision = skill._what_her_reading_chose(PAGE, "pick a video")
    assert decision["actions"] == [{"index": 0, "type": "click"}]
    assert decision["why"] == "The jet boat has the best arc."
    assert skill._what_her_reading_chose(PAGE, "pick a video") is None


def test_a_reading_not_asked_to_act_is_the_reading_it_was(monkeypatch):
    skill, router = _skill(monkeypatch, READ)
    assert _read(skill, acts_from="") == READ
    asked = router.asked[0]
    assert '"actions"' not in asked["prompt"]
    assert asked["schema"] is skill._UNDERSTANDING_SCHEMA
    assert asked["max_tokens"] == skill.DECISION_MAX_TOKENS
    assert skill._what_her_reading_chose(PAGE, "pick a video") is None


@pytest.mark.parametrize("answer", [
    {**READ, "actions": [{"index": 9, "type": "click"}], "why": "the ninth"},
    {**READ, "actions": [], "why": "nothing yet"},
])
def test_a_choice_naming_nothing_doable_here_leaves_the_decision_to_be_asked(monkeypatch, answer):
    skill, _router = _skill(monkeypatch, answer)
    _read(skill)
    assert skill._what_her_reading_chose(PAGE, "pick a video") is None


def test_a_reading_cut_off_is_kept_as_far_as_it_went_and_its_choice_is_not_acted_on(monkeypatch):
    cut = '{"here": "results for the channel", "to_progress": "open one", "actions": [{"index": 0, "type": "click"}], "wh'
    skill, _router = _skill(monkeypatch, cut)
    assert _read(skill) == {"here": "results for the channel", "to_progress": "open one"}
    assert skill._what_her_reading_chose(PAGE, "pick a video") is None


def test_a_choice_is_acted_on_only_on_the_page_it_was_made_on(monkeypatch):
    skill, _router = _skill(monkeypatch, {**READ, "actions": [{"index": 0, "type": "click"}], "why": "this one"})
    _read(skill)
    assert skill._what_her_reading_chose({**PAGE, "url": "https://videos.test/watch?v=1"}, "pick a video") is None


def test_the_acting_schema_asks_what_a_decision_asks_but_a_placement():
    schema = SovereignBrowserSkill._READING_THAT_ACTS_SCHEMA["properties"]
    decision = SovereignBrowserSkill._DECISION_SCHEMA["properties"]
    assert {"actions", "why", "expect", "done"} <= set(schema) and "stand" not in schema
    assert schema["actions"] == decision["actions"]
    assert set(SovereignBrowserSkill._UNDERSTANDING_SCHEMA["properties"]) <= set(schema)

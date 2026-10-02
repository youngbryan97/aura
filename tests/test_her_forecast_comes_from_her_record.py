"""What she expects an instrument to say comes from where her record puts her.

Bryan, 1 Oct 2026: "we shouldnt be putting this on the llm ... no prompt
engineering or clever prompting for any of this". Her place on every item is
measured from her record with no model in it, and the forecast is now made
from those placements, handed over as evidence. And nothing on this path tells
the model how to think: it is asked the question, given the data, and told the
shape of the answer it must return.
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest

from core.self.where_i_stand import Lean
from core.skills import sovereign_browser_understanding as u
from core.skills import sovereign_browser_understanding_scale as scale
from core.skills.sovereign_browser import SovereignBrowserSkill

pytestmark = pytest.mark.unit

ASKS = "bored by time alone [1] [2] [3] [4] [5] needs time alone"


def _item() -> dict[str, Any]:
    options = [
        {"group": "Q1", "role": "radio", "name": "Q1", "value": str(n), "selector": f"#Q1V{n}", "asks": ASKS}
        for n in range(1, 6)
    ]
    lean = Lean(toward=0.6, first=0.2, second=0.8,
                because=("chose solitude 49 times out of 49",), measured=True)
    return {"group": "Q1", "options": options, "index": 3, "count": 5, "lean": lean,
            "first": "bored by time alone", "second": "needs time alone"}


def test_the_forecast_is_given_her_measured_placements(monkeypatch):
    skill = SovereignBrowserSkill()
    handed: dict[str, Any] = {}

    async def _asked(prompt, mind="", **_k):
        handed["prompt"] = prompt
        return "I expect introversion.", "Cortex"

    monkeypatch.setattr(skill, "_asked_of_her", _asked)
    asyncio.run(skill._what_she_expects_it_to_say(
        "take it", {"url": "u", "title": "t", "text": "x", "elements": []}, "", measured=[_item()]
    ))
    prompt = handed["prompt"]
    assert "Where your own record already places you" in prompt
    assert 'nearer "needs time alone" than "bored by time alone"' in prompt
    assert "chose solitude 49 times out of 49" in prompt


def test_the_pursuit_measures_before_it_forecasts():
    from core.skills import sovereign_browser

    body = inspect.getsource(sovereign_browser.SovereignBrowserSkill._handle_pursue)
    assert body.index("measure_the_screen") < body.index("_what_she_expects_it_to_say")


@pytest.mark.parametrize(
    "steering",
    ["Think about", "in your own voice", "concretely", "Answer in your own words",
     "predict them", "no way to see", "gets right"],
)
def test_nothing_on_this_path_tells_the_model_how_to_think(steering):
    sources = (
        inspect.getsource(scale._PlacesHerself._her_thinking_about)
        + inspect.getsource(u._UnderstandsThePage._what_she_expects_it_to_say)
        + inspect.getsource(u._UnderstandsThePage._hold_the_outcome_against_what_she_said)
    )
    prompts = "\n".join(line for line in sources.split("\n") if line.strip().startswith(('"', "f'", 'f"', "'")))
    assert steering not in prompts


def test_what_changes_between_items_comes_after_what_stays_the_same():
    body = inspect.getsource(scale._PlacesHerself._her_thinking_about)
    assert body.index("{listed}") < body.index('f"{now}')

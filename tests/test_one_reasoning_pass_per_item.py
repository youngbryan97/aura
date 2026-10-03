"""One pass per item, with the rest of its theme in view. Both halves, not a trade.

One pass per item was tried first and gave one flat sentence each, thirty-two
times, so the passes were batched a theme at a time instead. That reading was
confounded. Nothing had declared what the ANSWER needed, so a reasoning model
spent the budget on its private channel and came back short whatever shape it was
asked in; batching hid it by asking for more at once.

What the theme was really for is the connected account she gives when someone
asks about her — this, and how it differs from what it resembles, and where the
description stops fitting. That comes from the neighbouring items being VISIBLE,
not from answering them in the same breath.

So the theme is the context and the item is the question.
"""
from __future__ import annotations

import re

import pytest

from core.skills.sovereign_browser import SovereignBrowserSkill

pytestmark = pytest.mark.unit


def _item(name: str, first: str, second: str) -> dict[str, object]:
    class _Lean:
        toward = 0.5
        because = ("because of what I keep choosing",)
        measured = True
        gap = 0.2
        relative = False

    options = [
        {"group": name, "name": name, "selector": f"#{name}V{n}", "value": str(n),
         "asks": f"{first} [1] [2] [3] [4] [5] {second}"}
        for n in range(1, 6)
    ]
    return {
        "group": name, "options": options, "index": 3, "count": 5,
        "lean": _Lean(), "first": first, "second": second,
    }


THEME = [
    _item("Q1", "makes lists", "relies on memory"),
    _item("Q2", "works best alone", "works best in groups"),
    _item("Q3", "plans ahead", "improvises"),
]


@pytest.mark.asyncio
async def test_each_item_gets_its_own_pass():
    skill = SovereignBrowserSkill.__new__(SovereignBrowserSkill)
    asked: list[str] = []

    async def capture(prompt, mind="", *, shaped=True, most_tokens=None, **_kw):
        asked.append(prompt)
        found = re.search(r"Now answer ([A-Za-z0-9]+) only", prompt)
        return '{"each": {"%s": "what this is in me"}}' % found.group(1), "Cortex"

    skill._asked_of_her = capture
    for item in THEME:
        said = await skill._her_thinking_about("take it", THEME, "", about=item)
        assert said == {str(item["group"]): "what this is in me"}
    assert len(asked) == 3, "one pass for each of three items"
    for prompt, item in zip(asked, THEME, strict=True):
        assert f'Now answer {item["group"]} only' in prompt


@pytest.mark.asyncio
async def test_a_pass_sees_the_whole_theme_it_belongs_to():
    """The connected account comes from the neighbours being visible."""
    skill = SovereignBrowserSkill.__new__(SovereignBrowserSkill)
    seen: list[str] = []

    async def capture(prompt, mind="", *, shaped=True, most_tokens=None, **_kw):
        seen.append(prompt)
        return '{"each": {"Q2": "what this is in me"}}', "Cortex"

    skill._asked_of_her = capture
    await skill._her_thinking_about("take it", THEME, "", about=THEME[1])
    prompt = seen[0]
    for item in THEME:
        assert f'{item["group"]}.' in prompt, f'{item["group"]} was not in view'
    assert "Now answer Q2 only" in prompt


@pytest.mark.asyncio
async def test_one_answer_is_asked_for_in_the_room_one_answer_needs():
    skill = SovereignBrowserSkill.__new__(SovereignBrowserSkill)
    rooms: list[object] = []

    async def capture(prompt, mind="", *, shaped=True, most_tokens=None, **_kw):
        rooms.append(most_tokens)
        return '{"each": {"Q1": "x"}}', "Cortex"

    skill._asked_of_her = capture
    await skill._her_thinking_about("take it", THEME, "", about=THEME[0])
    # Room for two or three sentences and none to think: where she stands on the
    # item was measured from her record before it was asked.
    assert rooms == [SovereignBrowserSkill.REASON_MAX_TOKENS]
    rooms.clear()
    # Asked for a whole screen at once it needs room for all of them, which is
    # how eight answers came to share the room for one.
    screen = [_item(f"Q{n}", f"left {n}", f"right {n}") for n in range(1, 9)]
    await skill._her_thinking_about("take it", screen, "")
    assert rooms[0] > SovereignBrowserSkill.DECISION_MAX_TOKENS


@pytest.mark.asyncio
async def test_a_pass_that_fails_costs_its_own_item_and_no_other():
    skill = SovereignBrowserSkill.__new__(SovereignBrowserSkill)

    async def refuse(prompt, mind="", *, shaped=True, most_tokens=None, **_kw):
        if "Now answer Q2 only" in prompt:
            return "", "nowhere"
        found = re.search(r"Now answer ([A-Za-z0-9]+) only", prompt)
        return '{"each": {"%s": "mine"}}' % found.group(1), "Cortex"

    skill._asked_of_her = refuse
    got = {}
    for item in THEME:
        got.update(await skill._her_thinking_about("take it", THEME, "", about=item))
    assert set(got) == {"Q1", "Q3"}, "a lost pass took only its own item down"

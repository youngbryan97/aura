"""A reasoning model charges its thinking to the answer's budget, so the answer needs a floor.

LIVE 2026-09-29. She finished a sixty-item instrument, read her result, and was
asked what she made of it against what she had predicted — the thing the person
had asked for. The worker reported `decode=900 tokens` and the runtime reported
`Cortex response received (len=10)`. Nine hundred tokens of thinking, ten
characters of answer: "Okay. Here". The reply fell back to reciting the rounds.

The runtime already has the mechanism. A declared completion floor above
`A_CLOSED_QUESTIONS_FLOOR` tells it the answer is worked out in this call, and
then the decoder closes the private channel at a bound and the clock buys the
reserve on top of the answer's budget. Declared nothing, the channel is neither
opened nor bounded, the model reasons in the answer, and the budget is gone
before it concludes.

Her forecast and her verdict on the result are that kind of call. Her reason
for one item is not: where she stands on it was measured from her record before
she was asked, so the call says what was settled, and on 1 Oct an open channel
cost the first item 111 seconds for three sentences.
"""
from __future__ import annotations

import pytest

from core.runtime.structured_input import A_CLOSED_QUESTIONS_FLOOR
from core.skills.sovereign_browser import SovereignBrowserSkill

pytestmark = pytest.mark.unit


class _Router:
    def __init__(self) -> None:
        self.asked: list[dict[str, object]] = []

    async def think(self, prompt: str = "", **kwargs: object) -> str:
        self.asked.append(dict(kwargs))
        return "what I make of it"


@pytest.fixture
def router(monkeypatch):
    held = _Router()
    monkeypatch.setattr(
        "core.skills.sovereign_browser_understanding.optional_service",
        lambda *names, default=None: held if "llm_router" in names else default,
    )
    monkeypatch.setattr(
        "core.brain.generation_provenance.generation_metadata_of",
        lambda _reply: {"endpoint": "Cortex"},
    )
    return held


@pytest.mark.asyncio
async def test_every_question_asked_of_her_declares_the_room_its_answer_needs(router):
    skill = SovereignBrowserSkill()
    text, lane = await skill._asked_of_her("what do you make of it", "her mind", shaped=False)
    assert text == "what I make of it"
    assert lane == "Cortex"
    asked = router.asked[-1]
    floor = int(asked["user_surface_completion_floor"])
    assert floor > A_CLOSED_QUESTIONS_FLOOR, (
        "at or below the closed-question floor the runtime reads the call as one "
        "whose answer was settled upstream and leaves the private channel "
        "unbounded — which returned ten characters out of nine hundred tokens"
    )


@pytest.mark.asyncio
async def test_a_bigger_answer_declares_a_bigger_floor(router):
    skill = SovereignBrowserSkill()
    await skill._asked_of_her("say a lot", "her mind", shaped=False, most_tokens=2400)
    assert int(router.asked[-1]["user_surface_completion_floor"]) >= 2400, (
        "a floor under the caller's own budget lets the clock pay for less than "
        "the answer was asked for"
    )


@pytest.mark.asyncio
async def test_the_floor_is_taken_from_the_runtimes_own_constant():
    assert SovereignBrowserSkill._ROOM_AN_ANSWER_NEEDS == A_CLOSED_QUESTIONS_FLOOR + 1, (
        "a number written here instead of read from there is a number that "
        "drifts away from the gate it has to clear"
    )


@pytest.mark.asyncio
async def test_a_reason_for_a_measured_place_does_not_open_the_channel(router):
    skill = SovereignBrowserSkill()
    await skill._asked_of_her(
        "why is that answer yours", "her mind", shaped=False,
        most_tokens=SovereignBrowserSkill.REASON_MAX_TOKENS, worked_out_here=False,
        held_to="json_object",
    )
    asked = router.asked[-1]
    assert asked["output_shape"] == "json_object"
    assert int(asked["user_surface_completion_floor"]) <= A_CLOSED_QUESTIONS_FLOOR
    assert asked["cognitive_mode"] == "fast"
    assert asked["hard_output_token_ceiling"] == asked["max_tokens"] == SovereignBrowserSkill.REASON_MAX_TOKENS


@pytest.mark.asyncio
async def test_what_she_is_given_room_for_is_a_ceiling(router):
    """Undeclared, the gate sized a forecast at 675 words and she wrote 2,046."""
    skill = SovereignBrowserSkill()
    await skill._asked_of_her("what do you expect", "her mind", shaped=False)
    asked = router.asked[-1]
    assert asked["hard_output_token_ceiling"] == asked["max_tokens"]
    assert "cognitive_mode" not in asked, "a forecast is worked out here"


@pytest.mark.asyncio
async def test_what_she_wrote_for_herself_is_not_what_she_says(monkeypatch):
    class Leaky:
        async def think(self, prompt: str = "", **kwargs: object) -> str:
            return "<internal_critique>First instinct: INTP.</internal_critique>I expect INTJ."

    monkeypatch.setattr(
        "core.skills.sovereign_browser_understanding.optional_service",
        lambda *names, default=None: Leaky() if "llm_router" in names else default,
    )
    monkeypatch.setattr(
        "core.brain.generation_provenance.generation_metadata_of",
        lambda _reply: {"endpoint": "Cortex"},
    )
    text, _lane = await SovereignBrowserSkill()._asked_of_her("predict", "mind", shaped=False)
    assert text == "I expect INTJ."

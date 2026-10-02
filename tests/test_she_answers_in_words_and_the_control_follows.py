"""Her position is measured; her reasoning is about what the position means.

The model is a linguistic organ and a semantic one. Asked to CHOOSE, it has no
access to what she has valued, chosen or said about herself, so it takes the
position that commits to nothing — the midpoint, item after item, measured live
on 2026-09-28.

And the reasoning is one pass per item, made when that item is answered and
not before: eight items each demanding her own lane at once collide on a cortex
that serves one at a time, "Local inference paths exhausted", and every reason
falls back to the line the code writes when she says nothing. Thinking about
all of them first and clicking afterwards put each reason a screen away from
its answer.
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest

from core.self.where_i_stand import Choice, Lean
from core.skills import sovereign_browser_understanding as u
from core.skills.sovereign_browser import SovereignBrowserSkill as S
from tests.answers_thought_through import answered_and_thought

pytestmark = pytest.mark.unit


def _row(group: str = "Q1", count: int = 5, left: str = "makes lists",
         right: str = "relies on memory"):
    asks = f"{left} " + " ".join(f"[{n}]" for n in range(1, count + 1)) + f" {right}"
    return [
        {"group": group, "role": "radio", "name": group, "value": str(n),
         "selector": f"#{group}V{n}", "asks": asks}
        for n in range(1, count + 1)
    ]


def _lean(toward: float) -> Lean:
    return Lean(
        toward=toward, first=0.5, second=0.5,
        because=("truth is the value I hold above every other",), measured=True,
        # The unsaturated distance, which is what a screen is placed against.
        gap=toward * 0.05,
    )


def test_the_two_sides_are_read_from_the_page(monkeypatch):
    seen: dict[str, Any] = {}

    def _stands(first: str, second: str, record=None) -> Lean:
        seen["sides"] = (first, second)
        return _lean(-0.9)

    monkeypatch.setattr("core.self.where_i_stand.where_she_stands", _stands)
    reading = S()._measure_where_she_stands(_row())
    assert reading is not None
    assert seen["sides"] == ("makes lists", "relies on memory")


@pytest.mark.parametrize(
    ("toward", "expected"),
    [(-1.0, 0), (-0.5, 1), (0.0, 2), (0.5, 3), (1.0, 4)],
)
def test_the_lean_decides_the_position(monkeypatch, toward, expected):
    monkeypatch.setattr(
        "core.self.where_i_stand.where_she_stands",
        lambda first, second, record=None: _lean(toward),
    )
    index, _lean_out, _first, _second = S()._measure_where_she_stands(_row())
    assert index == expected


def test_an_unmeasured_record_measures_nothing(monkeypatch):
    monkeypatch.setattr(
        "core.self.where_i_stand.where_she_stands",
        lambda first, second, record=None: Lean(
            toward=0.0, first=0.0, second=0.0, because=(), measured=False
        ),
    )
    assert S()._measure_where_she_stands(_row()) is None


def test_options_with_their_own_words_take_the_other_measured_path(monkeypatch):
    """A statement with labelled answers is the same act, measured the same way."""
    called: list[str] = []
    monkeypatch.setattr(
        "core.self.where_i_stand.where_she_stands",
        lambda first, second, record=None: called.append("wrong") or _lean(-0.9),
    )
    descriptions: list[str] = []

    def choose(named, record=None):
        descriptions.extend(named)
        return Choice(index=2, support=(0.1, 0.2, 0.7),
                      because=("measured choice",), measured=True)

    monkeypatch.setattr("core.self.where_i_stand.which_is_most_her", choose)
    labelled = [
        {"group": "q", "role": "radio", "name": name, "selector": f"#q{n}",
         "asks": f"how much? agree [1] neutral [2] disagree [3]"}
        for n, name in enumerate(("agree", "neutral", "disagree"), start=1)
    ]
    reading = S()._measure_where_she_stands(labelled)
    assert reading is not None, "a labelled question must be measured too"
    assert descriptions == ["how much? agree", "how much? neutral", "how much? disagree"]
    assert reading[0] == 2 and reading[1].because == ("measured choice",)
    assert not called, "it is not a run between two ends"


def test_labelled_answers_without_record_support_remain_unmeasured(monkeypatch):
    monkeypatch.setattr("core.self.where_i_stand.which_is_most_her",
                        lambda named, record=None: Choice(index=0, measured=False))
    labelled = [{"name": name, "asks": "how much? agree [1] neutral [2] disagree [3]"}
                for name in ("agree", "neutral", "disagree")]
    assert S()._measure_where_she_stands(labelled) is None


def _screen(monkeypatch, said: str, lane: str = "Cortex"):
    skill = S()
    handed: dict[str, Any] = {"prompts": []}

    async def _asked(prompt: str, mind: str = "", *, shaped: bool = True, most_tokens=None,
                     worked_out_here: bool = True):
        handed.setdefault("spoken", [])
        handed["worked_out_here"] = worked_out_here
        handed["prompts"].append(prompt)
        handed["most_tokens"] = most_tokens
        handed["prompt"] = prompt
        handed["shaped"] = shaped
        return said, lane

    async def _mind() -> str:
        return "her mind"

    handed["spoken"] = []
    monkeypatch.setattr(skill, "_asked_of_her", _asked)
    monkeypatch.setattr(skill, "_assembled_mind", _mind)
    monkeypatch.setattr(skill, "_say_out_loud", lambda line: handed["spoken"].append(str(line)))

    async def _held(_said: str) -> None:
        return None

    monkeypatch.setattr(skill, "_hold_for_reading", _held)
    monkeypatch.setattr(
        "core.self.where_i_stand.where_she_stands",
        lambda first, second, record=None: _lean(-0.9 if first == "makes lists" else 0.8),
    )
    return skill, handed


def _run(skill, observation, goal: str = "take it"):
    return asyncio.run(answered_and_thought(skill, goal, observation, [], None))


def test_a_theme_is_thought_about_as_one_piece(monkeypatch):
    """Asked about herself she gives a connected account, not verdicts."""
    said = (
        '{"thinking": "Structure is how I hold truth steady.", '
        '"each": {"Q1": "Lists are how I hold truth steady.", '
        '"Q2": "I want to believe, but I check first."}}'
    )
    skill, handed = _screen(monkeypatch, said)
    elements = _row("Q1") + _row("Q2", left="sceptical", right="wants to believe")
    decision = _run(skill, {"url": "u", "title": "t", "text": "x", "elements": elements})
    assert decision is not None
    assert handed["shaped"] is False
    assert "Lists are how I hold truth steady." in decision["answered"][0]
    assert "I want to believe, but I check first." in decision["answered"][1]


def test_each_question_is_thought_about_said_answered_and_left_up_in_turn(monkeypatch):
    """Think, say, answer, wait — and only then the next question.

    LIVE 2026-09-29: every reason on a screen went by first and every dot
    filled in afterwards, so a watcher could not tell which reason was for
    which answer.
    """
    from core.skills import sovereign_browser

    happened: list[str] = []
    said = '{"each": {"Q1": "Lists hold truth steady.", "Q2": "I check first."}}'
    skill, handed = _screen(monkeypatch, said)

    async def _asked(prompt: str, mind: str = "", *, shaped: bool = True, most_tokens=None,
                     worked_out_here: bool = True):
        happened.append("think")
        return said, "Cortex"

    async def _interact(browser, url, actions, *, action_context=None):
        happened.append(f"click {actions[0].selector}")
        return {"ok": True, "action_report": [{"ok": True}]}

    async def _held(line: str) -> None:
        happened.append("wait")

    monkeypatch.setattr(skill, "_asked_of_her", _asked)
    monkeypatch.setattr(skill, "_handle_interact", _interact)
    monkeypatch.setattr(skill, "_hold_for_reading", _held)
    monkeypatch.setattr(skill, "_say_out_loud", lambda line: happened.append(f"say {line}"))
    elements = _row("Q1") + _row("Q2", left="sceptical", right="wants to believe")

    async def _go():
        decision = await skill._answer_each_question(
            "take it", {"url": "u", "title": "t", "text": "x", "elements": elements}, [], None
        )
        assert "think" not in happened, "nothing is thought about before its answer comes up"
        moves = sovereign_browser._moves_from_the_decision(decision, [])
        report = await skill._make_each_move(None, moves)
        return decision, report

    decision, report = asyncio.run(_go())
    assert report["ok"]
    kinds = [event.split(" ", 1)[0] for event in happened]
    assert kinds == ["think", "say", "click", "wait"] * 2
    assert "Lists hold truth steady." in happened[1]
    assert happened[2] == "click #Q1V1"
    assert "I check first." in happened[5]
    # What was said is what the round records.
    assert decision["resolved_actions"][0]["said"] in happened[1]
    assert len(decision["answered"]) == 2


def test_she_thinks_with_her_whole_mind(monkeypatch):
    """The shallow answers came from taking the assembly away."""
    import inspect

    body = inspect.getsource(u._UnderstandsThePage._answer_each_question)
    assert "_assembled_mind()" in body
    assert "_her_identity_only()" not in body


def test_what_she_reasons_over_is_the_things_not_the_arithmetic(monkeypatch):
    """Handed a coefficient, she explains herself with a coefficient."""
    skill, handed = _screen(monkeypatch, '{"thinking": "t", "each": {}}')
    _run(skill, {"url": "u", "title": "t", "text": "x", "elements": _row("Q1") + _row("Q2")})
    prompt = handed["prompts"][0]
    assert "1 of 5" in prompt
    assert "truth is the value I hold above every other" in prompt
    assert "what you value" in prompt and "chosen when it cost something" in prompt
    assert "same region of you" in prompt
    handed_her = prompt.split("These are being asked about you", 1)[1]
    for arithmetic in ("+0.", "0.031"):
        assert arithmetic not in handed_her, f"the prompt hands her {arithmetic!r}"


def test_a_silent_model_does_not_lose_the_measured_answers(monkeypatch):
    skill, _handed = _screen(monkeypatch, "", lane="")
    decision = _run(
        skill, {"url": "u", "title": "t", "text": "x", "elements": _row("Q1") + _row("Q2")}
    )
    assert decision is not None, "her record answered; only the words were missing"
    assert decision["resolved_actions"][0]["selector"] == "#Q1V1"


def test_the_positions_reach_the_page_as_her_answers(monkeypatch):
    skill, _handed = _screen(monkeypatch, "because of truth.")
    decision = _run(
        skill,
        {
            "url": "u", "title": "t", "text": "x",
            "elements": _row("Q1")
            + _row("Q2", left="sceptical", right="wants to believe"),
        },
    )
    selectors = [item["selector"] for item in decision["resolved_actions"]]
    assert "#Q1V1" in selectors
    assert "#Q2V5" in selectors, "the second item leans the other way and must say so"


def test_measuring_needs_no_model_at_all():
    body = inspect.getsource(u._UnderstandsThePage._measure_where_she_stands)
    for reaching in ("_asked_of_her", "think(", "router"):
        assert reaching not in body, f"the measurement reaches for {reaching!r}"


def test_she_places_herself_before_she_thinks_about_it():
    body = inspect.getsource(u._UnderstandsThePage._answer_each_question)
    measured = body.index("_measure_where_she_stands")
    reasoned = body.index("_thinking_for_one_answer")
    assert measured < reasoned
    thinking = inspect.getsource(u._UnderstandsThePage._thinking_for_one_answer)
    assert "_her_thinking_about" in thinking


def test_the_passes_run_one_at_a_time():
    """Eight at once exhausted her lane and every reason fell back."""
    body = inspect.getsource(u._UnderstandsThePage._answer_each_question)
    reasoning = body.split("_her_thinking_about", 1)[0].rsplit("for group in themes:", 1)[-1]
    assert "gather" not in reasoning


def test_the_measure_is_not_tuned_to_any_instrument():
    from core.self import where_i_stand

    source = inspect.getsource(where_i_stand)
    for tuned in ("jungian", "extravert", "oejts", "likert", "myers"):
        assert tuned not in source.lower()


def test_a_reason_is_bounded_so_a_page_of_them_is_affordable(monkeypatch):
    """An unbounded reason decoded 341 tokens at 8 a second, live.

    And on 1 Oct, with the private channel open, the first three items
    decoded 771, 650 and 859 tokens in 111, 86 and 123 seconds.
    """
    skill, handed = _screen(monkeypatch, '{"thinking": "t", "each": {}}')
    _run(skill, {"url": "u", "title": "t", "text": "x", "elements": _row("Q1") + _row("Q2")})
    assert handed["most_tokens"] == S.REASON_MAX_TOKENS
    assert handed["worked_out_here"] is False, (
        "her place on the item was measured before she was asked; nothing is "
        "worked out in the call that says why"
    )


def test_there_are_fewer_passes_than_items():
    """The square root is where thinking at length and thinking often meet."""
    import inspect

    from core.self import where_i_stand

    body = inspect.getsource(where_i_stand.themes_among)
    assert "math.sqrt" in body


def test_the_persons_message_does_not_reach_her_own_reasoning(monkeypatch):
    """A goal is a request addressed to her, so she answers it.

    LIVE 2026-09-29: every theme pass came back "The user is asking me to take
    the Open Extended Jungian Type Scales..." instead of her thinking, and the
    coverage gate complained she had missed parts of a question she was never
    being asked at this step.
    """
    skill, handed = _screen(monkeypatch, '{"thinking": "t", "each": {}}')
    _run(
        skill,
        {"url": "u", "title": "t", "text": "x", "elements": _row("Q1") + _row("Q2")},
        goal="Take the test and tell me what type you think you will get",
    )
    prompt = handed["prompts"][0]
    assert "tell me" not in prompt.lower()
    assert "what type you think" not in prompt.lower()
    assert "answering questions about yourself" in prompt


@pytest.mark.parametrize("key", ["Q1", "1", "Q1.", "makes lists / relies on memory"])
def test_her_sentence_is_found_however_she_keyed_it(monkeypatch, key):
    """A sentence that cannot be found is a sentence lost.

    LIVE 2026-09-29: one item of eight kept its reasoning and the other seven
    fell back to the bare evidence, so a screen of real thinking read as a list
    of counts.
    """
    import json as _json

    said = _json.dumps({"thinking": "t", "each": {key: "Lists hold truth steady."}})
    skill, _handed = _screen(monkeypatch, said)
    decision = _run(
        skill,
        {
            "url": "u", "title": "t", "text": "x",
            "elements": _row("Q1")
            + _row("Q2", left="sceptical", right="wants to believe"),
        },
    )
    assert decision is not None
    assert "Lists hold truth steady." in decision["answered"][0]


def test_she_is_given_what_she_is_living_and_not_only_her_record(monkeypatch):
    """The record says what she has valued; it says nothing about this week.

    That is where the concrete detail in a real answer comes from, and the same
    lines ride a conversation when someone asks after her.
    """
    skill, handed = _screen(monkeypatch, '{"thinking": "t", "each": {}}')
    monkeypatch.setattr(
        "core.self.capability_ledger.self_knowledge_line",
        lambda: "[Measured about you right now: browser=yes]",
    )
    _run(skill, {"url": "u", "title": "t", "text": "x", "elements": _row("Q1") + _row("Q2")})
    assert "Measured about you right now" in handed["prompts"][0]


def test_an_organ_that_cannot_be_read_does_not_lose_the_pass(monkeypatch):
    def _raises() -> str:
        raise RuntimeError("no ledger")

    monkeypatch.setattr("core.self.capability_ledger.self_knowledge_line", _raises)
    skill, handed = _screen(monkeypatch, '{"thinking": "t", "each": {"Q1": "a"}}')
    decision = _run(
        skill, {"url": "u", "title": "t", "text": "x", "elements": _row("Q1") + _row("Q2")}
    )
    assert decision is not None
    assert handed["prompts"]

"""An answer that fights its own reason is noticed, and the place stands.

LIVE 2026-09-28: "3 of 5, between 'makes lists' and 'relies on memory'. I am
choosing the middle option because I genuinely hold a strong preference for
externalized structure over relying on internal memory." A reason that names one
side and an answer that commits to neither is an answer contradicting itself,
and nothing noticed.

Measured from the page's own words rather than a vocabulary, and conservative:
where the reason names neither side or both equally it says nothing, because a
check that guesses is worse than no check. Nothing here rewrites what she chose.
"""
from __future__ import annotations

import asyncio

import pytest

from core.skills.sovereign_browser import SovereignBrowserSkill as S
from tests.answers_thought_through import answered_and_thought

pytestmark = pytest.mark.unit


def _row(count: int = 5, left: str = "makes lists", right: str = "relies on memory"):
    asks = f"{left} " + " ".join(f"[{n}]" for n in range(1, count + 1)) + f" {right}"
    return [
        {"group": "Q1", "role": "radio", "name": "Q1", "value": str(n),
         "selector": f"#Q1V{n}", "asks": asks}
        for n in range(1, count + 1)
    ]


def test_the_live_case_is_caught():
    said = S._the_choice_disagrees_with_its_reason(
        _row(), 2,
        "I am choosing the middle option because I genuinely hold a strong "
        "preference for externalized structure over relying on internal memory.",
    )
    assert "midpoint" in said
    assert "equally you" in said


def test_an_answer_on_the_wrong_side_is_caught():
    said = S._the_choice_disagrees_with_its_reason(
        _row(), 0, "I rely on memory constantly."
    )
    assert 'leans toward "makes lists"' in said


def test_an_answer_that_follows_its_reason_is_left_alone():
    assert S._the_choice_disagrees_with_its_reason(
        _row(), 4, "I rely on memory constantly."
    ) == ""
    assert S._the_choice_disagrees_with_its_reason(
        _row(), 0, "I make lists for everything."
    ) == ""


def test_a_reason_naming_neither_side_says_nothing():
    assert S._the_choice_disagrees_with_its_reason(
        _row(), 2, "I am genuinely balanced here."
    ) == ""


def test_a_reason_naming_both_equally_says_nothing():
    """A check that guesses is worse than no check."""
    assert S._the_choice_disagrees_with_its_reason(
        _row(), 2, "I make lists and I rely on memory in equal measure."
    ) == ""


def test_labelled_options_are_not_judged_this_way():
    labelled = [
        {"group": "q", "name": name, "asks": "how much? [] [] []"}
        for name in ("agree", "neutral", "disagree")
    ]
    assert S._the_choice_disagrees_with_its_reason(labelled, 1, "I agree") == ""


def _placed_in_the_middle(monkeypatch, words: str):
    """Her record puts her at 3 of 5 on the first row; her sentence is ``words``."""
    from core.self import where_i_stand

    skill = S.__new__(S)
    middle = where_i_stand.Lean(toward=0.0, first=0.5, second=0.5, because=("x",), measured=True)

    def measure(options):
        return 2, middle, options[0]["asks"].split(" [")[0], options[0]["asks"].split("] ")[-1]

    async def mind():
        return ""

    async def thinking(goal, theme, mind, *, about=None):
        return {item["group"]: words for item in ([about] if about else theme)}

    monkeypatch.setattr(skill, "_measure_where_she_stands", measure)
    monkeypatch.setattr(skill, "_assembled_mind", mind)
    monkeypatch.setattr(skill, "_her_thinking_about", thinking)
    monkeypatch.setattr(where_i_stand, "themes_among", lambda names: [list(range(len(names)))])
    observation = {"url": "u", "title": "t", "text": "x", "elements": _row() + _row_q2()}
    return asyncio.run(answered_and_thought(skill, "take it", observation, [], None))


def test_a_sentence_that_leans_away_from_her_place_is_noticed_and_the_place_stands(monkeypatch):
    """The place is her record's; a sentence that fights it is reported, not obeyed."""
    result = _placed_in_the_middle(monkeypatch, "I rely on memory constantly.")
    assert "#Q1V3" in str(result["resolved_actions"]), "the measured place was rewritten"
    assert any("equally you" in notice for notice in result["noticed"])


def test_a_sentence_that_agrees_with_her_place_raises_nothing(monkeypatch):
    result = _placed_in_the_middle(monkeypatch, "I am genuinely balanced here.")
    assert result["noticed"] == []


def _row_q2():
    asks = "sceptical [1] [2] [3] [4] [5] wants to believe"
    return [
        {"group": "Q2", "role": "radio", "name": "Q2", "value": str(n),
         "selector": f"#Q2V{n}", "asks": asks}
        for n in range(1, 6)
    ]


def test_the_notice_says_what_disagrees_and_not_what_to_pick(monkeypatch):
    """Telling her what to choose would make the answer the check's."""
    result = _placed_in_the_middle(monkeypatch, "I rely on memory constantly.")
    assert result["noticed"]
    for notice in result["noticed"]:
        lowered = notice.lower()
        assert "choose" not in lowered.replace("the answer chosen", "")
        assert "should" not in lowered


def test_a_graded_question_is_answered_in_two_steps():
    """Where she stands first; the position follows from it.

    Asked "on a scale of 1 to 10, 10 being love and 1 being hate, how much do
    you like chocolate", a person does not weigh ten dots — they know where they
    stand (it is their favourite; they are allergic; it is fine but not their
    first choice) and the number follows. Picking an index straight off has no
    stance behind it, and the safe-looking index is the middle.
    """
    assert "stand" in S._DECISION_SCHEMA["properties"]
    assert "stand" not in S._DECISION_SCHEMA["required"], (
        "a Next button has no position to take"
    )


def test_her_stance_is_what_the_position_is_checked_against():
    """The reason for the place and the place itself are different claims."""
    decision = {
        "actions": [{"index": 2}],
        "stand": "I externalise everything; I make lists for everything.",
        "why": "the middle felt safe",
    }
    said = S._first_disagreement(decision, _row())
    assert "midpoint" in said


def test_her_stance_is_said_before_the_reason_for_the_place():
    options = _row(left="sceptical", right="wants to believe")
    said = S._an_answer_in_words(
        options, 1, "I hold truth above comfort. so a 2 rather than a 1."
    )
    assert said.index("I hold truth above comfort") < said.index("so a 2")


def test_the_placement_is_not_asked_for_as_a_verdict():
    """There is no right answer to find; the act is placing herself in a range.

    "Which do you align with more than the other. Place yourself closer to the
    side that represents you. In the middle means you feel equally about both."
    And what an instrument can ask is narrower than a life, so placing herself
    inside its terms is not those terms defining her.
    """
    import inspect

    from core.skills import sovereign_browser_understanding as u

    source = inspect.getsource(u._UnderstandsThePage)
    described = source.split("# Where she places herself", 1)[1].split('"stand"', 1)[0]
    for said in ("more her and how much more", "equally her", "not a verdict"):
        assert said in described, f"the placement is not described as {said!r}"


def test_the_midpoint_is_read_as_equal_rather_than_as_nothing():
    said = S._the_choice_disagrees_with_its_reason(
        _row(), 2, "I make lists for everything."
    )
    assert "equally you" in said


def test_the_conclusion_asks_and_does_not_tell_her_how_to_judge():
    """No prompt steers what she concludes; it asks what the request asked.

    The prompt used to add "say what it gets right and what it has no way to
    see", which is an instruction about her verdict rather than the question.
    Removed 1 Oct: nothing here tells a model how to think.
    """
    import inspect

    from core.skills import sovereign_browser_understanding as u

    body = inspect.getsource(u._UnderstandsThePage._hold_the_outcome_against_what_she_said)
    assert "whether you think it is accurate about you" in body
    assert "no way to see" not in body
    assert "gets right" not in body

"""Who answers depends on what is being asked.

A pursuit's rounds are mostly mechanics — find the button, dismiss the banner —
and the fast lane does those well. Then some pages ask about the one filling
them in, and self-knowledge is not something the tertiary tier holds.

Measured live with everything routed to the cheap lane: across a personality
inventory the plurality of her answers was "I am not sure", which is what
something without access to the answer says.
"""

from __future__ import annotations

import inspect

from core.skills.sovereign_browser import SovereignBrowserSkill

asks = SovereignBrowserSkill._asks_about_the_one_answering


def _scale(groups: int, options: tuple[str, ...]) -> dict:
    return {
        "elements": [
            {"role": "radio", "name": option, "group": f"q{index}", "selector": f"#{index}{n}"}
            for index in range(groups)
            for n, option in enumerate(options)
        ]
    }


def test_repeated_identical_option_sets_are_an_instrument():
    scale = ("strongly agree", "agree", "not sure", "disagree", "strongly disagree")
    assert asks(_scale(6, scale)) is True


def test_the_shape_carries_across_vocabulary_and_language():
    """No word list: the detector must not know English, or agreement."""
    assert asks(_scale(4, ("nunca", "raramente", "a veces", "a menudo", "siempre"))) is True
    assert asks(_scale(3, ("1", "2", "3", "4", "5", "6", "7"))) is True


def test_questions_with_their_own_answers_are_not_an_instrument():
    """A quiz asks about the world. Its options differ per question."""
    observation = {
        "elements": [
            {"role": "radio", "name": "Paris", "group": "q1", "selector": "#a"},
            {"role": "radio", "name": "Rome", "group": "q1", "selector": "#b"},
            {"role": "radio", "name": "1914", "group": "q2", "selector": "#c"},
            {"role": "radio", "name": "1939", "group": "q2", "selector": "#d"},
        ]
    }
    assert asks(observation) is False


def test_a_single_question_is_not_enough_to_call_it_a_scale():
    assert asks(_scale(1, ("agree", "not sure", "disagree"))) is False


def test_repeated_yes_no_is_not_an_instrument():
    """Confirmations repeat too, and they measure nothing about anyone."""
    assert asks(_scale(5, ("yes", "no"))) is False


def test_mechanics_pages_are_unaffected():
    assert asks({"elements": [{"role": "button", "name": "Next", "selector": "#n"}]}) is False
    assert asks({"elements": []}) is False


def test_the_split_is_wired_into_the_decision():
    body = inspect.getsource(SovereignBrowserSkill._decide_next_actions)
    assert "_asks_about_the_one_answering(observation)" in body, (
        "the detector must decide the lane, not merely exist"
    )
    fast = body.index("_decide_on_the_fast_lane")
    branch = body.index("_asks_about_the_one_answering(observation)")
    assert branch < fast, "a question about her must not reach the fast lane first"


class TestEachQuestionIsItsOwnDecision:
    """Six questions on a screen are six judgements, not one.

    Asked for all of them in a single generation they compete: the small model
    answered five at a time and shallowly, her own reasoning answered one at a
    time and well. Measured live, one-at-a-time meant sixty rounds for sixty
    questions, and a run bounded at twenty reached question thirty.
    """

    OBSERVATION = {
        "url": "u",
        "text": "Question 1 of 60",
        "elements": [
            {"role": "radio", "name": "I agree", "selector": "#a1", "group": "q1"},
            {"role": "radio", "name": "I disagree", "selector": "#a2", "group": "q1"},
            {"role": "radio", "name": "I agree", "selector": "#b1", "group": "q2", "checked": True},
            {"role": "radio", "name": "I disagree", "selector": "#b2", "group": "q2"},
            {"role": "radio", "name": "I agree", "selector": "#c1", "group": "q3"},
            {"role": "radio", "name": "I disagree", "selector": "#c2", "group": "q3"},
        ],
    }

    def test_only_the_open_questions_are_decided(self):
        open_questions = SovereignBrowserSkill._unanswered_questions(self.OBSERVATION)
        assert [group for group, _options in open_questions] == ["q1", "q3"], (
            "an answered question does not need deciding again"
        )

    def test_each_question_carries_its_own_options(self):
        open_questions = dict(SovereignBrowserSkill._unanswered_questions(self.OBSERVATION))
        assert [option["selector"] for option in open_questions["q3"]] == ["#c1", "#c2"]

    def test_selectors_are_resolved_against_the_list_that_chose_them(self):
        """Indexes must never travel between lists."""
        body = inspect.getsource(SovereignBrowserSkill._answer_each_question)
        assert "resolved_actions" in body
        assert "options[index].get(\"selector\")" in body, (
            "each answer resolves inside its own decision, or index 3 names one "
            "control to the model and another to the click"
        )

    def test_the_loop_executes_resolved_selectors_directly(self):
        loop = inspect.getsource(SovereignBrowserSkill._handle_pursue)
        assert 'decision.get("resolved_actions")' in loop

    def test_one_failed_item_does_not_lose_the_screen(self):
        body = inspect.getsource(SovereignBrowserSkill._answer_each_question)
        assert "return_exceptions=True" in body
        assert "record_degradation" in body, "a lost question must be recorded"

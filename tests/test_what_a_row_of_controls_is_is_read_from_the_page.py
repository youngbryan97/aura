"""What a row of controls IS, is read from the page — not decided in the code.

A run of unlabelled controls can be a scale between two opposites, a set of
choices, a "more like me / less like me" ranking, or whatever the page's own
instructions say. Deciding here that it is any one of those puts a rule where
her reading of the page belongs, and the rule is wrong on the next site.

So the code states what can be seen — how many controls, labelled or not, one or
several, and the words on either side of the run — and what it MEANS is hers,
worked out once and carried across the rounds that answer it.
"""
from __future__ import annotations

import pytest

from core.skills.sovereign_browser import SovereignBrowserSkill as S

pytestmark = pytest.mark.unit


def _row(count: int = 5, left: str = "makes lists", right: str = "relies on memory",
         role: str = "radio", labels: list[str] | None = None):
    asks = f"{left} " + " ".join(f"[{n}]" for n in range(1, count + 1))
    asks = f"{asks} {right}" if right else asks
    return [
        {
            "group": "Q1",
            "role": role,
            "name": (labels[n - 1] if labels else "Q1"),
            "value": str(n),
            "asks": asks,
        }
        for n in range(1, count + 1)
    ]


def test_the_layout_is_stated_without_being_interpreted():
    said = S._how_the_options_are_laid_out(_row())
    assert "5 controls" in said
    assert "none of them labelled" in said
    assert "one may be chosen" in said
    assert 'between "makes lists" and "relies on memory"' in said


def test_nothing_it_says_calls_the_row_a_scale():
    """The word is hers to use or not, from what the page says."""
    for shape in (
        _row(),
        _row(count=3, role="checkbox"),
        _row(count=3, labels=["agree", "neutral", "disagree"]),
        _row(count=3, right=""),
    ):
        said = S._how_the_options_are_laid_out(shape).lower()
        for claim in ("scale", "midpoint", "opposite", "entirely", "more", "less"):
            assert claim not in said, (
                f"the layout description asserts {claim!r}, which is a reading "
                "of the page and not a fact of it"
            )


def test_several_choosable_controls_say_so():
    said = S._how_the_options_are_laid_out(_row(count=3, role="checkbox"))
    assert "several may be chosen" in said


def test_labelled_options_say_they_carry_labels():
    said = S._how_the_options_are_laid_out(
        _row(count=3, labels=["agree", "neutral", "disagree"])
    )
    assert "each with its own label" in said


def test_words_on_one_side_only_are_reported_as_that():
    said = S._how_the_options_are_laid_out(_row(count=3, right=""))
    assert 'after "makes lists"' in said
    assert "between" not in said


def test_one_control_is_not_a_group_to_describe():
    assert S._how_the_options_are_laid_out(_row(count=1)) == ""


def test_the_page_she_is_shown_carries_the_layout():
    observation = {
        "url": "https://example.test",
        "title": "t",
        "text": "x",
        "elements": [
            {**option, "selector": f"#Q1V{n}"}
            for n, option in enumerate(_row(), start=1)
        ],
    }
    rendered = S._render_observation(observation, "take the test")
    assert "question Q1 offers:" in rendered
    assert 'between "makes lists" and "relies on memory"' in rendered


def test_an_unlabelled_answer_says_what_it_sits_between():
    """And which end it is nearer, which "between" alone did not say."""
    options = _row(left="sceptical", right="wants to believe")
    said = S._an_answer_in_words(options, 1, "")
    assert "2 of 5" in said
    assert 'nearer "sceptical" than "wants to believe"' in said


def test_a_labelled_answer_still_says_its_own_label():
    options = _row(count=3, labels=["agree", "neutral", "disagree"])
    said = S._an_answer_in_words(options, 0, "")
    assert "agree" in said
    assert "between" not in said


def test_how_the_page_wants_to_be_answered_is_part_of_her_understanding():
    assert "how_to_answer" in S._UNDERSTANDING_SCHEMA["properties"]
    rendered = S._render_understanding(
        {"here": "a survey", "how_to_answer": "each row is a scale; pick where I sit"}
    )
    assert "How this page wants to be answered" in rendered
    assert "pick where I sit" in rendered


def test_each_control_says_where_it_sits_in_the_run():
    """Picking item k from a list is not saying where in a range you are.

    A control whose only name is its group's reads as one nameless thing among
    five. Its place in the run is a fact of the layout; what being there means
    is hers.
    """
    observation = {
        "url": "https://example.test",
        "title": "t",
        "text": "x",
        "elements": [
            {**option, "selector": f"#Q1V{n}"}
            for n, option in enumerate(_row(), start=1)
        ],
    }
    rendered = S._render_observation(observation, "take the test")
    assert "position 1 of 5" in rendered
    assert "position 5 of 5" in rendered


def test_labelled_options_are_not_given_positions():
    """Their labels say what they are; a place in a run would say less."""
    observation = {
        "url": "https://example.test",
        "title": "t",
        "text": "x",
        "elements": [
            {**option, "selector": f"#q{n}"}
            for n, option in enumerate(
                _row(count=3, labels=["agree", "neutral", "disagree"]), start=1
            )
        ],
    }
    rendered = S._render_observation(observation, "take the test")
    assert "position 1 of 3" not in rendered


def test_a_pair_is_a_choice_and_gets_no_positions():
    observation = {
        "url": "https://example.test",
        "title": "t",
        "text": "x",
        "elements": [
            {**option, "selector": f"#q{n}"}
            for n, option in enumerate(_row(count=2), start=1)
        ],
    }
    assert "position 1 of 2" not in S._render_observation(observation, "take it")


def test_a_side_that_carries_quote_marks_is_kept_whole():
    """LIVE 2026-10-02: 'likes to know "who?", "what?", "when?"' was cut at its first quote."""
    left = 'likes to know "who?", "what?", "when?"'
    options = _row(left=left, right='likes to know "why?"')
    assert S._the_two_sides(options) == (left, 'likes to know "why?"')
    said = S._an_answer_in_words(options, 4, "")
    assert 'nearer "likes to know "why?"" than' in said


def test_labelled_options_have_no_two_sides():
    assert S._the_two_sides(_row(count=3, labels=["agree", "neutral", "disagree"])) is None
    assert S._the_two_sides(_row(count=3, right="")) is None

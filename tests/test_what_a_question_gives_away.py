"""What somebody asking tells her, quite apart from the answer.

The good Cluedo players are not the ones who track the answers. Everybody
tracks the answers. They watch what other people ASK, because a question is
not free.
"""

from __future__ import annotations

from core.cognition.what_a_question_gives_away import WhatTheirAskingSays


def _heard() -> WhatTheirAskingSays:
    said = WhatTheirAskingSays()
    said.they_asked("green", ["the dagger", "the library", "peacock"])
    said.they_asked("mustard", ["the rope"])
    said.they_asked("green", ["the dagger", "the study", "peacock"])
    said.they_asked("mustard", ["the rope", "the hall"])
    said.they_asked("green", ["the dagger"])
    return said


def test_asking_about_a_thing_says_they_do_not_have_it() -> None:
    said = _heard()
    assert "the dagger" in said.they_have_not_got("green")
    assert "the rope" in said.they_have_not_got("mustard")


def test_asking_twice_says_the_first_answer_did_not_settle_it() -> None:
    """A fact about their situation rather than about the thing."""
    said = _heard()
    assert said.what_they_are_stuck_on("green")[0] == "the dagger"
    assert "the study" not in said.what_they_are_stuck_on("green")


def test_stopping_asking_is_the_moment_their_knowledge_changed() -> None:
    """Visible from outside without anybody saying anything."""
    said = _heard()
    stopped = said.what_they_have_stopped_asking("green")
    assert "the library" in stopped
    assert "the dagger" not in stopped, "still asking about it"


def test_a_short_list_of_questions_is_a_nearly_finished_one() -> None:
    """How a table knows somebody is about to win before they say so."""
    said = _heard()
    assert said.who_is_furthest_along() == "mustard"


def test_somebody_who_has_asked_nothing_says_nothing() -> None:
    said = WhatTheirAskingSays()
    assert said.they_have_not_got("nobody") == ()
    assert "asked nothing yet" in said.describe("nobody")
    assert said.who_is_furthest_along() == ""

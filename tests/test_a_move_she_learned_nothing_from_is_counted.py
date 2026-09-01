"""A move thrown away in silence.

Two things have to be true before a move teaches her how the world moves: she
has to have been looking at the thing rather than the page around it, and the
readings either side have to be in one frame. When either fails the pair is
discarded, and nothing said so — the only trace was the rule staying
unworked-out, which reads as a hard world rather than as evidence going
missing on the way to the learner.

LIVE 2026-08-31: fifty-four moves of the real game, one of them watched.
"""

from __future__ import annotations

from core.skills.screen_pursuit import _what_she_could_not_learn_from


def test_nothing_lost_says_nothing():
    assert _what_she_could_not_learn_from({}) == ""
    assert _what_she_could_not_learn_from({"a different frame": 0}) == ""


def test_a_loss_is_named_and_counted():
    said = _what_she_could_not_learn_from({"a different frame": 53})
    assert "53" in said and "different frame" in said


def test_each_reason_is_kept_apart():
    """Which of the two it was decides what to go and fix."""
    said = _what_she_could_not_learn_from(
        {"a different frame": 12, "not the thing itself": 7}
    )
    assert "12" in said and "7" in said
    assert "different frame" in said and "not the thing itself" in said


def test_a_reason_that_never_fired_is_not_reported():
    said = _what_she_could_not_learn_from(
        {"a different frame": 4, "not the thing itself": 0}
    )
    assert "not the thing itself" not in said


def test_it_survives_being_handed_nothing():
    assert _what_she_could_not_learn_from(None) == ""

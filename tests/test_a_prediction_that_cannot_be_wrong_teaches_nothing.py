"""What she claimed reaches the check, and being right by it means something.

Measured live on 2026-08-26: every screen move carried the claim that the view
would differ, which is satisfied by almost any keystroke on almost any screen.
Seventeen of twenty predictions held, and holding told her nothing — while the
length of her plans, the part of the screen she believed answered to her, and
the record of what her moves lead to were all reading that verdict as if it
were confidence.

She had been predicting specifically the whole time. None of it was checked.
"""

from __future__ import annotations

import pytest

from core.agency.deliberate_action import ActionOption, Deliberation, Expectation
from core.agency.standing_strategy import claim_in
from core.perception.what_is_there import arranged

BEFORE = arranged([
    (0.30, 0.20, "4"), (0.30, 0.35, "2"),
    (0.45, 0.20, "4"), (0.45, 0.35, "8"),
    (0.60, 0.20, "64"), (0.60, 0.35, "16"),
])
AFTER_MERGED = arranged([
    (0.30, 0.20, "8"), (0.30, 0.35, "2"),
    (0.45, 0.35, "8"),
    (0.60, 0.20, "64"), (0.60, 0.35, "16"),
])
AFTER_NOTHING = BEFORE


# ── reading a claim out of her own words ─────────────────────────────────

@pytest.mark.parametrize(
    ("said", "keep", "avoid", "place"),
    [
        ("the two 4s in column 1 will merge into an 8", ("8",), (), ""),
        ("keep the 64 in the bottom-left corner", ("64",), (), "bottom-left"),
        ("press left so the board does not fill up with 2s", (), ("2",), ""),
        ("move to row 3 and check seat 12", (), (), ""),
        ("going left", (), (), ""),
        ("", (), (), ""),
    ],
)
def test_what_she_said_becomes_something_checkable(said, keep, avoid, place):
    claim = claim_in(said)
    assert claim.contains == keep
    assert claim.absent == avoid
    assert claim.at_place == place


def test_a_number_that_says_where_is_not_a_thing_to_watch_for():
    """"column 1" names a place. Read as a value it claimed a 1 would appear."""
    assert "1" not in claim_in("the two 4s in column 1 will merge into an 8").contains


def test_a_plural_is_the_same_thing_said_of_several():
    assert claim_in("two 4s are showing").contains == ("4",)


@pytest.mark.parametrize(
    ("said", "result"),
    [
        ("the two 4s will merge into an 8", ("8",)),
        ("the 8 and the 8 combine to give a 16", ("16",)),
        ("left turns the row into 4 2 8", ("4", "2", "8")),
    ],
)
def test_in_a_transformation_the_result_is_the_claim(said, result):
    """What went in was spent. It is not a claim that it will still be there."""
    assert claim_in(said).contains == result


def test_a_claim_that_says_nothing_admits_it():
    assert not claim_in("going left").says_something()
    assert not Expectation(changed=True).says_something()
    assert claim_in("keep the 64 in the corner").says_something()


# ── checking it against the thing ────────────────────────────────────────

def test_a_claim_that_came_true_holds():
    verdict = claim_in("the two 4s will merge into an 8").check_in(BEFORE, AFTER_MERGED)
    assert verdict.held
    assert verdict.observed_change


def test_a_claim_that_did_not_come_true_says_which_part():
    verdict = claim_in("this will make a 256").check_in(BEFORE, AFTER_MERGED)
    assert not verdict.held
    assert any("256" in part for part in verdict.missing)


def test_a_claim_about_a_place_is_a_question_with_an_answer():
    holding = claim_in("keep the 64 in the bottom-left corner")
    assert holding.check_in(BEFORE, AFTER_MERGED).held


def test_the_same_claim_fails_when_the_thing_moves_off_that_place():
    moved = arranged([(0.30, 0.20, "64"), (0.60, 0.35, "16")])
    verdict = claim_in("keep the 64 in the bottom-left corner").check_in(BEFORE, moved)
    assert not verdict.held
    assert any("64" in part for part in verdict.missing)


def test_a_move_that_changed_nothing_is_stalled_whatever_else_is_true():
    verdict = claim_in("the 64 stays put").check_in(BEFORE, AFTER_NOTHING)
    assert verdict.stalled
    assert not verdict.held


def test_something_that_should_be_gone_and_is_not():
    verdict = claim_in("after this there are no 2s left").check_in(BEFORE, AFTER_MERGED)
    assert not verdict.held
    assert any("2" in part for part in verdict.lingering)


# ── the claim travels with the decision ──────────────────────────────────

def test_a_decision_carries_the_sharper_claim_when_there_is_one():
    move = ActionOption(name="left", expectation=Expectation(changed=True))
    made = Deliberation(
        goal="get to 256",
        situation="",
        chosen=move,
        rationale="the two 4s will merge into an 8",
        expected=claim_in("the two 4s will merge into an 8"),
    )
    assert made.expected is not None
    assert made.expected.contains == ("8",)


def test_without_a_sharper_claim_the_option_keeps_its_own():
    move = ActionOption(
        name="left", expectation=Expectation(changed=True, describes="the view differs")
    )
    made = Deliberation(goal="g", situation="", chosen=move, rationale="going left")
    assert made.expected is None


def test_the_weak_claim_is_still_there_for_a_move_that_said_nothing():
    """A default that cannot be interestingly wrong is better than no check."""
    weak = Expectation(changed=True)
    assert weak.check_in(BEFORE, AFTER_MERGED).held
    assert weak.check_in(BEFORE, AFTER_NOTHING).stalled

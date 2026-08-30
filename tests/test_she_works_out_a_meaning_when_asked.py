"""induce_from and admit had no production caller. Now one asks at the right moment.

A demonstrated facility is not a live one. The machinery for giving a kind of
rule a meaning existed, passed its tests, and nothing in an ordinary turn ever
invoked it upon representational failure — which is the same shape of defect as
the emergent-goal engine that was fed for months and never asked what its
observations came to.

The right moment is the last one: after every rule she has has been tried and
none of them says it. Offered earlier, a wider net would lose the simpler answer
that was already right, which is why the ordering search sits behind the same
point.

And the state that mattered most was the one nobody had handled: the positional
proof needs the cells to stay put to say anything, so a family that CHANGES the
values is neither settled nor refuted by it. That exit returned an empty string.
"I have not proved I cannot" is not a reason to stop looking.
"""

from __future__ import annotations

import pytest

from core.cognition.an_invented_kind import KINDS
from core.cognition.sequence_induction import answer_sequence_question

PAIRWISE = (
    "[1,5,2,9] becomes [5,5,9,9]. [3,1,4,1] becomes [3,3,4,4]. "
    "[7,2,8,6] becomes [7,7,8,8]. [2,6,1,3] becomes [6,6,3,3]. "
    "What does [4,9,1,2] become?"
)
AGAIN = "[3,7,1,4] becomes [7,7,4,4]. What does [2,8,5,1] become?"
MIRRORED = (
    "[1,2,3] becomes [3,2,1]. [4,5,6] becomes [6,5,4]. "
    "[7,8,9] becomes [9,8,7]. [1,1,2] becomes [2,1,1]. "
    "What does [5,6,7] become?"
)


@pytest.fixture(autouse=True)
def _a_clean_registry(tmp_path, monkeypatch):
    from core.cognition import what_she_gave_meaning

    monkeypatch.setattr(what_she_gave_meaning, "_KEPT_AT", tmp_path / "meanings.json")
    held = dict(KINDS)
    KINDS.clear()
    yield
    KINDS.clear()
    KINDS.update(held)


# ── she works one out rather than saying she cannot ──────────────────────

def test_a_family_no_rule_of_hers_can_say_is_worked_out():
    said = answer_sequence_question(PAIRWISE)
    assert said.startswith("[9, 9, 2, 2]")


def test_and_she_says_that_is_what_she_did():
    said = answer_sequence_question(PAIRWISE)
    assert "worked out what the examples are doing" in said
    assert "gave it a meaning" in said


def test_and_names_the_meaning_in_words():
    answer_sequence_question(PAIRWISE)
    assert any("take" in kind for kind in KINDS)


def test_a_reversal_gives_a_different_meaning():
    said = answer_sequence_question(MIRRORED)
    assert said.startswith("[7, 6, 5]")


# ── and the second question of a kind is cheaper ─────────────────────────

def test_one_example_is_enough_once_she_has_met_the_family():
    answer_sequence_question(PAIRWISE)
    said = answer_sequence_question(AGAIN)
    assert said.startswith("[8, 8, 5, 5]")
    assert "met this before" in said


def test_meeting_it_before_needs_no_working_out():
    answer_sequence_question(PAIRWISE)
    before = dict(KINDS)
    answer_sequence_question(AGAIN)
    assert dict(KINDS) == before, "she invented a second meaning for a family she knew"


def test_a_known_meaning_that_does_not_fit_is_not_used():
    """Having met one family is not a licence to answer with it."""
    answer_sequence_question(PAIRWISE)
    said = answer_sequence_question(MIRRORED)
    assert said.startswith("[7, 6, 5]")


# ── and nothing is invented where the language already answers ───────────

def test_a_family_her_own_rules_explain_is_not_given_a_new_meaning():
    plain = (
        "[1,2,3] becomes [2,3,1]. [4,5,6] becomes [5,6,4]. "
        "[7,8,9] becomes [8,9,7]. What does [1,1,2] become?"
    )
    answer_sequence_question(plain)
    assert KINDS == {}, "a meaning was invented where an ordinary rule sufficed"


def test_what_she_worked_out_is_kept():
    answer_sequence_question(PAIRWISE)
    from core.cognition.what_she_gave_meaning import _KEPT_AT

    assert _KEPT_AT.exists()


# ── and she says when the examples do not settle it ──────────────────────

def test_meanings_are_told_apart_by_what_they_do_not_how_they_are_written():
    """Taking the smaller of a pair reads two ways round and is one meaning.

    Reporting that as two readings of the evidence would invent a doubt that
    is not there.
    """
    from core.cognition.an_invented_kind import everything_that_fits

    same = [((1, 2, 3), (1, 1, 1)), ((4, 5, 6), (4, 4, 4)),
            ((7, 8, 9), (7, 7, 7)), ((2, 9, 4), (2, 2, 2))]
    assert len(everything_that_fits(same)) == 1


def test_genuinely_thin_evidence_fits_several_meanings():
    from core.cognition.an_invented_kind import everything_that_fits

    thin = [((1, 2, 1), (1, 2, 1)), ((3, 4, 3), (3, 4, 3))]
    assert len(everything_that_fits(thin)) > 1


def test_and_a_case_that_would_settle_it_can_be_found():
    from core.cognition.an_invented_kind import (
        everything_that_fits,
        what_would_tell_them_apart,
    )

    fits = everything_that_fits([((1, 2, 1), (1, 2, 1)), ((3, 4, 3), (3, 4, 3))])
    telling = what_would_tell_them_apart(fits[0], fits[1], of_length=3)
    assert telling is not None
    assert fits[0].read(telling) != fits[1].read(telling)


def test_two_meanings_that_never_disagree_have_nothing_to_settle():
    from core.cognition.an_invented_kind import Induced, what_would_tell_them_apart

    one = Induced("here", "its partner", "the smaller of it and its neighbour")
    same = Induced("its partner", "here", "the smaller of it and its neighbour")
    assert what_would_tell_them_apart(one, same, of_length=4) is None

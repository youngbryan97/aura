"""Registered, available, and having actually worked are three different facts.

Asked on 2026-08-27 how many of her skills had never once executed
successfully, she counted the .py files in a directory and said so. That is an
honest method for a different question, and she had thirty-two thousand
outcomes on disk that nothing could read back as an answer about herself.

What the record covers is stated rather than implied: decisions and
sensorimotor actions, not skill executions, which resolve elsewhere.
"""

from __future__ import annotations

import sqlite3

import pytest

from core.self.what_has_ever_worked import (
    HowItHasGone,
    never_worked,
    says,
    what_has_ever_worked,
)


class Learner:
    """An outcome record with a known history in it."""

    def __init__(self, tmp_path, rows=()):
        self._db_path = str(tmp_path / "outcomes.db")
        held = sqlite3.connect(self._db_path)
        held.execute("CREATE TABLE outcomes (action TEXT, success INTEGER)")
        held.executemany("INSERT INTO outcomes VALUES (?, ?)", rows)
        held.commit()
        held.close()


@pytest.fixture
def record(tmp_path):
    return Learner(
        tmp_path,
        rows=[("writes", 1)] + [("writes", 0)] * 9 + [("tries", 0)] * 4 + [("always", 1)] * 3,
    )


# ── reading it back ──────────────────────────────────────────────────────

def test_every_action_on_record_is_counted(record):
    gone = what_has_ever_worked(record)
    assert set(gone) == {"writes", "tries", "always"}
    assert gone["writes"].tried == 10
    assert gone["writes"].worked == 1


def test_a_thing_that_worked_once_has_worked(record):
    assert what_has_ever_worked(record)["writes"].has_ever_worked is True


def test_a_thing_that_never_did_has_not(record):
    assert what_has_ever_worked(record)["tries"].has_ever_worked is False


def test_the_share_is_reported_as_it_is(record):
    assert what_has_ever_worked(record)["writes"].share == pytest.approx(0.1)


def test_it_says_what_it_found(record):
    assert what_has_ever_worked(record)["writes"].says() == "writes: 1 of 10 worked (10%)"
    assert what_has_ever_worked(record)["tries"].says() == "tries: tried 4×, never worked"


# ── the three facts, kept apart ──────────────────────────────────────────

def test_tried_and_never_worked_is_not_the_same_as_never_tried(record):
    split = never_worked(known=("writes", "tries", "always", "nobody_ran_this"), learner=record)
    assert split["tried and never worked"] == ("tries",)
    assert split["never tried"] == ("nobody_ran_this",)
    assert set(split["worked"]) == {"writes", "always"}


def test_something_never_tried_is_invisible_without_a_registry(record):
    """Which is exactly the gap that makes a registry sound like evidence."""
    assert never_worked(learner=record)["never tried"] == ()


def test_she_can_say_it_in_a_line(record):
    said = says(known=("writes", "tries", "always", "nobody_ran_this"), learner=record)
    assert "2 worked" in said
    assert "1 tried and never worked" in said
    assert "1 never tried" in said


# ── and it never guesses ─────────────────────────────────────────────────

def test_a_record_that_cannot_be_opened_reports_nothing(tmp_path):
    class Missing:
        _db_path = str(tmp_path / "not-here.db")

    assert what_has_ever_worked(Missing()) == {}


def test_and_nothing_is_not_a_verdict(tmp_path):
    """Not being able to check is not the same as nothing having worked."""
    class Missing:
        _db_path = ""

    assert says(known=("a", "b"), learner=Missing()) != ""
    assert "no record" in says(learner=Missing())


def test_an_empty_record_says_so(tmp_path):
    assert says(learner=Learner(tmp_path)) == "there is no record of anything having been tried"


def test_nothing_that_has_never_been_tried_is_called_broken(record):
    split = never_worked(known=("nobody_ran_this",), learner=record)
    assert "nobody_ran_this" not in split["tried and never worked"]


def test_a_thing_with_no_name_is_not_a_thing(tmp_path):
    assert what_has_ever_worked(Learner(tmp_path, rows=[("", 1), ("   ", 0)])) == {}


def test_how_it_has_gone_defaults_to_never_tried():
    assert HowItHasGone("x").says() == "x: never tried"

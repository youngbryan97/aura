"""What she worked out about a thing survives the run and the restart.

Everything she learns about a world she is acting in — which part answers to
her, how it moves when she pushes it — was dying with the process, so the
fortieth run started exactly as ignorant as the first and experience was
something she had during a run rather than something she had.

It comes back discounted. Something she worked out yesterday is evidence about
today, not a fact about it.
"""

from __future__ import annotations

import pytest

from core.perception.how_it_moves import HowItMoves, shifted_and_combined
from core.perception.what_is_there import arranged
from core.perception.where_it_responds import Responsive
from core.runtime import what_she_learned
from core.runtime.what_she_learned import TRUST_CARRIED_OVER, forget, named, recall, remember


@pytest.fixture(autouse=True)
def somewhere_else(tmp_path, monkeypatch):
    monkeypatch.setattr(what_she_learned, "_KEPT_IN", tmp_path / "worlds")


def board(rows):
    return arranged([
        (0.2 + r * 0.15, 0.2 + c * 0.15, said)
        for r, row in enumerate(rows)
        for c, said in enumerate(row)
        if said
    ])


START = board([["2", "2", "4", ""], ["", "4", "", "8"], ["8", "", "", "8"], ["", "", "2", "2"]])


def a_run(moves: int = 12) -> HowItMoves:
    model = HowItMoves()
    state = START
    for index in range(moves):
        move = ("left", "up", "right", "down")[index % 4]
        after = shifted_and_combined(state, move)
        model.watched(state, move, after)
        state = after
    return model


# ── naming the thing ─────────────────────────────────────────────────────

def test_a_thing_is_named_by_whatever_identifies_it():
    assert named("Google Chrome", "play2048.co") == "google-chrome-play2048-co"


def test_two_different_things_are_not_one():
    assert named("Chrome", "play2048.co") != named("Chrome", "docs.google.com")


def test_nothing_identifying_still_gets_a_name():
    assert named("", "") == "somewhere"


# ── keeping and getting back ─────────────────────────────────────────────

def test_somewhere_she_has_never_been_holds_nothing():
    assert recall("nowhere-at-all") == {}


def test_what_she_worked_out_comes_back():
    assert remember("a-place", {"moves": a_run().as_memory()})
    assert recall("a-place")["moves"]["seen"] == 12


def test_it_comes_back_discounted_rather_than_whole():
    learned = a_run()
    remember("a-place", {"moves": learned.as_memory()})
    again = HowItMoves.from_memory(recall("a-place")["moves"], TRUST_CARRIED_OVER)
    assert 0 < again.seen < learned.seen


def test_what_she_knew_is_a_starting_point_that_still_predicts():
    remember("a-place", {"moves": a_run().as_memory()})
    again = HowItMoves.from_memory(recall("a-place")["moves"], TRUST_CARRIED_OVER)
    assert again.rule() is not None
    assert again.expect(START, "left") is not None


def test_too_little_experience_does_not_survive_the_discount():
    """A handful of moves is not something to start the next run believing."""
    remember("a-place", {"moves": a_run(moves=4).as_memory()})
    again = HowItMoves.from_memory(recall("a-place")["moves"], TRUST_CARRIED_OVER)
    assert again.rule() is None


def test_a_few_moves_that_disagree_can_overturn_it():
    """Which is the whole reason it comes back discounted."""
    remember("a-place", {"moves": a_run().as_memory()})
    again = HowItMoves.from_memory(recall("a-place")["moves"], TRUST_CARRIED_OVER)
    state = START
    for _ in range(14):
        again.watched(state, "left", state)
    assert again.rule() is None or again.rule().name != "slides and combines"


def test_where_things_happen_survives_too():
    where = Responsive()
    where.answered = {(50, 50): 8, (10, 90): 1}
    where.regardless = {(10, 90): 6}
    where.effective, where.idle, where.acts = 8, 6, 14
    remember("a-place", {"responds": where.as_memory()})
    again = Responsive.from_memory(recall("a-place")["responds"], TRUST_CARRIED_OVER)
    assert again.answered[(50, 50)] == 4
    assert again.effective == 4


def test_the_line_she_was_taking_is_kept_with_it():
    remember("a-place", {"approach": "keep the largest in the bottom-left corner"})
    assert "bottom-left" in recall("a-place")["approach"]


def test_nothing_worth_keeping_is_not_kept():
    assert not remember("a-place", {})


def test_a_thing_can_be_forgotten():
    remember("a-place", {"approach": "something"})
    assert forget("a-place")
    assert recall("a-place") == {}


def test_nonsense_on_disk_is_not_a_memory(tmp_path):
    (tmp_path / "worlds").mkdir(parents=True, exist_ok=True)
    (tmp_path / "worlds" / "a-place.json").write_text("not json")
    assert recall("a-place") == {}


def test_a_memory_of_the_wrong_shape_is_not_trusted():
    assert HowItMoves.from_memory({"tried": "nonsense", "right": 4}, 1.0).rule() is None
    assert Responsive.from_memory({"answered": ["not", "a", "map"]}, 1.0).answered == {}

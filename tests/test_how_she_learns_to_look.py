"""The search order is hers to change; the ruler is not.

Two questions look alike and are not. What to try first is a guess, and a wrong
one costs time. What to keep is a judgement, and a wrong one is a language that
got worse while every number said it got better. Only the first is learned.
"""

from __future__ import annotations

import pytest

from core.cognition import an_invented_kind as kinds
from core.cognition import how_she_learns_to_look as look
from core.cognition import one_algebra as algebra
from core.cognition.what_it_costs_to_say import _symbols


@pytest.fixture(autouse=True)
def _nowhere_near_her_state(monkeypatch, tmp_path):
    monkeypatch.setattr(look, "state_root", lambda: tmp_path)
    look.forget_what_worked()
    yield
    look.forget_what_worked()


def test_search_history_round_trips_through_its_owned_state_root(tmp_path):
    look.remember_what_worked(["here"])
    assert look._kept_at() == tmp_path / "state" / "what_worked_when_she_invented.json"
    assert look._kept_at().is_file()
    look.forget_what_worked()
    assert look.recall() == 1
    assert look.how_often_it_worked("here").won == 1
    assert look.how_often_it_worked("here").of == 1


def test_search_history_does_not_cross_runtime_roots(monkeypatch, tmp_path):
    look.remember_what_worked(["here"])
    monkeypatch.setattr(look, "state_root", lambda: tmp_path / "other-runtime")
    look.forget_what_worked()
    assert look.recall() == 0
    assert look.what_is_remembered() == ()


SHOWN = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 1, 2, 6), (4, 7, 2, 8)]


def _wanted():
    along = kinds.WHERE_FROM["one along"]
    return algebra._where_each_came_from(
        [
            (one, tuple(one[along(at, len(one)) % len(one)] for at in range(len(one))))
            for one in SHOWN
        ]
    )


def _order():
    return look.in_the_order_worth_trying(
        kinds.addressings(), algebra._agrees_with, _wanted(), shortest=_symbols
    )


def test_with_no_history_the_order_is_exactly_the_agreement_order():
    """Laplace gives every word the same figure, so this is a generalisation of
    what was there rather than a change to it."""
    every = kinds.addressings()
    wanted = _wanted()
    by_agreement = sorted(
        every,
        key=lambda name: (-algebra._agrees_with(every[name], wanted), _symbols(name), name),
    )
    assert _order() == by_agreement


def test_a_word_that_never_won_falls_behind_one_that_did():
    every = kinds.addressings()
    wanted = _wanted()
    tied = [
        name
        for name in every
        if algebra._agrees_with(every[name], wanted)
        == algebra._agrees_with(every["here"], wanted)
    ]
    if len(tied) < 2:
        pytest.skip("nothing ties on agreement in the language she has")
    look.remember_what_worked([tied[-1]])
    assert _order().index(tied[-1]) < _order().index(tied[0])


def test_only_terms_that_passed_the_gate_are_remembered():
    """A term that computed the right thing and was then refused is evidence
    about what to try and then throw away."""
    assert look.how_often_it_worked("here").won == 0
    look.remember_what_worked(["here"])
    assert look.how_often_it_worked("here").won == 1
    assert look.how_often_it_worked("here").of == 1


def test_the_estimate_needs_no_number_chosen_to_smooth_it():
    assert look.how_often_it_worked("never seen").rate == pytest.approx(1 / 2)
    look.remember_what_worked(["here"])
    assert look.how_often_it_worked("here").rate == pytest.approx(2 / 3)
    assert look.how_often_it_worked("elsewhere").rate == pytest.approx(1 / 3)


def test_widening_offers_the_best_few_first():
    names = [f"word {n}" for n in range(20)]
    rounds = list(look.widening_word_lists(names, holes=2, within=60.0))
    assert rounds[0] == names[:3]
    assert [len(one) for one in rounds] == sorted(len(one) for one in rounds)
    assert rounds[-1] == names


def test_widening_stops_when_the_time_does_rather_than_when_the_words_do():
    """A search over every word of a grown language is quadratic in hundreds
    and does not finish. Stopping on a word count would put a different number
    of mine back where the last one was."""
    names = [f"word {n}" for n in range(2000)]
    assert list(look.widening_word_lists(names, holes=2, within=0.0)) == []


def test_the_ruler_is_not_among_what_she_can_learn():
    """What is kept is decided the same way however the proposing changed, so a
    prior that learned to propose rubbish loses time and keeps nothing."""
    from core.cognition.the_ruler_she_cannot_move import what_it_costs_to_be

    before = what_it_costs_to_be(kinds.WHERE_FROM["here"], "here")
    look.remember_what_worked(["here"] * 50)
    assert what_it_costs_to_be(kinds.WHERE_FROM["here"], "here") == before

"""Starting again clears the verdict that nothing answers.

Measured live on 2026-08-26: she noticed a finished game, decided the attempt
was over, clicked New Game — and then made no move at all for the rest of the
run, because while that verdict stood she was only ever offered the ways out.
"""

from __future__ import annotations

from core.perception.where_it_responds import Responsive


def _dead() -> Responsive:
    state = Responsive()
    state.unanswered = Responsive.DEAD_AFTER
    return state


def test_a_dead_world_says_so():
    assert _dead().nothing_answers()


def test_starting_again_clears_it():
    state = _dead()
    state.began_again()
    assert not state.nothing_answers()


def test_where_things_happen_still_holds_across_a_restart():
    state = _dead()
    state.answered = {(50, 50): 6}
    state.regardless = {(50, 50): 1}
    state.effective = 6
    state.idle = 2
    state.began_again()
    assert state.answered == {(50, 50): 6}
    assert state.effective == 6
    assert state.settled()


def test_it_can_die_again_afterwards():
    state = _dead()
    state.began_again()
    state.unanswered = Responsive.DEAD_AFTER
    assert state.nothing_answers()

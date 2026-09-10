"""Thinking priced at nothing, because she had never gone without it.

What a language pass costs is measured against a move made without one. Both
halves had to be measured before either counted — so a run that thought about
its very first move could never find out that thinking was expensive. No quiet
move, so a pass costs one move, so the bar stays where a pass is cheap, so she
thinks again, and again.

LIVE 2026-09-07, playing the real game: forty-eight passes for nineteen moves,
about ten seconds each, on a game that takes hundreds of moves.

And two futures being close was itself a reason to think, which on a board
where the best two moves are nearly always close means a pass almost every
move — bought to settle a difference between two futures worth the same.
Thinking is for when the arithmetic is unreliable, not for when it is tied.
"""

from __future__ import annotations

from core.agency.worth_thinking_about import worth_a_pass
from core.skills.screen_pursuit import _a_pass_in_moves

FUTURES = {"up": (0.80, ""), "down": (0.78, ""), "left": (0.60, ""), "right": (0.40, "")}


def test_a_pass_is_dear_before_any_move_has_been_made_without_one():
    """The whole defect: she cannot learn thinking is expensive by thinking."""
    only_ever_thought = {
        "passes": 5.0, "pass_s": 50.0,
        "quiet": 0.0, "quiet_s": 0.0,
        "cycles": 5.0, "cycle_s": 55.0,
    }
    assert _a_pass_in_moves(only_ever_thought) > 5.0


def test_with_nothing_measured_it_claims_nothing():
    assert _a_pass_in_moves(
        {"passes": 0.0, "pass_s": 0.0, "quiet": 0.0, "quiet_s": 0.0, "cycles": 0.0, "cycle_s": 0.0}
    ) == 1.0


def test_a_measured_quiet_move_is_still_preferred():
    """The real comparison, once she has one."""
    both = {
        "passes": 4.0, "pass_s": 40.0,
        "quiet": 4.0, "quiet_s": 4.0,
        "cycles": 8.0, "cycle_s": 44.0,
    }
    assert _a_pass_in_moves(both) == 10.0


def test_a_cheap_pass_stays_cheap():
    """A small model answering in a fraction of a move must not read as dear."""
    cheap = {
        "passes": 5.0, "pass_s": 0.5,
        "quiet": 0.0, "quiet_s": 0.0,
        "cycles": 5.0, "cycle_s": 5.0,
    }
    assert _a_pass_in_moves(cheap) == 1.0


def test_two_futures_being_close_is_not_a_reason_to_think():
    """It is a reason not to: the difference is worth about nothing."""
    ask, why = worth_a_pass(
        FUTURES, stakes=0.6, since_words=1, horizon=5, how_sure=0.7, costs_moves=10.0
    )
    assert ask is False, why


def test_she_thinks_when_her_own_model_is_unreliable():
    """A rule she cannot trust makes scores she cannot sort by."""
    ask, why = worth_a_pass(
        FUTURES, stakes=0.6, since_words=1, horizon=5, how_sure=0.0, costs_moves=1.0
    )
    assert ask is True
    assert "gets wrong" in why


def test_a_settled_model_sorts_a_hair_apart_without_asking():
    ask, _why = worth_a_pass(
        FUTURES, stakes=0.6, since_words=1, horizon=5, how_sure=1.0, costs_moves=1.0
    )
    assert ask is False


def test_nothing_here_excuses_her_from_the_necessities():
    """A price must not buy away the reasons that are not about value."""
    seen, _ = worth_a_pass(FUTURES, stakes=0.6, unusual=True, how_sure=1.0, costs_moves=99.0)
    assert seen is True
    riding, _ = worth_a_pass(FUTURES, stakes=0.95, how_sure=1.0, costs_moves=99.0)
    assert riding is True
    blind, _ = worth_a_pass(None, stakes=0.1, how_sure=1.0, costs_moves=99.0)
    assert blind is True

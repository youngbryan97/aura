"""Whether an answer is worth asking for depends on what asking costs.

A language pass on a small model costs a fraction of a move; on a resident
model under memory pressure it costs the time of ten. The same "the best two
are close" that is worth buying at one is not worth buying at ten, and only
one half of that was ever looked at.

LIVE 2026-09-04, playing on a real board: a pass every move at about
twenty-seven seconds each, on a game that takes hundreds of moves.

The bars a price moves are the ones about VALUE. Nothing excuses her from
thinking when there is something she cannot see, when what worked before
disagrees with what she can see now, or when enough rides on it.
"""

from __future__ import annotations

from core.agency.worth_thinking_about import TOO_CLOSE_TO_CALL, worth_a_pass
from core.skills.screen_pursuit import _a_pass_in_moves

CLOSE = {"up": (1.0, ""), "down": (1.0 - TOO_CLOSE_TO_CALL / 2.0, "")}
CLEAR = {"up": (1.0, ""), "down": (1.0 - TOO_CLOSE_TO_CALL * 2.0, "")}


def test_a_close_call_is_worth_a_cheap_thought():
    asked, _why = worth_a_pass(CLOSE, costs_moves=1.0)
    assert asked is True


def test_the_same_close_call_is_not_worth_a_dear_one():
    asked, why = worth_a_pass(CLOSE, costs_moves=10.0)
    assert asked is False
    assert "10 move(s)" in why


def test_a_clear_best_is_never_worth_either():
    assert worth_a_pass(CLEAR, costs_moves=1.0)[0] is False
    assert worth_a_pass(CLEAR, costs_moves=10.0)[0] is False


def test_going_a_long_time_without_words_stretches_with_the_price():
    assert worth_a_pass(CLEAR, since_words=6, horizon=5, costs_moves=1.0)[0] is True
    assert worth_a_pass(CLEAR, since_words=6, horizon=5, costs_moves=10.0)[0] is False


def test_a_price_does_not_excuse_her_from_a_necessity():
    """Nothing to see, what worked before disagreeing, or enough riding on it."""
    assert worth_a_pass(None, costs_moves=100.0)[0] is True
    assert worth_a_pass(CLEAR, recognised="down", costs_moves=100.0)[0] is True
    assert worth_a_pass(CLEAR, stakes=0.9, costs_moves=100.0)[0] is True
    assert worth_a_pass(CLEAR, unusual=True, costs_moves=100.0)[0] is True


# ── and the price is measured, not assumed ───────────────────────────────


def test_before_anything_is_measured_a_pass_costs_one_move():
    assert _a_pass_in_moves({"pass_s": 0.0, "passes": 0.0, "quiet_s": 0.0, "quiet": 0.0}) == 1.0


def test_a_pass_that_took_ten_times_a_quiet_move_costs_ten():
    assert _a_pass_in_moves(
        {"pass_s": 60.0, "passes": 2.0, "quiet_s": 12.0, "quiet": 4.0}
    ) == 10.0


def test_a_pass_cheaper_than_a_move_still_costs_one():
    assert _a_pass_in_moves(
        {"pass_s": 1.0, "passes": 4.0, "quiet_s": 40.0, "quiet": 4.0}
    ) == 1.0


def test_the_pursuit_weighs_the_price_it_measured():
    import inspect

    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit.pursue_on_screen)
    assert "costs_moves=_a_pass_in_moves(costs)" in source
    assert 'costs["quiet_s"] +=' in source
    assert 'costs["pass_s"] +=' in source

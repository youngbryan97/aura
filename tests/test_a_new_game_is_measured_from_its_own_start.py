"""What she has been getting per act is counted within one game.

The baseline was the first board of the run. A run that begins on a finished
game and starts a new one then measured every board of the new game against
the old game's total, read steady progress as a loss, and leaned toward the
world's swing for the whole game (live, 2026-09-23: +0.03 rising to +0.33 over
175 moves, and the game ended at 256).
"""

from __future__ import annotations

from types import SimpleNamespace

from screen_pursuit_support import pursuit_loop_source

from core.agency.looking_ahead import whether_to_take_the_wide_option
from core.skills.screen_pursuit_looking import _how_it_has_been_going


def _board(*tiles: float):
    return SimpleNamespace(numbers=lambda: list(tiles))


def test_measured_from_an_old_game_a_new_one_reads_as_losing_and_gambles():
    began_at = {"worth": 1200.0, "seen": 40}
    going = _how_it_has_been_going(began_at, _board(64, 32, 16, 8, 4))
    assert going < 0
    assert whether_to_take_the_wide_option(0.6, going) > 0


def test_measured_from_its_own_start_it_reads_as_gaining_and_steadies():
    began_at = {"worth": 4.0, "seen": 40}
    going = _how_it_has_been_going(began_at, _board(64, 32, 16, 8, 4))
    assert going > 0
    assert whether_to_take_the_wide_option(0.6, going) < 0


def test_a_confirmed_restart_resets_where_she_measures_from():
    source = pursuit_loop_source()
    restart = source.index('pending["judge_on_the_next_board"] = True')
    reset = source.index('began_at["worth"], began_at["seen"] = None, 0')
    assert 0 < reset - restart < 1000

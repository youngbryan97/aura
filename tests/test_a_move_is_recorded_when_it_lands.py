"""What she did is written from what her body did, first move included.

Measured live 2026-08-26: thirty-five moves in the record, a board that had
not changed once, and no correction said out loud. The follow-ons of a
sequence were written from what landed; the first was written before the body
was asked to do anything — and a plan of one move has no follow-ons.
"""

from __future__ import annotations

from screen_pursuit_support import pursuit_function_source, pursuit_source

import re
from pathlib import Path

# The decision and the act were closures in `pursue_on_screen` and are
# their own functions now. Asked for by name rather than sliced out of a
# file between two `async def`s.
SOURCE = pursuit_source()
DECIDE = pursuit_function_source("decide_the_next_move")
ACT = pursuit_function_source("carry_out_the_move")


def test_the_first_move_is_not_written_before_it_is_made():
    decide = DECIDE
    assert "moves.append" not in decide
    assert 'about_to = {"key": key' in decide


def test_it_is_written_where_the_landings_are_counted():
    act = ACT
    where = act.index("moves.append(about_to)")
    assert act.index("sequence[:arrived]") < where


def test_a_keystroke_that_did_not_land_is_corrected_out_loud():
    act = ACT
    assert re.search(r"for step in sequence\[arrived:\]:\s*\n(\s*#[^\n]*\n)*\s*_say_it_did_not_land", act)


def test_every_landed_step_counts_as_a_step_taken():
    act = ACT
    landed = act[act.index("sequence[:arrived]") : act.index("sequence[arrived:]")]
    assert landed.count("doing.a_step_taken()") == 1
    assert landed.count("moves.append") == 2


def test_nothing_landing_records_nothing():
    """sequence[:0] is empty, so no move is written and every step is corrected."""
    act = ACT
    assert "arrived = 1 if await press(" in act
    assert "else 0" in act

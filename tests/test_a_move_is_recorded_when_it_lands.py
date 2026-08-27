"""What she did is written from what her body did, first move included.

Measured live 2026-08-26: thirty-five moves in the record, a board that had
not changed once, and no correction said out loud. The follow-ons of a
sequence were written from what landed; the first was written before the body
was asked to do anything — and a plan of one move has no follow-ons.
"""

from __future__ import annotations

import re
from pathlib import Path

SOURCE = Path("core/skills/screen_pursuit.py").read_text()
BODY = SOURCE[SOURCE.index("async def pursue_on_screen") :]


def test_the_first_move_is_not_written_before_it_is_made():
    decide = BODY[BODY.index("async def decide") : BODY.index("async def act")]
    assert "moves.append" not in decide
    assert 'about_to = {"key": key' in decide


def test_it_is_written_where_the_landings_are_counted():
    act = BODY[BODY.index("async def act") :]
    where = act.index("moves.append(about_to)")
    assert act.index("sequence[:arrived]") < where


def test_a_keystroke_that_did_not_land_is_corrected_out_loud():
    act = BODY[BODY.index("async def act") :]
    assert re.search(r"for step in sequence\[arrived:\]:\s*\n(\s*#[^\n]*\n)*\s*_say_it_did_not_land", act)


def test_every_landed_step_counts_as_a_step_taken():
    act = BODY[BODY.index("async def act") :]
    landed = act[act.index("sequence[:arrived]") : act.index("sequence[arrived:]")]
    assert landed.count("doing.a_step_taken()") == 1
    assert landed.count("moves.append") == 2


def test_nothing_landing_records_nothing():
    """sequence[:0] is empty, so no move is written and every step is corrected."""
    act = BODY[BODY.index("async def act") :]
    assert "arrived = 1 if await press(" in act
    assert "else 0" in act

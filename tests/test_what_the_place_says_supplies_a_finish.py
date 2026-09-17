"""A request that names no finish takes the one the place states.

"Play until you win" and "can you beat it?" name no thing on screen to wait
for, and the place usually says what finishing is: a board saying to get to a
tile, a download page saying when it is done. Where the person named a finish,
theirs stands, and how the two bear on each other is said.
"""
from __future__ import annotations

from core.cognition.what_the_place_says import AGREES, DISAGREES, ON_THE_WAY, SAYS_NOTHING, SUPPLIES, what_this_place_tells_her
from core.runtime.watched_goal import read_watched_goal

BOARD = "2048\nSCORE\n1268\nJoin the numbers and get to the 2048 tile!\nNew Game\n4\n8"


def test_a_place_supplies_the_finish_a_request_left_out():
    told = what_this_place_tells_her(BOARD, asked="play until you win")
    assert (told.states, told.bearing, told.aim) == ("2048", SUPPLIES, "2048")
    assert "Join the numbers" in told.said_out_loud()


def test_the_persons_finish_stands_and_the_bearing_is_said():
    assert what_this_place_tells_her(BOARD, success_when="2048").bearing == AGREES
    nearer = what_this_place_tells_her(BOARD, success_when="256")
    assert (nearer.bearing, nearer.aim) == (ON_THE_WAY, "256")
    assert what_this_place_tells_her(BOARD, success_when="4096").aim == "4096"
    assert what_this_place_tells_her(BOARD, success_when="4096").bearing == DISAGREES


def test_it_is_about_places_in_general():
    page = "Downloading update\nYour download will finish when the progress reaches 100%"
    assert what_this_place_tells_her(page).states == "100"
    assert what_this_place_tells_her("Welcome back").bearing == SAYS_NOTHING


def test_a_condition_ends_where_its_clause_does():
    """"until you win, and narrate each move" waited for the word "move"."""
    goal = read_watched_goal("Play 2048 until you win, and narrate each move.")
    assert goal is not None
    assert goal.success_when != "move"


def test_winning_is_an_end_even_when_nothing_on_screen_is_named():
    for asked in ("Play 2048 until you win, and narrate each move.", "Can you beat 2048? Tell me each move as you go."):
        goal = read_watched_goal(asked)
        assert goal is not None and goal.success_when == ""
        assert goal.max_cycles > 200, asked

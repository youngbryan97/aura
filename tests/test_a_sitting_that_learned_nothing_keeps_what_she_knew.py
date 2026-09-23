"""A sitting that learned nothing writes back what she knew, not half of it.

LIVE, found 2026-09-23: her rule for how the 2048 board moves held "slides and
combines", 90 of 92 moves right, in the memory of that kind of world, while
every count in the game's own memory was nought. What she carries comes back
at half trust and was written back at half, so each sitting halved it — and a
run of sittings that ended at their first reading ("already true after 0
moves") took it to nothing. Each new run then spent its first moves finding
out again how the board moves.
"""
from __future__ import annotations

from core.skills.screen_pursuit import _kept_as_it_was


def test_a_part_that_did_not_move_is_written_back_as_it_was_remembered():
    knew = {"moves": {"right": {"slides and combines": 90}, "seen": 92}}
    loaded_as = {"moves": {"right": {"slides and combines": 45}, "seen": 46}}
    now = {"moves": {"right": {"slides and combines": 45}, "seen": 46}, "furthest": 256}
    kept = _kept_as_it_was(knew, loaded_as, now)
    assert kept["moves"] == knew["moves"]
    assert kept["furthest"] == 256


def test_a_part_that_learned_something_is_written_as_it_now_is():
    knew = {"moves": {"right": {"slides and combines": 90}, "seen": 92}}
    loaded_as = {"moves": {"right": {"slides and combines": 45}, "seen": 46}}
    now = {"moves": {"right": {"slides and combines": 60}, "seen": 61}}
    assert _kept_as_it_was(knew, loaded_as, now)["moves"] == now["moves"]


def test_nothing_remembered_is_nothing_restored():
    loaded_as = {"world": {"arrives": {}, "acts": 0}}
    now = {"world": {"arrives": {}, "acts": 0}}
    assert _kept_as_it_was({}, loaded_as, now)["world"] == now["world"]

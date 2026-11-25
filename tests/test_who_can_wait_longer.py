"""Who can wait longer (core/cognition/who_can_wait_longer.py), in a world of lanes, not a board."""
from __future__ import annotations

import pytest

from core.cognition.where_forces_can_meet import Mover
from core.cognition.who_can_wait_longer import ENDLESS, spare_moves, who_waits_longer

pytestmark = pytest.mark.unit

ROWS, LANES = 6, 4


def lane_reach(mover, place, _standing):
    """Side 1's runners only go down their lane; side 0's free movers go either way along a lane."""
    row, lane = place
    steps = [(row - 1, lane)] if str(mover.name).startswith("runner") else [(row - 1, lane), (row + 1, lane)]
    return [there for there in steps if 0 <= there[0] < ROWS]


def test_a_runner_that_only_goes_one_way_has_its_way_forward_and_no_more() -> None:
    movers = [Mover("runner a", 1, (4, 0)), Mover("wall", 0, (1, 0))]
    # Down from row 4 to row 2, stopped by what stands on row 1.
    assert spare_moves(movers, lane_reach, 1) == 2


def test_a_mover_that_can_go_back_and_forth_waits_without_end() -> None:
    movers = [Mover("free", 0, (3, 2)), Mover("runner a", 1, (5, 0))]
    assert spare_moves(movers, lane_reach, 0) == ENDLESS


def test_the_side_with_endless_waiting_outlasts_runners_that_run_out() -> None:
    """Her gate on the bottom row holds their runners up; her free mover can wait as long as she likes."""
    movers = [Mover("gate 0", 0, (0, 0)), Mover("gate 1", 0, (0, 1)), Mover("free", 0, (4, 3)),
              Mover("runner a", 1, (2, 0)), Mover("runner b", 1, (3, 1))]
    assert spare_moves(movers, lane_reach, 1) == 1 + 2
    assert who_waits_longer(movers, lane_reach) == 0


def test_where_both_can_wait_without_end_neither_outlasts() -> None:
    movers = [Mover("free", 0, (3, 2)), Mover("other free", 1, (3, 3))]
    assert who_waits_longer(movers, lane_reach) is None


def test_a_way_through_goes_only_by_places_where_nothing_can_take_it_on_arriving() -> None:
    """Taking needs room behind what is taken: a place next to a watcher of theirs is safe only where the place beyond
    it, away from the watcher, is held, by either side. Walking on cover is going only by such places."""
    from core.cognition.where_forces_can_meet import time_to_reach_safely

    watcher = (2, 2)  # beside the walker's lane, two up

    def safe(_mover, place, standing):
        r, lane = place
        if abs(r - watcher[0]) + abs(lane - watcher[1]) != 1:
            return True
        beyond = (2 * r - watcher[0], 2 * lane - watcher[1])
        return beyond in standing

    walker = Mover("walker", 0, (0, 1))
    open_way = time_to_reach_safely(walker, lane_reach, [watcher], 6, safe)
    assert (1, 1) in open_way and (2, 1) not in open_way and (5, 1) not in open_way
    # A piece of hers on the far side of that place covers it: the walker goes through, to the end of its lane.
    covered = time_to_reach_safely(walker, lane_reach, [watcher, (2, 0)], 6, safe)
    assert covered.get((5, 1)) == 5

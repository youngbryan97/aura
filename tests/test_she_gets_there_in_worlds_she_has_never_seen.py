"""The same machinery, in worlds it was never tuned on, reaches what it was asked to reach.

`tools/measure_getting_there.py` is the full measurement. These are the
cheap end of it: goals a few hundred moves away, so the suite can afford
them, in worlds that break a player tuned on one of them. Her rule is
learned from nothing each time, what arrives is learned the same way, and
every term of her judging starts level. Her search is held at two moves
ahead, so what the machine is busy with does not decide the result.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tools.measure_getting_there import WORLDS, live_in

#: Worlds that differ from the one this stack was built on in shape, in what
#: arrives, in how far a push carries and in what her acts are called.
NEAR_GOALS = {
    "three by three": 128,
    "three by five": 256,
    "threes arrive": 384,
    "keys that say nothing": 256,
    "keys the other way": 256,
    "one step at a time": 256,
}


@pytest.mark.parametrize("name", sorted(NEAR_GOALS))
def test_she_gets_there(name):
    world = replace(WORLDS[name], goal=NEAR_GOALS[name])
    lived = live_in(world, seed=0, judging="even", depth=2, within_s=60.0)
    assert lived["rule"], f"worked out nothing in {name}"
    assert lived["got there"], f"{name}: reached {lived['furthest']:g} of {world.goal:g}"


def test_and_what_she_does_there_is_looked_ahead_not_guessed():
    world = replace(WORLDS["keys the other way"], goal=256)
    lived = live_in(world, seed=1, judging="even", depth=2, within_s=60.0)
    assert int(lived["searched"]) >= 0.8 * int(lived["moves"])


def test_nothing_in_the_harness_tells_her_about_any_world():
    """She is handed act names and a goal; the rule, the arrivals and the key bindings are hers to find."""
    import inspect

    from tools import measure_getting_there

    body = inspect.getsource(measure_getting_there.live_in)
    assert "world.acts[" not in body and "world.rule" not in body
    assert "world.arrives" not in body


@pytest.mark.parametrize("seed", range(6))
def test_a_puzzle_whose_goal_is_where_things_are_is_solved(seed):
    """Nothing combines, nothing arrives, and no number is worth reaching."""
    world = WORLDS["a sliding puzzle"]
    lived = live_in(world, seed=seed, judging="even", depth=4, within_s=60.0, most_moves=400)
    assert lived["rule"] == "one thing steps"
    assert lived["got there"], f"came {lived['nearest']:.0%} of the way in {lived['moves']} moves"

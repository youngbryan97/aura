"""Acts that take time, and what the world does during them.

A Ghosts 'n Goblins jump cannot be called off. Once it starts the knight goes
where he was going, and for about a second he cannot answer anything — so the
decision to jump is a decision to be unable to respond for a second, taken
while knowing what is coming.
"""

from __future__ import annotations

from core.cognition.what_happens_while_she_acts import (
    WhatItCostsToBeBusy,
    how_exposed_each_act_is,
)

ACTS = ["step", "jump"]


def _a_world(seconds_per_thing: float = 0.5) -> WhatItCostsToBeBusy:
    busy = WhatItCostsToBeBusy()
    busy.an_act_took("step", 0.1)
    busy.an_act_took("step", 0.1)
    busy.an_act_took("jump", 1.0)
    busy.an_act_took("jump", 1.0)
    busy.the_world_moved(10.0, times=int(10.0 / seconds_per_thing))
    return busy


def test_the_cost_is_what_the_world_does_and_not_the_time() -> None:
    """Five minutes in a world that changes hourly costs nothing; one second
    in a world that deals a card every half second costs two cards."""
    quick = _a_world(seconds_per_thing=0.5)
    assert quick.exposure("jump") == 2.0
    assert quick.exposure("step") == 0.2

    still = _a_world(seconds_per_thing=1000.0)
    assert still.exposure("jump") < 0.01
    assert not still.worth_thinking_about(ACTS)
    assert "long enough" in still.describe(ACTS)


def test_being_busy_is_taken_off_what_the_act_is_worth() -> None:
    """A thing the world does while she is busy is a thing that happens to
    her rather than one she chose."""
    busy = _a_world()
    # The jump is worth more on every other ground, and still loses.
    ranked = how_exposed_each_act_is(ACTS, busy, worth={"jump": 3.0, "step": 1.5})
    assert ranked[0][0] == "step", ranked
    # And where it is worth enough, it wins anyway.
    ranked = how_exposed_each_act_is(ACTS, busy, worth={"jump": 9.0, "step": 1.5})
    assert ranked[0][0] == "jump", ranked


def test_an_act_she_has_never_timed_costs_nothing_yet() -> None:
    busy = WhatItCostsToBeBusy()
    busy.the_world_moved(10.0, times=20)
    assert busy.exposure("jump") == 0.0
    assert busy.how_long("jump") == 0.0


def test_a_world_never_seen_to_move_by_itself_exposes_nothing() -> None:
    busy = WhatItCostsToBeBusy()
    busy.an_act_took("jump", 5.0)
    assert busy.how_fast_the_world_is == 0.0
    assert busy.exposure("jump") == 0.0
    assert not busy.worth_thinking_about(ACTS)


def test_it_says_when_the_weighing_could_not_matter() -> None:
    """A weighing that cannot change the answer should not be made."""
    busy = _a_world(seconds_per_thing=0.5)
    assert busy.worth_thinking_about(ACTS)
    assert "jump" in busy.describe(ACTS)
    assert "unanswered" in busy.describe(ACTS)


def test_with_nothing_else_to_go_on_it_ranks_by_exposure() -> None:
    busy = _a_world()
    assert [one for one, _ in how_exposed_each_act_is(ACTS, busy)] == ["step", "jump"]

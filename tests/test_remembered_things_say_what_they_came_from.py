"""Remembered things say what they were gathered under, or they are dropped.

Three places this was not true, found by reading what she had actually written
down about playing 2048 on this machine.
"""

from __future__ import annotations

import json

from core.agency.what_worked_before import WhatWorkedBefore
from core.perception.what_is_there import Arrangement, Cell
from core.perception.what_the_world_does import WhatTheWorldDoes
from core.runtime.what_she_learned import named, recall, remember


def _board(rows: list[list[int]]) -> Arrangement:
    return Arrangement(
        rows=len(rows),
        columns=len(rows[0]),
        cells=tuple(
            Cell(row=r, column=c, says=str(value), at=(c * 0.1, r * 0.1))
            for r, row in enumerate(rows)
            for c, value in enumerate(row)
            if value
        ),
    )


def test_the_store_does_not_take_a_name_a_caller_might_mean(tmp_path, monkeypatch) -> None:
    """It kept its own bookkeeping under "world", which is also what the
    pursuit calls the model of what the world does on its own. Whichever way
    that collision resolved, one of the two was lost — and both happened: one
    world file on this machine has the model where the name should be, and
    another has the name where the model should be.
    """
    from core.runtime import what_she_learned

    monkeypatch.setattr(what_she_learned, "_KEPT_IN", tmp_path)
    where = "a thing to test the store with"
    mine = {"arrives": {"2": 9}, "acts": 12}
    assert remember(where, {"world": mine, "moves": {"seen": 3}})
    got = recall(where)
    assert got["world"] == mine, "the caller's own key survives"
    assert got["moves"] == {"seen": 3}
    assert "_kept_for" not in got, "and the bookkeeping does not come back"
    kept = json.loads((tmp_path / f"{named(where)}.json").read_text())
    assert kept["_kept_for"] == named(where)
    assert kept["world"] == mine


def test_what_worked_before_is_dropped_because_its_names_mean_nothing() -> None:
    """It is looked up by the SHAPE of the situation, and a shape is a
    description of a grid."""
    skilled = WhatWorkedBefore()
    skilled.learned("4x7 filled:9/28 largest:128@left", "down", True)
    assert skilled.known
    skilled.forget_what_was_read_differently()
    assert not skilled.known
    assert skilled.recognised == 0
    assert skilled.suggests("4x4 filled:9/16 largest:128@left", ("up", "down")) == ""
    # And dropping nothing is quiet.
    skilled.forget_what_was_read_differently()


def test_what_the_world_does_is_dropped_because_it_read_the_wrong_thing() -> None:
    """One world file on this machine has her believing History, Help, File
    and Edit turn up on their own. That is a browser's menu bar, read as
    though it were the game."""
    world = WhatTheWorldDoes()
    world.watched(
        _board([[2, 0], [0, 0]]),
        _board([[2, 0], [0, 4]]),
    )
    assert world.arrives
    world.forget_what_was_read_differently()
    assert not world.arrives
    assert world.acts == 0
    assert not world.might_do(_board([[2, 0], [0, 0]]))
    world.forget_what_was_read_differently()


def test_the_pursuit_drops_the_whole_group_together() -> None:
    """All three were read through the same grid, so all three go together."""
    from core.skills import screen_pursuit

    with open(screen_pursuit.__file__, encoding="utf-8") as handle:
        text = handle.read()
    at = text.index("learned_through_a_different_reading()")
    near = text[at : at + 900]
    assert "skilled.forget_what_was_read_differently()" in near
    assert "world.forget_what_was_read_differently()" in near


def test_she_will_not_borrow_from_a_world_shaped_differently() -> None:
    """A rule that survived somewhere else is about the thing it was watched
    in. Read through a grid of another shape it is not weak evidence about
    this one, it is about something that is not here."""
    from core.skills import screen_pursuit

    with open(screen_pursuit.__file__, encoding="utf-8") as handle:
        text = handle.read()
    at = text.index("_no_more_than_a_fresh_one_is_worth(elsewhere")
    near = text[at - 900 : at]
    assert "read_through" in near
    assert "elsewhere = {}" in near

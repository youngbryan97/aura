"""What she has already started, and has not landed yet.

Three fleets on a Stellaris screen, every one reading Estimated Arrival Date.
None of them is anywhere yet, and all three are facts about the world being
planned in — because sending a fourth to a place three are arriving at is not
caution, it is doing the thing twice.
"""

from __future__ import annotations

from core.cognition.what_she_has_set_in_motion import WhatIsComing


def _started() -> WhatIsComing:
    coming = WhatIsComing()
    coming.she_started("the build", at=0.0, lands_at=100.0, brings="a binary")
    coming.she_started("the fetch", at=0.0, lands_at=10.0, brings="the data")
    return coming


def test_something_already_on_its_way_is_not_started_again() -> None:
    coming = _started()
    assert coming.already_coming("the build")
    assert not coming.already_coming("something else")


def test_what_the_world_will_be_includes_what_is_coming() -> None:
    """A gap something is on its way to fill is not a gap."""
    coming = _started()
    assert set(coming.what_it_will_be(["a config"])) == {
        "a config",
        "a binary",
        "the data",
    }


def test_overdue_is_told_apart_from_still_in_flight() -> None:
    """The difference between waiting and being stuck."""
    coming = _started()
    assert [one.what for one in coming.overdue(now=50.0)] == ["the fetch"]
    assert [one.what for one in coming.still_coming(now=50.0)] == ["the build"]
    assert "1 overdue" in coming.describe(now=50.0)


def test_landing_takes_it_off_the_list() -> None:
    coming = _started()
    coming.it_landed("the fetch")
    assert not coming.already_coming("the fetch")
    assert coming.already_coming("the build")


def test_nothing_in_flight_says_so() -> None:
    assert "nothing in flight" in WhatIsComing().describe(now=0.0)


def test_what_is_in_flight_survives_the_process() -> None:
    coming = _started()
    again = WhatIsComing.from_memory(coming.as_memory())
    assert again.already_coming("the build")
    assert [one.what for one in again.overdue(now=50.0)] == ["the fetch"]
    assert WhatIsComing.from_memory("nothing").on_the_way == []

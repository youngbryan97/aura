"""A parallel group reads one state and its races have names."""
from __future__ import annotations

import threading

import pytest

from core.state.what_they_all_read import (
    TheyDisagreed,
    a_superstep_over,
    forget_everything,
    how_the_supersteps_have_gone,
)


@pytest.fixture(autouse=True)
def _clean():
    forget_everything()
    yield
    forget_everything()


def test_a_sibling_write_is_invisible_until_the_barrier() -> None:
    state = {"depth": 1, "mood": "flat"}
    with a_superstep_over(state, "deliberation") as step:
        step.write("planner", "depth", 4)
        assert step.read("depth") == 1, "a sibling must still see the snapshot"
        assert state["depth"] == 1, "and the live state must not have moved"
    assert state["depth"] == 4, "the barrier commits it"


def test_two_phases_writing_the_same_field_differently_is_named_not_picked() -> None:
    state = {"depth": 1}
    with pytest.raises(TheyDisagreed) as caught:
        with a_superstep_over(state, "deliberation") as step:
            step.write("planner", "depth", 4)
            step.write("critic", "depth", 9)
    assert "depth" in str(caught.value)
    assert "planner=4" in str(caught.value)
    assert "critic=9" in str(caught.value)
    assert state["depth"] == 1, "a disagreement commits nothing"


def test_two_phases_agreeing_on_a_value_is_not_a_conflict() -> None:
    state = {"depth": 1}
    with a_superstep_over(state, "deliberation") as step:
        step.write("planner", "depth", 4)
        step.write("critic", "depth", 4)
    assert state["depth"] == 4


def test_lenient_mode_records_the_clash_and_settles_deterministically() -> None:
    state = {"depth": 1}
    for _ in range(5):
        state["depth"] = 1
        with a_superstep_over(state, "group", strict=False) as step:
            step.write("zeta", "depth", 9)
            step.write("alpha", "depth", 4)
        assert state["depth"] == 4, "sorted by phase name, so it never flips"
    seen = how_the_supersteps_have_gone()
    assert seen["with_conflicts"] == 5
    assert any("depth" in c for c in seen["conflicts"])


def test_it_works_on_an_object_as_well_as_a_mapping() -> None:
    class State:
        depth = 1
        mood = "flat"

    state = State()
    with a_superstep_over(state, "group", fields=("depth",)) as step:
        assert step.keys() == ("depth",)
        step.write("planner", "depth", 7)
    assert state.depth == 7


def test_real_threads_all_see_the_same_snapshot() -> None:
    state = {"n": 0}
    seen: list[int] = []
    with a_superstep_over(state, "parallel") as step:
        def run(name: str, value: int) -> None:
            seen.append(step.read("n"))
            step.write(name, f"out_{name}", value)

        threads = [threading.Thread(target=run, args=(f"p{i}", i)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert seen == [0] * 8, "every phase read the state as it was at the start"
    assert state["out_p3"] == 3


def test_an_exception_inside_the_group_commits_nothing() -> None:
    state = {"depth": 1}
    with pytest.raises(ZeroDivisionError):
        with a_superstep_over(state, "group") as step:
            step.write("planner", "depth", 4)
            raise ZeroDivisionError
    assert state["depth"] == 1


def test_the_report_names_the_groups_that_ran() -> None:
    state = {"a": 0}
    with a_superstep_over(state, "deliberation") as step:
        step.write("planner", "a", 1)
    seen = how_the_supersteps_have_gone()
    assert seen["supersteps"] == 1
    assert seen["groups"] == ["deliberation"]
    assert seen["fields_committed"] == 1
    assert seen["with_conflicts"] == 0

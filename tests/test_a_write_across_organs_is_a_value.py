"""A cross-organ write is something you can log, refuse, replay and merge.

LangGraph's nodes return `Partial[State]` and the framework applies it with a
declared reducer per key. The value is not the immutability — it is that a
write becomes a VALUE instead of an assignment that has already happened by
the time anyone could object.

Inside one organ, direct mutation is the organ doing its job and this does not
touch it. What had no representation is the write another organ has to reason
about.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.runtime.the_shape_of_one_turn import LAST_IN_THE_ORDER, declare_write_mode
from core.state.a_patch_to_the_state import (
    APatch,
    apply_a_patch,
    how_much_still_bypasses_this,
    note_a_direct_write,
    read_a_path,
)


@pytest.fixture()
def state():
    return SimpleNamespace(
        affect=SimpleNamespace(arousal=0.2, mood="calm"),
        cognition={"modifiers": {}, "depth": 1},
    )


def test_a_patch_is_a_value_before_it_is_a_change(state):
    patch = APatch(
        changes={"affect.arousal": 0.7},
        by="AffectUpdatePhase",
        because="a percept arrived",
    )
    # Readable, comparable and loggable without touching anything.
    assert patch.paths == ("affect.arousal",)
    assert patch.to_dict()["because"] == "a percept arrived"
    assert state.affect.arousal == 0.2, "reading a patch changed the state"

    said = apply_a_patch(state, patch)
    assert said["changed"]["affect.arousal"]["was"] == 0.2
    assert said["changed"]["affect.arousal"]["now"] == 0.7
    assert state.affect.arousal == 0.7


def test_it_reaches_into_mappings_as_well_as_objects(state):
    apply_a_patch(state, APatch(changes={"cognition.depth": 4}, by="Routing"))
    assert state.cognition["depth"] == 4
    assert read_a_path(state, "cognition.depth") == 4


def test_the_write_mode_is_the_same_declaration_the_plan_uses(state):
    """One rule with one place to change it, not two that can disagree."""

    apply_a_patch(state, APatch(changes={"affect.arousal": 0.7}, by="First"))
    declare_write_mode("affect.arousal", "highest")
    try:
        said = apply_a_patch(state, APatch(changes={"affect.arousal": 0.4}, by="Second"))
        assert said["changed"]["affect.arousal"]["combined_by"] == "highest"
        assert state.affect.arousal == 0.7, "a lower value won under 'highest'"

        said = apply_a_patch(state, APatch(changes={"affect.arousal": 0.9}, by="Third"))
        assert state.affect.arousal == 0.9
    finally:
        declare_write_mode("affect.arousal", LAST_IN_THE_ORDER)


def test_a_path_whose_parent_does_not_resolve_is_refused_rather_than_created(state):
    """Creating it turns a typo into a field nothing reads."""

    said = apply_a_patch(state, APatch(changes={"nowhere.at.all": 1}, by="Someone"))
    assert said["refused"] == ["nowhere.at.all"]
    assert not said["changed"]
    assert not hasattr(state, "nowhere")


def test_several_paths_apply_in_a_fixed_order(state):
    """Two runs of one patch produce one result, whatever the dict order was."""

    patch = APatch(changes={"affect.mood": "alert", "affect.arousal": 0.5}, by="One")
    first = apply_a_patch(state, patch)
    assert list(first["changed"]) == sorted(first["changed"])


def test_what_still_bypasses_this_is_counted_rather_than_claimed_finished():
    before = dict(how_much_still_bypasses_this())
    note_a_direct_write("SomePhase")
    after = how_much_still_bypasses_this()
    assert after.get("SomePhase", 0) == before.get("SomePhase", 0) + 1

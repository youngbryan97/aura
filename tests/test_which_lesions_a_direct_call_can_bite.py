"""One of ten faculties can be ablated without booting the whole mind."""
from __future__ import annotations

from core.verify.which_lesions_a_direct_call_can_bite import (
    HOW_A_LESION_IS_BOUND,
    WHAT_A_HARNESS_REACHES,
    how_the_lesions_are_reachable,
    what_a_direct_call_can_bite,
    where_each_channel_acts,
)


def test_every_declared_channel_is_lesioned_somewhere() -> None:
    """Declared and never lesioned means an arm removing it equals intact."""
    inert = how_the_lesions_are_reachable()["declared_and_inert"]
    assert inert == [], f"declared and nothing lesions them: {inert}"


def test_all_three_ways_a_lesion_is_bound_are_counted() -> None:
    """Counting only apply_channel called three live channels inert.

    register_flag_lesion binds a channel to a flag the faculty reads, and
    @lesionable binds it to a class's own lesion(). Reading one of the three
    is the same mistake as measuring a faculty by the one path you looked at.
    """
    assert set(HOW_A_LESION_IS_BOUND) == {
        "apply_channel",
        "register_flag_lesion",
        "lesionable",
    }
    said = how_the_lesions_are_reachable()
    assert said["applied_somewhere"] == said["declared"]


def test_a_direct_model_call_can_move_exactly_the_channels_the_gate_applies() -> None:
    """Which is why the matched protocol reports one arm and two refusals."""
    direct = what_a_direct_call_can_bite()
    assert "affect.circumplex_sampling" in direct
    assert "live_mind.recurrent_loops" not in direct


def test_going_through_the_engine_is_what_the_other_nine_are_worth() -> None:
    said = how_the_lesions_are_reachable()["reachable_by"]
    assert len(said["a turn through the engine"]) > len(said["a direct model call"])
    assert set(said["a direct model call"]) <= set(said["a turn through the engine"])


def test_every_channel_says_where_it_is_applied() -> None:
    for one in where_each_channel_acts():
        assert one.applied_in, one.channel
        assert one.to_dict()["reachable_by"], one.channel


def test_a_harness_nobody_declared_reaches_nothing() -> None:
    """Rather than reaching everything, which is how a gap reads as coverage."""
    for one in where_each_channel_acts():
        assert not one.reachable_by("a harness that does not exist")


def test_the_harness_table_names_real_files() -> None:
    import pathlib

    for harness, files in WHAT_A_HARNESS_REACHES.items():
        for one in files:
            assert pathlib.Path(one).exists(), f"{harness} names {one}"

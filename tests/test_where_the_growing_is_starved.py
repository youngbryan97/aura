"""The loop that widens her language, and the link that binds it."""
from __future__ import annotations

from core.cognition.where_the_growing_is_starved import (
    ENOUGH_TO_JUDGE_ON,
    THE_STEPS,
    how_the_growing_stands,
    where_it_is_starved,
)


def test_every_step_is_counted() -> None:
    counted = how_the_growing_stands()["counts"]
    for step in THE_STEPS:
        assert step in counted, step


def test_the_step_with_nothing_coming_into_it_is_named() -> None:
    said = how_the_growing_stands()
    if said["turns_over"]:
        assert said["starved_at"] == ""
    else:
        assert said["starved_at"] in THE_STEPS


def test_the_gate_needs_enough_families_to_be_able_to_say_yes() -> None:
    """Three binary observations cannot lift a posterior over the threshold.

    A gate judging on too few held-out families refuses a change that helps
    and a change that does nothing alike, so its refusals carry no
    information. The same families are what the developmental evidence gate
    weighs, which is why this is upstream of everything else.
    """
    said = how_the_growing_stands()
    assert said["enough_to_judge_on"] == ENOUGH_TO_JUDGE_ON
    assert said["the_gate_can_say_yes"] == (
        said["families_to_judge_on"] >= ENOUGH_TO_JUDGE_ON
    )


def test_the_reading_survives_a_process_that_has_recalled_nothing() -> None:
    """Zeros are true of that process, and the report must not raise."""
    said = how_the_growing_stands()
    assert isinstance(said["counts"]["heads she has written"], int)
    assert isinstance(said["families_to_judge_on"], int)


def test_where_it_is_starved_and_the_counts_agree() -> None:
    said = how_the_growing_stands()
    starved = where_it_is_starved()
    assert starved == said["starved_at"]
    if starved:
        assert said["counts"][starved] == 0

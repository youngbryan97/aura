"""One answer to "is a person waiting on this turn".

The question was answered in eleven places. Between them they knew forty-three
origin names, and exactly two — "user" and "voice" — appeared in all eleven;
twenty-one were known to a single file. Measured on 2026-08-29, that cost a
turn: the conscience's own list did not contain the names the desktop lane
emits, so a request whose words were "use that library to record this" was
held at worst-case harm 0.80 while every other layer in the same turn treated
it as user-facing.
"""

from __future__ import annotations

import pytest

from core.runtime.turn_origin import a_person_is_waiting, normalise_origin


@pytest.mark.parametrize(
    "origin",
    ["user", "voice", "desktop", "desktop_ui", "desktop-ui", "gui", "api", "ws"],
)
def test_the_names_every_layer_knew(origin: str) -> None:
    assert a_person_is_waiting(origin) is True


@pytest.mark.parametrize("origin", ["desktop_quick_user", "api_chat", "voice_bridge"])
def test_a_lane_names_its_entry_points_after_itself(origin: str) -> None:
    assert a_person_is_waiting(origin) is True


@pytest.mark.parametrize(
    "origin",
    ["autonomous_initiative_loop", "curiosity", "dream_cycle", "system", "unknown", ""],
)
def test_nothing_autonomous_qualifies(origin: str) -> None:
    assert a_person_is_waiting(origin) is False


def test_the_same_word_spelled_two_ways_is_one_word() -> None:
    """The lists carried both spellings because callers differed."""

    assert normalise_origin("desktop-ui") == normalise_origin("desktop_ui")
    assert a_person_is_waiting("native-shell") is a_person_is_waiting("native_shell")


def test_a_stated_fact_is_believed_in_both_directions() -> None:
    """A caller that knows beats a name, including when it says no."""

    assert a_person_is_waiting("dream_cycle", stated=True) is True
    assert a_person_is_waiting("user", stated=False) is False


def test_a_lanes_internal_phase_name_is_not_guessed_at() -> None:
    """Nothing about the string says it belongs to a person's turn.

    "response_generation_user" is a user turn seen from inside the lane, and
    no rule over the name establishes that without guessing. Inventing a
    suffix rule to cover it would be the same drift in a new place, so the
    turn states the fact instead.
    """

    assert a_person_is_waiting("response_generation_user") is False
    assert a_person_is_waiting("response_generation_user", stated=True) is True

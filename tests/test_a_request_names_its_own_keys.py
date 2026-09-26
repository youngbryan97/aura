"""A request that names the keys a thing is played with is played with those keys.

Every request that was not a form was played with the arrows, and a game
played with w, a, s and d listens to none of them. Her rules now learn which
way each key pushes from what it does (test_an_act_is_not_its_name.py); this
is the other half, pressing the keys she was told to.
"""

from __future__ import annotations

import pytest

from core.runtime.watched_goal import BOARD_KEYS, keys_named_in, read_watched_goal


@pytest.mark.parametrize(
    ("said", "keys"),
    [
        ("play it with w a s d until 512", ("w", "a", "s", "d")),
        ("use the keys h, j, k and l", ("h", "j", "k", "l")),
        ("press w/a/s/d to move", ("w", "a", "s", "d")),
        ("the wasd keys move the pieces", ("w", "a", "s", "d")),
    ],
)
def test_keys_a_request_names_are_read(said, keys):
    assert keys_named_in(said) == keys


@pytest.mark.parametrize(
    "said",
    [
        "play it until a 512",
        "use the arrow keys",
        "i want a 2048 tile",
        "get a b in the corner",
    ],
)
def test_words_that_are_not_keys_are_not_read_as_keys(said):
    assert keys_named_in(said) == ()


def test_a_watched_goal_carries_the_keys_it_was_given():
    assert read_watched_goal("play the game with w a s d until 512").move_keys == ("w", "a", "s", "d")


def test_and_the_arrows_where_it_names_none():
    assert read_watched_goal("play it until 128").move_keys == BOARD_KEYS


def test_a_plain_letter_is_a_key_she_may_press_and_a_chord_is_not():
    from core.skills.screen_pursuit_surface import PRESSABLE_KEYS

    assert {"w", "a", "s", "d", "h", "j", "k", "l"} <= set(PRESSABLE_KEYS)
    assert not any("cmd" in key or "+" in key for key in PRESSABLE_KEYS)

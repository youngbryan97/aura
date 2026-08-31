"""Asking her to play something, in the ways people ask.

Whether a request was a thing to keep at was decided by a list of ways of
saying so, and whether it named somewhere to act by a list of verbs. "play
2048" was on both lists. "beat 2048" was on neither, and produced no goal at
all — which is also what tells the classifier a message is not about the
desktop, so she would have answered in conversation while the game sat in
front of her.

These are checked against what is actually installed on the machine running
them, so they say nothing where the game is not there.
"""

from __future__ import annotations

import pathlib

import pytest

from core.runtime.watched_goal import (
    _a_game_here,
    _is_asking,
    _says_it_is_a_game,
    read_watched_goal,
)

A_GAME = pathlib.Path("/Applications/2048.app")
here = pytest.mark.skipif(not A_GAME.is_dir(), reason="that game is not installed here")


@here
def test_the_bundle_says_it_is_a_game() -> None:
    assert _says_it_is_a_game(A_GAME)
    assert not _says_it_is_a_game(pathlib.Path("/System/Applications/Mail.app"))


@here
@pytest.mark.parametrize(
    "said",
    [
        "play 2048",
        "beat 2048",
        "win at 2048",
        "have a go at 2048",
        "crush 2048",
        "get me a 2048 tile",
        "can you beat 2048?",
        "could you play 2048",
        "play 2048 until you get 512",
    ],
)
def test_every_way_of_asking_her_to_play_reaches_the_screen(said: str) -> None:
    goal = read_watched_goal(said)
    assert goal is not None, said
    assert goal.target_app == "2048 Game", goal.target_app


@here
@pytest.mark.parametrize(
    "said",
    [
        "what is 2048 anyway",
        "how does 2048 work",
        "is 2048 installed?",
        "have you played 2048?",
        "do you know what 2048 is",
    ],
)
def test_asking_about_it_is_not_asking_for_it_to_be_played(said: str) -> None:
    assert read_watched_goal(said) is None, said


def test_a_polite_instruction_is_not_a_question() -> None:
    """The question mark on the end of a politely worded instruction does not
    make it a question."""
    assert not _is_asking("can you play it?")
    assert not _is_asking("could you beat it")
    assert _is_asking("can it be beaten?")
    assert _is_asking("what is it")
    assert not _is_asking("have a go at it")
    assert _is_asking("have you had a go at it?")


def test_ordinary_words_are_not_applications() -> None:
    """Half of Apple's applications are named after ordinary words, so a whole
    sentence looked up against all of them opens something for every turn."""
    for said in (
        "help me find my keys",
        "check my mail",
        "take notes on this",
        "read the file /tmp/x.py",
    ):
        assert read_watched_goal(said) is None, said


@here
def test_the_name_has_to_appear_whole() -> None:
    """Matching a beginning is useful when the words are already known to be a
    name and dangerous when they are a whole sentence: "have a go at 2048" was
    answered with Google Chrome, on "go"."""
    assert _a_game_here("have a go at 2048") == "2048 Game"

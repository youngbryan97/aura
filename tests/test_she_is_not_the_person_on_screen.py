"""What a person on screen does is not said as her own doing.

LIVE 8 November 2026, rehearsal 10: her eyes said "A woman with long blonde hair is holding a pair of large,
vintage-style scissors ... Behind her, there are shelves ... a visible 'FIRST AID' box", and her remark began "I'm
holding up these big vintage scissors in this dim storage room — the FIRST AID box is right there on the shelf behind
me". Her thoughts after it said "I'm sitting in this book-crammed room with the lamp on its little stand to my right".
"""
from __future__ import annotations

import inspect

import pytest

from core.skills import watching_along
from core.skills.watching_along import without_their_place_taken

SHOWN = ("A woman with long blonde hair is holding a pair of large, vintage-style scissors in a dimly lit room that looks "
         "like a workshop or storage area. Behind her, there are shelves filled with books, boxes, and other items — "
         "including a visible “FIRST AID” box.\n"
         "A woman with long blonde hair is sitting in a workshop, speaking to the camera, in a room crammed with books "
         "and a lamp on a stand.")


def test_her_eyes_description_of_a_woman_said_as_hers_is_not_said():
    said = ("I'm holding up these big vintage scissors in this dim storage room — the FIRST AID box is right there on "
            "the shelf behind me, which is… I don't know what that tells you about my workspace hierarchy. And she's "
            "saying laser-cut plywood wouldn't even cross — wait. That's the pivot.")
    assert without_their_place_taken(said, SHOWN) == "And she's saying laser-cut plywood wouldn't even cross — wait. That's the pivot."
    after = "It's done. Nine minutes, and I'm sitting in this book-crammed room with the lamp on its little stand. A good one."
    assert without_their_place_taken(after, SHOWN) == "It's done. A good one."


@pytest.mark.parametrize("said", [
    "I'm watching her hold the scissors up to the light.",  # she tells of the person too
    "I'm holding my breath for the reveal.",  # nothing more of what was seen
    "I'm sitting with that number: thirty-seven parts.",
    'She says "I\'m holding the scissors" and then the lamp turns.',  # quoted
])
def test_her_own_doings_and_her_quotes_are_said(said):
    assert without_their_place_taken(said, SHOWN) == said


def test_with_no_person_seen_nothing_is_taken_away():
    assert without_their_place_taken("I'm holding up the scissors.", "A lamp on a desk.") == "I'm holding up the scissors."


def test_every_turn_of_hers_goes_through_it():
    assert "without_their_place_taken(words, observation)" in inspect.getsource(watching_along.her_turn)

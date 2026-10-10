"""She knows which thing on the screen is her own, and acts through it: as people playing these games do, before they
press anything.

What a game names her, what her eyes take for the player's own, and what answers to her keys and her mouse button,
each in its place. Nothing here is any one game.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.cognition.a_guide_to_a_place import TOLD, Guide, pointer_acts

pytestmark = pytest.mark.unit


def test_a_click_that_does_something_to_her_makes_the_mouse_button_one_of_her_keys():
    """LIVE 2026-10-10 a runner whose fruit jumped at a click had her take the cursor for herself, three games over."""
    jump = Guide()
    jump.take_in(TOLD, "Click to jump.")
    assert pointer_acts(jump) == "jump"
    steer = Guide()
    steer.take_in(TOLD, "Use the mouse to move Frankie back and forth.")
    assert pointer_acts(steer) == ""
    aim = Guide()
    aim.take_in(TOLD, "Use the arrow keys to move and the mouse to aim.")
    assert pointer_acts(aim) == ""


def test_the_place_names_who_she_plays_as():
    guide = Guide()
    guide.take_in(TOLD, ["Playing as: Robin", "Use the mouse to move Frankie back and forth to catch the items."])
    assert guide.names["me"][:2] == ["robin", "frankie"]


def _moves(things):
    return SimpleNamespace(things={t.number: t for t in things}, share=lambda x, y: (x / 300, y / 200))


def test_what_her_eyes_point_at_is_matched_to_the_thing_play_follows():
    from core.perception.where_i_am_on_screen import avatar_from, which_thing_is_shown

    avatar = avatar_from('{"avatar": "purple dog", "bbox_2d": [200, 400, 300, 700]}')
    assert avatar is not None and avatar.what == "purple dog"
    dog = SimpleNamespace(number=4, kind=2, x=75.0, y=110.0)
    mummy = SimpleNamespace(number=5, kind=3, x=240.0, y=110.0)
    assert which_thing_is_shown(avatar, _moves([dog, mummy])) is dog
    assert avatar_from('{"avatar": "dog", "bbox_2d": [500, 400, 300, 700]}') is None     # a box turned inside out
    assert avatar_from("I cannot tell.") is None


def test_what_her_eyes_took_for_hers_is_taken_where_nothing_answered_and_never_over_what_did():
    from core.agency.which_one_answers_to_her import WhichIsHers

    hers = WhichIsHers()
    dog = SimpleNamespace(number=4, kind=2)
    hers.seen_to_be_her(dog, 10.0, "purple dog")
    assert hers.number == 4 and hers.kind == 2
    other = WhichIsHers()
    other.number, other.kind = 7, 1
    other.seen_to_be_her(dog, 10.0, "purple dog")
    assert other.number == 7                                                  # what answered to her stands

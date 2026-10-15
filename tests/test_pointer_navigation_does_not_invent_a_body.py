"""View navigation cannot spend the interaction budget looking for a body."""
from __future__ import annotations

import pytest

from core.agency.how_a_game_works import FOLLOWS, POINTS, understood
from core.agency.the_controls_a_game_names import (
    _navigation_motion_invariant,
    the_pointer_only_clicks,
)
from core.agency.the_way_it_is_played import AS_IT_HAPPENS, ways_asked
from core.agency.what_i_can_do_here import WhatWorksHere, a_click_on
from core.cognition.a_guide_to_a_place import SCREEN, THE_GUIDE, Guide, moved_by_the_pointer


@pytest.mark.parametrize('said', [
    'Use your MOUSE to move from room to room. Click the icon to access your inventory.',
    'Use the pointer to move between pages.',
    'Use the mouse to move from one screen to another screen.',
    'Use your mouse to move to the next panel.',
    'Move between sections with the mouse.',
    'Move from place to place with the pointer.',
    'Use the cursor to move from the previous view to the next view.',
])
def test_navigation_reaches_the_normal_pointing_choice_without_a_following_body(said):
    guide = Guide(place='an invented surface')
    guide.take_in(SCREEN, said)
    token = THE_GUIDE.set(guide)
    try:
        assert moved_by_the_pointer(said) == '' and 'moves' not in guide.pointer
        assert the_pointer_only_clicks(said) and AS_IT_HAPPENS not in ways_asked(said)
        how = understood(guide)
        assert how.me.value == POINTS and how.has_a_body() is False
        assert not how.arrows_are_a_guess_worth_trying()
        here = WhatWorksHere(told=('up', 'down', 'left', 'right'))
        offered = (a_click_on('the one that stands out at 20% across, 40% down'),)
        here.looked_at(offered, said)
        here.looked_at(offered, said)
        here.asked_for_by(said, offered)
        assert offered[0] in here.available()
        assert not {'up', 'down', 'left', 'right'} & set(here.available())
    finally:
        THE_GUIDE.reset(token)
    assert _navigation_motion_invariant() == ()


@pytest.mark.parametrize('said, body', [
    ('Use your mouse to move the robot from room to room.', 'robot'),
    ('Move your ship between rooms with the mouse.', 'ship'),
    ('Use the mouse to move your basket and catch apples.', 'basket'),
    ('Move the mouse to steer.', 'you'),
])
def test_a_direct_body_object_still_reaches_continuous_control(said, body):
    guide = Guide()
    guide.take_in(SCREEN, said)
    assert moved_by_the_pointer(said) == body
    assert understood(guide).me.value == FOLLOWS
    assert not the_pointer_only_clicks(said) and AS_IT_HAPPENS in ways_asked(said)


@pytest.mark.parametrize('active', [
    'Click to shoot.', 'Aim with the mouse and click to throw.',
    'Press the arrow keys to move.', 'Use WASD to move your body.',
])
def test_a_navigation_clause_cannot_erase_separate_continuous_controls(active):
    said = 'Use the mouse to move between rooms. '+active
    assert not the_pointer_only_clicks(said) and AS_IT_HAPPENS in ways_asked(said)


@pytest.mark.parametrize('preposition', ['from', 'between'])
def test_old_cached_prepositions_cannot_return_as_a_body(preposition):
    guide = Guide.from_memory({'pointer': {'moves': preposition, 'clicks': 'from its program'}})
    assert guide.pointer == {'clicks': 'from its program'}
    assert understood(guide).me.value != FOLLOWS

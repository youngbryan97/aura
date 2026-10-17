"""A blinking measured way on survives the general scene-motion filter."""
from types import SimpleNamespace

import pytest

from core.skills.screen_pursuit_bearings import (
    _animated_navigation_invariant,
    things_to_click,
    where_to_click,
)
from core.skills.screen_pursuit_on_a_page import _what_the_stream_shows
from core.skills.screen_step_evidence import current_controls


@pytest.mark.parametrize('label', ['CONTINUE •', 'Next', 'PLAY', 'Resume', 'Try again',
                                  'BTART', 'NEX7', 'PL4Y', 'RESUMEe', 'TRY ABAIN'])
def test_the_stream_can_mark_motion_without_deleting_a_whole_navigation_label(label):
    control = dict(text=label, x=.74, y=.94, width=.14, height=.04, center_x=.81, center_y=.96)
    prose = dict(text='A figure crosses the room.', x=.3, y=.5, width=.5, height=.1)
    observation = dict(layout=[control, prose])
    watcher = SimpleNamespace(keeps_changing_on_its_own=lambda _box: True, acts=[])
    _what_the_stream_shows(observation, watcher)
    assert control['of_its_own'] and prose['of_its_own']
    move = f'click "{label}"'
    assert things_to_click(observation, True) == (move,)
    assert current_controls(observation, True) == (move,)
    assert where_to_click(observation, label) == (.81, .96)


@pytest.mark.parametrize('label', ['Continue exploring the house.', 'SCORE 120', 'Time left: 12', 'Display', 'A new scene'])
def test_a_word_inside_animated_prose_or_a_readout_cannot_supply_a_navigation_control(label):
    assert things_to_click({'layout': [dict(text=label, of_its_own=True)]}, True) == ()


def test_animation_does_not_remove_the_required_drawing_scope():
    observation = dict(layout=[dict(text='Continue', of_its_own=True)])
    assert things_to_click(observation, None) == ()
    assert _animated_navigation_invariant() == ()

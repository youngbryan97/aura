"""Current motion directions follow source capture order rather than wording novelty."""
import math

import pytest

from core.agency.a_requested_relation_in_play import requested_relation, take_motion_directions
from core.agency.playing_as_it_happens import _Run
from core.agency.reading_while_playing import (
    _current_directions_invariant,
    kept_reading,
    rules_as_told_since,
)
from core.agency.what_meeting_things_does import WhatMeetingDoes

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('words', [
    'Go to find the missing cargo.', 'Use the keys to guide the vehicle to collect coins.',
    'Use the mouse to move and click the icons on the panel.', 'Go to it.', 'Walk at somewhere.',
])
def test_a_purpose_or_an_unbound_pronoun_is_not_compiled_as_a_named_place(words):
    assert requested_relation(words) is None


def test_a_new_standalone_direction_reaches_the_existing_meeting_state():
    run, meeting = _Run(keys=[], began=0.), WhatMeetingDoes()
    take_motion_directions(meeting, 'Land on the blue platform.')
    kept_reading(run, 2., 'Land on the green platform.')
    rules_as_told_since(run, None, meeting=meeting)
    assert meeting.spatial_request.destination == 'the green platform'
    current = meeting.spatial_request
    kept_reading(run, 3., 'Score: 10')
    rules_as_told_since(run, None, meeting=meeting)
    assert meeting.spatial_request is current


def test_a_direction_can_become_current_again_without_reteaching_the_same_words():
    run, meeting = _Run(keys=[], began=0.), WhatMeetingDoes()
    for at, colour in ((1., 'blue'), (2., 'green'), (3., 'blue')):
        kept_reading(run, at, f'Land on the {colour} platform.')
        rules_as_told_since(run, None, meeting=meeting)
        assert meeting.spatial_request.destination == f'the {colour} platform'
    assert run.rules_read_at == 3.


def test_late_delivery_of_an_old_reading_cannot_replace_a_newer_direction():
    run, meeting = _Run(keys=[], began=0.), WhatMeetingDoes()
    kept_reading(run, 3., 'Land on the current platform.')
    kept_reading(run, 2., 'Land on the old platform.')
    rules_as_told_since(run, None, meeting=meeting)
    assert meeting.spatial_request.destination == 'the current platform'
    assert run.rules_read_at == 3.
    kept_reading(run, 1., 'Land on the first platform.')
    rules_as_told_since(run, None, meeting=meeting)
    assert meeting.spatial_request.destination == 'the current platform'


@pytest.mark.parametrize('at', [math.nan, math.inf, True, '4'])
def test_an_invalid_capture_time_cannot_revive_or_retire_a_current_direction(at):
    run, meeting = _Run(keys=[], began=0.), WhatMeetingDoes()
    take_motion_directions(meeting, 'Land on the current platform.')
    run.rules_read_at = 3.
    run.words_read.append((at, 'Land on the other platform.'))
    rules_as_told_since(run, None, meeting=meeting)
    assert meeting.spatial_request.destination == 'the current platform'
    assert run.rules_read_at == 3.


def test_the_current_direction_connection_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _current_directions_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'current_directions_follow_capture_order')
    assert check.predict(None) is True

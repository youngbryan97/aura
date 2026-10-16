"""An explicit relation reaches target selection without naming arbitrary scenery."""
from types import SimpleNamespace

import numpy as np
import pytest

from core.agency.a_requested_relation_in_play import (
    SpatialRequest,
    _requested_destination_invariant,
    bind_destination,
    position_for_relation,
    requested_relation,
    take_motion_directions,
)
from core.agency.choosing_where_to_be import _Choosing
from core.agency.what_meeting_things_does import AVOID, IGNORE, MEET, WhatMeetingDoes
from core.agency.which_one_answers_to_her import WhichIsHers
from core.cognition.a_guide_to_a_place import THE_GUIDE, Guide
from core.perception.what_moves_in_the_picture import Thing

pytestmark = pytest.mark.unit


def _thing(number, kind, x, y, width=20., height=10.):
    return Thing(number, x, y, width, height, np.zeros(64), (120, 120, 120),
                 np.zeros((1, 1, 3), np.uint8), 0., 5., kind=kind, moved=number == 1)


@pytest.mark.parametrize('words,relation,name', [
    ('Use the arrow keys to guide the vessel safely onto the landing platform!', 'on', 'the landing platform'),
    ('Use the aтrow keys to guide the vessel safely onto the landing platform!', 'on', 'the landing platform'),
    ('Using the direction keys to move the icon into the blue panel.', 'inside', 'the blue panel'),
    ('Walk beside the fence.', 'beside', 'the fence'),
    ('Dock at the station.', 'at', 'the station'),
    ('Go above the sign.', 'above', 'the sign'),
    ('Move to the terminal.', 'at', 'the terminal'),
    ('Drive into the garage.', 'inside', 'the garage'),
])
def test_explicit_motion_directions_compile_without_a_model(words, relation, name):
    request = requested_relation(words)
    assert request and (request.relation, request.destination) == (relation, name)


@pytest.mark.parametrize('words', [
    'Put the crate onto the platform.', 'Move the file into the folder.',
    'If you land on the platform you win.', "Don't land on the platform.",
    'Land on the red platform. Land on the blue platform.', 'Collect the coins.',
])
def test_indirect_conditional_negated_and_conflicting_requests_abstain(words):
    assert requested_relation(words) is None


def _scene(names):
    body = _thing(1, 0, 60., 60., 12., 12.)
    destination = _thing(2, 1, 140., 150., 40., 10.)
    nearby = _thing(3, 2, 65., 70.)
    moves = SimpleNamespace(shape=(200, 200), things={1: body, 2: destination, 3: nearby},
                            kinds=[SimpleNamespace(number=n) for n in range(3)], out_of_sight={})
    hers, meeting = WhichIsHers(), WhatMeetingDoes()
    hers.number, hers.kind = 1, 0
    take_motion_directions(meeting, 'Use the keys to guide the vessel gently onto the platform.')
    guide = Guide()
    for kind, name in names.items():
        guide.seen.saw(kind, name)
    return body, destination, nearby, moves, hers, meeting, guide


def test_normal_chooser_uses_the_named_support_instead_of_nearby_unknown_material():
    body, target, nearby, moves, hers, meeting, guide = _scene({1: 'platform', 2: 'ground fragment'})
    token = THE_GUIDE.set(guide)
    try:
        choosing = _Choosing(moves, hers, meeting, [])
        choosing.at = body.seen
        point, why, aim = choosing.target()
        assert (point, why, aim) == ((140., 139.), 'meet', target)
        assert choosing.stance(nearby) == IGNORE
        assert choosing.stance(target) == MEET
        assert choosing.last_target[2] is target
        assert meeting.told_gently and meeting.spatial_request is not None
    finally:
        THE_GUIDE.reset(token)


def test_the_actual_input_choice_keeps_the_bound_goal_and_ignores_unearned_contacts():
    body, target, nearby, moves, hers, meeting, guide = _scene({1: 'platform', 2: 'ground fragment'})
    # These are supplied direct-motion measurements. This checks the planner
    # connection, not control discovery or a visual classification result.
    hers.way_of = {'a': (-60., 0.), 'b': (60., 0.), 'c': (0., 60.)}.get
    token = THE_GUIDE.set(guide)
    try:
        choosing = _Choosing(moves, hers, meeting, ['a', 'b', 'c'])
        choosing.at = body.seen
        key, why, aim = choosing.key('')
        assert key in ('b', 'c') and why == 'meet' and aim is target
        assert choosing.last_target == ((140., 139.), 'meet', target)
        assert choosing.stance(nearby) == IGNORE
    finally:
        THE_GUIDE.reset(token)


def test_an_explicit_named_destination_is_not_removed_for_sharing_the_controlled_bodys_visual_kind():
    body, target, _, moves, hers, meeting, guide = _scene({0: 'platform'})
    target.kind = body.kind
    token = THE_GUIDE.set(guide)
    try:
        choosing = _Choosing(moves, hers, meeting, [])
        choosing.at = body.seen
        assert target not in choosing.others()  # Ordinary contact filtering is unchanged.
        assert choosing.target() == ((140., 139.), 'meet', target)
        assert choosing.goal_binding.thing is target
    finally:
        THE_GUIDE.reset(token)


def test_a_filtered_named_rival_cannot_turn_an_ambiguous_request_into_a_unique_destination():
    body, target, nearby, moves, hers, meeting, guide = _scene({1: 'platform', 2: 'platform'})
    meeting.writing.add(nearby.number)
    token = THE_GUIDE.set(guide)
    try:
        choosing = _Choosing(moves, hers, meeting, [])
        choosing.at = body.seen
        assert target in choosing.others() and nearby not in choosing.others()
        assert choosing.target() == ((body.x, body.y), 'wait', None)
        assert choosing.goal_binding is None
    finally:
        THE_GUIDE.reset(token)


@pytest.mark.parametrize('change', ['unnamed', 'ambiguous', 'stale', 'bad_box', 'too_small', 'danger'])
def test_failed_binding_cannot_fall_back_to_an_arbitrary_contact_target(change):
    body, target, nearby, moves, hers, meeting, guide = _scene({1: 'platform'})
    if change == 'unnamed':
        guide.seen.by_kind.clear()
        # A name allocated for narration is not an observation.
        guide.names['goal'] = ['platform']
        guide.name_for('goal', 'grey')
    elif change == 'ambiguous':
        guide.seen.saw(2, 'platform')
    elif change == 'stale':
        target.seen -= .1
    elif change == 'bad_box':
        target.w = float('nan')
    elif change == 'too_small':
        target.w = body.w / 2
    else:
        meeting.told[target.kind] = AVOID
    token = THE_GUIDE.set(guide)
    try:
        choosing = _Choosing(moves, hers, meeting, [])
        choosing.at = body.seen
        assert choosing.target() == ((body.x, body.y), 'wait', None)
        assert choosing.stance(nearby) == IGNORE
        assert not meeting.known(nearby.kind)
    finally:
        THE_GUIDE.reset(token)


def test_a_descriptive_modifier_cannot_be_discarded_to_match_a_different_named_target():
    body, target, _, _, _, _, _ = _scene({})
    request = SpatialRequest('at', 'red platform', 'go at the red platform')
    assert bind_destination(request, body, [target], {1: 'blue platform'}, 5.) is None
    assert bind_destination(request, body, [target], {1: 'platform'}, 5.) is None
    assert bind_destination(request, body, [target], {1: 'small red platform'}, 5.) is not None


@pytest.mark.parametrize('relation,point', [
    ('on', (140., 139.)), ('above', (140., 133.)), ('below', (140., 167.)),
    ('beside', (114., 150.)), ('at', (140., 150.)),
])
def test_relations_compute_body_centre_destinations_instead_of_reusing_object_centres(relation, point):
    body, target, _, _, _, _, _ = _scene({})
    assert position_for_relation(relation, body, target) == point


def test_containment_requires_the_body_to_fit_and_claims_are_registered():
    body, target, _, _, _, _, _ = _scene({})
    assert position_for_relation('inside', body, target) is None
    target.h = 20.
    assert position_for_relation('inside', body, target) == (140., 150.)
    assert _requested_destination_invariant() == ()
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'requested_destinations_require_current_unique_referents')
    assert check.predict(None) is True


@pytest.mark.parametrize('at', [None, True, '5', float('nan'), float('inf')])
def test_binding_refuses_unmeasured_capture_times(at):
    body, target, _, _, _, meeting, _ = _scene({})
    assert bind_destination(meeting.spatial_request, body, [target], {1: 'platform'}, at) is None

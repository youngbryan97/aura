"""Semantic answers retain measured instance ownership; the provider is stood in for."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from core.perception.names_for_observed_instances import (
    InstanceNames, _instance_name_invariant, query_for,
)
from core.perception.what_moves_in_the_picture import Thing
from core.perception.what_things_look_like import LookingAtThings, sighting_from

pytestmark = pytest.mark.unit


def _thing(number=1, kind=0, born=1., at=4., x=40., y=40., w=16., h=16.):
    return Thing(number, x, y, w, h, np.zeros(64), (120, 120, 120),
                 np.zeros((1, 1, 3), np.uint8), born, at, kind=kind, moved=number == 1)


def _scene(*things):
    return SimpleNamespace(things={t.number: t for t in things}, shape=(200, 300), scale=1.,
                           kinds=[SimpleNamespace(number=0, colour=(120, 120, 120), size=256.)],
                           out_of_sight={}, share=lambda x, y: (x/300, y/200))


def _query(thing, observer):
    query = query_for(thing, observer, thing.seen)
    assert query is not None
    return query.with_picture(np.zeros((200, 300, 3), np.uint8))


def test_only_the_queried_instance_receives_a_name_even_when_another_shares_its_kind():
    a, b = _thing(), _thing(2, x=100.)
    scene, names = _scene(a, b), InstanceNames()
    names.begin(scene)
    query = _query(a, scene)
    assert names.record_for(scene, query, 'vessel')
    packet = names.scene(scene, {0: 'vessel'})
    assert packet.for_thing(a) == 'vessel' and packet.for_thing(b) == ''
    assert names.consensus(scene, 0) is None
    assert names.record_for(scene, _query(b, scene), 'platform')
    assert names.scene(scene, {}).for_thing(b) == 'platform'
    assert names.consensus(scene, 0) is None
    assert not hasattr(a, 'controlled') and a.kind == b.kind == 0


def test_capture_bytes_are_owned_and_provider_json_cannot_supply_a_query():
    a = _thing()
    scene = _scene(a)
    picture = np.zeros((200, 300, 3), np.uint8)
    query = query_for(a, scene, 4.)
    captured = query.with_picture(picture)
    picture[:] = 255
    assert query.picture_sha256 == ''
    assert captured.picture_sha256 != query.with_picture(picture).picture_sha256
    answer = sighting_from('{"is":"platform","query":{"number":1,"kind":0,"picture_sha256":"fake"}}')
    assert answer.what == 'platform' and answer.query is None
    with pytest.raises(ValueError):
        query.with_picture(picture.astype(float))


@pytest.mark.parametrize('change', [
    {'observer': 0}, {'epoch': 'bad'}, {'number': True}, {'kind': -1}, {'born': -1.},
    {'at': float('nan')}, {'at': 0.}, {'box': (0., 0., 0., 1.)},
    {'box': (0., 0., float('inf'), 1.)}, {'picture_sha256': 'bad'},
])
def test_malformed_capture_queries_are_refused(change):
    a = _thing()
    with pytest.raises(ValueError):
        replace(_query(a, _scene(a)), **change)


@pytest.mark.parametrize('field,value', [('seen', 3.), ('seen', True), ('seen', float('nan')),
                                        ('born', 6.), ('number', True), ('kind', -1)])
def test_a_query_requires_a_current_finite_track(field, value):
    a = _thing()
    setattr(a, field, value)
    assert query_for(a, _scene(a), 4.) is None


@pytest.mark.parametrize('change', ['birth', 'kind', 'removed', 'clock', 'epoch', 'observer'])
def test_late_answers_cannot_attach_to_a_replacement_track_or_observer(change):
    a = _thing()
    scene, names = _scene(a), InstanceNames()
    names.begin(scene)
    query = _query(a, scene)
    if change == 'birth':
        scene.things[1] = _thing(born=4.)
    elif change == 'kind':
        a.kind = 1
    elif change == 'removed':
        scene.things.clear()
    elif change == 'clock':
        a.seen = 3.
    elif change == 'epoch':
        scene._instance_name_epoch = 'a'*32
        names.begin(scene)
    else:
        scene = _scene(_thing())
        names.begin(scene)
    assert not names.record_for(scene, query, 'platform')
    assert names.scene(scene, {0: 'platform'}).for_thing(a) == ''


def test_current_correspondence_can_retain_a_delayed_name_but_not_overwrite_a_newer_one():
    a = _thing()
    scene, names = _scene(a), InstanceNames()
    names.begin(scene)
    earlier = _query(a, scene)
    a.seen, a.x = 8., 120.
    assert names.record_for(scene, earlier, 'vessel')
    assert names.of(scene, a).what == 'vessel'
    assert names.record_for(scene, _query(a, scene), 'vehicle')
    assert not names.record_for(scene, earlier, 'platform')
    assert names.of(scene, a).what == 'vehicle'
    assert not names.record_for(scene, _query(a, scene), 'platform')
    assert names.of(scene, a) is None  # The same capture cannot settle conflicting names.


def test_instance_storage_is_bounded_and_a_retired_observer_cannot_lend_its_labels():
    names, scenes = InstanceNames(), []
    for _ in range(17):
        a = _thing()
        scene = _scene(a)
        scenes.append(scene)
        names.begin(scene)
        assert names.record_for(scene, _query(a, scene), 'vessel')
    assert len(names.observers) == len(names.records) == 16
    assert names.active(scenes[0]) and names.of(scenes[0], scenes[0].things[1]) is None
    assert names.scene(scenes[0], {0: 'another observers name'}).for_thing(scenes[0].things[1]) == ''
    scene = scenes[-1]
    for number in range(300):
        a = _thing(number)
        scene.things = {number: a}
        assert names.record_for(scene, _query(a, scene), 'vessel')
    assert len(names.records) == 256


def test_peer_consensus_is_revoked_when_an_unknown_or_different_instance_arrives():
    from core.cognition.what_things_are import WhatSheSees

    a = _thing()
    scene, seen = _scene(a), WhatSheSees()
    seen.instances.begin(scene)
    assert seen.saw_instance(scene, _query(a, scene), 'vessel') == ''
    assert seen.refresh_instances(scene) == set() and seen.as_seen(0) == 'vessel'
    b = _thing(2)
    scene.things[2] = b
    assert seen.refresh_instances(scene) == {0} and seen.as_seen(0) == ''
    seen.saw_instance(scene, _query(b, scene), 'vessel')
    assert seen.refresh_instances(scene) == set() and seen.as_seen(0) == 'vessel'
    b.seen = 5.
    seen.saw_instance(scene, _query(b, scene), 'platform')
    assert seen.refresh_instances(scene) == {0} and seen.as_seen(0) == ''
    assert seen.instances.of(scene, a).what == 'vessel'


def test_a_revised_class_guess_cannot_keep_a_previous_proper_name_or_unusual_feature():
    from core.cognition.what_things_are import WhatSheSees

    seen = WhatSheSees()
    seen.saw(0, 'machine', 'has eyes', 'Example')
    assert seen.as_seen(0) == 'Example'
    seen.saw(0, 'platform')
    assert seen.as_seen(0) == 'platform' and seen.odd == {} and seen.who == {}


@pytest.mark.asyncio
async def test_normal_query_to_guide_to_chooser_keeps_same_kind_destinations_separate(tmp_path, monkeypatch):
    from core.agency import naming_what_she_sees as naming
    from core.agency.a_requested_relation_in_play import observed_names, take_motion_directions
    from core.agency.choosing_where_to_be import _Choosing
    from core.agency.what_meeting_things_does import AVOID, WhatMeetingDoes
    from core.agency.which_one_answers_to_her import WhichIsHers
    from core.cognition.a_guide_to_a_place import THE_GUIDE, Guide
    import core.runtime.what_she_learned as learned

    monkeypatch.setattr(learned, '_KEPT_IN', tmp_path)
    # Control identity is supplied here. This test exercises semantic routing,
    # not the independent avatar model or causal control discovery.
    monkeypatch.setattr(naming, '_who_i_am_to_my_eyes', lambda *args: None)
    body, target, other = _thing(), _thing(2, x=140., y=150., w=40., h=10.), _thing(3, x=220.)
    scene, guide, meeting = _scene(body, target, other), Guide(), WhatMeetingDoes()
    take_motion_directions(meeting, 'Use the keys to guide the vessel gently onto the platform.')
    assert meeting.spatial_request is not None
    hers = WhichIsHers()
    hers.number, hers.kind = 1, 0
    meeting.supposed[0] = AVOID
    # Input/collision measurements remain outside semantic revocation.
    meeting.told[4] = AVOID
    evidence = meeting.evidence[0]
    evidence.touch_sum, evidence.touches_settled = 2., 2
    answers, encoded = iter(('vessel', 'platform', 'plaque')), []

    async def see(_prompt, image):
        encoded.append(image)
        return '{"is":"'+next(answers)+'"}'

    looking = LookingAtThings(see=see, _began=0.)
    run = SimpleNamespace(seeing=naming.SeeingInPlay(looking=looking))
    picture = np.zeros((200, 300, 3), np.uint8)
    picture[32:48, 32:48] = 180
    picture[145:155, 120:160] = 150
    token = THE_GUIDE.set(guide)
    try:
        naming.seen_in_play(run, scene, hers, meeting, picture, 4., lambda *args: None)
        await looking._asking
        for thing in scene.things.values():
            thing.seen = 6.
        naming.seen_in_play(run, scene, hers, meeting, picture, 6., lambda *args: None)
        packet = observed_names(scene)
        assert packet.scoped and [packet.for_thing(t) for t in (body, target, other)] == ['vessel', 'platform', 'plaque']
        assert len(set(encoded)) == 3
        assert guide.seen.as_seen(0) == '' and guide.reading.looks == {}
        assert naming.describe(scene, 0, target) == 'grey platform'
        assert 0 not in meeting.supposed and meeting.told[4] == AVOID
        assert meeting.evidence[0] is evidence and evidence.touches_settled == 2
        choosing = _Choosing(scene, hers, meeting, [])
        choosing.at = 6.
        point, why, chosen = choosing.target()
        assert chosen is target and why == 'meet' and point == (140., 137.)
        assert hers.number == 1 and hers.kind == 0
        assert not naming.still_looking_at(run, 0, 6., mine=1)
        assert guide.as_memory().get('seen') is None
    finally:
        THE_GUIDE.reset(token)


def test_the_claim_executes_its_instance_custody_canary():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _instance_name_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(test for test in suite.tests() if test.name == 'visual_names_keep_the_queried_instance')
    assert check.predict(None) is True

"""Missing current motion cannot erase earlier measurements of the same press."""
from types import SimpleNamespace

import pytest

from core.agency.facing_seen_under_its_input import observe_facing
from core.agency.which_one_answers_to_her import WhichIsHers
from core.agency.which_way_she_faces import WhichWaySheFaces

pytestmark = pytest.mark.unit


def _observed_then_missing(faces, *, began=0., key='opaque'):
    faces.saw(80., 0., key, began, began=began)
    faces.saw(80., 0., key, began+.25, began=began)
    faces.saw(80., 0., key, began+.4, began=began)
    faces.saw(0., 0., key, began+.55, began=began, measured=False)


def test_a_later_unmeasured_picture_keeps_one_past_motion_measurement():
    faces = WhichWaySheFaces()
    _observed_then_missing(faces)
    assert not faces.going and not faces.turns_measured
    assert not faces.along and faces.ahead_key(least=1) is None
    faces.saw(0., 0., '', .7, began=.7, measured=False)
    assert faces.along['opaque'] == [80.]
    assert faces.ahead_key(least=1) == 'opaque' and faces.ahead_key() is None
    assert not faces.turns


def test_disjoint_single_samples_cannot_be_combined_into_a_measured_segment():
    faces = WhichWaySheFaces()
    for at, measured in ((0., True), (.25, True), (.3, False), (.4, True), (.45, False)):
        faces.saw(80., 0., 'opaque', at, began=0., measured=measured)
    faces.saw(0., 0., '', .6, began=.6)
    assert not faces.along and not faces.ways


def test_many_segments_still_supply_only_one_sample_of_one_physical_press():
    faces = WhichWaySheFaces()
    _observed_then_missing(faces)
    for at in (.6, .7, .8):
        faces.saw(80., 0., 'opaque', at, began=0.)
    faces.saw(0., 0., 'opaque', .9, began=0., measured=False)
    faces.saw(0., 0., '', 1., began=1.)
    assert faces.along['opaque'] == [80.] and len(faces.ways['opaque']) == 1
    _observed_then_missing(faces, began=1.1)
    faces.saw(0., 0., '', 1.8, began=1.8)
    assert faces.along['opaque'] == [80., 80.] and faces.ahead_key() == 'opaque'


def test_a_new_input_receipt_cannot_borrow_the_prior_segment():
    faces = WhichWaySheFaces()
    _observed_then_missing(faces)
    faces.saw(0., 0., 'opaque', .6, began=.6, measured=False)
    faces.saw(0., 0., '', .8, began=.8, measured=False)
    assert faces.along['opaque'] == [80.]


def test_changed_body_birth_cannot_borrow_a_retained_motion_segment():
    hers, faces = WhichIsHers(), WhichWaySheFaces()
    hers.number, hers.kind = 1, 1
    hers.holding('opaque', 0.)
    body = SimpleNamespace(number=1, born=0., seen=0., vx=80., vy=0., motion_evidence='direct')
    for at in (0., .25, .4):
        body.seen = at
        observe_facing(faces, body, hers, at)
    body.seen, body.motion_evidence = .55, 'unavailable'
    body.motion_measured_at = .4
    observe_facing(faces, body, hers, .55)
    assert faces.observed_going is not None
    body.born, body.seen = .6, .65
    observe_facing(faces, body, hers, .65)
    hers.holding('', .7)
    body.seen = .75
    observe_facing(faces, body, hers, .75)
    assert not faces.along and faces.observed_going is None


def test_retained_travel_has_a_registered_measured_claim():
    from core.agency.which_way_she_faces import _past_travel_invariant
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _past_travel_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'observed_travel_is_not_erased_by_a_later_missing_picture')
    assert check.predict(None) is True and any(claim.test == check.name for claim in suite.claims())

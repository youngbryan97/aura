"""Each refusal names its missing evidence without promoting a partial label."""
from types import SimpleNamespace

import pytest

from core.agency.a_requested_relation_in_play import SpatialRequest, read_destination
from core.agency.evidence_for_a_requested_relation import (
    _referent_evidence_invariant, relation_region_in_its_scene,
)

pytestmark = pytest.mark.unit


def body(number, *, width=20., seen=5., kind=None):
    return SimpleNamespace(number=number, born=0., kind=number if kind is None else kind,
                           x=50., y=60., w=width, h=10., seen=seen)


@pytest.mark.parametrize('status,operation', [
    ('no_named_referent', 'identify_whole_referent'),
    ('incomplete_name', 'resolve_missing_properties'),
    ('ambiguous_named_referent', 'distinguish_referents'),
    ('referent_not_current', 'reacquire_referent'),
    ('relation_geometry_unavailable', 'measure_whole_extent'),
])
def test_the_real_binder_names_the_specific_recovery_without_authorizing_a_candidate(status, operation):
    mine, target = body(1), body(2, width=40.)
    things, names = [target], {2: 'blue platform'}
    if status == 'no_named_referent':
        names = {2: 'blue base'}
    elif status == 'incomplete_name':
        names = {2: 'platform'}
    elif status == 'ambiguous_named_referent':
        things.append(body(3, kind=2))
    elif status == 'referent_not_current':
        target.seen = 4.
    else:
        target.w = 5.
    reading = read_destination(SpatialRequest('on', 'blue platform', ''), mine, things, names, 5.)
    assert reading.binding is None and reading.status == status
    assert reading.need.reason == status and reading.need.operation == operation
    assert reading.need.at == 5. and reading.need.candidates[0].owner == (2, 0., 2)
    if status == 'incomplete_name':
        assert reading.need.missing_words == ('blue',)
    if status == 'no_named_referent':
        assert reading.need.candidates[0].matched_words == ('blue',)
        assert reading.need.missing_words == ('platform',)


def test_geometric_contact_interval_reaches_binding_with_physical_acceptance_still_unknown():
    mine, target = body(1, width=10.), body(2, width=40.)
    reading = read_destination(SpatialRequest('on', 'platform', ''), mine, [target], {2: 'platform'}, 5.)
    region = reading.binding.region
    assert region.body_owner == (1, 0., 1) and region.target_owner == (2, 0., 2)
    assert region.bounds == (35., 50., 65., 50.)
    assert region.contact_verified is None and region.maximum_speed is None and region.maximum_angle is None
    assert region.basis is None and 'coordinate_basis' in region.unknowns
    moves = SimpleNamespace(shape=(100, 200), scale=1.)
    current = relation_region_in_its_scene(region, moves, 5.)
    assert current.basis[:4] == (id(moves), None, (100, 200), 1.)
    assert 'observed_contact' in current.unknowns and 'coordinate_basis' not in current.unknowns


def test_truncation_cannot_turn_many_ambiguous_candidates_into_one():
    rivals = [body(n, kind=2) for n in range(2, 82)]
    reading = read_destination(SpatialRequest('on', 'platform', ''), body(1), rivals, {2: 'platform'}, 5.)
    assert reading.binding is None and len(reading.need.candidates) == 32
    assert reading.need.candidate_count == 80


@pytest.mark.parametrize('at', [None, True, float('nan'), float('inf'), -1., 10**400])
def test_unavailable_capture_time_never_supplies_an_evidence_timestamp(at):
    reading = read_destination(SpatialRequest('on', 'platform', ''), body(1), [body(2)], {2: 'platform'}, at)
    assert reading.binding is None and reading.need.at is None


def test_evidence_contract_has_an_executable_measurement():
    assert _referent_evidence_invariant() == ()
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'referent_candidates_cannot_certify_contact')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())

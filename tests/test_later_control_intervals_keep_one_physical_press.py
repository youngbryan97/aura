"""A later clear interval can measure acceleration without bridging missing pictures."""
from types import SimpleNamespace

import pytest

from core.agency.which_one_answers_to_her import WhichIsHers, _Speeds

pytestmark = pytest.mark.unit


def _prefix(record, key='q', began=1.):
    for at in (began+.25, began+.3):
        record.add(key, 3., 0., press=began, when=at)
    record.unseen(key, began)


def _later(record, key='q', began=1., push=40.):
    for dt in (.7, .8, .9, 1.):
        record.add(key, push*dt, 0., press=began, when=began+dt)


def test_a_later_clear_interval_supplies_one_acceleration_estimate():
    record = _Speeds()
    _prefix(record)
    before = list(record.timed[('q', 1.)])
    _later(record)
    push, = record.pushes()['q']
    assert push == pytest.approx((40., 0.))
    assert record.push('q') is None, 'one held input became independent repeated presses'
    assert record.by_press[('q', 1.)] == [(3., 0.), (3., 0.)]
    assert record.timed[('q', 1.)] == before


def test_repeated_missing_intervals_cannot_be_added_to_make_four_samples():
    record = _Speeds()
    _prefix(record)
    for times in ((1.7, 1.8), (2.1, 2.2), (2.5, 2.6)):
        for at in times:
            record.add('q', 40.*at, 0., press=1., when=at)
        record.unseen('q', 1.)
    assert not record.pushes()


def test_two_physical_presses_can_establish_a_repeated_acceleration():
    record = _Speeds()
    for began in (1., 4.):
        _prefix(record, began=began)
        _later(record, began=began)
        record.end('q', began)
    assert record.push('q') == pytest.approx((40., 0.))
    assert len(record.pushes()['q']) == 2
    assert set(record.by_press) == {('q', 1.), ('q', 4.)}


def test_duplicate_times_cannot_reenter_after_a_later_gap():
    record = _Speeds()
    _prefix(record)
    for at in (2., 2.1, 2.2):
        record.add('q', at*40., 0., press=1., when=at)
    record.unseen('q', 1.)
    for at in (2., 2.1, 2.2, 2.3):
        record.add('q', at*40., 0., press=1., when=at)
    assert not record.pushes()
    assert len(record.resumed_samples[('q', 1.)]) == 1


@pytest.mark.parametrize('after', ['gap', 'release', 'contradiction'])
def test_the_interval_keeps_the_actual_input_end_and_contradiction(after):
    record = _Speeds()
    _prefix(record)
    _later(record)
    if after == 'release':
        record.end('q', 1.)
    else:
        record.unseen('q', 1., contradicted=after == 'contradiction')
    for at in (2.5, 2.6, 2.7):
        record.add('q', -900.*at, 0., press=1., when=at)
    if after == 'contradiction':
        assert not record.pushes() and ('q', 1.) in record.interrupted
    else:
        push, = record.pushes()['q']
        assert push == pytest.approx((40., 0.))


@pytest.mark.parametrize('damage', ['duplicate', 'backward', 'nonfinite', 'untimed', 'unsettled', 'pinned'])
def test_unqualified_later_samples_cannot_complete_an_acceleration(damage):
    record = _Speeds()
    _prefix(record)
    for at in (1.7, 1.8, 1.9):
        record.add('q', 40.*at, 0., press=1., when=at)
    when = {'duplicate': 1.9, 'backward': 1.85, 'nonfinite': float('nan'), 'untimed': None}.get(damage, 2.)
    record.add('q', 80., 0., press=1., when=when, settled=damage != 'unsettled', pinned=damage == 'pinned')
    assert not record.pushes()


def test_the_current_owned_trial_can_finish_from_a_later_continuous_interval():
    from core.agency.calibration_for_a_current_body import calibration_owner
    from core.agency.enough_observations_for_a_trial import trial_has_its_observations
    from core.agency.giving_a_trial_time import ControlTrial

    hers = WhichIsHers()
    hers.number, hers.kind, hers.carried = 1, 0, True
    mine = SimpleNamespace(number=1, born=0.)
    moves = SimpleNamespace(things={1: mine})
    owner = calibration_owner(hers, mine)
    trial = ControlTrial('q', 1., 2.8, hers.identification.epoch, owner, False, 1.5)
    _prefix(hers._hers)
    assert not trial_has_its_observations(trial, hers, moves, 1.6)
    _later(hers._hers)
    assert trial_has_its_observations(trial, hers, moves, 2.1)
    assert trial.until == 2.8 and hers._hers.push('q') is None


def test_old_intervals_and_sample_storage_remain_bounded():
    record = _Speeds()
    for began in range(100):
        _prefix(record, began=float(began))
        _later(record, began=float(began))
    assert len(record.resumed_pushes) <= 80 and len(record.resumed_samples) <= 80
    assert len(record.resumed_through) <= 80
    for at in (102.+i*.01 for i in range(400)):
        record.add('q', 40.*at, 0., press=99., when=at)
    assert max(map(len, record.resumed_samples.values())) <= 128
    assert max(map(len, record.resumed_pushes.values())) <= 128


def test_ordinary_control_identification_uses_repeated_later_intervals():
    import numpy as np

    from core.perception.what_moves_in_the_picture import Thing, WhatMoves

    hers, moves = WhichIsHers(), WhatMoves()
    moves.shape = (300, 300)
    body = Thing(7, 150., 150., 10., 10., np.ones(3), (100, 150, 200),
                 np.zeros((10, 10, 3), np.uint8), 0., 1., kind=0, moved=True)
    moves.things[7] = body
    for key, began, push in [('q', 1., 100.), ('z', 3., -100.), ('q', 5., 100.), ('z', 7., -100.)]:
        hers.holding(key, began, trying=True)
        for after in (.25, .3, .4, .7, .8, .9, 1.):
            body.seen = began+after
            body.motion_evidence = 'unavailable' if after == .4 else 'measured'
            body.motion_measured_at = None if after == .4 else body.seen
            body.vx = 3. if after < .4 else push*after
            body.x += body.vx*.05
            body.path.append((body.seen, body.x, body.y))
            hers.saw(moves, [], body.seen)
        hers.holding('', began+1.1, trying=False)
    assert hers.number == 7 and hers.carried
    assert hers.push_of('q') == pytest.approx((100., 0.))
    assert hers.push_of('z') == pytest.approx((-100., 0.))
    assert len(hers._by_thing[7].pushes()['q']) == 2
    assert len(hers._by_thing[7].pushes()['z']) == 2
    for press in hers._hers.resumed_pushes:
        assert hers._hers.resumed_pushes is not hers._by_thing[7].resumed_pushes
        assert isinstance(hers._hers.resumed_pushes[press], tuple)


def test_later_intervals_have_a_registered_measured_claim():
    from core.agency.which_one_answers_to_her import _later_control_interval_invariant
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _later_control_interval_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(test for test in suite.tests() if test.name == 'later_control_intervals_keep_one_input')
    assert check.predict(None) is True and any(claim.test == check.name for claim in suite.claims())

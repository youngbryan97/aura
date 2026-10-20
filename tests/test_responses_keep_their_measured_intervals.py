"""Dated pixel intervals retain time, ownership, precision and physical trials."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from core.agency.responses_from_measured_intervals import (
    CapturedIntervals, IntervalResponses, VelocityInterval, acceleration_from_intervals,
    _measured_interval_trials_invariant,
)
from core.agency.which_one_answers_to_her import WhichIsHers, _Speeds
from core.agency.controls_measured_on_one_thing import own_control_witness
from core.perception.frames_as_they_are_drawn import CanvasFrameTime
from core.perception.material_displacement_from_named_pixels import MaterialDisplacement

pytestmark = pytest.mark.unit


def _frame(at, **change):
    fields = dict(at=at, lower=at, upper=at+.002, arrived=at+.01, source='clock',
                  source_at=at+100., epoch=1, surface='canvas', sequence=round(at*100))
    return CanvasFrameTime(**(fields | change))


def _thing(before=1.1, at=1.3, **change):
    step = MaterialDisplacement(1, 0., .5, before, at, (10.,10.), (10.,12.), .2, 4)
    return SimpleNamespace(**(dict(number=1, born=0., seen=at, material_displacement=step,
                                  motion_position=step.end, motion_measured_at=at) | change))


def _measured(first=None, last=None, thing=None, held=None):
    first = _frame(1.1) if first is None else first
    last = _frame(1.3) if last is None else last
    captures, hers = CapturedIntervals(), WhichIsHers()
    for key, began in [('a', 1.)] if held is None else held:
        hers.holding(key, began, trying=True)
    captures.remember(first)
    captures.remember(last)
    return captures.measured(_thing() if thing is None else thing, last, hers._held_at, response_s=.05)


def test_interval_velocity_is_dated_at_its_midpoint_with_point_and_clock_error():
    interval = _measured()
    assert interval is not None and interval.middle == pytest.approx(1.2)
    assert interval.velocity == pytest.approx((0.,10.))
    assert interval.error == pytest.approx(.2/.2+10*.004/(.2-.004))
    assert interval.owner == (1,0.,.5) and interval.began == 1.
    assert interval.time_error == pytest.approx(.002)


@pytest.mark.parametrize('damage', ['plain_clock', 'missing_lower', 'missing_source', 'missing_previous',
    'surface', 'epoch', 'source', 'sequence', 'overlap', 'transition', 'unsettled', 'wrong_birth',
    'wrong_number', 'wrong_time', 'wrong_position', 'unmeasured', 'missing_step'])
def test_unowned_or_ambiguous_exposures_have_no_interval(damage):
    first, last, thing, held = _frame(1.1), _frame(1.3), _thing(), None
    if damage == 'plain_clock': last = 1.3
    elif damage == 'missing_lower': last = _frame(1.3,lower=None)
    elif damage == 'missing_source': last = _frame(1.3,source_at=None)
    elif damage == 'missing_previous': first = _frame(1.)
    elif damage == 'surface': last = _frame(1.3,surface='other')
    elif damage == 'epoch': last = _frame(1.3,epoch=2)
    elif damage == 'source': last = _frame(1.3,source='other')
    elif damage == 'sequence': last = _frame(1.3,sequence=2)
    elif damage == 'overlap': first = _frame(1.1,upper=1.31,arrived=1.32)
    elif damage == 'transition': held = [('a',1.), ('b',1.2)]
    elif damage == 'unsettled': held = [('a',1.08)]
    elif damage == 'wrong_birth': thing.born = .1
    elif damage == 'wrong_number': thing.number = 2
    elif damage == 'wrong_time': thing.seen = 1.2
    elif damage == 'wrong_position': thing.motion_position = (3.,4.)
    elif damage == 'unmeasured': thing.motion_measured_at = None
    elif damage == 'missing_step': thing.material_displacement = None
    assert _measured(first,last,thing,held) is None


def _samples(began, acceleration=60., key='a', *, error=.2, widths=(.16,.3,.12)):
    before = began+.1
    samples = []
    for width in widths:
        at = before+width
        samples.append(VelocityInterval(key,began,(1,0.,0.),before,at,
                       (3.,10.+acceleration*((before+at)/2-began)),error))
        before = at
    return samples


def test_irregular_cadence_uses_the_observed_interval_midpoints():
    samples = _samples(1.)
    estimate = acceleration_from_intervals(samples)
    assert estimate is not None and estimate.acceleration == pytest.approx((0.,60.))
    assert estimate.error > 0
    naive = (samples[-1].velocity[1]-samples[0].velocity[1])/(samples[-1].at-samples[0].at)
    assert abs(naive-60.) > 1.


def test_two_intervals_require_time_separation_and_explicit_precision():
    assert acceleration_from_intervals(_samples(1.,widths=(.2,.2))) is not None
    assert acceleration_from_intervals(_samples(1.,widths=(.04,.04))) is None
    with pytest.raises(ValueError): replace(_samples(1.)[0],error=float('nan'))
    with pytest.raises(ValueError): replace(_samples(1.)[0],error=True)


def test_midpoint_uncertainty_is_propagated_and_unresolved_time_axes_are_refused():
    samples = _samples(1.)
    exact = acceleration_from_intervals(samples)
    dated = acceleration_from_intervals([replace(s,time_error=.004) for s in samples])
    assert dated is not None and dated.error > exact.error
    assert acceleration_from_intervals([replace(s,time_error=1.) for s in samples]) is None


@pytest.mark.parametrize('damage',['gap','basis','birth','input','physical_press'])
def test_an_acceleration_fit_cannot_cross_discontinuous_intervals(damage):
    samples = _samples(1.,widths=(.2,.2))
    sample = samples[1]
    if damage == 'gap': sample = replace(sample,before=sample.before+.01)
    elif damage == 'basis': sample = replace(sample,owner=(1,0.,.8))
    elif damage == 'birth': sample = replace(sample,owner=(1,.1,.8))
    elif damage == 'input': sample = replace(sample,key='b')
    else: sample = replace(sample,began=1.01)
    assert acceleration_from_intervals([samples[0],sample]) is None


def _repeated(*, errors=.1, effect=100., repeats=2):
    responses = IntervalResponses()
    for i in range(repeats*2):
        key, acceleration = ('a',effect) if i%2 else ('',0.)
        for sample in _samples(1.+2*i,acceleration,key,error=errors): responses.add(sample)
    return responses


def test_independent_physical_holds_are_trials_and_repeat_frames_cannot_multiply_them():
    responses = IntervalResponses()
    samples = _samples(1.)
    for _ in range(10):
        for sample in samples: responses.add(sample)
    assert len(responses.best) == 1 and responses.contrast() == (0.,0.)
    assert _measured_interval_trials_invariant() == ()


def test_repeated_effects_must_clear_noise_and_error_adjusted_effect_gates():
    measured = _repeated()
    assert measured.contrast()[0] > 20 and measured.contrast()[1] > 20
    assert measured.push('a',statistic_over=20,effect_over=20) == pytest.approx((0.,100.))
    assert measured.push('',statistic_over=20,effect_over=20) == pytest.approx((0.,0.))
    for rejected in (_repeated(errors=50.),_repeated(effect=0.),_repeated(repeats=1)):
        assert rejected.push('a',statistic_over=20,effect_over=20) is None


def test_gaps_preserve_one_trial_and_contradiction_retires_it_until_physical_completion():
    responses = _repeated()
    press = ('a',3.)
    assert press in responses.best
    responses.unseen(press)
    assert press in responses.best and press not in responses.samples
    responses.unseen(press,contradicted=True)
    for sample in _samples(3.): responses.add(sample)
    assert press not in responses.best
    responses = _repeated()
    responses.end(press)
    responses.unseen(press,contradicted=True)
    assert press in responses.best


@pytest.mark.parametrize('retire',['contradiction','physical_end'])
def test_retirement_before_the_first_precise_sample_blocks_late_frames(retire):
    responses = IntervalResponses()
    if retire == 'contradiction': responses.unseen(('a',1.),contradicted=True)
    else: responses.end(('a',1.))
    for sample in _samples(1.): responses.add(sample)
    assert not responses.best and not responses.samples


def test_copied_history_cannot_resurrect_a_retired_trial():
    responses = IntervalResponses()
    responses.unseen(('a',3.),contradicted=True)
    responses.merge(_repeated())
    assert ('a',3.) not in responses.best and responses.contrast() == (0.,0.)


def test_retirements_without_samples_also_have_a_finite_storage_budget():
    responses = IntervalResponses()
    for i in range(1000): responses.unseen(('a',float(i)),contradicted=True)
    assert len(responses.through) == len(responses.invalid) == 80
    responses.end(('a',1.))
    assert ('a',1.) not in responses.through


def test_missing_stretches_cannot_supply_an_acceleration_or_a_second_trial():
    responses = IntervalResponses()
    first,last = _samples(1.,widths=(.2,.2))
    responses.add(first)
    responses.unseen(('a',1.))
    responses.add(last)
    assert not responses.best
    for sample in _samples(1.,widths=(.2,.2)):
        responses.add(replace(sample,before=sample.before+1.,at=sample.at+1.))
    assert len(responses.best) == 1


def test_retention_and_copying_are_bounded_without_aliasing():
    responses = IntervalResponses()
    for i in range(100):
        for sample in _samples(1.+2*i): responses.add(sample)
        responses.end(('a',1.+2*i))
    assert len(responses.through) == len(responses.best) == len(responses.ended) == 80
    for sample in _samples(1.): responses.add(sample)
    assert len(responses.best) == 80 and ('a',1.) not in responses.best
    other = IntervalResponses()
    other.merge(responses)
    responses.best.clear()
    assert len(other.best) == 80


def test_new_push_estimates_reach_speeds_without_replacing_the_existing_trial_gate():
    speeds = _Speeds()
    speeds.intervals = _repeated()
    assert speeds.by_push()[0] > 20 and speeds.push('a') == pytest.approx((0.,100.))
    assert speeds.ratio() == (0.,0.)
    assert own_control_witness(1,speeds,epoch=1) is None
    for i in range(4):
        key = 'a' if i%2 else ''
        for j in range(2): speeds.add(key,0.,1.,press=1.+2*i,when=1.3+2*i+j*.2)
    assert speeds.ratio()[0] > 20
    assert own_control_witness(1,speeds,epoch=1) is not None


def test_a_gap_seals_only_the_pixel_segment_while_input_end_closes_the_trial():
    speeds = _Speeds()
    for sample in _samples(1.): speeds.intervals.add(sample)
    for j in range(2): speeds.add('a',0.,10.,press=1.,when=1.3+j*.2)
    speeds.unseen('a',1.)
    assert ('a',1.) not in speeds.intervals.ended
    speeds.end('a',1.)
    assert ('a',1.) in speeds.intervals.ended


def test_measured_intervals_have_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'measured_intervals_keep_one_physical_trial')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())

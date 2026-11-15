"""Rotated input responses cannot rely on a fixed screen direction or borrowed trials."""
import math
from types import SimpleNamespace

import pytest

from core.agency.an_input_answers_relative_to_motion import (
    _motion_coordinates_invariant,
    response_in_motion_coordinates,
)
from core.agency.controls_measured_on_one_thing import own_control_witness
from core.agency.which_one_answers_to_her import _Speeds


def _trials(*, autonomous=False, copies=3):
    evidence = _Speeds()
    for index in range(3 * copies):
        key = ('push', 'bend', '')[index % 3]
        began, heading = 1. + index, index * 2.1
        for elapsed in (.2, .3, .4, .5, .6):
            speed = 50 + (40 if autonomous or key == 'push' else 0.) * elapsed
            angle = heading + (1.3 if autonomous or key == 'bend' else 0.) * elapsed
            evidence.add(key, speed * math.cos(angle), speed * math.sin(angle), press=began, when=began + elapsed)
    evidence.lowest, evidence.highest = [0., 0.], [30., 30.]
    return evidence


def test_changes_in_speed_and_curvature_answer_the_same_key_across_different_headings():
    evidence = _trials()
    statistic, effect = response_in_motion_coordinates(evidence)
    assert statistic > 20. and effect > 20.
    witness = own_control_witness(7, evidence, epoch=3, movement_over=4.)
    assert witness is not None and witness.identity == 7 and witness.epoch == 3
    # Since 5d54bf2f3 (20 October) the witness is the contrast that clears both admission gates by the widest margin,
    # not the one with the largest statistic, so it is held to that: at least the motion response's margin.
    from core.agency.a_contrast_that_meets_both_tests import contrast_rank

    assert contrast_rank((witness.statistic, witness.effect), statistic_over=20., effect_over=20.) >= contrast_rank(
        (statistic, effect), statistic_over=20., effect_over=20.)


def test_autonomous_acceleration_and_turning_supply_no_control_difference():
    statistic, effect = response_in_motion_coordinates(_trials(autonomous=True))
    assert statistic < 1. and effect < 1.


def test_one_press_or_missing_neutral_history_cannot_supply_the_response():
    assert response_in_motion_coordinates(_trials(copies=1)) == (0., 0.)
    evidence = _trials()
    evidence.timed = {k: v for k, v in evidence.timed.items() if k[0]}
    assert response_in_motion_coordinates(evidence) == (0., 0.)
    assert _motion_coordinates_invariant() == ()


@pytest.mark.parametrize('departure', ['clock', 'repeat', 'nonfinite', 'discontinuity', 'alias', 'interrupted'])
def test_unmeasured_clock_or_discontinuous_motion_cannot_be_a_repeated_response(departure):
    evidence = _trials()
    for key, values in evidence.timed.items():
        if not key[0]:
            continue
        if departure == 'clock':
            values[-1] = (values[0][0], *values[-1][1:])
        elif departure == 'repeat':
            values[:] = values[:3]
        elif departure == 'nonfinite':
            values[-1] = (math.inf, *values[-1][1:])
        elif departure == 'discontinuity':
            values[-1] = (values[-1][0], 900., 700.)
        elif departure == 'alias':
            values[-1] = (values[-1][0], -values[-2][1], -values[-2][2])
        else:
            evidence.interrupted.add(key)
    assert response_in_motion_coordinates(evidence) == (0., 0.)


def test_a_pooled_kind_cannot_supply_any_motion_trials_to_a_new_entity():
    kind = _trials()
    blank = SimpleNamespace(timed={}, by_press={}, ratio=lambda: response_in_motion_coordinates(kind))
    assert own_control_witness(9, blank, epoch=0) is None


def test_the_current_visual_state_reaches_control_identification_without_granting_identity():
    from core.agency.which_one_answers_to_her import WhichIsHers
    from core.perception.motion_kept_between_pictures import current_motion_estimate
    from tests.test_measured_motion_proposes_visual_correspondence import _tracked_body

    eyes, body = _tracked_body()
    body.vx, body.vy = 800., -800.
    estimate = current_motion_estimate(body, body.seen)
    hers = WhichIsHers()
    hers.holding('opaque', 1., trying=True)
    hers.saw(eyes, [], body.seen)
    retained = hers._by_thing[body.number].by_press['opaque', 1.]
    assert retained[-1] == pytest.approx(estimate.velocity)
    assert hers.number is None  # one observation is not independent repeated input proof
    body.motion_evidence, body.motion_measured_at = 'unavailable', None
    hers.saw(eyes, [], body.seen + .1)
    assert hers.number is None and len(retained) == 1


def test_an_autonomous_measured_trajectory_cannot_borrow_raw_velocities_correlated_with_inputs():
    from core.agency.which_one_answers_to_her import WhichIsHers
    from core.perception.motion_kept_between_pictures import observe_thing_motion
    from tests.test_measured_motion_proposes_visual_correspondence import _tracked_body

    eyes, body = _tracked_body()
    hers, at = WhichIsHers(), body.seen
    for key in ['opaque-a', '', 'opaque-b', ''] * 3:
        hers.holding(key, at, trying=True)
        for _ in range(16):
            at += .05
            body.x += 4.  # the same autonomous trajectory under every input
            body.seen = body.motion_measured_at = at
            body.vx = {'opaque-a': 200., 'opaque-b': -200., '': 0.}[key]
            body.vy = 0.
            observe_thing_motion(body, at, eyes.shape)
            hers.saw(eyes, [], at)
    evidence = hers._by_thing[body.number]
    assert sum(len(v) >= 2 for v in evidence.by_press.values()) >= 8
    assert hers.number is None
    assert not any(r.get('selected') is not None for r in hers.identification.receipts)


def test_motion_coordinates_have_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'rotated_control_trials_keep_their_own_motion')
    assert check.predict(None) and any(c.test == check.name for c in suite.claims())

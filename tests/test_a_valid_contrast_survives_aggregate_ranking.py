"""A high statistic cannot hide another contrast that passes both admission tests."""
import pytest

from core.agency.responses_from_measured_intervals import AccelerationEstimate, IntervalResponses
from core.agency.which_one_answers_to_her import _Speeds

pytestmark = pytest.mark.unit


def responses():
    evidence = IntervalResponses()
    for i, (key, acceleration, error) in enumerate((('', (0., 0.), .5),
            ('small', (30., 0.), .5), ('large', (0., 160.), 30.))):
        for trial in range(2):
            evidence.best[key, float(i*2+trial+1)] = AccelerationEstimate(acceleration, error, 4)
    return evidence


def test_a_valid_interval_pair_is_not_hidden_by_a_larger_statistic_with_insufficient_effect():
    statistic, effect = responses().contrast()
    assert statistic > 20. and effect > 20.


def test_rest_can_use_its_qualified_comparison_with_another_key():
    evidence = responses()
    assert evidence.push('', statistic_over=20., effect_over=20.) == (0., 0.)
    assert len(evidence.best) == 6


def test_raw_push_summary_cannot_hide_a_qualified_precision_bearing_interval():
    speeds = _Speeds()
    speeds.pushes = lambda: {'small': [(30., 0.)]*2, '': [(0., 0.)]*2}
    speeds.intervals.contrast = lambda **_kwargs: (20.4512, 72.3741)
    assert speeds.by_push() == (20.4512, 72.3741)


def test_the_causal_ratio_keeps_a_qualified_effect_from_its_available_witnesses():
    speeds = _Speeds()
    for key, vx in (('a', 0.), ('b', 15.)):
        for began in (1., 3.):
            speeds.by_press[key, began] = [(vx, 0.)]*2
    speeds.by_push = lambda: (25., 60.)
    assert speeds.ratio() == (25., 60.)


def test_a_small_repeatable_trajectory_does_not_hide_a_qualified_longer_response():
    from types import SimpleNamespace
    from core.agency.controls_measured_on_one_thing import own_trajectory_witness
    from core.agency.what_a_press_does import PressTrajectory

    samples = [('', 0.), ('', 0.), ('small', 15.), ('small', 15.), ('large', 74.), ('large', 86.)]
    trials = [PressTrajectory(key, 1, 3, 1.+2*i, 1.4+2*i, 0.,
              ((.1, 0., amount/2), (.2, 0., amount), (.4, 0., 0.))) for i, (key, amount) in enumerate(samples)]
    witness = own_trajectory_witness(SimpleNamespace(number=1, born=0., seen=14.), trials, epoch=3)
    assert witness.statistic > 20. and witness.effect > 20.
    assert len(witness.independent_trials) == 2


def test_joint_ranking_preserves_admission_across_order_and_threshold_boundaries():
    from core.agency.a_contrast_that_meets_both_tests import (
        _joint_contrast_invariant, strongest_eligible_contrast,
    )

    assert _joint_contrast_invariant() == ()
    pairs = [(float('nan'), 80.), (1000., 10.), (30., 21.), (True, 80.), (float('inf'), 80.), (10**400, 80.)]
    assert strongest_eligible_contrast(pairs, statistic_over=20., effect_over=20.) == (30., 21.)
    assert strongest_eligible_contrast([], statistic_over=20., effect_over=20.) == (0., 0.)
    assert strongest_eligible_contrast([(20., 50.), (21., 21.)], statistic_over=20., effect_over=20., strict=True) == (21., 21.)


def test_joint_contrast_has_a_registered_executable_measurement():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'contrast_ranking_preserves_joint_admission')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())

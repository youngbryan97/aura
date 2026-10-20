"""Future physical holds compare both models without training on their result."""
import math

import pytest

from core.agency.motion_models_on_the_same_intervals import (
    MotionModelsOnTheSameIntervals, _forward_motion_models_invariant,
)
from core.agency.responses_from_measured_intervals import VelocityInterval
from core.agency.which_one_answers_to_her import _Speeds

pytestmark = pytest.mark.unit


def held(models, began, *, family='acceleration', owner=(1, 0., 0.), close=True, impulse=0.):
    dates = [round(began+.1+j*.1, 6) for j in range(7)]
    for before, at in zip(dates, dates[1:]):
        speed = 40.*(before+at)/2 if family == 'acceleration' else 50.
        if impulse and before > began+.3:
            speed += impulse
        models.add(VelocityInterval('a', began, owner, before, at, (speed, 0.), .1))
    if close:
        models.end(('a', began))


def test_independent_later_holds_select_acceleration_on_comparable_predictions():
    assert _forward_motion_models_invariant() == ()


def test_direct_motion_is_not_mistaken_for_acceleration():
    models = MotionModelsOnTheSameIntervals()
    for began in (1., 3., 5., 7.):
        held(models, began, family='direct')
    result = models.read(admitted_acceleration=False, qualified_push=True)
    # Constant pace also fits zero acceleration. Comparable errors abstain,
    # rather than inventing evidence for either family.
    assert result.selected is None and result.reason == 'forward_predictions_ambiguous'


def test_a_scored_hold_cannot_vote_twice_or_train_the_model_predicting_itself():
    models = MotionModelsOnTheSameIntervals()
    for began in (1., 3., 5.):
        held(models, began)
    assert models.votes == 1 and models.selected is None
    held(models, 5.)
    models.end(('a', 5.))
    assert models.votes == 1 and models.selected is None
    assert models.models[('a', 5.)][0] == pytest.approx((40., 0.))
    # A huge counterexample is retained, never removed as a guessed contact.
    held(models, 7., impulse=200.)
    assert models.last[-2] > 1000. and models.last[-1] > 1000.


def test_an_admitted_law_survives_an_unrelated_pace_statistic_or_temporary_effect_loss():
    speeds = _Speeds()
    for i in range(4):
        began, key = 1.+2*i, 'a' if i % 2 else ''
        dates = [round(began+.1+j*.1, 6) for j in range(7)]
        for before, at in zip(dates, dates[1:]):
            speeds.intervals.add(VelocityInterval(key, began, (1, 0., 0.), before, at,
                ((80. if key else 0.)*((before+at)/2-began), 0.), .1))
        speeds.end(key, began)
    speeds.by_push = lambda: (34., 43.)
    speeds._by_pace_alone = lambda: 18.
    speeds._stops_when_let_go = lambda: False
    assert speeds.carried()
    speeds._by_pace_alone = lambda: 55.
    assert speeds.carried()
    speeds.by_push = lambda: (22., 14.)
    assert speeds.carried()
    assert speeds.last_family_reading.prediction.selected is True


def test_without_measured_interval_ownership_the_old_admission_rules_still_apply():
    speeds = _Speeds()
    speeds.by_push = lambda: (34., 43.)
    speeds._by_pace_alone = lambda: 55.
    assert not speeds.carried()


def test_switching_requires_two_independently_completed_contrary_holds():
    models = MotionModelsOnTheSameIntervals()
    held(models, 1., family='direct')
    assert models.read(admitted_acceleration=True, qualified_push=True).selected is True
    # Two prior direct holds fit a stable direct law, while acceleration
    # predicts the held-out disturbance from its measured initial velocity.
    held(models, 3., family='direct')
    for began in (5., 7.):
        dates = [round(began+.1+j*.1, 6) for j in range(7)]
        for i, (before, at) in enumerate(zip(dates, dates[1:])):
            speed = 100. if i == 0 else 50.
            models.add(VelocityInterval('a', began, (1, 0., 0.), before, at, (speed, 0.), .1))
        models.end(('a', began))
        if began == 5.:
            assert models.selected is True and models.contrary == 1
    assert models.selected is False


def test_new_owner_or_reference_cannot_inherit_old_model_authority():
    models = MotionModelsOnTheSameIntervals()
    held(models, 1.)
    assert models.read(admitted_acceleration=True, qualified_push=True).selected is True
    held(models, 3., owner=(2, 2., 2.))
    assert models.read(admitted_acceleration=False, qualified_push=False).selected is None
    assert models.owner == (2, 2., 2.) and models.votes == 0


def test_old_contrast_cannot_readmit_a_law_for_a_new_material_owner():
    speeds = _Speeds()
    speeds.by_push = lambda: (34., 43.)
    speeds._by_pace_alone = lambda: 18.
    speeds._stops_when_let_go = lambda: False
    held(speeds.intervals.families, 3., owner=(2, 2., 2.))
    assert not speeds.carried()
    assert speeds.last_family_reading.reason == 'current_material_push_unresolved'


def test_memory_is_bounded_and_recorded_losses_are_finite():
    models = MotionModelsOnTheSameIntervals()
    for began in range(1, 205, 2):
        held(models, float(began))
    assert len(models.samples) == 80
    assert len(models.models) <= 80 and len(models.losses) <= 80 and len(models.closed) <= 80
    assert all(math.isfinite(value) for value in models.last)


def test_predictive_motion_claim_runs_its_registered_measurement():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'motion_models_share_forward_heldout_observations')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())


def test_real_interval_response_and_family_paths_share_the_delivered_hold_boundaries():
    speeds = _Speeds()
    for i in range(10):
        began, key = 1.+2*i, 'a' if i % 2 else ''
        dates = [round(began+.1+j*.1, 6) for j in range(7)]
        for before, at in zip(dates, dates[1:]):
            acceleration = 80. if key else 0.
            speeds.intervals.add(VelocityInterval(key, began, (1, 0., 0.), before, at,
                (10.*i+acceleration*((before+at)/2-began), 0.), .1))
        speeds.end(key, began)
    assert speeds.by_push()[0] >= 20. and speeds.by_push()[1] >= 20.
    assert speeds.carried()
    assert speeds.last_family_reading.prediction.holds >= 2
    assert len(speeds.intervals.families.closed) == 10

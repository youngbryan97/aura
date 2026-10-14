"""A fixed observation budget retains excitation without hiding the latest disagreement."""
import math

import pytest

from core.agency.direction_linked_to_observed_pose import DirectionFromPose

pytestmark = pytest.mark.unit


def test_repeated_nearby_readings_cannot_discard_the_measured_rotation_span():
    relation = DirectionFromPose()
    for i, phase in enumerate((0., .7, 1., *[.85+j*.001 for j in range(40)])):
        relation.observe('body', 1.+i, phase, .01, direction=phase+.3)
    assert len(relation.samples) == 16
    assert relation.current == pytest.approx(phase+.3)
    assert max(abs((a.phase-b.phase+math.pi)%(2*math.pi)-math.pi)
               for a in relation.samples for b in relation.samples) >= .7


def test_the_latest_contradiction_is_retained_and_revokes_the_relation():
    relation = DirectionFromPose()
    for i in range(40):
        phase = i*.03
        relation.observe('body', 1.+i, phase, .01, direction=phase+.3)
    assert relation.current is not None
    relation.observe('body', 50., 1., .01, direction=2.)
    assert relation.current is None and relation.samples[-1].direction == 2.
    relation.observe('replacement', 51., 1., .01, direction=None)
    assert relation.current is None and not relation.samples


def test_angular_coverage_does_not_make_an_unrelated_animation_a_direction():
    relation = DirectionFromPose()
    for i in range(50):
        relation.observe('animation', 1.+i, i*.04, .01, direction=.3)
    assert relation.current is None and len(relation.samples) == 16


def test_calibration_coverage_has_a_registered_validation():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'pose_calibration_keeps_measured_phase_coverage')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())

"""Local motion precision follows observed correspondence, not a fresh coordinate guess."""
import math
from dataclasses import replace

import numpy as np
import pytest

from core.perception.material_displacement_from_named_pixels import (
    MaterialDisplacement, displacement_from_named_pixels, _material_displacement_precision_invariant,
)
from core.perception.position_from_the_same_points import PointPose
from core.perception.points_that_keep_their_motion import PointMotion
from core.perception.rigid_motion_from_named_points import RigidMotion

pytestmark = pytest.mark.unit


def _poses(angle=0., reference_error=.2):
    marks = dict(enumerate(((10., 10.), (30., 10.), (10., 30.), (30., 30.))))
    transform = RigidMotion(angle, (2., 1.), (), 0.)
    previous = PointPose(0., 1., (20., 20.), marks.copy(), (20., 20.), reference_at=.5,
                         visible_marks=marks.copy(), reference_error=reference_error)
    current = PointPose(0., 1.2, (20., 20.), marks.copy(), tuple(transform.apply((20., 20.))),
        reference_at=.5, visible_marks={n: tuple(transform.apply(p)) for n, p in marks.items()})
    transform = replace(transform, inliers=tuple(range(4)))
    motion = PointMotion(2.,1.,0.,4,1.,1.2,tuple(marks),tuple(current.visible_marks.values()),rigid=transform)
    return previous, current, motion


def test_a_shared_translation_offset_cannot_be_counted_as_new_displacement_noise():
    previous, current, motion = _poses(reference_error=100.)
    observed = displacement_from_named_pixels(7, previous, current, motion)
    assert observed is not None and observed.error < .3
    assert observed.start == (20., 20.) and observed.end == (22., 21.)
    assert observed.before == 1. and observed.at == 1.2 and observed.reference_at == .5
    assert _material_displacement_precision_invariant() == ()


@pytest.mark.parametrize('angle', [-math.pi, -.4, .2, math.pi/2])
def test_rotation_propagates_the_shared_reference_offset_at_its_actual_lever(angle):
    previous, current, motion = _poses(angle=angle, reference_error=12.)
    observed = displacement_from_named_pixels(7, previous, current, motion)
    previous.reference_error = 0.
    local = displacement_from_named_pixels(7, previous, current, motion)
    assert observed is not None and local is not None
    assert observed.error-local.error == pytest.approx(2*abs(math.sin(angle/2))*12.)


def test_disagreement_between_the_local_fit_and_the_reported_origin_is_included():
    previous, current, motion = _poses()
    baseline = displacement_from_named_pixels(1, previous, current, motion)
    current.position = (current.position[0]+.6, current.position[1]+.8)
    observed = displacement_from_named_pixels(1, previous, current, motion)
    assert observed is not None and observed.error == pytest.approx(baseline.error+1.)


@pytest.mark.parametrize('damage', ['birth', 'basis', 'gap', 'time', 'unknown_error', 'two_marks',
                                  'unknown_correspondence', 'collinear', 'wrong_owner', 'boolean_error',
                                  'wrong_pixels'])
def test_departed_or_underconstrained_material_supplies_no_precision(damage):
    previous, current, motion = _poses()
    number = 1
    if damage == 'birth': current.born = .1
    elif damage == 'basis': current.reference_at = 1.1
    elif damage == 'gap':
        current.at = 1.4
        motion = replace(motion,at=1.4)
    elif damage == 'time': motion = replace(motion,before=.9)
    elif damage == 'unknown_error': previous.reference_error = float('nan')
    elif damage == 'boolean_error': previous.reference_error = True
    elif damage == 'two_marks': current.visible_marks = dict(list(current.visible_marks.items())[:2])
    elif damage == 'unknown_correspondence': motion = replace(motion,rigid=None)
    elif damage == 'wrong_owner': number = True
    elif damage == 'wrong_pixels': motion = replace(motion,locations=((0.,0.),)*4)
    else:
        previous.visible_marks = {n: (float(n*10), 0.) for n in range(4)}
        current.visible_marks = {n: (float(n*10+2), 1.) for n in range(4)}
        motion = replace(motion,locations=tuple(current.visible_marks.values()))
    assert displacement_from_named_pixels(number, previous, current, motion) is None


@pytest.mark.parametrize('change', [dict(error=float('inf')), dict(support=2), dict(before=1.3),
                                   dict(at=1.6), dict(reference_at=2.), dict(start=(True, 0.))])
def test_invalid_displacement_receipts_are_refused(change):
    values = dict(number=1,born=0.,reference_at=.5,before=1.,at=1.2,start=(2.,3.),end=(4.,5.),error=.2,support=4)
    with pytest.raises(ValueError): MaterialDisplacement(**(values | change))


def test_the_precision_helper_does_not_rewrite_the_observers_pose():
    previous, current, motion = _poses()
    before = replace(previous, anchors=previous.anchors.copy(), visible_marks=previous.visible_marks.copy())
    after = replace(current, anchors=current.anchors.copy(), visible_marks=current.visible_marks.copy())
    displacement_from_named_pixels(1,previous,current,motion)
    assert vars(previous) == vars(before) and vars(current) == vars(after)


def test_the_ordinary_pixel_observer_publishes_local_displacement_precision():
    from tests.test_resting_material_keeps_its_measured_origin import _moving_texture

    eyes, body, picture = _moving_texture()
    step = body.material_displacement
    assert step is not None and step.at == body.seen and step.end == body.motion_position
    assert step.before == 1.1 and step.at == 1.2 and step.error > 0
    eyes._blobs = lambda *_args: []
    eyes.see(picture,1.3)
    assert body.material_displacement is not None and body.material_displacement.at == 1.3


def test_the_observation_worker_keeps_the_receipt_across_a_detached_process_snapshot():
    import pickle
    from core.perception.play_observation_without_holding_the_loop import _detached_state
    from tests.test_resting_material_keeps_its_measured_origin import _moving_texture

    eyes, body, _picture = _moving_texture()
    mirrored = pickle.loads(pickle.dumps(_detached_state(eyes)))['things'][body.number]
    assert mirrored is not body and mirrored.material_displacement == body.material_displacement
    assert mirrored.point_pose is not body.point_pose


def test_local_precision_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'material_displacement_separates_shared_reference_error')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())

"""A geometric reference survives lost marks only while its precision remains measured."""
from types import SimpleNamespace

import numpy as np
import pytest

from core.perception.points_that_keep_their_motion import PointMotion
from core.perception.position_from_the_same_points import PointPose, position_from_points
from core.perception.rigid_motion_from_named_points import RigidMotion

pytestmark = pytest.mark.unit


def _trial(*, names=(0, 2, 6, 8), origin=(40., 40.), prior_error=0., angle=.15, offset=(2., -1.)):
    marks = {i: (34.+6*(i % 3), 34.+6*(i // 3)) for i in range(9)}
    previous = PointPose(0., 1., origin, marks, origin, reference_at=1.,
                         visible_marks=marks.copy(), reference_error=prior_error)
    body = SimpleNamespace(born=0., seen=1., point_pose=previous)
    transform = RigidMotion(angle, offset, (), 0.)
    ends = tuple(tuple(transform.apply(marks[i])) for i in names)
    delta = transform.apply(origin)-origin
    motion = PointMotion(*delta, .01, len(names), 1., 1.1, names, ends)
    centre = transform.apply((40., 40.))
    blob = dict(x=float(centre[0]), y=float(centre[1]), w=60., h=60., support=np.ones((60, 60), bool))
    return body, blob, motion, transform


@pytest.mark.parametrize('angle', [0., -.2, .15, .8])
def test_a_minority_of_spread_original_marks_still_locates_the_reference(angle):
    body, blob, motion, transform = _trial(angle=angle)
    result = position_from_points(body, blob, motion, 1.1)
    assert result[0] == (40., 40.)
    assert result[1] == pytest.approx(transform.apply((40., 40.)))
    assert body.point_pose.reference_at == 1. and 0 < body.point_pose.reference_error < 1.


@pytest.mark.parametrize('departure', ['line', 'unseen_names', 'far_target', 'exhausted_error', 'wrong_motion'])
def test_surviving_marks_cannot_bridge_unobserved_or_uncertain_geometry(departure):
    options = {'names': (0, 1, 2)} if departure == 'line' else {}
    if departure == 'far_target':
        options['origin'] = (1000., 1000.)
    if departure == 'exhausted_error':
        options['prior_error'] = .999
    body, blob, motion, _ = _trial(**options)
    if departure == 'unseen_names':
        from dataclasses import replace

        motion = replace(motion, points=(100, 101, 102, 103))
    elif departure == 'wrong_motion':
        from dataclasses import replace

        motion = replace(motion, dx=100.)
    assert position_from_points(body, blob, motion, 1.1) is None


def test_long_sequences_cannot_reset_the_error_budget_by_dropping_marks():
    body, blob, motion, _ = _trial(prior_error=.9)
    old_error = body.point_pose.reference_error
    assert position_from_points(body, blob, motion, 1.1) is not None
    assert body.point_pose.reference_error > old_error


def test_the_surviving_reference_contract_is_registered():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite
    from core.perception.position_from_the_same_points import _surviving_reference_invariant

    assert _surviving_reference_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'surviving_marks_need_target_precision')
    assert check.predict(None) and any(c.test == check.name for c in suite.claims())


@pytest.mark.parametrize('departure', ['', 'two_survive', 'collinear', 'spent_precision', 'other_birth'])
def test_geometric_qualification_reaches_point_motion_material_matching_and_position(departure):
    from core.perception.points_that_keep_their_motion import FollowingPoints, SeenPoint
    from tests.test_continuing_material_survives_a_changed_outline import scene

    eyes, body, blob, original = scene()
    pose = body.point_pose
    for n, point in enumerate(((17., 29.), (23., 29.), (17., 33.), (23., 33.), (20., 30.), (20., 32.)), 4):
        pose.visible_marks[n] = point
        pose.anchors[n] = (point[0]-10., point[1])
    if departure == 'spent_precision':
        pose.reference_error = .999
    elif departure == 'other_birth':
        pose.born = 0.
    names = (0, 1) if departure == 'two_survive' else (0, 1, 2, 3)
    if departure == 'collinear':
        for n in names:
            pose.visible_marks[n] = (16.+2*n, 31.)
    following = FollowingPoints()
    following.points = tuple(SeenPoint(n, x+3., y, x, y, .5, 1., 1.1, .01, 3)
                             for n in names for x, y in [pose.visible_marks[n]])
    motion = following.motion_of(body, 1.1)
    if departure:
        assert motion is None
        return
    assert motion is not None and motion.support == 4  # four of ten original anchors survive
    eyes._point_motion = {body.number: motion}
    assert eyes._cost(body, blob, 1.1) == float('inf')  # outline alone still refuses the change
    assert eyes._match([blob], 1.1) == ({body.number: 0}, [])
    eyes._moved(body, blob, 1.1)
    assert body.point_pose.reference_at == .8 and body.born == .5
    assert body.motion_position == pytest.approx((23., 20.)) and body.motion_measured_at == 1.1

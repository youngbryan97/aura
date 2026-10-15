"""Limited image deformation needs redundant, independently predicted correspondences."""
import numpy as np
import pytest

from core.perception.affine_motion_from_named_points import (
    _affine_material_invariant,
    affine_motion,
)
from core.perception.rigid_motion_from_named_points import rigid_motion

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('linear', [((1., .1), (0., .8)), ((1.3, 0.), (0., .9)), ((.8, -.2), (.2, .8))])
def test_affine_material_and_its_inverse_are_observed_separately_from_a_rigid_pose(linear):
    starts = np.array([(x, y) for x in (0., 8., 16.) for y in (0., 8., 16.)])
    ends = starts@np.array(linear).T+(4., -3.)
    fit = affine_motion(starts, ends)
    assert fit is not None and fit.apply(starts) == pytest.approx(ends)
    assert fit.inverse(ends) == pytest.approx(starts)
    assert rigid_motion(starts, ends) is None


@pytest.mark.parametrize('departure', ['few', 'line', 'collapse', 'reflection', 'excessive', 'unrelated', 'nonfinite'])
def test_unsupported_or_degenerate_geometry_remains_unknown(departure):
    starts = np.array([(x, y) for x in (0., 8., 16.) for y in (0., 8., 16.)])
    ends = starts+(4., -3.)
    if departure == 'few':
        starts, ends = starts[:5], ends[:5]
    elif departure == 'line':
        starts[:, 1] = 0.
    elif departure == 'collapse':
        ends[:, 1] = 0.
    elif departure == 'reflection':
        ends = starts*(-1., 1.)
    elif departure == 'excessive':
        ends = starts*4.
    elif departure == 'unrelated':
        ends = np.random.default_rng(39).uniform(-40, 40, starts.shape)
    else:
        ends[1, 0] = np.nan
    assert affine_motion(starts, ends) is None


def test_a_minor_outlier_is_explicitly_excluded_and_target_precision_degrades_away_from_the_marks():
    starts = np.array([(x, y) for x in (0., 8., 16.) for y in (0., 8., 16.)])
    ends = starts@np.array(((1., .05), (0., .8))).T+(4., -3.)
    ends[-1] += (7., -10.)
    fit = affine_motion(starts, ends)
    assert fit is not None and 8 not in fit.inliers
    assert fit.target_error((8., 8.), resolution=.25) < .5
    assert fit.target_error((500., 500.), resolution=.25) > 1.
    assert fit.target_error((np.nan, 8.), resolution=.25) == float('inf')
    assert fit.target_error(None, resolution=.25) == float('inf')
    assert _affine_material_invariant() == ()


def test_a_redundant_affine_observation_reaches_object_position_without_supplying_a_turn():
    from core.perception.orientation_from_continuing_marks import current_angular_phase
    from core.perception.points_that_keep_their_motion import FollowingPoints, SeenPoint
    from core.perception.position_from_the_same_points import PointPose
    from core.perception.what_moves_in_the_picture import WhatMoves

    eyes = WhatMoves()
    eyes.shape = (100, 100)
    old = dict(x=40., y=40., w=24., h=24., colour=(180, 80, 80), look=np.ones(64)/64,
               support=np.ones((24, 24), bool), patch=np.full((24, 24, 3), 180, np.uint8))
    body = eyes._born(old, .5)
    body.seen = 1.
    body.path.append((1., 40., 40.))
    starts = np.array([(x, y) for x in (32., 40., 48.) for y in (32., 40., 48.)])
    ends = starts@np.array(((1., 0.), (0., .8))).T+(2., 6.)
    marks = {n: tuple(p) for n, p in enumerate(starts)}
    body.point_pose = PointPose(.5, 1., (40., 40.), marks.copy(), (40., 40.), reference_at=1.,
                                visible_marks=marks.copy(), angular_reference_at=1.)
    body.motion_position, body.motion_measured_at = (40., 40.), 1.
    points = FollowingPoints()
    points.points = tuple(SeenPoint(n, *end, *start, .5, 1., 1.1, .01, 5)
                         for n, (start, end) in enumerate(zip(starts, ends, strict=True)))
    motion = points.motion_of(body, 1.1)
    assert motion is not None and motion.has_affine_pose() and not motion.has_rigid_pose()
    eyes._point_motion = {body.number: motion}
    eyes._moved(body, dict(old, x=42., y=38.), 1.1)
    assert body.motion_evidence == 'points' and body.motion_measured_at == 1.1
    assert body.motion_position == pytest.approx((42., 38.))
    assert body.point_pose.reference_at == 1. and body.born == .5
    assert current_angular_phase(body, 1.1) is None


@pytest.mark.parametrize('departure', ['error', 'matrix', 'negative_variance', 'few_names', 'both_models'])
def test_invalid_affine_receipts_cannot_acquire_material_authority(departure):
    from dataclasses import replace

    from core.perception.points_that_keep_their_motion import PointMotion
    from core.perception.rigid_motion_from_named_points import RigidMotion

    starts = np.array([(x, y) for x in (0., 8., 16.) for y in (0., 8., 16.)])
    ends = starts@np.array(((1., .05), (0., .8))).T+(4., -3.)
    fit = affine_motion(starts, ends)
    if departure == 'error':
        fit = replace(fit, error=2.)
    elif departure == 'matrix':
        fit = replace(fit, linear=((1., 0.), (0., -1.)))
    elif departure == 'negative_variance':
        fit = replace(fit, residual_variance=-1.)
    motion = PointMotion(4., -3., .01, 9, 1., 1.1, tuple(range(9)), tuple(map(tuple, ends)), affine=fit)
    if departure == 'few_names':
        motion = replace(motion, support=5, points=motion.points[:5], locations=motion.locations[:5])
    elif departure == 'both_models':
        motion = replace(motion, rigid=RigidMotion(0., (4., -3.), tuple(range(9)), .01))
    assert not motion.has_affine_pose()


def test_the_affine_observation_boundary_has_a_registered_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'affine_material_requires_redundant_geometry')
    assert check.predict(None) and any(c.test == check.name for c in suite.claims())


@pytest.mark.parametrize('seed', [7, 31, 83])
def test_source_pixels_supply_deforming_correspondences_without_exact_point_injection(seed):
    from types import SimpleNamespace

    from scipy import ndimage

    from core.perception.points_before_the_picture_is_reduced import PointsBeforeReduction
    from core.perception.position_from_the_same_points import position_from_points

    image = np.full((120, 160, 3), 25, np.uint8)
    texture = np.random.default_rng(seed).uniform(30, 240, (36, 32, 3))
    texture = ndimage.gaussian_filter(texture, (1.2, 1.2, 0))
    image[32:68, 44:76] = texture.astype(np.uint8)
    body = SimpleNamespace(born=1., seen=1., x=60., y=50., w=32., h=36.,
                           support=np.ones((36, 32), bool), point_pose=None)
    points = PointsBeforeReduction(limit=96)
    points.see_source(image, image, 1.)
    points.see_source(image, image, 1.1)
    seed_motion = points.motion_of(body, 1.1)
    assert seed_motion is not None
    blob = dict(x=60., y=50., w=32., h=36., support=body.support)
    assert position_from_points(body, blob, seed_motion, 1.1) is None
    body.seen = 1.1
    # Forward image map: x'=x+3, y'=.85*y+7.5. The tracker sees only the rendered pixels.
    warped = np.stack([ndimage.affine_transform(image[..., c], np.diag((1/.85, 1.)),
        offset=(-7.5/.85, -3.), order=1, mode='constant', cval=25) for c in range(3)], axis=2)
    points.see_source(warped, warped, 1.2)
    motion = points.motion_of(body, 1.2)
    assert motion is not None and motion.has_affine_pose()
    assert motion.affine.apply((60., 50.)) == pytest.approx((63., 50.), abs=.75)

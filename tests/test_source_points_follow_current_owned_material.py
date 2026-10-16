"""A supplied coordinate frame can use an observed track without changing control identity."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from core.perception.a_visual_reference_in_the_picture import _digest, reference_appearances
from core.perception.coordinates_in_a_visual_reference import SourceImageCoordinates
from core.perception.position_from_the_same_points import PointPose, position_from_points
from core.perception.points_that_keep_their_motion import PointMotion
from core.perception.pixels_consumed_by_the_observer import consumed_pixels
from core.perception.what_moves_in_the_picture import Thing

pytestmark = pytest.mark.unit


@pytest.fixture
def scene():
    reference = np.random.default_rng(719).integers(10, 230, (64, 64, 4), dtype=np.uint8)
    reference[..., 3] = 255
    picture = np.zeros((128, 160, 3), np.uint8)
    picture[30:94, 40:104] = reference[..., :3]
    appearance, = reference_appearances(picture, reference, scales=(1.,))
    # Source calibration is supplied; its independent raster measurement is
    # exercised by test_source_coordinates_survive_reference_export.py.
    coordinates = SourceImageCoordinates('source', _digest(reference), '0'*64, (64, 64),
                                          ((1., 0.), (0., 1.)), (12., 8.), .01)
    body = Thing(1, 72., 62., 64., 64., np.zeros(64), (120, 120, 120), reference[..., :3].copy(),
                 1., 4., kind=0, support=np.ones((64, 64), bool))
    marks = dict(enumerate(((48., 38.), (94., 38.), (48., 84.), (94., 84.))))
    body.point_pose = PointPose(1., 4., (72., 62.), marks.copy(), (72., 62.), reference_at=4.,
                               visible_marks=marks.copy())
    observer = SimpleNamespace(things={1: body}, scale=1., pixels_consumed=consumed_pixels(picture, 4.))
    return coordinates, appearance, picture, reference, body, observer


def _attach(scene, point=(0., 0.)):
    coordinates, appearance, picture, reference, _body, observer = scene
    return coordinates.on_track(point, appearance, picture, reference, observer, 4., source='source')


def test_current_reference_geometry_attaches_without_changing_the_motion_model(scene):
    _, _, _, _, body, observer = scene
    pose = body.point_pose
    result = _attach(scene)
    assert result.reason == 'qualified' and result.binding is not None
    current = result.binding.current(observer, 4., source='source')
    assert current.number == body.number and current.position == pytest.approx((52., 38.), abs=.05)
    assert 0 < current.error < 1.
    assert body.motion_position is None and body.point_pose is pose and pose.origin == (72., 62.)
    assert not hasattr(result.binding, 'controlled') and not hasattr(result.binding, 'role')


def test_the_existing_point_continuation_carries_the_registered_point(scene):
    _, _, _, _, body, observer = scene
    binding = _attach(scene).binding
    marks = body.point_pose.visible_marks
    moved = tuple((x+3., y+2.) for x, y in marks.values())
    motion = PointMotion(3., 2., .01, len(marks), 4., 4.1, tuple(marks), moved)
    blob = dict(x=75., y=64., w=64., h=64., support=np.ones((64, 64), bool))
    assert position_from_points(body, blob, motion, 4.1) == ((72., 62.), (75., 64.))
    body.x, body.y, body.seen = 75., 64., 4.1
    observer.pixels_consumed = consumed_pixels(np.roll(scene[2], (2, 3), axis=(0, 1)), 4.1)
    current = binding.current(observer, 4.1, source='source')
    assert current.position == pytest.approx((55., 40.), abs=.05)
    assert body.point_pose.origin == (72., 62.)  # Existing control coordinates retain their basis.


def test_a_turn_transports_the_source_point_about_the_observed_material_basis(scene):
    _, _, _, _, body, observer = scene
    binding = _attach(scene).binding
    angle = .4
    c, s = np.cos(angle), np.sin(angle)
    linear, offset = np.array(((c, -s), (s, c))), np.array((28., -20.))
    pose = body.point_pose
    pose.visible_marks = {n: tuple(linear @ p+offset) for n, p in pose.anchors.items()}
    pose.at = body.seen = 4.1
    body.x, body.y = tuple(linear @ np.array((72., 62.))+offset)
    body.w, body.h, body.support = 100., 100., np.ones((100, 100), bool)
    inverse, shift = np.linalg.inv(linear), -np.linalg.inv(linear) @ offset
    transform = (inverse[0, 0], inverse[0, 1], shift[0], inverse[1, 0], inverse[1, 1], shift[1])
    picture = np.array(Image.fromarray(scene[2]).transform((160, 128), Image.Transform.AFFINE, transform))
    observer.pixels_consumed = consumed_pixels(picture, 4.1)
    current = binding.current(observer, 4.1, source='source')
    assert current.position == pytest.approx(linear @ np.array((52., 38.))+offset, abs=.05)


@pytest.mark.parametrize('change,reason', [
    ('no_support', 'no_unique_material_owner'), ('support_hole', 'no_unique_material_owner'),
    ('competing_owner', 'no_unique_material_owner'), ('no_pose', 'no_current_point_basis'),
    ('stale_pose', 'no_current_point_basis'), ('new_basis', 'no_current_point_basis'),
    ('unsupported_marks', 'no_current_point_basis'), ('collinear_marks', 'no_current_point_basis'),
    ('changed_pixels', 'reference_not_current'), ('changed_reference', 'reference_not_current'),
    ('changed_source', 'reference_not_current'),
    ('unpaired_observer', 'observer_pixels_do_not_match'),
])
def test_missing_or_conflicting_measurements_explain_why_attachment_is_refused(scene, change, reason):
    coordinates, appearance, picture, reference, body, observer = scene
    if change == 'no_support':
        body.support = None
    elif change == 'support_hole':
        body.support[:] = False
    elif change == 'competing_owner':
        other = replace(body, number=2)
        observer.things[2] = other
    elif change == 'no_pose':
        body.point_pose = None
    elif change == 'stale_pose':
        body.point_pose.at = 3.9
    elif change == 'new_basis':
        body.point_pose.reference_at = 5.
    elif change == 'unsupported_marks':
        body.point_pose.visible_marks = {n: (0., 0.) for n in range(4)}
    elif change == 'collinear_marks':
        body.point_pose.anchors = {n: (float(n), 0.) for n in range(4)}
        body.point_pose.visible_marks = {n: (48.+n, 38.) for n in range(4)}
    elif change == 'changed_pixels':
        picture[0, 0] = 10
    elif change == 'changed_reference':
        reference[0, 0] = 10
    elif change == 'unpaired_observer':
        observer.pixels_consumed = consumed_pixels(np.ones_like(picture), 4.)
    else:
        coordinates = replace(coordinates, source='changed')
        scene = coordinates, appearance, picture, reference, body, observer
    result = _attach(scene)
    assert result.binding is None and result.reason == reason


def test_a_small_competing_owner_cannot_be_hidden_by_a_dominant_owner(scene):
    _, appearance, _, _, body, observer = scene
    other = replace(body, number=2, support=np.zeros((64, 64), bool))
    x, y = appearance.observed_points[0]
    other.support[round(y-30), round(x-40)] = True
    observer.things[2] = other
    result = _attach(scene)
    assert result.binding is None and result.reason == 'ambiguous_material_ownership'


@pytest.mark.parametrize('change', ['birth', 'observer', 'basis', 'stale', 'scale', 'source', 'removed'])
def test_an_attachment_does_not_survive_a_broken_track_or_coordinate_frame(scene, change):
    _, _, _, _, body, observer = scene
    binding = _attach(scene).binding
    source = 'source'
    if change == 'birth':
        body.born = 2.
    elif change == 'observer':
        observer = SimpleNamespace(things=observer.things, scale=1.)
    elif change == 'basis':
        body.point_pose.reference_at = 3.9
    elif change == 'stale':
        body.seen = 4.1
    elif change == 'scale':
        observer.scale = .5
    elif change == 'source':
        source = 'changed'
    else:
        observer.things.clear()
    assert binding.current(observer, 4., source=source) is None


def test_a_visual_reclassification_does_not_erase_a_continuing_material_reference(scene):
    _, _, _, _, body, observer = scene
    binding = _attach(scene).binding
    body.kind = 1
    assert binding.current(observer, 4., source='source').position == pytest.approx((52., 38.), abs=.05)
    body.point_pose.reference_at = 3.9
    assert binding.current(observer, 4., source='source') is None


def test_an_extrapolated_source_origin_requires_enough_precision(scene):
    result = _attach(scene, (10000., 10000.))
    assert result.binding is None and result.reason == 'reference_point_precision'


@pytest.mark.parametrize('at', [None, True, '4', float('nan'), float('inf')])
def test_unmeasured_times_cannot_refresh_an_attachment(scene, at):
    binding = _attach(scene).binding
    assert binding.current(scene[-1], at, source='source') is None
    assert binding.inspect(scene[-1], at, source='source').reason == 'capture_not_measured'


def test_a_current_looking_pose_cannot_borrow_an_older_picture_receipt(scene):
    _, _, _, _, body, observer = scene
    binding = _attach(scene).binding
    body.seen = body.point_pose.at = 4.1
    assert binding.inspect(observer, 4.1, source='source').reason == 'capture_not_observed'


@pytest.mark.parametrize('changes', [
    {'source_points': ()}, {'linear': ((1.,),)}, {'offset': (0.,)}, {'observed_points': ((0., 0.),)},
    {'error': float('nan')}, {'source_points': ((1., 2.),)*6, 'observed_points': ((1., 2.),)*6},
])
def test_malformed_correspondence_receipts_refuse_before_attachment(scene, changes):
    coordinates, appearance, picture, reference, body, observer = scene
    bad = coordinates, replace(appearance, **changes), picture, reference, body, observer
    result = _attach(bad)
    assert result.binding is None and result.reason == 'invalid_correspondence'


def test_the_registered_claim_executes_the_current_basis_check():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    claim = next(test for test in suite.tests() if test.name == 'source_points_need_their_current_material_basis')
    assert claim.predict(None) is True

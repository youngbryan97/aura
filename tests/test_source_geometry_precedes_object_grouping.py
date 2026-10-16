"""Named pixels can carry source coordinates without declaring a controlled object."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from core.perception.a_material_frame_kept_for_its_owner import keep_material_frame
from core.perception.a_source_point_on_visible_marks import attach_source_point_to_visible_marks
from core.perception.a_visual_reference_in_the_picture import _digest, reference_appearances
from core.perception.coordinates_in_a_visual_reference import SourceImageCoordinates
from core.perception.named_material_in_a_consumed_frame import MaterialMarks, _named_material_capture_invariant
from core.perception.pixels_consumed_by_the_observer import consumed_pixels
from core.perception.points_that_keep_their_motion import SeenPoint
from core.perception.the_pixel_grid_a_mark_uses import SampleGrid

pytestmark = pytest.mark.unit


@pytest.fixture
def scene():
    reference = np.random.default_rng(719).integers(10, 230, (64, 64, 4), dtype=np.uint8)
    reference[..., 3] = 255
    picture = np.zeros((128, 160, 3), np.uint8)
    picture[30:94, 40:104] = reference[..., :3]
    pose, = reference_appearances(picture, reference, scales=(1.,))
    coordinates = SourceImageCoordinates('source', _digest(reference), '0'*64, (64, 64),
                                          ((1., 0.), (0., 1.)), (12., 8.), .01)
    places = ((48., 38.), (72., 38.), (94., 38.), (48., 62.), (94., 62.), (48., 84.), (72., 84.), (94., 84.))
    points = tuple(SeenPoint(n, x, y, x, y, 3.9, 3.9, 4., .001, 2) for n, (x, y) in enumerate(places))
    receipt = consumed_pixels(picture, 4.)
    marks = MaterialMarks('1'*32, 20, receipt, 1., 1., points)
    observer = SimpleNamespace(material_marks=marks, pixels_consumed=receipt, scale=1., things={})
    return coordinates, pose, picture, reference, observer


def attach(scene, observer=None):
    coordinates, pose, picture, reference, original = scene
    return attach_source_point_to_visible_marks(coordinates, (0., 0.), pose, picture, reference,
        original if observer is None else observer, 4., source='source')


def moved(scene, *, missing=0, sequence=21, steps=1):
    observer = scene[-1]
    picture = np.roll(scene[2], (2, 3), axis=(0, 1))
    points = tuple(replace(p, x=p.x+3., y=p.y+2., old_x=p.x, old_y=p.y,
                           before=4., at=4.1, pictures=p.pictures+steps)
                   for p in observer.material_marks.points[missing:])
    receipt = consumed_pixels(picture, 4.1)
    observer.material_marks = replace(observer.material_marks, sequence=sequence, pixels=receipt, points=points)
    observer.pixels_consumed = receipt
    return observer


def test_reference_geometry_exists_before_any_object_was_grouped(scene):
    result = attach(scene)
    assert result.binding is not None and result.support == 8 and result.reason == 'qualified'
    point = result.binding.current(scene[-1], 4., source='source')
    assert point.position == pytest.approx((52., 38.), abs=.05) and point.error < 1.
    assert not scene[-1].things
    assert not any(hasattr(point, key) for key in ('number', 'kind', 'controlled', 'role'))


def test_original_pixel_identities_carry_a_source_origin_without_object_ids(scene):
    binding = attach(scene).binding
    observer = moved(scene)
    current = binding.current(observer, 4.1, source='source')
    assert current.position == pytest.approx((55., 40.), abs=.05) and current.support == 8
    assert observer.things == {}


def test_missing_marks_cannot_be_replaced_by_new_corners_at_the_same_positions(scene):
    binding = attach(scene).binding
    observer = moved(scene)
    observer.material_marks = replace(observer.material_marks,
        points=tuple(replace(p, number=p.number+100, born=4.1, before=4.1, old_x=p.x, old_y=p.y, pictures=1)
                     for p in observer.material_marks.points))
    assert binding.inspect(observer, 4.1, source='source').reason == 'original_marks_missing'


@pytest.mark.parametrize('damage,reason', [('epoch', 'marks_epoch_changed'), ('sequence', 'original_marks_missing'),
    ('partial', 'original_marks_missing'), ('stale', 'marks_not_current'), ('scale', 'marks_sampling_changed'),
    ('pixels', 'marks_not_current'), ('source', 'source_changed')])
def test_broken_pixel_continuity_refuses_geometry(scene, damage, reason):
    binding = attach(scene).binding
    observer = moved(scene, missing=3 if damage == 'partial' else 0, sequence=22 if damage == 'sequence' else 21)
    source = 'source'
    if damage == 'epoch':
        observer.material_marks = replace(observer.material_marks, epoch='2'*32)
    elif damage == 'stale':
        observer.pixels_consumed = consumed_pixels(scene[2], 4.)
    elif damage == 'scale':
        observer.material_marks = replace(observer.material_marks, scale=2., pixel_size=2.,
            grid=SampleGrid((128, 160), (128, 160), (256, 320)))
    elif damage == 'pixels':
        observer.pixels_consumed = None
    elif damage == 'source':
        source = 'another source'
    assert binding.inspect(observer, 4.1, source=source).reason == reason


def test_a_retained_majority_survives_partial_cover_without_inventing_new_marks(scene):
    binding = attach(scene).binding
    observer = moved(scene, missing=2)
    current = binding.current(observer, 4.1, source='source')
    assert current.support == 6 and current.position == pytest.approx((55., 40.), abs=.05)


def test_a_rigid_triangle_can_carry_geometry_but_cannot_claim_affine_deformation(scene):
    observer = scene[-1]
    observer.material_marks = replace(observer.material_marks,
        points=tuple(observer.material_marks.points[n] for n in (0, 2, 5)))
    result = attach(scene)
    assert result.binding is not None and result.support == 3
    observer = moved(scene)
    assert result.binding.current(observer, 4.1, source='source').position == pytest.approx((55., 40.), abs=.05)
    observer.material_marks = replace(observer.material_marks,
        points=tuple(replace(p, x=p.x+.3*p.y) for p in observer.material_marks.points))
    assert result.binding.inspect(observer, 4.1, source='source').reason == 'marks_do_not_share_a_frame'


def test_two_marks_cannot_determine_a_source_frame_with_redundant_support(scene):
    observer = scene[-1]
    observer.material_marks = replace(observer.material_marks, points=observer.material_marks.points[:2])
    assert attach(scene).reason == 'reference_marks_missing'


def test_the_saved_capture_keeps_the_same_pixel_stream_after_ordinary_state_advances(scene):
    observer = scene[-1]
    captured = keep_material_frame(observer, scene[2], 4.)
    assert captured.material_marks is observer.material_marks
    moved(scene)
    result = attach(scene, captured)
    assert result.binding.current(observer, 4.1, source='source').position == pytest.approx((55., 40.), abs=.05)


def test_patch_colours_must_be_supported_even_when_an_outline_would_overlap(scene):
    coordinates, pose, picture, reference, observer = scene
    wrong = np.full_like(picture, 255)
    receipt = consumed_pixels(wrong, 4.)
    observer.material_marks = replace(observer.material_marks, pixels=receipt)
    observer.pixels_consumed = receipt
    # A supplied proposal still has to prove every attached named patch.
    proposal = replace(pose, picture_sha256=_digest(wrong))
    result = attach_source_point_to_visible_marks(coordinates, (0., 0.), proposal, wrong, reference,
                                                  observer, 4., source='source')
    assert result.binding is None and result.reason == 'reference_marks_missing'


def test_an_unverified_reference_fit_cannot_supply_pixel_geometry(scene):
    coordinates, pose, picture, reference, observer = scene
    result = attach_source_point_to_visible_marks(coordinates, (0., 0.), replace(pose, source_points=()),
        picture, reference, observer, 4., source='source')
    assert result.binding is None and result.reason == 'invalid_correspondence'


def test_mark_receipts_cannot_contain_points_from_a_different_capture(scene):
    assert _named_material_capture_invariant() == ()
    marks = scene[-1].material_marks
    with pytest.raises(ValueError):
        replace(marks, points=(replace(marks.points[0], at=3.9),))
    with pytest.raises(ValueError):
        replace(marks, points=(marks.points[0], marks.points[0]))


def test_ordinary_observation_publishes_material_before_its_first_object_grouping():
    from core.perception.what_moves_in_the_picture import WhatMoves

    picture = np.random.default_rng(17).integers(0, 255, (96, 128, 3), np.uint8)
    observer = WhatMoves()
    observer.see(picture, 1.)
    assert observer.material_marks is not None and observer.material_marks.points and not observer.things
    first = observer.material_marks
    observer.see(picture, 1.1)
    second = observer.material_marks
    assert second.sequence == first.sequence+1 and second.epoch == first.epoch
    assert second.pixels == observer.pixels_consumed and first.pixels.at == 1.
    observer._start_again()
    assert observer.material_marks is None and observer._material_mark_epoch != first.epoch

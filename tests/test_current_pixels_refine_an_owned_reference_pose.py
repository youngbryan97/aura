"""A current photometric correction retains measured patch uncertainty."""
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from core.perception.a_reference_pose_refined_in_current_pixels import (
    _pixel_pose_refinement_invariant, refine_reference_pose,
)
from core.perception.a_visual_reference_in_the_picture import ReferenceAppearance, _digest
from core.perception.material_samples_in_a_reference_fit import material_samples

pytestmark = pytest.mark.unit


@pytest.fixture
def view():
    reference = np.random.default_rng(843).integers(15, 235, (64, 64, 4), dtype=np.uint8)
    reference[..., 3] = 255
    picture = np.zeros((240, 320, 3), np.uint8)
    picture[40:168, 60:188] = np.asarray(Image.fromarray(reference[..., :3]).resize((128, 128), Image.Resampling.NEAREST))
    points = np.array(((8., 8.), (52., 8.), (8., 52.), (52., 52.), (18., 34.), (40., 30.), (28., 18.), (20., 46.)))
    observed = points*2+(60.5, 40.5)
    linear, offset = np.array(((1.99, .005), (.002, 2.01))), np.array((60.9, 39.8))
    error = float(np.linalg.norm(points @ linear.T+offset-observed, axis=1).max())
    pose = ReferenceAppearance(_digest(reference), _digest(picture), tuple(map(tuple, linear)), tuple(offset),
        (60., 39., 129., 130.), tuple(map(tuple, points)), tuple(map(tuple, observed)), error)
    assert material_samples(pose) is not None
    return picture, reference, pose


def test_the_current_picture_corrects_pose_and_keeps_original_patch_observations(view):
    picture, reference, pose = view
    result = refine_reference_pose(picture, reference, pose)
    assert result.pose is not None, result.receipt()
    refined = result.pose
    assert result.heldout_after < result.heldout_before and result.raw_heldout_after < result.raw_heldout_before
    assert result.maximum_shift <= 4. and result.evaluations <= 24
    assert refined.source_points == pose.source_points and refined.observed_points == pose.observed_points
    residual = np.linalg.norm(np.asarray(refined.source_points) @ np.asarray(refined.linear).T+refined.offset
                              -np.asarray(refined.observed_points), axis=1)
    assert refined.error == pytest.approx(float(residual.max()), abs=1e-12)
    assert material_samples(refined) is not None
    assert refined.picture_sha256 == _digest(picture) and refined.reference_sha256 == _digest(reference)
    assert np.linalg.norm(np.asarray(refined.project((32., 32.)))-(124.5, 104.5)) < .5
    assert not any(key in result.receipt() for key in ('body', 'role', 'contact_verified', 'pose'))


@pytest.mark.parametrize('damage', ['picture', 'reference', 'witness', 'uniform', 'huge-scale'])
def test_missing_owned_evidence_cannot_supply_a_photometric_correction(view, damage):
    picture, reference, pose = view
    if damage == 'picture':
        picture = np.zeros_like(picture)
    elif damage == 'reference':
        reference = np.zeros_like(reference)
    elif damage == 'witness':
        pose = replace(pose, observed_points=pose.observed_points[:5])
    elif damage == 'uniform':
        reference = np.full_like(reference, 255)
        pose = replace(pose, reference_sha256=_digest(reference))
    else:
        linear = ((100., 0.), (0., 100.))
        observed = tuple(tuple(np.asarray(p)*100+pose.offset) for p in pose.source_points)
        pose = replace(pose, linear=linear, observed_points=observed, error=0.)
    result = refine_reference_pose(picture, reference, pose)
    assert result.pose is None
    assert result.reason in ('current_reference_correspondence_unavailable', 'photometric_texture_unavailable',
                             'photometric_scale_unavailable')


def test_solver_exhaustion_is_unknown_and_cannot_be_scored_as_a_better_pose(view):
    result = refine_reference_pose(*view, max_evaluations=1)
    assert result.pose is None and result.reason == 'photometric_fit_unresolved'
    assert result.heldout_after is None and result.evaluations == 1


def test_the_refinement_records_real_uncertainty_instead_of_preserving_an_old_smaller_number(view):
    picture, reference, pose = view
    # Noisy patch measurements can fit a coarse transform better than the
    # whole current picture. The new residual is retained even when larger.
    observed = tuple(pose.project(p) for p in pose.source_points)
    exact_coarse = replace(pose, observed_points=observed, error=0.)
    result = refine_reference_pose(picture, reference, exact_coarse)
    assert result.pose is not None, result.receipt()
    assert result.pose.error > .1 and result.patch_residual == result.pose.error
    assert result.pose.observed_points == observed and material_samples(result.pose) is not None


def test_training_tiles_cannot_move_a_pose_against_its_heldout_visible_pixels(view):
    picture, reference, original = view
    picture = picture.copy()
    region = picture[40:168, 60:188].copy()
    yy, xx = np.mgrid[:128, :128]
    training = ((xx//8+yy//8) % 2) == 0
    displaced = np.roll(region, 2, axis=1)
    picture[40:168, 60:188][training] = displaced[training]
    pose = replace(original, linear=((2., 0.), (0., 2.)), offset=(60.5, 40.5), error=0.,
        picture_sha256=_digest(picture))
    result = refine_reference_pose(picture, reference, pose)
    assert result.pose is None and result.reason == 'photometric_holdout_did_not_improve', result.receipt()
    assert result.raw_heldout_before == pytest.approx(0., abs=1e-10)
    assert result.raw_heldout_after >= result.raw_heldout_before


@pytest.mark.parametrize('budget', [0, 31, True, None])
def test_an_unbounded_solver_request_is_refused(view, budget):
    with pytest.raises(ValueError, match='bounded'):
        refine_reference_pose(*view, max_evaluations=budget)


def test_a_refinement_boundary_has_an_executable_measurement():
    assert _pixel_pose_refinement_invariant() == ()

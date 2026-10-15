"""References propose current appearances, without creating task roles or body identity."""
import math

import numpy as np
import pytest
from PIL import Image

from core.perception.a_visual_reference_in_the_picture import (
    ReferenceAppearance,
    _reference_appearance_invariant,
    reference_appearances,
)

pytestmark = pytest.mark.unit


def _reference(seed=17):
    rng = np.random.default_rng(seed)
    reference = rng.integers(0, 255, (80, 88, 4), dtype=np.uint8)
    reference[..., 3] = 0
    reference[6:-6, 6:-6, 3] = 255
    return reference


def _paste(picture, reference, x, y, scale=1.):
    size = (round(reference.shape[1]*scale), round(reference.shape[0]*scale))
    rgba = np.asarray(Image.fromarray(reference).resize(size, Image.Resampling.BOX))
    alpha = rgba[..., 3:4].astype(float)/255
    part = picture[y:y+size[1], x:x+size[0]]
    part[:] = np.round(rgba[..., :3]*alpha+part*(1-alpha)).astype(np.uint8)
    return x+size[0]/2, y+size[1]/2


@pytest.mark.parametrize('scale', [.75, 1., 1.25])
def test_named_material_can_be_found_at_a_different_scale_and_location(scale):
    reference = _reference()
    picture = np.full((240, 360, 3), 30, np.uint8)
    centre = _paste(picture, reference, 90, 80, scale)
    poses = reference_appearances(picture, reference, scales=(scale,))
    assert len(poses) == 1
    pose = poses[0]
    assert math.dist(pose.project((44., 40.)), centre) < 1.
    assert len(pose.source_points) == len(pose.observed_points) >= 6
    projected = np.asarray([pose.project(point) for point in pose.source_points])
    assert np.max(np.linalg.norm(projected-pose.observed_points, axis=1)) <= pose.error+1e-5
    assert pose.reference_sha256 != pose.picture_sha256
    assert not hasattr(pose, 'controlled') and not hasattr(pose, 'role')


def test_two_visible_copies_remain_two_competing_referents():
    reference = _reference()
    picture = np.full((240, 360, 3), 30, np.uint8)
    expected = [_paste(picture, reference, 30, 70), _paste(picture, reference, 230, 70)]
    poses = reference_appearances(picture, reference, scales=(1.,))
    assert len(poses) == 2
    assert all(min(math.dist(pose.project((44., 40.)), point) for pose in poses) < 1. for point in expected)


@pytest.mark.parametrize('seed', range(5))
def test_an_unrelated_picture_cannot_be_named_from_a_reference(seed):
    picture = np.random.default_rng(seed+100).integers(0, 255, (160, 240, 3), dtype=np.uint8)
    assert not reference_appearances(picture, _reference(), scales=(.75, 1., 1.25))


@pytest.mark.parametrize('change', ['transparent', 'flat', 'bad_dtype', 'nan_scale', 'many_scales', 'huge_edge'])
def test_missing_support_and_unbounded_search_remain_unmeasured(change):
    reference = _reference()
    picture = np.full((160, 240, 3), 30, np.uint8)
    _paste(picture, reference, 60, 40)
    scales, edge = (1.,), 480
    if change == 'transparent':
        reference[..., 3] = 0
    elif change == 'flat':
        reference[..., :3] = 30
    elif change == 'bad_dtype':
        reference = reference.astype(float)
    elif change == 'nan_scale':
        scales = (math.nan,)
    elif change == 'many_scales':
        scales = (1.,)*33
    else:
        edge = 10000
    assert reference_appearances(picture, reference, scales=scales, edge=edge) == ()


def test_reference_contract_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _reference_appearance_invariant() == ()
    assert ReferenceAppearance.__dataclass_fields__['picture_sha256']
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'reference_appearance_is_not_temporal_identity')
    assert check.predict(None) is True

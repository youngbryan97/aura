"""Previous geometry proposes places; current material must qualify there again."""
import math
from dataclasses import replace

import numpy as np
import pytest

from core.perception.a_visual_reference_in_the_picture import _digest, reference_appearances
from core.perception.reference_material_near_a_measured_pose import (
    _near_reference_material_invariant, near_reference_appearances,
)


def _scene():
    ref = np.random.default_rng(107).integers(0, 255, (40, 48, 4), dtype=np.uint8)
    ref[..., 3] = 255
    picture = np.zeros((120, 180, 3), np.uint8)
    picture[20:60, 30:78] = ref[..., :3]
    poses = reference_appearances(picture, ref, scales=(1.,))
    assert len(poses) == 1
    return ref, picture, poses


def test_a_previous_pose_locates_material_again_in_fresh_captured_bytes():
    ref, picture, poses = _scene()
    current = np.zeros_like(picture)
    current[22:62, 33:81] = ref[..., :3]
    found = near_reference_appearances(current, ref, poses)
    assert len(found) == 1 and len(found[0].source_points) >= 6
    assert math.dist(found[0].project((11., 13.)), (44., 35.)) < .3
    assert found[0].picture_sha256 == _digest(current) != poses[0].picture_sha256
    assert found[0].reference_sha256 == poses[0].reference_sha256


def test_motion_beyond_the_local_window_still_requires_and_can_find_current_material():
    ref, picture, poses = _scene()
    current = np.zeros_like(picture)
    current[65:105, 112:160] = ref[..., :3]
    found = near_reference_appearances(current, ref, poses)
    assert len(found) == 1 and len(found[0].source_points) >= 6
    assert math.dist(found[0].project((11., 13.)), (123., 78.)) < .3
    assert found[0].picture_sha256 == _digest(current) != poses[0].picture_sha256
    assert not {'number', 'born', 'controlled', 'role'} & vars(found[0]).keys()


def test_a_wider_search_keeps_two_distant_current_copies_separate():
    ref, _, poses = _scene()
    current = np.zeros((180, 300, 3), np.uint8)
    current[60:100, 140:188] = ref[..., :3]
    current[120:160, 220:268] = ref[..., :3]
    found = near_reference_appearances(current, ref, poses)
    assert len(found) == 2
    expected = [(151., 73.), (231., 133.)]
    for point in expected:
        assert min(math.dist(candidate.project((11., 13.)), point) for candidate in found) < .3


def test_the_previous_pose_cannot_publish_material_that_disappeared():
    ref, picture, poses = _scene()
    assert near_reference_appearances(np.zeros_like(picture), ref, poses) == ()
    assert _near_reference_material_invariant() == ()


def test_different_reference_bytes_cannot_inherit_the_old_proposal():
    ref, picture, poses = _scene()
    changed = ref.copy()
    changed[0, 0, 0] ^= 255
    assert near_reference_appearances(picture, changed, poses) == ()


def test_two_proposals_keep_separate_current_copies_without_counting_duplicate_poses():
    ref, picture, _ = _scene()
    picture[70:110, 110:158] = ref[..., :3]
    poses = reference_appearances(picture, ref, scales=(1.,))
    assert len(poses) == 2
    found = near_reference_appearances(picture, ref, (*poses, poses[0]))
    assert len(found) == 2
    assert all(len(pose.source_points) >= 6 for pose in found)


@pytest.mark.parametrize('change', [
    dict(linear=((0., 0.), (0., 0.))), dict(linear=((-1., 0.), (0., 1.))),
    dict(linear=((float('nan'), 0.), (0., 1.))), dict(offset=(float('inf'), 0.)),
    dict(offset=(9000., 0.)),
])
def test_malformed_or_unbounded_geometry_has_no_material_authority(change):
    ref, picture, poses = _scene()
    assert near_reference_appearances(picture, ref, (replace(poses[0], **change),)) == ()


def test_a_bounded_proposal_bank_refuses_extra_work():
    ref, picture, poses = _scene()
    assert near_reference_appearances(picture, ref, poses*5) == ()
    assert near_reference_appearances(picture, ref, list(poses)) == ()


def test_a_failed_local_search_keeps_the_wider_fine_work_bounded(monkeypatch):
    from core.perception import reference_material_near_a_measured_pose as module

    ref, picture, poses = _scene()
    attempts = []
    monkeypatch.setattr(module, '_supported_fit', lambda *args: attempts.append(args[2:]) or None)
    assert near_reference_appearances(np.zeros_like(picture), ref, poses*4) == ()
    assert 0 < len(attempts) <= 24


@pytest.mark.parametrize('shape', [(16, 4096), (4096, 16)])
def test_extreme_capture_aspect_cannot_round_a_resampled_dimension_to_zero(shape):
    ref, _, poses = _scene()
    picture = np.zeros((*shape, 3), np.uint8)
    assert reference_appearances(picture, ref, scales=(1.,), edge=64) == ()
    assert near_reference_appearances(picture, ref, poses, edge=64) == ()

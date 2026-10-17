"""Visible named material keeps its origin while an object pauses."""
from dataclasses import replace
import numpy as np
import pytest

from core.perception.what_moves_in_the_picture import WhatMoves

pytestmark = pytest.mark.unit


def _moving_texture():
    background = np.full((120, 160, 3), 25, np.uint8)
    texture = np.random.default_rng(918).integers(35, 240, (30, 24, 3), dtype=np.uint8)
    eyes = WhatMoves(width=160)
    for step in range(8):
        eyes.see(background, step*.1)
    picture = None
    for offset, at in ((0, 1.), (2, 1.1), (4, 1.2)):
        picture = background.copy()
        picture[35:65, 40+offset:64+offset] = texture
        eyes.see(picture, at)
    body, = eyes.things.values()
    assert body.motion_position is not None and body.point_pose is not None
    return eyes, body, picture


def test_a_measured_point_origin_survives_a_still_visible_unmatched_outline():
    eyes, body, picture = _moving_texture()
    origin, reference, before = body.motion_position, body.point_pose.reference_at, tuple(body.motion_path)
    eyes._blobs = lambda *_args: []
    eyes.see(picture, 1.3)
    assert body.seen == 1.3 and body.still
    assert body.motion_measured_at == 1.3 and body.motion_position == pytest.approx(origin, abs=.1)
    assert body.point_pose is not None and body.point_pose.reference_at == reference
    assert tuple(body.motion_path)[:len(before)] == before


@pytest.mark.parametrize('damage', ['missing', 'time', 'support', 'movement'])
def test_a_still_box_without_current_agreeing_point_support_cannot_keep_the_old_path(monkeypatch, damage):
    eyes, body, picture = _moving_texture()
    original = eyes.points.motion_of
    def motion(thing, at):
        value = original(thing, at)
        assert value is not None
        if damage == 'missing':
            return None
        return replace(value, **{'time': dict(before=1.), 'support': dict(support=2),
                                 'movement': dict(dx=8.)}[damage])
    monkeypatch.setattr(eyes.points, 'motion_of', motion)
    eyes._blobs = lambda *_args: []
    eyes.see(picture, 1.3)
    assert body.motion_measured_at is None and body.motion_position is None


def test_a_projected_box_does_not_move_the_patch_used_to_continue_a_still_origin():
    eyes, body, picture = _moving_texture()
    origin = body.motion_position
    body.x += 80.
    eyes._blobs = lambda *_args: []
    eyes.see(picture, 1.3)
    assert body.motion_measured_at == 1.3 and body.motion_position == pytest.approx(origin, abs=.1)
    assert body.x == 56.


def test_resting_material_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite
    from core.perception.a_reference_kept_by_visible_support import _resting_material_invariant

    assert _resting_material_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(test for test in suite.tests() if test.name == 'resting_material_keeps_a_measured_origin')
    assert check.predict(None) is True and any(claim.test == check.name for claim in suite.claims())

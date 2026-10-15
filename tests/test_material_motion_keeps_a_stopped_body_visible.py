"""Changing an outline cannot erase the material that answered independent inputs."""
from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from core.agency.which_one_answers_to_her import WhichIsHers
from core.perception.position_from_the_same_points import PointPose
from core.perception.what_moves_in_the_picture import WhatMoves, _behind_it

pytestmark = pytest.mark.unit


def test_outline_motion_cannot_supply_travel_when_the_material_stayed_in_place():
    from types import SimpleNamespace

    pose = PointPose(0., 1., (40., 40.), {}, (40., 40.))
    body = SimpleNamespace(path=deque(((.5, 40., 40.), (1., 40., 64.))),
                           motion_path=deque(((.5, 40., 40.), (1., 40., 40.))),
                           point_pose=pose, seen=1., born=0., motion_measured_at=1.,
                           motion_position=(40., 40.), steady=(12., 12.), least_dims=(12., 12.),
                           box=lambda: (34., 34., 46., 82.))
    area = slice(30, 90), slice(30, 50)
    assert _behind_it(body, area, (60, 20)) is None
    body.point_pose = None
    assert _behind_it(body, area, (60, 20)) is not None


@pytest.mark.parametrize('seed', [347, 348])
@pytest.mark.parametrize('gap', [.1, .268130064])
def test_the_actual_pixel_path_finds_and_keeps_the_responder_as_its_outline_changes(seed, gap):
    eyes, hers = WhatMoves(width=240), WhichIsHers()
    tile = np.random.default_rng(seed).integers(90, 240, (24, 24, 3), dtype=np.uint8)
    for i in range(8):
        eyes.see(np.full((180, 240, 3), 25, np.uint8), 1.+i*.1)
    at, x, body_attributions = 1.8, 100., []
    for interval, key in enumerate(('', 'A', '', 'B', '', 'A', '', 'B', '', 'A', '', 'B')):
        hers.holding(key, at, trying=True)
        for step in range(10):
            x += {'A': 4., 'B': -4.}.get(key, 0.)
            at += gap
            image = np.full((180, 240, 3), 25, np.uint8)
            left = round(x)-12
            image[104:128, left:left+24] = tile
            if (interval*10+step) % 8 < 4:
                image[56:104, round(x)-2:round(x)+2] = 120
            happened = eyes.see(image, at)
            hers.saw(eyes, happened, at)
            if interval >= 8:
                assert hers.number is not None, 'the last four control intervals must keep the measured body'
            if hers.number is not None:
                mine = eyes.things.get(hers.number)
                assert mine is not None and mine.seen == at
                assert mine.x == pytest.approx(x, abs=.5)
                body_attributions.append(hers.number)
    assert len(body_attributions) >= 40
    assert len(set(body_attributions)) == 1 and hers.identification.epoch == 0
    receipt = next(r for r in hers.identification.receipts if r.get('reason') == 'independent repeated trials')
    assert sum(count >= 2 for count in receipt['independent_trials'].values()) >= 2
    assert receipt['statistic'] > 20. and receipt['effect'] > 20.


def test_material_travel_has_a_registered_validation():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'trail_cleanup_uses_current_material_travel')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())

"""Separated foreground pieces retain identity only through current owned material."""
from dataclasses import replace

import numpy as np
import pytest

from core.perception.a_material_observation_from_its_parts import material_observations_from_parts
from tests.test_continuing_material_survives_a_changed_outline import scene

pytestmark = pytest.mark.unit


def _parts():
    eyes, body, whole, motion = scene()
    image = np.full((80, 100, 3), 25, np.uint8)
    blobs = []
    for top, bottom in ((27, 30), (33, 36)):
        image[top:bottom, 17:29] = (170, 80, 80)
        blobs.append(dict(whole, x=23., y=(top+bottom)/2, w=12., h=float(bottom-top),
                          support=np.ones((bottom-top, 12), bool), patch=image[top:bottom, 17:29].copy()))
    return eyes, body, image, blobs, motion


def test_disjoint_pieces_share_a_body_only_through_the_measured_material():
    eyes, body, image, blobs, motion = _parts()
    assert eyes._match(blobs, 1.1)[0] == {}
    joined = material_observations_from_parts(image, blobs, eyes.things, {body.number: motion}, 1.1,
                                              outline_cost=eyes._cost)
    assert len(joined) == 1 and joined[0]['support'].sum() == 72
    assert not joined[0]['support'][3:6].any()  # visible gaps remain gaps
    assert eyes._match(joined, 1.1) == ({body.number: 0}, [])
    eyes._moved(body, joined[0], 1.1)
    assert body.born == .5 and body.point_pose.reference_at == .8
    assert body.motion_position == pytest.approx((23., 20.))


@pytest.mark.parametrize('departure', ['absent', 'stale', 'unsupported', 'one_per_piece', 'rival', 'two_owners',
                                    'deformed', 'spent_precision', 'unseen_names'])
def test_unknown_or_competing_material_cannot_merge_visible_objects(departure):
    eyes, body, image, blobs, motion = _parts()
    motions = {body.number: motion}
    if departure == 'absent':
        motions.clear()
    elif departure == 'stale':
        motions[body.number] = replace(motion, at=1.)
    elif departure == 'unsupported':
        blobs[0]['support'][:] = False
    elif departure == 'one_per_piece':
        motions[body.number] = replace(motion, support=3, points=(0, 1, 2), locations=((19., 28.), (27., 28.), (19., 34.)))
    elif departure == 'rival':
        rival = eyes._born(dict(blobs[0]), 1.05)
        eyes.things[rival.number] = rival
    elif departure == 'two_owners':
        from copy import deepcopy

        rival = deepcopy(body)
        rival.number = body.number+100
        eyes.things[rival.number] = rival
        motions[rival.number] = motion
    elif departure == 'deformed':
        motions[body.number] = replace(motion, locations=((18., 27.), (27., 27.), (18., 35.), (30., 35.)))
    elif departure == 'spent_precision':
        body.point_pose.reference_error = .999
    else:
        motions[body.number] = replace(motion, points=(100, 101, 102, 103))
    joined = material_observations_from_parts(image, blobs, eyes.things, motions, 1.1, outline_cost=eyes._cost)
    assert len(joined) == len(blobs) and all(a is b for a, b in zip(joined, blobs, strict=True))


def test_the_material_parts_boundary_has_a_registered_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite
    from core.perception.a_material_observation_from_its_parts import _material_parts_invariant

    assert _material_parts_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'material_parts_need_one_current_owner')
    assert check.predict(None) and any(c.test == check.name for c in suite.claims())

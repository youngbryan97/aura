"""A current-frame refit has one fine position; later visits search the rest."""
from dataclasses import replace

import numpy as np
import pytest

from core.perception import reference_material_near_a_measured_pose as near
from core.perception import source_geometry_in_current_pixels as geometry
from core.perception.a_source_point_on_visible_marks import VisibleSourcePoint
from tests.test_previous_pose_only_proposes_current_reference_pixels import _scene
from tests.test_fresh_source_geometry_reaches_ordinary_observation import scene, _state

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('position', range(6))
def test_one_position_cannot_enter_more_than_one_fine_solver(monkeypatch, position):
    ref, picture, poses = _scene()
    attempts, phases = [], []

    def candidates(*args, whole_picture=False):
        phases.append(whole_picture)
        return ((10, 12), (20, 22), (30, 32))

    monkeypatch.setattr(near, '_near_places', candidates)
    monkeypatch.setattr(near, '_supported_fit', lambda *args: attempts.append(args[2:]) or None)
    assert near.near_reference_appearances(picture, ref, poses, position=position) == ()
    assert attempts == [((10, 12), (20, 22), (30, 32))[position % 3]]
    assert phases == [bool(position // 3)]


@pytest.mark.parametrize('position', [-1, 6, True, 1.5, float('nan')])
def test_invalid_position_budget_cannot_start_a_fine_fit(monkeypatch, position):
    ref, picture, poses = _scene()
    monkeypatch.setattr(near, '_supported_fit', lambda *args: pytest.fail('invalid budget'))
    assert near.near_reference_appearances(picture, ref, poses, position=position) == ()


def test_a_budgeted_call_refuses_multiple_poses_instead_of_multiplying_the_work(monkeypatch):
    ref, picture, poses = _scene()
    monkeypatch.setattr(near, '_supported_fit', lambda *args: pytest.fail('multiple budgeted poses'))
    assert near.near_reference_appearances(picture, ref, poses*2, position=0) == ()


def test_a_wide_visit_still_finds_distant_current_pixels_and_rejects_absence():
    ref, picture, poses = _scene()
    current = np.zeros_like(picture)
    current[65:105, 112:160] = ref[..., :3]
    found = near.near_reference_appearances(current, ref, poses, position=3)
    assert len(found) == 1 and found[0].project((11., 13.)) == pytest.approx((123., 78.), abs=.3)
    assert near.near_reference_appearances(current*0, ref, poses, position=3) == ()


def test_missing_fits_visit_each_local_and_wide_position_for_every_pose(scene, monkeypatch):
    proposal, table, _, current = scene
    pose = proposal.appearances[0]
    poses = tuple(replace(pose, offset=(pose.offset[0]+100*n, pose.offset[1])) for n in range(4))
    proposal = replace(proposal, appearances=poses)
    calls = []

    def refit(_picture, _reference, candidates, *, position):
        calls.append((candidates[0].offset, position))
        return ()

    monkeypatch.setattr(geometry, 'near_reference_appearances', refit)
    worker = geometry.SourceGeometryInCurrentPixels()
    for n in range(24):
        at = 2.+n*.1
        frames, unavailable = worker.measure(current, at, _state(current, at), (proposal,), (table,),
                                             one_position=True)
        assert not frames and 'fresh_reference_positions_unsearched' in unavailable
    assert calls == [(pose.offset, position) for position in range(6) for pose in poses]
    worker.measure(current, 5., _state(current, 5.), (), ())
    assert not worker.search_cursors


def test_only_a_precision_qualified_fit_returns_its_pose_to_the_first_local_peak(scene, monkeypatch):
    proposal, table, _, current = scene
    calls, qualified = [], [False]

    def refit(_picture, _reference, candidates, *, position):
        calls.append(position)
        return candidates

    def point(_reference, _pose, _picture, at, _grid):
        return VisibleSourcePoint(at, (30., 30.), .1, 6) if qualified[0] else None

    monkeypatch.setattr(geometry, 'near_reference_appearances', refit)
    monkeypatch.setattr(geometry, '_current_point', point)
    worker = geometry.SourceGeometryInCurrentPixels()
    for n in range(5):
        at = 2.+n*.1
        qualified[0] = n >= 2
        frames, _ = worker.measure(current, at, _state(current, at), (proposal,), (table,), one_position=True)
        assert bool(frames) == qualified[0]
        if frames:
            assert frames[0].point.at == at and not hasattr(frames[0], 'controlled')
    assert calls == [0, 1, 2, 0, 0]


def test_one_position_refitting_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert near._single_position_refit_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'optional_reference_refits_one_position')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())

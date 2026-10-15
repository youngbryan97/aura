"""Learned correspondences feed actual point support without renewing hidden motion."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.perception.learned_tracking import points_for_the_motion_eyes as bridging
from core.perception.learned_tracking.contracts import PointResult, PointTrack
from core.perception.learned_tracking.points_for_the_motion_eyes import PointsForMotionEyes


class InventedTracker:
    def __init__(self):
        self.seeds = ()
        self.hidden = False
        self.wrong_capture = False
        self.calls = []

    def start(self, frame, seeds):
        self.seeds = seeds
        self.calls.append(('start', frame))
        return self._result(frame, 0.)

    def track(self, frame):
        self.calls.append(('track', frame))
        return self._result(frame, 4. * (frame.number - self.calls[0][1].number))

    def _result(self, frame, shift):
        tracks = tuple(PointTrack(seed.number, seed.x + shift, seed.y, .1 if self.hidden else .99,
                                  not self.hidden) for seed in self.seeds)
        return PointResult('invented', frame.epoch, frame.number + int(self.wrong_capture),
                           frame.at, frame.shape, tracks, .01)


@pytest.fixture
def images(monkeypatch):
    monkeypatch.setattr(bridging, '_new_marks', lambda *_args: [(32., 32.), (48., 32.), (32., 48.), (48., 48.)])
    source = np.random.default_rng(44).integers(30, 230, (96, 128, 3), dtype=np.uint8)
    return source, np.zeros((48, 64, 3), np.uint8)


def test_native_points_reach_the_actual_motion_support_in_working_coordinates(images):
    tracker = InventedTracker()
    points = PointsForMotionEyes(tracker, limit=4)
    first = points.see_source(*images, 1.)
    second = points.see_source(*images, 1.1)
    assert [(point.x, point.y) for point in first] == [(16., 16.), (24., 16.), (16., 24.), (24., 24.)]
    assert all(point.pictures == 2 and point.before == 1. and point.at == 1.1 for point in second)
    body = SimpleNamespace(x=20., y=20., w=20., h=20., seen=1., support=np.ones((20, 20), bool))
    motion = points.motion_of(body, 1.1)
    assert motion is not None and motion.support == 4 and motion.dx == 2. and motion.dy == 0.
    assert motion.before == 1. and motion.at == 1.1


def test_continuously_visible_learned_marks_bridge_a_missing_object_detection(images):
    points = PointsForMotionEyes(InventedTracker(), limit=4)
    for at in (1., 1.1, 1.2):
        points.see_source(*images, at)
    body = SimpleNamespace(x=20., y=20., w=20., h=20., seen=1., support=np.ones((20, 20), bool))
    motion = points.motion_of(body, 1.2)
    assert motion is not None and motion.support == 4 and motion.dx == 4. and motion.dy == 0.
    assert motion.before == 1. and motion.at == 1.2


def test_hidden_predictions_cannot_extend_a_path_and_reappearance_needs_new_visible_support(images):
    tracker = InventedTracker()
    points = PointsForMotionEyes(tracker, limit=4)
    first = points.see_source(*images, 1.)
    tracker.hidden = True
    assert not points.see_source(*images, 1.1)
    tracker.hidden = False
    returned = points.see_source(*images, 1.2)
    assert {point.number for point in first} == {point.number for point in returned}
    assert all(point.pictures == 1 and point.before == 1.2 for point in returned)
    body = SimpleNamespace(x=20., y=20., w=30., h=30., seen=1.1, support=np.ones((30, 30), bool))
    assert points.motion_of(body, 1.2) is None
    body.seen = 1.
    assert points.motion_of(body, 1.2) is None


def test_a_gap_or_geometry_change_starts_fresh_queries_without_reusing_point_identities(images):
    tracker = InventedTracker()
    points = PointsForMotionEyes(tracker, limit=4)
    first = points.see_source(*images, 1.)
    later = points.see_source(*images, 3.)
    assert not {point.number for point in first} & {point.number for point in later}
    assert all(point.pictures == 1 and point.born == 3. for point in later)
    assert tracker.calls[0][1].epoch != tracker.calls[-1][1].epoch


def test_another_capture_cannot_publish_points_to_the_current_observer(images):
    tracker = InventedTracker()
    tracker.wrong_capture = True
    points = PointsForMotionEyes(tracker, limit=4)
    with pytest.raises(ValueError, match='another capture'):
        points.see_source(*images, 1.)
    assert not points.points and not points._slots


def test_new_changing_texture_gets_resting_slots_without_evicting_continuing_motion(images):
    class ReplaceableTracker(InventedTracker):
        def _replace_queries(self):
            pass

    tracker = ReplaceableTracker()
    points = PointsForMotionEyes(tracker, limit=4)
    source, working = images
    points.see_source(source, working, 1.)
    points._native_visible[points._slots[0].number] = SimpleNamespace(
        x=20., y=20., old_x=18., old_y=20., pictures=8)
    changed = source.copy()
    changed[28:54, 28:54] = 255 - changed[28:54, 28:54]
    replacements = points._new_texture_queries(changed, bridging._grey(changed), 1.7)
    assert replacements and 0 not in replacements
    assert all(seed.number not in {s.number for s in points._slots} for seed in replacements.values())


def test_the_actual_factory_keeps_the_model_on_its_worker_and_out_of_the_public_mirror(monkeypatch):
    from core.perception import seeing_play_elsewhere
    from core.perception.learned_tracking import models, selection
    from core.perception.what_moves_in_the_picture import WhatMoves

    monkeypatch.setenv('AURA_PLAY_POINT_BACKEND', 'online_tapir')
    monkeypatch.setenv('AURA_PLAY_POINT_WEIGHTS', '/invented-verified-weights')
    monkeypatch.setenv('AURA_PLAY_POINT_DEVICE', 'cpu')
    monkeypatch.setenv('AURA_PLAY_POINT_MODE', 'direct')
    loaded = []

    def load(name, weights, *, device):
        loaded.append((name, weights, device))
        return InventedTracker()

    monkeypatch.setattr(models, 'load_backend', load)
    eyes = seeing_play_elsewhere.eyes_for_play()
    assert isinstance(eyes.points, PointsForMotionEyes)
    assert loaded == [('online_tapir', Path('/invented-verified-weights'), 'cpu')]
    assert 'points' not in seeing_play_elsewhere._state_of(eyes)
    assert not isinstance(WhatMoves().points, PointsForMotionEyes)
    monkeypatch.setenv('AURA_PLAY_POINT_BACKEND', 'unsupported')
    with pytest.raises(ValueError):
        selection.selected_motion_eyes()


def test_the_learned_capture_claim_is_registered():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert bridging._learned_point_capture_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'learned_points_keep_their_capture')
    assert check.predict(None) and any(c.test == check.name for c in suite.claims())

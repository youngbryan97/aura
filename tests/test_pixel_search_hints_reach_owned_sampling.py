"""A frame-bound search region reaches the actual observer and preserves other sampling."""
from __future__ import annotations

import copy
import dataclasses
import os
import pickle
import subprocess
import sys

import numpy as np
import pytest

from core.perception import seeing_play_elsewhere as elsewhere
from core.perception.a_search_hint_for_these_pixels import (
    PixelSearchHint,
    _pixel_search_hint_invariant,
    frame_priority_mask,
    make_search_hint,
    sampling_hint_matches,
)
from core.perception.frames_as_they_are_drawn import CanvasFrameTime
from core.perception.play_observation_without_holding_the_loop import open_play_observation
from core.perception.points_before_the_picture_is_reduced import (
    PointsBeforeReduction,
    _new_marks,
    _pixel_search_capacity_invariant,
    _sampling_hint_receipt_invariant,
)
from core.perception.what_moves_in_the_picture import WhatMoves

pytestmark = pytest.mark.unit


def captured(at=10., **changes):
    values = dict(lower=at, upper=at + .01, arrived=at + .02, source="canvas-bracket",
                  source_at=at + 20., epoch=2, surface="document/canvas", sequence=4)
    values.update(changes)
    return CanvasFrameTime(at, **values)


def pixels(seed=47):
    return np.random.default_rng(seed).integers(20, 240, (80, 120, 3), dtype=np.uint8)


def hint_for(picture, at):
    hint = make_search_hint(picture, at, normalized_region=(2 / 3, .5, .95, .95))
    assert isinstance(hint, PixelSearchHint)
    return hint


def test_hint_and_capture_keep_exact_custody_through_worker_and_wire_copy():
    picture, at = pixels(), captured()
    original = hint_for(picture, at)
    for hint, clock in (copy.deepcopy((original, at)), pickle.loads(pickle.dumps((original, at)))):
        priority = frame_priority_mask(hint, picture, clock)
        assert priority is not None and 0 < priority.sum() < priority.size / 2
        assert hint == original and dataclasses.asdict(hint) == dataclasses.asdict(original)
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            hint.sequence = 5
        assert not hasattr(hint, "thing") and not hasattr(hint, "mask") and not hasattr(hint, "key")
    assert _pixel_search_hint_invariant() == ()


@pytest.mark.parametrize("change", [
    {"surface": "another"}, {"epoch": 3}, {"sequence": 5}, {"upper": 10.015},
    {"arrived": 10.03}, {"source_at": 31.}, {"source": "browser-wall"},
    {"lower": None}, {"upper": float("nan")}, {"arrived": float("inf")},
])
def test_departing_from_the_exact_capture_refuses_the_priority(change):
    picture, at = pixels(), captured()
    hint = hint_for(picture, at)
    assert frame_priority_mask(hint, picture, captured(**change)) is None
    assert frame_priority_mask(hint, picture, float(at)) is None


@pytest.mark.parametrize("change", [
    {"sequence": True}, {"epoch": True}, {"lower": True}, {"shape": (True, 120, 3)},
    {"digest": "0" * 64}, {"region": (0., 0., 1., 1.)}, {"source": []},
])
def test_malformed_hint_values_stay_unknown(change):
    picture, at = pixels(), captured()
    hint = dataclasses.replace(hint_for(picture, at), **change)
    assert frame_priority_mask(hint, picture, at) is None


def test_equal_shape_and_visual_similarity_cannot_replace_exact_rgb_bytes():
    picture, at = pixels(), captured()
    hint = hint_for(picture, at)
    changed = picture.copy()
    changed[0, 0, 0] ^= 1
    assert frame_priority_mask(hint, changed, at) is None
    assert frame_priority_mask(hint, picture[::-1].copy(), at) is None
    assert frame_priority_mask(hint, picture.astype(np.float32), at) is None
    assert frame_priority_mask(hint, picture[:, :-1], at) is None


def test_new_corner_search_and_surviving_marks_both_share_priority_capacity():
    image = np.random.default_rng(71).random((80, 120)).astype(np.float32)
    priority = np.zeros(image.shape, bool)
    priority[40:76, 80:116] = True
    marks = _new_marks(image, [], 18, (1., 1.), priority=priority)
    assert len(marks) == 18
    assert sum(bool(priority[int(y), int(x)]) for x, y in marks) == 6
    assert _pixel_search_capacity_invariant() == ()
    picture = np.repeat((image[..., None] * 255).astype(np.uint8), 3, axis=2)
    tracker = PointsBeforeReduction(limit=18)
    sampled = tracker.see_source(picture, picture, 10., priority_mask=priority)
    assert len(sampled) == 18
    assert sum(bool(priority[int(point.y), int(point.x)]) for point in sampled) == 6
    assert all(point.before == point.at == point.born == 10. and point.pictures == 1 for point in sampled)
    assert any(not priority[int(point.y), int(point.x)] for point in sampled)


def test_source_priority_scales_to_the_bounded_tracking_grid_and_expires():
    picture, at = pixels(), captured()
    priority = frame_priority_mask(hint_for(picture, at), picture, at)
    tracker = PointsBeforeReduction(limit=18, pixel_budget=2400)
    tracker.see_source(picture, picture[::2, ::2].copy(), 10., priority_mask=priority)
    assert tracker._priority is not None and tracker._priority.shape == tracker._geometry[1]
    assert tracker._priority.size <= tracker.pixel_budget
    tracker.see_source(picture, picture[::2, ::2].copy(), 10.1)
    assert tracker._priority is None


def test_refused_hints_leave_the_ordinary_observer_and_sampler_unchanged():
    picture, at = pixels(), captured()
    hint = dataclasses.replace(hint_for(picture, at), digest="0" * 64)
    ordinary, refused = WhatMoves(), WhatMoves()
    assert ordinary.see(picture, at) == refused.see(picture, at, search_hint=hint)
    assert ordinary.points.points == refused.points.points
    assert refused.points._priority is None
    first, second = PointsBeforeReduction(limit=18), PointsBeforeReduction(limit=18)
    baseline = first.see_source(picture, picture, 10.)
    assert baseline == second.see_source(picture, picture, 10., priority_mask=np.ones(picture.shape[:2], bool))


@pytest.mark.asyncio
async def test_hint_reaches_the_private_thread_observer_before_publication():
    picture, at = pixels(), captured()
    hint = hint_for(picture, at)
    observation = await open_play_observation(factory=lambda **kwargs: WhatMoves(**kwargs), frame_wait_s=2.)
    try:
        assert await observation.see(picture, at, search_hint=hint) is not None
        actual = observation._eyes.points._priority
        assert actual is not None and np.array_equal(actual, frame_priority_mask(hint, picture, at))
        assert observation._mirror.points._priority is None
        assert observation._eyes is not observation._mirror
        assert sampling_hint_matches(observation.sampling_hint_receipt, hint)
        assert observation.sampling_hint_receipt.fresh_priority_points > 0
        assert observation.sampling_hint_receipt == observation._eyes.points.sampling_hint_receipt
        assert observation.sampling_hint_receipt is not observation._eyes.points.sampling_hint_receipt
    finally:
        await observation.close()


def test_hint_and_measured_receipt_reach_the_real_child_through_shared_memory(monkeypatch):
    # The production sampler emits the receipt in its child-owned state.
    program = """
from core.perception import seeing_play_elsewhere as elsewhere
elsewhere._serve()
"""
    environment = dict(os.environ)
    environment["AURA_PLAY_EYES_IN_THIS_PROCESS"] = "1"
    environment["AURA_EYES_IN_THIS_PROCESS"] = "1"
    child = subprocess.Popen([sys.executable, "-c", program], env=environment,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    child.answered = False
    monkeypatch.setattr(elsewhere, "_start_them", lambda: child)
    monkeypatch.setattr(elsewhere, "_CHILD", child)
    monkeypatch.setattr(elsewhere, "_ENDED", [])
    picture, at = pixels(), captured()
    hint = hint_for(picture, at)
    try:
        observer = elsewhere.SeenElsewhere()
        assert observer.elsewhere
        observer.see(picture, at, search_hint=hint)
        assert sampling_hint_matches(observer.sampling_hint_receipt, hint)
        assert observer.sampling_hint_receipt.priority_pixels == int(frame_priority_mask(hint, picture, at).sum())
        assert observer.sampling_hint_receipt.fresh_priority_points > 0
        observer.see(pixels(48), captured(10.2, sequence=5), search_hint=hint)
        assert observer.sampling_hint_receipt is None
        assert observer.elsewhere
    finally:
        child.stdin.close()
        child.wait(timeout=10)
        child.stdout.close()


def test_sampling_receipt_counts_the_actual_marks_after_spread_and_has_no_pose_authority():
    picture, at = pixels(), captured()
    hint = hint_for(picture, at)
    tracker = PointsBeforeReduction(limit=18)
    tracker.see_source(picture, picture, at, search_hint=hint)
    receipt = tracker.sampling_hint_receipt
    assert sampling_hint_matches(receipt, hint)
    preferred = [point for point in tracker._native_points if tracker._priority[int(round(point.y)), int(round(point.x))]]
    assert receipt.kept_priority_points == receipt.fresh_priority_points == len(preferred) > 0
    assert receipt.background_points == len(tracker.points) - len(preferred) > 0
    assert receipt.priority_pixels == int(tracker._priority.sum())
    assert all(type(point.at) is type(point.before) is type(point.born) is float for point in tracker.points)
    assert not hasattr(receipt, "thing") and not hasattr(receipt, "pose") and not hasattr(receipt, "key")
    for restored in (copy.deepcopy(receipt), pickle.loads(pickle.dumps(receipt))):
        assert restored == receipt and sampling_hint_matches(restored, hint)
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            restored.consumed = False
    assert _sampling_hint_receipt_invariant() == ()


@pytest.mark.parametrize("change", [
    {"consumed": False}, {"sequence": 5}, {"upper": 10.015}, {"priority_pixels": 0},
    {"fresh_priority_points": -1}, {"kept_priority_points": True}, {"background_points": 385},
])
def test_receipt_match_refuses_stale_unconsumed_or_invalid_counts(change):
    picture, at = pixels(), captured()
    hint = hint_for(picture, at)
    tracker = PointsBeforeReduction(limit=18)
    tracker.see_source(picture, picture, at, search_hint=hint)
    receipt = dataclasses.replace(tracker.sampling_hint_receipt, **change)
    assert not sampling_hint_matches(receipt, hint)
    assert not sampling_hint_matches(None, hint)


def test_empty_priority_after_bounded_grid_sampling_has_no_consumption_receipt():
    picture = np.random.default_rng(47).integers(20, 240, (64, 1024, 3), dtype=np.uint8)
    at = captured()
    hint = make_search_hint(picture, at, normalized_region=(0., 0., 1 / 1024, 1.))
    assert frame_priority_mask(hint, picture, at).sum() == 64
    tracker = PointsBeforeReduction(limit=18, pixel_budget=16_384)
    tracker.see_source(picture, picture[:, ::4].copy(), at, search_hint=hint)
    assert tracker._priority is None and tracker.sampling_hint_receipt is None


def test_repeated_view_and_unsupported_tracker_leave_no_stale_receipt():
    picture, at = pixels(), captured()
    observer = WhatMoves()
    observer.see(picture, at, search_hint=hint_for(picture, at))
    assert observer.sampling_hint_receipt is not None
    after = captured(10.2, sequence=5)
    observer.view_moved_at = 10.1
    assert observer.see(picture, after, search_hint=hint_for(picture, after)) == []
    assert observer.sampling_hint_receipt is None

    class OtherTracker(PointsBeforeReduction):
        def see_source(self, source, working, at):
            return super().see_source(source, working, at)

    unsupported = WhatMoves(points=OtherTracker())
    assert unsupported.see(picture, at, search_hint=hint_for(picture, at)) is not None
    assert unsupported.sampling_hint_receipt is None

"""Admitted pixel experiments cannot borrow or omit motion measurements."""
from types import SimpleNamespace

import numpy as np

from core.agency.discovering_control_in_current_pixels import PixelResponseDiscovery
from core.agency.pixels_inside_control_trials import (
    PixelPress,
    RandomizedPixelExperiment,
    freeze_frame,
)
from core.perception.frames_as_they_are_drawn import CanvasFrameTime


def pair(plan, index):
    base = index + 1.
    image = np.zeros((96, 120, 3), np.uint8)
    def frame(sequence, at):
        return freeze_frame(image, CanvasFrameTime(at, lower=at, upper=at + .01, arrived=at + .02,
                            source="canvas-bracket", source_at=None, epoch=2,
                            surface="canvas/test", sequence=sequence))
    return PixelPress(plan.schedule[index], base + .05, base + .06, base + .4, base + .41, base,
                      frame(2 * index + 1, base + .01), frame(2 * index + 2, base + .25))


def test_missing_motion_retires_instead_of_dropping_an_inconvenient_trial():
    plan = RandomizedPixelExperiment(("a", "b"), seed=11)
    discovery = PixelResponseDiscovery(plan)
    admitted = pair(plan, 0)
    assert plan.record(admitted)
    assert not discovery.measure_last_press(admitted)
    assert plan.refused == "pixel_motion_unavailable" and discovery.region() is None


def test_measurement_cannot_use_an_unadmitted_or_substituted_pair():
    plan = RandomizedPixelExperiment(("a", "b"), seed=11)
    discovery = PixelResponseDiscovery(plan)
    assert not discovery.measure_last_press(pair(plan, 0))
    assert plan.refused == "motion_trial_order"


def test_complete_measurements_retain_missing_camera_as_unknown(monkeypatch):
    plan = RandomizedPixelExperiment(("a", "b"), seed=11)
    discovery = PixelResponseDiscovery(plan)
    calls = iter(plan.schedule)
    def signature(*args, **kwargs):
        press = next(calls)
        features = np.zeros((4, 5, 3))
        features[1:3, 2:4, 2] = 2 if press.key == "a" else -2
        return SimpleNamespace(shape=(4, 5), working_shape=(96, 120), features=tuple(map(tuple, features.reshape(-1, 3))),
                               support=(4,) * 20, angular_support=(4,) * 20, camera=None)
    monkeypatch.setattr("core.agency.discovering_control_in_current_pixels.pixel_motion_signature", signature)
    for index in range(24):
        admitted = pair(plan, index)
        assert plan.record(admitted) and discovery.measure_last_press(admitted)
    region = discovery.region()
    assert region is not None and region.status == "search_hint_only"
    assert not hasattr(discovery, "number") and not hasattr(region, "identity")


def test_motion_reference_cannot_silently_switch_halfway(monkeypatch):
    plan = RandomizedPixelExperiment(("a", "b"), seed=11)
    discovery = PixelResponseDiscovery(plan)
    results = iter((SimpleNamespace(camera=None, shape=(4, 5), working_shape=(96, 120)),
                    SimpleNamespace(camera=SimpleNamespace(dx=0., dy=0.), shape=(4, 5), working_shape=(96, 120))))
    monkeypatch.setattr("core.agency.discovering_control_in_current_pixels.pixel_motion_signature", lambda *a, **k: next(results))
    admitted = pair(plan, 0)
    assert plan.record(admitted) and discovery.measure_last_press(admitted)
    admitted = pair(plan, 1)
    assert plan.record(admitted) and not discovery.measure_last_press(admitted)
    assert plan.refused == "changed_motion_reference"

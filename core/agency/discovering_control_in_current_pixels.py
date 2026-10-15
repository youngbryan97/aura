"""Turn admitted independent presses into bounded visual search measurements.

This path learns no body identity. Each response uses one acknowledged press
and its own frozen RGB captures; confirmation cannot revise discovery.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from core.agency.pixels_inside_control_trials import PixelPress, RandomizedPixelExperiment
from core.perception.a_region_that_answers_independent_inputs import (
    ResponsiveRegion,
    region_from_press_features,
)
from core.perception.pixel_motion_before_object_identity import (
    PixelMotionSignature,
    pixel_motion_signature,
)
from core.verify import invariant


class PixelResponseDiscovery:
    """Keep motion measurements in the experiment's fixed acquisition order."""

    def __init__(self, experiment: RandomizedPixelExperiment, *, tile: int = 24) -> None:
        if not isinstance(experiment, RandomizedPixelExperiment) or type(tile) is not int or not 12 <= tile <= 32:
            raise ValueError("discovery requires a prespecified bounded experiment")
        self.experiment, self.tile = experiment, tile
        self._signatures: list[PixelMotionSignature] = []
        self._reference: str | None = None

    def measure_last_press(self, press: PixelPress) -> bool:
        """Run after the collector admits this press, before acquiring another."""
        return self.accept_last_press(press, pixel_motion_signature(
            press.before.rgb(), press.after.rgb(), tile=self.tile, point_limit=256))

    def accept_last_press(self, press: PixelPress, result: PixelMotionSignature | None) -> bool:
        """Accept a completed off-loop measurement in acquisition order."""
        if (self.experiment.refused is not None or not self.experiment.trials
                or len(self.experiment.trials) != len(self._signatures) + 1
                or self.experiment.trials[-1] is not press):
            self.experiment.retire("motion_trial_order")
            return False
        if result is None:
            self.experiment.retire("pixel_motion_unavailable")
            return False
        reference = "distributed_translation" if result.camera is not None else "screen_unreferenced"
        if self._reference is not None and reference != self._reference:
            self.experiment.retire("changed_motion_reference")
            return False
        if self._signatures and (result.shape, result.working_shape) != (
                self._signatures[0].shape, self._signatures[0].working_shape):
            self.experiment.retire("changed_motion_extent")
            return False
        self._reference = reference
        self._signatures.append(result)
        return True

    def region(self) -> ResponsiveRegion | None:
        """A completed independent experiment may guide new point allocation."""
        if not self.experiment.complete or len(self._signatures) != len(self.experiment.schedule):
            return None
        shape = self._signatures[0].shape
        values = np.asarray([s.features for s in self._signatures], dtype=float).reshape(-1, *shape, 3)
        support = np.asarray([s.support for s in self._signatures], dtype=int).reshape(-1, *shape)
        angles = np.asarray([s.angular_support for s in self._signatures], dtype=int).reshape(-1, *shape)
        available = np.repeat((support >= 3)[..., None], 3, axis=-1)
        available[..., 2] = angles >= 4
        cameras = (np.asarray([(s.camera.dx, s.camera.dy) for s in self._signatures])
                   if self._reference == "distributed_translation" else None)
        region = region_from_press_features(values, support, cameras,
                                          tuple(p.key for p in self.experiment.schedule),
                                          tuple(p.cohort for p in self.experiment.schedule),
                                          first_key=self.experiment.keys[0], second_key=self.experiment.keys[1],
                                          tile=self.tile, available=available)
        if region is None:
            return None
        height, width = self._signatures[0].working_shape
        left, top, right, bottom = region.bounds
        return replace(region, bounds=(left, top, min(width, right), min(height, bottom)))


@invariant("agency.pixel_discovery_requires_its_complete_admitted_experiment", scope="agency",
           owner="core/agency/discovering_control_in_current_pixels.py", observational=False)
def _pixel_discovery_invariant() -> tuple:
    discovery = PixelResponseDiscovery(RandomizedPixelExperiment(("a", "b"), seed=41))
    assert discovery.region() is None
    discovery.experiment.retire("capture_unavailable")
    assert discovery.region() is None
    return ()

"""Calibration preserves measured probability and cannot cancel bin errors."""

import copy
import math

import pytest

from core.agency.how_far_her_model_carries import HowFarHerModelCarries
from core.world_model.prediction_quality import CalibrationCurve, PredictionOutcome


def test_perfect_calibration_is_not_false_because_its_error_is_zero():
    curve = CalibrationCurve()
    for i in range(40):
        curve.observe(PredictionOutcome(float(i % 2), float(not i % 2)))
    assert curve.expected_calibration_error == 0.0
    assert curve.to_dict()["calibrated"] is True


def test_opposing_bin_errors_cannot_cancel():
    carried = HowFarHerModelCarries()
    for _ in range(20):
        carried.it_predicted(distance=1, confidence=0.1, was_right=True)
        carried.it_predicted(distance=1, confidence=0.9, was_right=False)
    curve = carried.calibration
    assert curve.overconfidence == pytest.approx(0.0)
    assert curve.expected_calibration_error == pytest.approx(0.9)
    assert curve.to_dict()["calibrated"] is False
    assert carried.how_sure_she_should_be(0.9) == 0.0
    assert carried.how_sure_she_should_be(0.1) == 1.0
    assert carried.how_sure_she_should_be(0.6) == 0.6


def test_diagram_uses_actual_confidence_not_bin_midpoint():
    curve = CalibrationCurve()
    for confidence in (0.91, 0.93):
        curve.observe(PredictionOutcome(confidence, 0.0))
    assert curve.to_dict()["diagram"][9]["stated"] == pytest.approx(0.92)
    assert curve.to_dict()["diagram"][8]["stated"] is None


@pytest.mark.parametrize("confidence", [math.nan, math.inf, -0.1, 1.1])
@pytest.mark.parametrize("distance", [1, 2])
def test_invalid_confidence_cannot_partially_mutate_measurements(confidence, distance):
    carried = HowFarHerModelCarries()
    before = carried.as_memory()
    with pytest.raises(ValueError):
        carried.it_predicted(distance=distance, confidence=confidence, was_right=True)
    assert carried.as_memory() == before


@pytest.mark.parametrize("distance", [0, -1, 1.5, True])
def test_distance_is_a_count_not_a_coerced_claim(distance):
    with pytest.raises(ValueError):
        HowFarHerModelCarries().it_predicted(distance=distance, confidence=0.5, was_right=True)


@pytest.mark.parametrize("field", ["error", "tolerance"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -0.1])
def test_invalid_errors_and_tolerances_are_rejected(field, value):
    args = dict(confidence=0.5, error=0.0, tolerance=0.1)
    args[field] = value
    with pytest.raises(ValueError):
        PredictionOutcome(**args)


def test_roundtrip_retains_bin_means_and_calibrated_decisions():
    carried = HowFarHerModelCarries()
    for i in range(40):
        carried.it_predicted(distance=1, confidence=0.91, was_right=i % 2 == 0)
    again = HowFarHerModelCarries.from_memory(carried.as_memory())
    assert again.as_memory() == carried.as_memory()
    assert again.calibration.to_dict() == carried.calibration.to_dict()
    assert again.how_sure_she_should_be(0.92) == 0.5


def test_legacy_bin_means_are_unmeasured_not_invented():
    curve = CalibrationCurve()
    curve.observe(PredictionOutcome(0.91, 0.0))
    held = curve.as_memory()
    del held["confidence_sums"]
    again = CalibrationCurve.from_memory(held)
    assert again.brier == curve.brier
    assert again.overconfidence == curve.overconfidence
    assert again.expected_calibration_error is None
    assert again.to_dict()["diagram"][9]["stated"] is None
    again.observe(PredictionOutcome(0.93, 0.0))
    assert again.expected_calibration_error is None


@pytest.mark.parametrize("field,value", [("n", 9), ("squared", math.nan), ("correct_sum", 0.0),
    ("counts", [1] * 10), ("hits", [2] * 10), ("confidence_sums", [0.0] * 10)])
def test_inconsistent_saved_statistics_are_rejected(field, value):
    curve = CalibrationCurve()
    curve.observe(PredictionOutcome(0.91, 0.0))
    held = curve.as_memory()
    held[field] = value
    with pytest.raises(ValueError):
        CalibrationCurve.from_memory(held)


def test_old_runtime_horizons_do_not_reappear_as_corrected_measurements():
    carried = HowFarHerModelCarries()
    for _ in range(10):
        carried.it_predicted(distance=4, confidence=0.9, was_right=True)
    old = carried.as_memory()
    del old["measurement_version"]
    saved = copy.deepcopy(old)
    again = HowFarHerModelCarries.from_memory(old)
    assert again.carries_to() == 0
    assert again.graded == {}
    assert old == saved

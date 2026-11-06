"""Her anomaly score grades how far from usual an event is (third day run, 6 November 2026).

Her error rate, resource pressure and polarity sit at 0 for hours, so their variance in the window was the 1e-5
regularization, any move read as a hundred sigma, and every alarm of the day read threat=1.0000 at distances of 77 to
651. Her ICE layer reads that threat level.
"""
from __future__ import annotations

import time

import numpy as np

from core.cognitive.anomaly_detector import FEATURE_DIM, _NormalityModel


def _usual_window() -> _NormalityModel:
    model = _NormalityModel(dim=FEATURE_DIM, window_size=500, decay_half_life=250)
    rng = np.random.default_rng(7)
    now = time.time()
    for i in range(200):
        vec = np.zeros(FEATURE_DIM)
        vec[0] = 5.0 + rng.normal(0, 0.5)   # length varies
        vec[1] = 0.7 + rng.normal(0, 0.05)  # lexical variety varies
        vec[3] = 3.0 + rng.normal(0, 0.4)   # time since the last event varies
        # error rate, resource pressure, polarity and the rest stay at 0
        model.push(vec, now + i)
    return model


def _usual_event() -> np.ndarray:
    vec = np.zeros(FEATURE_DIM)
    vec[0], vec[1], vec[3] = 5.0, 0.7, 3.0
    return vec


def test_a_small_move_in_a_feature_that_never_moved_is_a_few_sigma_not_a_hundred():
    model = _usual_window()
    vec = _usual_event()
    vec[4] = 0.3  # error rate 0.3
    assert 2.0 < model.mahalanobis(vec) < 4.0


def test_a_large_move_still_reads_as_far_from_usual():
    model = _usual_window()
    vec = _usual_event()
    vec[4] = 1.0
    assert model.mahalanobis(vec) > 8.0


def test_a_feature_that_varies_keeps_its_own_spread():
    model = _usual_window()
    vec = _usual_event()
    vec[0] = 5.0 + 3 * 0.5  # three of its own standard deviations
    assert 2.0 < model.mahalanobis(vec) < 4.5

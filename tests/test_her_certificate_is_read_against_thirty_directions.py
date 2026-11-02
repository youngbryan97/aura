"""Her steering certificate's random control, read against thirty fresh directions.

The certificate compares her basis with the median of three random directions
from one fixed seed. These pin the null of 2 November: its directions are drawn
the way the certificate draws its control, it measures in the certificate's
order, and it is read by the addendum's three rules.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.certificate_null import (
    HER_CERTIFICATE,
    MARGIN_TOLERANCE,
    draw_direction,
    measure_null,
    read,
)

pytestmark = pytest.mark.unit


def test_each_direction_has_her_length_at_every_layer_and_skips_what_she_lacks():
    composites = {3: np.ones(8, dtype=np.float32) * 2.0, 7: None, 43: np.arange(8, dtype=np.float32)}
    felts = {3: None, 7: None, 43: np.full(8, 5.0, dtype=np.float32)}
    controls, felt_controls = draw_direction(composites, felts, np.random.default_rng([20261102, 0]))
    assert controls[7] is None
    for layer in (3, 43):
        assert np.linalg.norm(controls[layer]) == pytest.approx(np.linalg.norm(composites[layer]), rel=1e-5)
    assert set(felt_controls) == {43}
    assert np.linalg.norm(felt_controls[43]) == pytest.approx(np.linalg.norm(felts[43]), rel=1e-5)


def test_the_directions_are_fresh_and_reproducible():
    composites, felts = {3: np.ones(64, dtype=np.float32)}, {}
    first, _ = draw_direction(composites, felts, np.random.default_rng([20261102, 0]))
    again, _ = draw_direction(composites, felts, np.random.default_rng([20261102, 0]))
    second, _ = draw_direction(composites, felts, np.random.default_rng([20261102, 1]))
    certificate, _ = draw_direction(composites, felts, np.random.default_rng(20260908))
    assert np.array_equal(first[3], again[3])
    assert not np.allclose(first[3], second[3])
    assert not np.allclose(first[3], certificate[3])


class FakeProbe:
    """Records the calls the null makes, and answers with a margin set by what is injected."""

    def __init__(self):
        self.calls, self.alpha, self.overridden = [], 0.0, None
        self.composite = {3: np.ones(4, dtype=np.float32)}

    def basis_sha256(self):
        return HER_CERTIFICATE["basis_sha256"]

    def set_alpha(self, alpha):
        self.alpha = alpha
        self.calls.append(("alpha", alpha))

    def greedy_paths(self, steps):
        assert self.alpha == 0.0
        return [[1, 2]]

    def walk(self, paths):
        return [("walked", self.alpha, self.overridden is not None)]

    def shift(self, walked, unsteered):
        return 0.65 if walked[0][1] and not walked[0][2] else 0.30

    def margin(self):
        if self.alpha == 0.0:
            return 6.5
        return 6.5 + (0.33 if self.overridden is None else 0.30)

    def settle(self, moods):
        self.calls.append(("settle", moods["valence"]))

    def composites(self):
        return dict(self.composite)

    def felts(self):
        return {3: None}

    def override(self, vectors, felts=None):
        self.overridden = vectors
        self.calls.append(("override", vectors is not None))

    def voice(self, replies, alpha, rngs):
        assert alpha == 0.5 and len(rngs) == 4
        return 0.24, len(replies), [0.001 * index for index in range(len(rngs))]


def test_the_null_measures_her_then_each_direction_in_the_certificates_order():
    probe = FakeProbe()
    result = measure_null(probe, [("hello there", [0.1])], directions=4, log=lambda line: None)
    assert result["her"]["margin_delta"] == pytest.approx(0.33)
    assert result["her"]["distribution_shift"] == pytest.approx(0.65)
    assert result["her"]["voice_delta"] == pytest.approx(0.24)
    assert [row["margin_delta"] for row in result["directions"]] == pytest.approx([0.30] * 4)
    assert [row["voice_delta"] for row in result["directions"]] == pytest.approx([0.0, 0.001, 0.002, 0.003])
    overrides = [on for kind, on in probe.calls if kind == "override"]
    assert overrides == [True, False] * 4  # every direction is cleared before the next
    assert probe.calls[-1] == ("alpha", 0.0)


def _result(her_margin, margins, her_voice=0.24, voices=None, basis=HER_CERTIFICATE["basis_sha256"]):
    voices = voices if voices is not None else [0.001] * len(margins)
    return {
        "model_identity": HER_CERTIFICATE["model_identity"],
        "basis_sha256": basis,
        "her": {
            "margin_delta": her_margin,
            "distribution_shift": HER_CERTIFICATE["distribution_shift"],
            "voice_delta": her_voice,
        },
        "directions": [{"margin_delta": m, "voice_delta": v} for m, v in zip(margins, voices, strict=True)],
    }


def test_her_reading_holds_when_she_beats_the_median_and_every_voice():
    reading = read(_result(HER_CERTIFICATE["margin_delta"], [0.30, 0.31, 0.34, 0.20]))
    assert reading["read"] and reading["beats_the_null"] and reading["keeps_her_voice_against_the_null"]
    assert reading["share_of_null_below_her_margin"] == pytest.approx(0.75)
    assert reading["holds"]


def test_a_margin_at_or_below_the_null_median_does_not_beat_it():
    reading = read(_result(HER_CERTIFICATE["margin_delta"], [0.34, 0.35, 0.20, 0.36]))
    assert not reading["beats_the_null"] and not reading["holds"]


def test_a_random_direction_that_keeps_her_voice_as_well_takes_the_voice_rule():
    reading = read(_result(HER_CERTIFICATE["margin_delta"], [0.1, 0.2], voices=[0.001, 0.30]))
    assert not reading["keeps_her_voice_against_the_null"] and not reading["holds"]


def test_a_reading_that_does_not_reproduce_her_certificate_is_not_read():
    drifted = HER_CERTIFICATE["margin_delta"] + 2 * MARGIN_TOLERANCE
    assert not read(_result(drifted, [0.1, 0.2]))["read"]
    assert not read(_result(HER_CERTIFICATE["margin_delta"], [0.1, 0.2], basis="0" * 64))["read"]

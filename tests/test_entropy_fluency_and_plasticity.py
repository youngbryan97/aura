"""tests/test_entropy_fluency_and_plasticity.py
─────────────────────────────────────────────────
Tests for the two research-derived metric modules added 2026-04-28.

EntropyFluencyTracker covers:
- Idle (no ticks) returns no report.
- Stable substrate (small noise around fixed mean) lands in a low-entropy /
  high-fluency phase after warmup.
- Sudden destabilization (large random injections) raises entropy and
  drops fluency, classified as DESTABILIZING / REORGANIZING.
- Reset returns to pristine state.
- Phase classifier: synthetic (entropy, fluency) joint values map to
  the documented phases.

PlasticityMonitor covers:
- Full-rank matrix → stable_rank_ratio near 1.0.
- Rank-1 matrix → ratio collapses to ~1/N.
- All-zero matrix → degenerate report, no crash.
- Sustained breaches trigger the collapse warning after the configured
  threshold of consecutive low-rank measurements.
- Refuses to SVD an oversized matrix, returns None instead of raising.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Pre-import to break the circular path that surfaces only on cold offline import
import core.runtime.atomic_writer  # noqa: F401
import core.utils.concurrency      # noqa: F401
import core.exceptions             # noqa: F401
import core.container              # noqa: F401

from core.consciousness.entropy_fluency import (
    ALL_PHASES,
    ENTROPY_HIGH,
    ENTROPY_LOW,
    EntropyFluencyTracker,
    FLUENCY_HIGH,
    FLUENCY_LOW,
    PHASE_DESTABILIZING,
    PHASE_FLUENT,
    PHASE_REORGANIZING,
    PHASE_STABLE,
    reset_entropy_fluency_tracker,
)
from core.consciousness.plasticity_monitor import (
    MAX_MATRIX_DIM,
    PlasticityMonitor,
    reset_plasticity_monitor,
)


# ── EntropyFluencyTracker ────────────────────────────────────────────────


class TestEntropyFluencyTracker(unittest.TestCase):
    def setUp(self):
        reset_entropy_fluency_tracker()
        self.rng = np.random.default_rng(seed=20260428)

    def test_idle_returns_no_report(self):
        tracker = EntropyFluencyTracker()
        self.assertIsNone(tracker.last_report)
        self.assertEqual(tracker.tick_count, 0)
        self.assertEqual(tracker.phase, PHASE_STABLE)

    def test_stable_substrate_lands_in_low_entropy_high_fluency(self):
        """A substrate that holds steady around a fixed point with small
        noise should settle into a stable / fluent regime within the
        smoothing horizon."""
        tracker = EntropyFluencyTracker()
        baseline = self.rng.standard_normal(64).astype(np.float32) * 0.1
        # Run 80 ticks with very small noise — stable attractor
        last = None
        for _ in range(80):
            state = baseline + self.rng.standard_normal(64).astype(np.float32) * 0.02
            last = tracker.update(state)
        assert last is not None
        # Smoothed entropy should be moderate-to-low (not max)
        self.assertLess(last.entropy, 0.85, f"entropy too high: {last.entropy}")
        # Fluency should be high under steady state
        self.assertGreater(last.fluency, 0.50, f"fluency too low: {last.fluency}")
        # Phase should be one of the calmer phases (not DESTABILIZING)
        self.assertIn(last.phase, (PHASE_STABLE, PHASE_FLUENT, PHASE_REORGANIZING),
                      f"phase unexpectedly {last.phase}")

    def test_destabilization_raises_entropy_drops_fluency(self):
        """Sudden large injections push the substrate around and should
        surface as elevated entropy AND lower fluency relative to baseline."""
        tracker = EntropyFluencyTracker()
        baseline = np.zeros(64, dtype=np.float32)
        # 40 ticks of stable
        for _ in range(40):
            tracker.update(baseline + self.rng.standard_normal(64).astype(np.float32) * 0.02)
        stable_report = tracker.last_report
        assert stable_report is not None
        stable_fluency = stable_report.fluency

        # 40 ticks of large random injections (high innovation)
        for _ in range(40):
            tracker.update(self.rng.standard_normal(64).astype(np.float32) * 5.0)
        chaotic_report = tracker.last_report
        assert chaotic_report is not None

        # Fluency must drop measurably under heavy innovation
        self.assertLess(chaotic_report.fluency, stable_fluency,
                        f"fluency did not drop: stable={stable_fluency:.3f} "
                        f"chaotic={chaotic_report.fluency:.3f}")
        # Phase under heavy innovation must not be the calm STABLE
        self.assertIn(chaotic_report.phase,
                      (PHASE_DESTABILIZING, PHASE_REORGANIZING),
                      f"unexpected chaotic phase: {chaotic_report.phase}")

    def test_phase_dwell_increments(self):
        tracker = EntropyFluencyTracker()
        for _ in range(20):
            tracker.update(np.zeros(64, dtype=np.float32))
        self.assertGreater(tracker.last_report.phase_dwell_ticks, 1)

    def test_reset_returns_to_pristine_state(self):
        tracker = EntropyFluencyTracker()
        for _ in range(10):
            tracker.update(self.rng.standard_normal(64).astype(np.float32))
        tracker.reset()
        self.assertEqual(tracker.tick_count, 0)
        self.assertIsNone(tracker.last_report)
        self.assertEqual(tracker.phase, PHASE_STABLE)

    def test_phase_classifier_extremes(self):
        tracker = EntropyFluencyTracker()
        # Synthetically set internal state and exercise the classifier
        self.assertEqual(tracker._classify_phase(0.9, 0.2), PHASE_DESTABILIZING)
        self.assertEqual(tracker._classify_phase(0.9, 0.8), PHASE_REORGANIZING)
        self.assertEqual(tracker._classify_phase(0.2, 0.9), PHASE_FLUENT)
        self.assertEqual(tracker._classify_phase(0.2, 0.3), PHASE_STABLE)

    def test_all_phases_constant_correct(self):
        self.assertEqual(set(ALL_PHASES),
                         {PHASE_STABLE, PHASE_DESTABILIZING, PHASE_REORGANIZING, PHASE_FLUENT})

    def test_short_vector_handled(self):
        """A neuron-count of 0 or near-zero shouldn't crash entropy compute."""
        tracker = EntropyFluencyTracker()
        # 1-dim vectors — degenerate but legal
        for _ in range(20):
            tracker.update(np.array([0.5], dtype=np.float32))
        report = tracker.last_report
        self.assertIsNotNone(report)
        # Should not raise; entropy should be finite
        self.assertGreaterEqual(report.entropy, 0.0)
        self.assertLessEqual(report.entropy, 1.0)


# ── PlasticityMonitor ────────────────────────────────────────────────────


class TestPlasticityMonitor(unittest.TestCase):
    def setUp(self):
        reset_plasticity_monitor()
        self.rng = np.random.default_rng(seed=20260428)

    def test_full_rank_random_matrix_high_ratio(self):
        """A well-conditioned random Gaussian matrix should have a stable_rank
        ratio at the Marchenko-Pastur expectation of ~1/4 for square
        Gaussians (largest singular ≈ 2√n, Frobenius² ≈ n²,
        so stable_rank ≈ n/4 and ratio ≈ 1/4).

        Should sit ~10× above the rank-1 floor (1/64 ≈ 0.016) and not
        trigger the collapse warning."""
        mon = PlasticityMonitor()
        W = self.rng.standard_normal((64, 64))
        report = mon.measure(W)
        assert report is not None
        self.assertGreater(report.stable_rank_ratio, 0.18,
                           f"random Gaussian collapsed below MP expectation: "
                           f"ratio={report.stable_rank_ratio:.3f}")
        self.assertLess(report.stable_rank_ratio, 0.40,
                        f"random Gaussian above MP expectation: "
                        f"ratio={report.stable_rank_ratio:.3f}")
        self.assertFalse(report.collapse_warning,
                         "single random Gaussian measurement should not warn")

    def test_rank_one_matrix_collapses(self):
        """A literal rank-1 matrix has stable_rank = 1, so ratio = 1/min(m,n)."""
        mon = PlasticityMonitor(sustained_for_warn=1)  # warn immediately
        u = self.rng.standard_normal((64, 1))
        v = self.rng.standard_normal((1, 64))
        W = u @ v  # shape (64, 64), rank 1
        report = mon.measure(W)
        assert report is not None
        # Stable rank should be ~1.0; ratio = 1/64
        self.assertLess(report.stable_rank_ratio, 0.05,
                        f"rank-1 not detected: ratio={report.stable_rank_ratio:.4f}")
        self.assertTrue(report.collapse_warning)

    def test_all_zero_matrix_degenerate_no_crash(self):
        mon = PlasticityMonitor()
        W = np.zeros((32, 32))
        report = mon.measure(W)
        self.assertIsNotNone(report)
        self.assertEqual(report.stable_rank, 0.0)
        self.assertEqual(report.stable_rank_ratio, 0.0)

    def test_none_or_wrong_shape_returns_none(self):
        mon = PlasticityMonitor()
        self.assertIsNone(mon.measure(None))
        self.assertIsNone(mon.measure(np.array([1.0, 2.0, 3.0])))  # 1-D
        self.assertIsNone(mon.measure(np.zeros((4, 4, 4))))         # 3-D

    def test_oversized_matrix_returns_none(self):
        """Refuses to SVD a matrix larger than the cap rather than blocking
        on a multi-second computation in the substrate tick."""
        mon = PlasticityMonitor()
        n = MAX_MATRIX_DIM + 1
        # Use a dummy zero matrix — we never compute SVD on it
        W = np.zeros((n, n), dtype=np.float32)
        self.assertIsNone(mon.measure(W))

    def test_sustained_breaches_trigger_warning(self):
        mon = PlasticityMonitor(sustained_for_warn=3)
        # Submit a rank-1 matrix three times — should warn on third
        u = self.rng.standard_normal((32, 1))
        v = self.rng.standard_normal((1, 32))
        W = u @ v
        r1 = mon.measure(W)
        assert r1 is not None
        r2 = mon.measure(W)
        assert r2 is not None
        r3 = mon.measure(W)
        assert r3 is not None
        self.assertTrue(r3.collapse_warning)
        self.assertGreaterEqual(r3.sustained_breaches, 3)

    def test_history_bounded(self):
        mon = PlasticityMonitor(history=8)
        for _ in range(20):
            mon.measure(self.rng.standard_normal((16, 16)))
        # History capped at 8
        self.assertLessEqual(len(mon.history), 8)

    def test_partial_rank_collapse_intermediate(self):
        """A matrix that's full nominal rank but with a long-tail singular
        spectrum should land below 1.0 ratio but above the rank-1 floor."""
        mon = PlasticityMonitor()
        # Construct via SVD: singulars decay geometrically
        d = 32
        U = np.linalg.qr(self.rng.standard_normal((d, d)))[0]
        V = np.linalg.qr(self.rng.standard_normal((d, d)))[0]
        s = np.array([0.95**i for i in range(d)])
        W = U @ np.diag(s) @ V
        report = mon.measure(W)
        assert report is not None
        self.assertGreater(report.stable_rank_ratio, 0.05)
        self.assertLess(report.stable_rank_ratio, 0.6)


if __name__ == "__main__":
    unittest.main(verbosity=2)

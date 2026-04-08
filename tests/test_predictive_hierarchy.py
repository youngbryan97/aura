"""Tests for core/consciousness/predictive_hierarchy.py

Covers:
  - Initialization and level structure
  - Bottom-up error computation
  - Top-down prediction updates
  - Full tick cycle with all inputs
  - Free energy aggregation and smoothing
  - Precision adaptation (high-error → low precision)
  - Context block generation
  - Snapshot telemetry
  - Singleton accessor
  - Graceful degradation with None inputs
  - Weight norm clipping
  - Meta-level self-referential input
"""

import numpy as np
import pytest


class TestPredictiveLevel:
    """Unit tests for individual PredictiveLevel."""

    def _make_level(self, dim=16):
        from core.consciousness.predictive_hierarchy import PredictiveLevel
        return PredictiveLevel(name="test", index=0, dim=dim)

    def test_initialization(self):
        lv = self._make_level(dim=16)
        assert lv.name == "test"
        assert lv.dim == 16
        assert lv.prediction_vector.shape == (16,)
        assert lv.error_vector.shape == (16,)
        assert 0.0 < lv.precision < 1.0
        assert lv._W_pred.shape == (16, 16)

    def test_compute_error_basic(self):
        lv = self._make_level(dim=8)
        actual = np.ones(8, dtype=np.float32) * 0.5
        # prediction_vector starts at zero, so error = actual - 0 = actual
        error = lv.compute_error(actual)
        np.testing.assert_allclose(error, actual, atol=1e-6)

    def test_compute_error_updates_precision(self):
        lv = self._make_level(dim=8)
        initial_prec = lv.precision
        # Feed large error repeatedly → precision should drop
        for _ in range(20):
            lv.compute_error(np.ones(8, dtype=np.float32) * 2.0)
        assert lv.precision < initial_prec, "Precision should decrease with sustained high error"

    def test_compute_error_dimension_mismatch_pads(self):
        """If input has fewer dims than the level, it gets zero-padded."""
        lv = self._make_level(dim=16)
        short_input = np.ones(4, dtype=np.float32) * 0.7
        error = lv.compute_error(short_input)
        assert error.shape == (16,)
        # First 4 should be ~0.7 (actual - prediction of 0), rest ~0
        assert abs(error[0] - 0.7) < 0.01
        assert abs(error[15]) < 0.01

    def test_update_prediction_changes_state(self):
        lv = self._make_level(dim=8)
        initial = lv.prediction_vector.copy()
        parent_pred = np.random.randn(8).astype(np.float32)
        lv.update_prediction(parent_pred, learning_rate=0.5)
        assert not np.allclose(lv.prediction_vector, initial, atol=1e-6), \
            "Prediction should change after top-down update"

    def test_weighted_error_energy_zero_initially(self):
        lv = self._make_level(dim=8)
        # Before any error computation, error_vector is zero
        assert lv.weighted_error_energy() == 0.0

    def test_weighted_error_energy_positive_after_error(self):
        lv = self._make_level(dim=8)
        lv.compute_error(np.ones(8, dtype=np.float32))
        assert lv.weighted_error_energy() > 0.0

    def test_snapshot_format(self):
        lv = self._make_level(dim=8)
        lv.compute_error(np.ones(8, dtype=np.float32) * 0.3)
        snap = lv.get_snapshot()
        assert "name" in snap
        assert "precision" in snap
        assert "error_magnitude" in snap
        assert "prediction_magnitude" in snap
        assert "weighted_error" in snap
        assert isinstance(snap["precision"], float)


class TestPredictiveHierarchy:
    """Integration tests for the full hierarchy."""

    def _make_hierarchy(self, dim=16):
        from core.consciousness.predictive_hierarchy import PredictiveHierarchy
        return PredictiveHierarchy(dim=dim)

    def test_initialization(self):
        ph = self._make_hierarchy(dim=16)
        assert len(ph.levels) == 5
        assert ph.levels[0].name == "sensory"
        assert ph.levels[4].name == "meta"
        assert ph._tick_count == 0

    def test_level_names(self):
        ph = self._make_hierarchy()
        names = [lv.name for lv in ph.levels]
        assert names == ["sensory", "association", "executive", "narrative", "meta"]

    def test_tick_with_all_none_inputs(self):
        """Should not crash with no inputs (graceful degradation)."""
        ph = self._make_hierarchy(dim=8)
        fe = ph.tick()
        assert isinstance(fe, float)
        assert fe >= 0.0
        assert ph._tick_count == 1

    def test_tick_with_real_inputs(self):
        ph = self._make_hierarchy(dim=8)
        rng = np.random.default_rng(42)
        fe = ph.tick(
            sensory_input=rng.random(8).astype(np.float32),
            association_input=rng.random(8).astype(np.float32),
            executive_state=rng.random(8).astype(np.float32),
            narrative_state=rng.random(8).astype(np.float32),
        )
        assert fe > 0.0, "Non-zero inputs should produce non-zero free energy"

    def test_tick_reduces_error_over_time(self):
        """Repeated ticking with stable input should reduce total FE as
        the hierarchy learns to predict the input."""
        ph = self._make_hierarchy(dim=8)
        stable_input = np.ones(8, dtype=np.float32) * 0.3

        fe_early = []
        fe_late = []
        for i in range(100):
            fe = ph.tick(
                sensory_input=stable_input,
                association_input=stable_input,
                executive_state=stable_input,
                narrative_state=stable_input,
            )
            if i < 10:
                fe_early.append(fe)
            if i >= 90:
                fe_late.append(fe)

        avg_early = np.mean(fe_early)
        avg_late = np.mean(fe_late)
        assert avg_late < avg_early, (
            f"FE should decrease with learning: early={avg_early:.4f}, late={avg_late:.4f}"
        )

    def test_highest_error_level_changes(self):
        """Injecting high error at a specific level should make it the highest."""
        ph = self._make_hierarchy(dim=8)
        # Sensory gets large input, others zero
        ph.tick(
            sensory_input=np.ones(8, dtype=np.float32) * 5.0,
            association_input=np.zeros(8, dtype=np.float32),
            executive_state=np.zeros(8, dtype=np.float32),
            narrative_state=np.zeros(8, dtype=np.float32),
        )
        # Sensory should have highest error since prediction starts at zero
        # (meta may also be high due to error-of-errors)
        assert ph._highest_error_level in ("sensory", "meta")

    def test_precision_adapts_per_level(self):
        """Levels receiving consistent input should develop higher precision
        than levels receiving noisy input."""
        ph = self._make_hierarchy(dim=8)
        rng = np.random.default_rng(99)

        for _ in range(50):
            ph.tick(
                sensory_input=np.ones(8, dtype=np.float32) * 0.2,  # stable
                association_input=rng.random(8).astype(np.float32) * 2.0,  # noisy
                executive_state=np.ones(8, dtype=np.float32) * 0.2,  # stable
                narrative_state=rng.random(8).astype(np.float32) * 2.0,  # noisy
            )

        sensory_prec = ph.levels[0].precision
        assoc_prec = ph.levels[1].precision
        # Stable input → higher precision; noisy → lower precision
        assert sensory_prec > assoc_prec, (
            f"Stable level should have higher precision: sensory={sensory_prec}, assoc={assoc_prec}"
        )

    def test_meta_level_receives_error_magnitudes(self):
        """Meta level should respond to the error pattern of other levels."""
        ph = self._make_hierarchy(dim=8)
        # First tick to populate errors
        ph.tick(sensory_input=np.ones(8, dtype=np.float32))
        # Meta level should have non-zero prediction now
        meta = ph.levels[4]
        # After one tick, meta got error magnitudes as input, so it should
        # have computed a non-zero error
        err_mag = float(np.sqrt(np.mean(meta.error_vector ** 2)))
        # Might be zero on first tick if everything was zero; tick again
        ph.tick(sensory_input=np.ones(8, dtype=np.float32))
        err_mag2 = float(np.sqrt(np.mean(meta.error_vector ** 2)))
        # At least one tick should produce non-zero meta error
        assert err_mag > 0 or err_mag2 > 0, "Meta level should process error magnitudes"

    def test_get_context_block_empty_before_tick(self):
        ph = self._make_hierarchy(dim=8)
        assert ph.get_context_block() == ""

    def test_get_context_block_after_tick(self):
        ph = self._make_hierarchy(dim=8)
        ph.tick(sensory_input=np.ones(8, dtype=np.float32) * 0.5)
        ctx = ph.get_context_block()
        assert "PREDICTIVE HIERARCHY" in ctx
        assert "Highest-error" in ctx

    def test_get_snapshot_format(self):
        ph = self._make_hierarchy(dim=8)
        ph.tick()
        snap = ph.get_snapshot()
        assert "tick_count" in snap
        assert "total_free_energy" in snap
        assert "smoothed_fe" in snap
        assert "highest_error_level" in snap
        assert "levels" in snap
        assert len(snap["levels"]) == 5

    def test_get_level_precisions(self):
        ph = self._make_hierarchy(dim=8)
        ph.tick(sensory_input=np.ones(8, dtype=np.float32))
        precs = ph.get_level_precisions()
        assert len(precs) == 5
        assert "sensory" in precs
        assert "meta" in precs
        for v in precs.values():
            assert 0.0 < v < 1.0

    def test_weight_norm_clipping(self):
        """Extreme inputs should not cause weight explosion."""
        ph = self._make_hierarchy(dim=8)
        huge = np.ones(8, dtype=np.float32) * 100.0
        for _ in range(50):
            ph.tick(sensory_input=huge, association_input=huge,
                    executive_state=huge, narrative_state=huge)
        for lv in ph.levels:
            w_norm = float(np.linalg.norm(lv._W_pred))
            assert w_norm <= 11.0, f"Weight norm should be clipped: {w_norm}"

    def test_smoothed_fe_tracks_total(self):
        """Smoothed FE should roughly follow total FE."""
        ph = self._make_hierarchy(dim=8)
        for _ in range(30):
            ph.tick(sensory_input=np.ones(8, dtype=np.float32) * 0.5)
        assert ph._smoothed_fe > 0.0
        # Smoothed should be in the same order of magnitude
        assert ph._smoothed_fe < ph._total_free_energy * 5  # generous bound


class TestSingleton:
    def test_singleton_returns_same_instance(self):
        from core.consciousness.predictive_hierarchy import get_predictive_hierarchy
        a = get_predictive_hierarchy()
        b = get_predictive_hierarchy()
        assert a is b

    def test_singleton_is_predictive_hierarchy(self):
        from core.consciousness.predictive_hierarchy import (
            get_predictive_hierarchy, PredictiveHierarchy,
        )
        assert isinstance(get_predictive_hierarchy(), PredictiveHierarchy)

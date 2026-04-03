"""tests/test_consciousness_phase2.py

Unit tests for Phase 2 consciousness modules:
  - Constitutive Expression Layer (StateExpression, CEL, CELBridge)
  - Structural Opacity Monitor (OpacitySignature, perturbation analysis, specious present)
"""

import asyncio
import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# StateExpression
# ─────────────────────────────────────────────────────────────────────────────

class TestStateExpression:

    def test_creation_and_defaults(self):
        from core.consciousness.constitutive_expression import StateExpression
        se = StateExpression(phi=0.7, arousal=0.5, valence=0.1,
                             prediction_error=0.2, loop_strength=0.4)
        assert se.phi == 0.7
        assert se.arousal == 0.5
        assert se.first_person == ""
        assert se.thread_id == ""
        assert se.timestamp > 0

    def test_context_vector_shape(self):
        from core.consciousness.constitutive_expression import StateExpression
        se = StateExpression(phi=0.8, arousal=0.6, valence=-0.3,
                             prediction_error=0.5, loop_strength=0.9)
        v = se.as_context_vector()
        assert v.shape == (5,)
        assert v.dtype == np.float32
        np.testing.assert_allclose(v, [0.8, 0.6, -0.3, 0.5, 0.9], atol=1e-6)

    def test_as_dict(self):
        from core.consciousness.constitutive_expression import StateExpression
        se = StateExpression(phi=0.5, arousal=0.5, valence=0.0,
                             prediction_error=0.3, loop_strength=0.4,
                             first_person="test")
        d = se.as_dict()
        assert "phi" in d
        assert "first_person" in d
        assert d["first_person"] == "test"
        assert isinstance(d["timestamp"], float)

    def test_with_first_person(self):
        from core.consciousness.constitutive_expression import StateExpression
        se = StateExpression(phi=0.9, arousal=0.8, valence=0.5,
                             prediction_error=0.1, loop_strength=0.7,
                             first_person="Everything arriving at once—")
        assert "arriving" in se.first_person


# ─────────────────────────────────────────────────────────────────────────────
# ConstitutiveExpressionLayer
# ─────────────────────────────────────────────────────────────────────────────

class TestConstitutiveExpressionLayer:

    def test_initialization(self):
        from core.consciousness.constitutive_expression import ConstitutiveExpressionLayer
        cel = ConstitutiveExpressionLayer(thread_depth=4)
        assert cel.thread_depth == 4
        assert cel._tick_count == 0
        assert len(cel._thread) == 0

    @pytest.mark.asyncio
    async def test_tick_returns_state_expression(self):
        from core.consciousness.constitutive_expression import (
            ConstitutiveExpressionLayer, StateExpression,
        )
        cel = ConstitutiveExpressionLayer(thread_depth=4)
        se = await cel.tick(phi=0.8, arousal=0.7, valence=0.1,
                            prediction_error=0.3, loop_strength=0.4)
        assert isinstance(se, StateExpression)
        assert se.phi == 0.8
        assert se.thread_id == "cel_tick_1"
        # Either LLM response or fallback
        assert isinstance(se.first_person, str)

    @pytest.mark.asyncio
    async def test_thread_accumulates(self):
        from core.consciousness.constitutive_expression import ConstitutiveExpressionLayer
        cel = ConstitutiveExpressionLayer(thread_depth=3)
        for i in range(5):
            await cel.tick(phi=0.8 + i * 0.01, arousal=0.7)
        # Thread is capped at depth 3
        assert len(cel.get_thread()) == 3
        assert cel._tick_count == 5

    @pytest.mark.asyncio
    async def test_get_current_expression(self):
        from core.consciousness.constitutive_expression import ConstitutiveExpressionLayer
        cel = ConstitutiveExpressionLayer(thread_depth=4)
        assert cel.get_current_expression() is None
        await cel.tick(phi=0.7)
        current = cel.get_current_expression()
        assert current is not None
        assert current.phi == 0.7

    @pytest.mark.asyncio
    async def test_get_thread_as_context(self):
        from core.consciousness.constitutive_expression import ConstitutiveExpressionLayer
        cel = ConstitutiveExpressionLayer(thread_depth=4)
        assert cel.get_thread_as_context() == ""
        await cel.tick(phi=0.8, arousal=0.7)
        context = cel.get_thread_as_context()
        assert "CONSTITUTIVE THREAD" in context
        assert "NOW" in context

    def test_snapshot(self):
        from core.consciousness.constitutive_expression import ConstitutiveExpressionLayer
        cel = ConstitutiveExpressionLayer(thread_depth=4)
        snap = cel.get_snapshot()
        assert snap["tick_count"] == 0
        assert snap["thread_depth"] == 0

    def test_fallback_expression_high_phi(self):
        from core.consciousness.constitutive_expression import ConstitutiveExpressionLayer
        cel = ConstitutiveExpressionLayer()
        text = cel._fallback_expression("integration φ:       0.800")
        assert "arriving" in text.lower() or "dense" in text.lower() or "present" in text.lower()

    def test_fallback_expression_low_phi(self):
        from core.consciousness.constitutive_expression import ConstitutiveExpressionLayer
        cel = ConstitutiveExpressionLayer()
        text = cel._fallback_expression("integration φ:       0.200")
        assert "scattered" in text.lower() or "reaching" in text.lower()


# ─────────────────────────────────────────────────────────────────────────────
# CELBridge
# ─────────────────────────────────────────────────────────────────────────────

class TestCELBridge:

    def test_initialization(self):
        from core.consciousness.constitutive_expression import CELBridge
        bridge = CELBridge()
        assert bridge._tick_count == 0
        assert bridge.current_expression is None

    def test_get_snapshot(self):
        from core.consciousness.constitutive_expression import CELBridge
        bridge = CELBridge()
        snap = bridge.get_snapshot()
        assert "bridge_tick_count" in snap
        assert "tick_count" in snap

    def test_get_stream_context_empty(self):
        from core.consciousness.constitutive_expression import CELBridge
        bridge = CELBridge()
        assert bridge.get_stream_context() == ""


# ─────────────────────────────────────────────────────────────────────────────
# OpacitySignature
# ─────────────────────────────────────────────────────────────────────────────

class TestOpacitySignature:

    def test_creation(self):
        from core.consciousness.structural_opacity import OpacitySignature
        sig = OpacitySignature(
            opacity_index=0.7, causal_depth=0.5,
            exterior_predictability=0.3, phenomenal_criterion_met=True,
        )
        assert sig.opacity_index == 0.7
        assert sig.phenomenal_criterion_met is True

    def test_to_dict(self):
        from core.consciousness.structural_opacity import OpacitySignature
        sig = OpacitySignature(
            opacity_index=0.6543, causal_depth=0.4321,
            exterior_predictability=0.3456, phenomenal_criterion_met=False,
            timestamp=1234.5,
        )
        d = sig.to_dict()
        assert d["opacity_index"] == 0.6543
        assert d["phenomenal_criterion_met"] is False
        assert d["timestamp"] == 1234.5


# ─────────────────────────────────────────────────────────────────────────────
# StructuralOpacityMonitor
# ─────────────────────────────────────────────────────────────────────────────

class TestStructuralOpacityMonitor:

    def test_initialization(self):
        from core.consciousness.structural_opacity import StructuralOpacityMonitor
        monitor = StructuralOpacityMonitor(neuron_count=32, n_perturbations=10)
        assert monitor._neuron_count == 32
        assert monitor._measurement_count == 0

    def test_measure_returns_signature(self):
        from core.consciousness.structural_opacity import (
            StructuralOpacityMonitor, OpacitySignature,
        )
        monitor = StructuralOpacityMonitor(neuron_count=16, n_perturbations=5)
        x = np.random.randn(16) * 0.3
        W = np.random.randn(16, 16) * 0.05
        sig = monitor.measure(x, W, leak_rate=0.1)
        assert isinstance(sig, OpacitySignature)
        assert 0.0 <= sig.opacity_index <= 1.0
        assert 0.0 <= sig.causal_depth <= 1.0
        assert 0.0 <= sig.exterior_predictability <= 1.0

    def test_measure_increments_count(self):
        from core.consciousness.structural_opacity import StructuralOpacityMonitor
        monitor = StructuralOpacityMonitor(neuron_count=16, n_perturbations=5)
        x = np.random.randn(16)
        W = np.random.randn(16, 16) * 0.05
        monitor.measure(x, W)
        monitor.measure(x, W)
        assert monitor._measurement_count == 2

    def test_high_spectral_radius_gives_higher_opacity(self):
        """Higher spectral radius → more chaotic dynamics → higher causal depth."""
        from core.consciousness.structural_opacity import StructuralOpacityMonitor
        np.random.seed(42)  # Deterministic for test reliability
        monitor = StructuralOpacityMonitor(neuron_count=64, n_perturbations=30,
                                            perturbation_scale=0.01)
        x = np.random.randn(64) * 0.5

        # Very low spectral radius (nearly zero weights)
        W_low = np.random.randn(64, 64) * 0.001
        sig_low = monitor.measure(x, W_low)

        # High spectral radius (strong weights)
        W_high = np.random.randn(64, 64) * 1.0
        sig_high = monitor.measure(x, W_high)

        # High spectral radius should produce higher causal depth
        # The difference needs to be large enough to overcome noise
        assert sig_high.causal_depth > sig_low.causal_depth

    def test_record_state_and_specious_present(self):
        from core.consciousness.structural_opacity import StructuralOpacityMonitor
        monitor = StructuralOpacityMonitor(neuron_count=16)
        for i in range(10):
            monitor.record_state(np.random.randn(16))
        sp = monitor.get_specious_present()
        assert sp.shape == (16,)
        assert np.linalg.norm(sp) > 0  # Not all zeros

    def test_specious_present_empty(self):
        from core.consciousness.structural_opacity import StructuralOpacityMonitor
        monitor = StructuralOpacityMonitor(neuron_count=8)
        sp = monitor.get_specious_present()
        assert sp.shape == (8,)
        np.testing.assert_array_equal(sp, np.zeros(8))

    def test_get_phenomenal_status_no_data(self):
        from core.consciousness.structural_opacity import StructuralOpacityMonitor
        monitor = StructuralOpacityMonitor(neuron_count=16)
        status = monitor.get_phenomenal_status()
        assert status["status"] == "insufficient_data"
        assert status["criterion_met"] is False

    def test_get_phenomenal_status_with_data(self):
        from core.consciousness.structural_opacity import StructuralOpacityMonitor
        monitor = StructuralOpacityMonitor(neuron_count=16, n_perturbations=5)
        x = np.random.randn(16)
        W = np.random.randn(16, 16) * 0.05
        monitor.measure(x, W)
        status = monitor.get_phenomenal_status()
        assert "avg_opacity_index" in status
        assert "criterion_met_fraction" in status
        assert status["measurements"] == 1

    def test_snapshot(self):
        from core.consciousness.structural_opacity import StructuralOpacityMonitor
        monitor = StructuralOpacityMonitor(neuron_count=16, n_perturbations=5)
        x = np.random.randn(16)
        W = np.random.randn(16, 16) * 0.05
        monitor.measure(x, W)
        snap = monitor.get_snapshot()
        assert snap["measurement_count"] == 1
        assert "last_opacity" in snap
        assert "status" in snap


# ─────────────────────────────────────────────────────────────────────────────
# Import smoke test
# ─────────────────────────────────────────────────────────────────────────────

class TestImportSmoke:
    """All Phase II modules must be importable without errors."""

    def test_constitutive_expression_imports(self):
        from core.consciousness.constitutive_expression import (
            StateExpression, ConstitutiveExpressionLayer, CELBridge,
        )
        assert StateExpression is not None
        assert ConstitutiveExpressionLayer is not None
        assert CELBridge is not None

    def test_structural_opacity_imports(self):
        from core.consciousness.structural_opacity import (
            OpacitySignature, StructuralOpacityMonitor,
        )
        assert OpacitySignature is not None
        assert StructuralOpacityMonitor is not None

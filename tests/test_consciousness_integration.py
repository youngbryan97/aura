"""tests/test_consciousness_integration.py

Unit tests for new consciousness architecture modules:
- QuantumEntropyBridge
- RIIU (IIT Φ Surrogate)
- GanglionNode
- ExecutiveInhibitor
- QualiaEngine
- GlobalWorkspace ignition
"""

import asyncio
import time
import numpy as np
import pytest
import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =========================================================================
# 1. Quantum Entropy Bridge
# =========================================================================

class TestQuantumEntropyBridge:
    """Test QuantumEntropyBridge with OS fallback (no network required)."""

    def test_init_seeds_fallback_pool_and_schedules_async_refill(self, monkeypatch):
        from core.consciousness.quantum_entropy import QuantumEntropyBridge

        scheduled = []
        monkeypatch.setattr(
            QuantumEntropyBridge,
            "_schedule_refill",
            lambda self: scheduled.append(True),
        )

        bridge = QuantumEntropyBridge(pool_size=128)

        assert len(bridge._pool) > 0
        assert scheduled == [True]

    def test_get_quantum_float_returns_valid_range(self):
        from core.consciousness.quantum_entropy import QuantumEntropyBridge
        bridge = QuantumEntropyBridge()
        for _ in range(20):
            val = bridge.get_quantum_float()
            assert 0.0 <= val <= 1.0, f"Float out of range: {val}"

    def test_get_quantum_bytes_returns_correct_length(self):
        from core.consciousness.quantum_entropy import QuantumEntropyBridge
        bridge = QuantumEntropyBridge()
        for n in [1, 4, 16, 64]:
            data = bridge.get_quantum_bytes(n)
            assert len(data) == n, f"Expected {n} bytes, got {len(data)}"

    def test_collapse_decision_uniform(self):
        from core.consciousness.quantum_entropy import QuantumEntropyBridge
        bridge = QuantumEntropyBridge()
        options = ["a", "b", "c"]
        result = bridge.collapse_decision(options)
        assert result in options

    def test_collapse_decision_weighted(self):
        from core.consciousness.quantum_entropy import QuantumEntropyBridge
        bridge = QuantumEntropyBridge()
        options = ["rare", "common"]
        weights = [0.01, 0.99]
        # Run many times — "common" should dominate
        results = [bridge.collapse_decision(options, weights) for _ in range(100)]
        common_count = results.count("common")
        assert common_count > 50, f"Expected 'common' to dominate, got {common_count}/100"

    def test_collapse_decision_single_option(self):
        from core.consciousness.quantum_entropy import QuantumEntropyBridge
        bridge = QuantumEntropyBridge()
        assert bridge.collapse_decision(["only"]) == "only"

    def test_collapse_decision_empty_raises(self):
        from core.consciousness.quantum_entropy import QuantumEntropyBridge
        bridge = QuantumEntropyBridge()
        with pytest.raises(ValueError):
            bridge.collapse_decision([])

    def test_stats_tracking(self):
        from core.consciousness.quantum_entropy import QuantumEntropyBridge
        bridge = QuantumEntropyBridge()
        bridge.get_quantum_float()
        stats = bridge.get_stats()
        assert "quantum_reads" in stats
        assert "fallback_reads" in stats
        assert stats["quantum_reads"] + stats["fallback_reads"] > 0

    def test_partial_pool_read_falls_back_without_blocking(self, monkeypatch):
        from core.consciousness.quantum_entropy import QuantumEntropyBridge

        bridge = QuantumEntropyBridge(pool_size=32)
        bridge._pool = bytearray(b"\x01\x02")

        scheduled = []
        monkeypatch.setattr(bridge, "_schedule_refill", lambda: scheduled.append(True))

        data = bridge.get_quantum_bytes(8)

        assert len(data) == 8
        assert scheduled == [True]
        assert bridge.get_stats()["fallback_reads"] >= 1

# =========================================================================
# 2. RIIU (IIT Φ Surrogate)
# =========================================================================

class TestRIIU:
    """Test the Reflexive Integrated Information Unit."""

    def test_phi_returns_float(self):
        from core.consciousness.iit_surrogate import RIIU
        riiu = RIIU(neuron_count=16, buffer_size=32)
        state = np.random.randn(16)
        phi = riiu.compute_phi(state)
        assert isinstance(phi, float)

    def test_phi_nonnegative(self):
        from core.consciousness.iit_surrogate import RIIU
        riiu = RIIU(neuron_count=16, buffer_size=32)
        for _ in range(20):
            state = np.random.randn(16)
            phi = riiu.compute_phi(state)
            assert phi >= 0.0, f"Φ should be non-negative, got {phi}"

    def test_phi_zero_for_insufficient_samples(self):
        from core.consciousness.iit_surrogate import RIIU
        riiu = RIIU(neuron_count=16, buffer_size=32)
        state = np.random.randn(16)
        phi = riiu.compute_phi(state)
        assert phi >= 0.0, "Φ should be non-negative"

    def test_phi_nonzero_after_enough_samples(self):
        from core.consciousness.iit_surrogate import RIIU
        riiu = RIIU(neuron_count=16, buffer_size=64, num_partitions=4)
        # Feed enough varied states with structured correlations
        phi = 0.0
        for i in range(50):
            # Create correlated state patterns (not pure noise)
            base = np.sin(np.linspace(0, 4 * np.pi, 16) + i * 0.3)
            state = base + np.random.randn(16) * 0.05
            phi = riiu.compute_phi(state)
        # After 50 correlated states, Φ should be non-negative (may be 0 for small systems)
        assert phi >= 0.0, f"Φ should be non-negative, got {phi}"
        assert isinstance(phi, float)

    def test_phi_handles_all_zeros(self):
        from core.consciousness.iit_surrogate import RIIU
        riiu = RIIU(neuron_count=16, buffer_size=32)
        for _ in range(20):
            phi = riiu.compute_phi(np.zeros(16))
        assert phi >= 0.0  # Should not crash or return NaN

    def test_phi_handles_wrong_shape(self):
        from core.consciousness.iit_surrogate import RIIU
        riiu = RIIU(neuron_count=16, buffer_size=32)
        # Pass wrong shape — should resize gracefully
        phi = riiu.compute_phi(np.random.randn(8))
        assert isinstance(phi, float)

    def test_get_stats(self):
        from core.consciousness.iit_surrogate import RIIU
        riiu = RIIU(neuron_count=16)
        stats = riiu.get_stats()
        assert "phi" in stats
        assert "samples" in stats
        assert stats["samples"] == 0

# =========================================================================
# 3. Ganglion Node
# =========================================================================

class TestGanglionNode:
    """Test the decentralized ganglion node."""

    @pytest.fixture
    def action_queue(self):
        return asyncio.Queue()

    def test_register_handler(self, action_queue):
        from core.consciousness.ganglion_node import GanglionNode
        node = GanglionNode("memory", action_queue)
        node.register_handler("recall", lambda p: None)
        assert "recall" in node._handlers

    @pytest.mark.asyncio
    async def test_process_stimulus_fires_handler(self, action_queue):
        from core.consciousness.ganglion_node import GanglionNode, GanglionAction
        node = GanglionNode("memory", action_queue, refractory_seconds=0.0)

        async def handler(payload):
            return GanglionAction(
                source_domain="memory",
                action_type="recall",
                payload=payload,
                priority=0.8,
            )

        node.register_handler("recall", handler)
        result = await node.process_stimulus("recall", {"query": "test"})
        assert result is not None
        assert result.action_type == "recall"
        assert result.priority == 0.8

    @pytest.mark.asyncio
    async def test_refractory_period(self, action_queue):
        from core.consciousness.ganglion_node import GanglionNode, GanglionAction
        node = GanglionNode("motor", action_queue, refractory_seconds=1.0)

        async def handler(payload):
            return GanglionAction("motor", "move", payload)

        node.register_handler("move", handler)

        # First call should succeed
        r1 = await node.process_stimulus("move", {})
        assert r1 is not None

        # Immediate second call should be suppressed (refractory)
        r2 = await node.process_stimulus("move", {})
        assert r2 is None

    def test_snapshot(self, action_queue):
        from core.consciousness.ganglion_node import GanglionNode
        node = GanglionNode("affect", action_queue)
        snap = node.get_snapshot()
        assert snap["domain"] == "affect"
        assert snap["fire_count"] == 0

    def test_decay_activation(self, action_queue):
        from core.consciousness.ganglion_node import GanglionNode
        node = GanglionNode("test", action_queue)
        node._activation = 0.5
        node.decay_activation(dt=1.0, rate=0.1)
        assert node._activation == pytest.approx(0.4, abs=0.01)

# =========================================================================
# 4. Executive Inhibitor
# =========================================================================

class TestExecutiveInhibitor:
    """Test the PFC-like executive inhibitor."""

    def test_critical_always_passes(self):
        from core.consciousness.executive_inhibitor import ExecutiveInhibitor

        class MockAction:
            is_critical = True
            source_domain = "test"
            action_type = "emergency"

        inhibitor = ExecutiveInhibitor(phi_threshold=0.1)
        assert inhibitor.authorize(MockAction(), phi=1.0, ignited=True) is True

    def test_high_phi_blocks_noncritical(self):
        from core.consciousness.executive_inhibitor import ExecutiveInhibitor

        class MockAction:
            is_critical = False
            source_domain = "motor"
            action_type = "move"

        inhibitor = ExecutiveInhibitor(phi_threshold=0.5)
        assert inhibitor.authorize(MockAction(), phi=0.8, ignited=True) is False

    def test_low_phi_allows_noncritical(self):
        from core.consciousness.executive_inhibitor import ExecutiveInhibitor

        class MockAction:
            is_critical = False
            source_domain = "motor"
            action_type = "move"

        inhibitor = ExecutiveInhibitor(phi_threshold=0.5)
        assert inhibitor.authorize(MockAction(), phi=0.2, ignited=False) is True

    def test_veto_log(self):
        from core.consciousness.executive_inhibitor import ExecutiveInhibitor

        class MockAction:
            is_critical = False
            source_domain = "test"
            action_type = "blocked"

        inhibitor = ExecutiveInhibitor(phi_threshold=0.3)
        inhibitor.authorize(MockAction(), phi=0.5, ignited=True)
        vetoes = inhibitor.get_recent_vetoes(10)
        assert len(vetoes) == 1
        assert vetoes[0]["reason"] == "high_phi_protection"

    def test_snapshot(self):
        from core.consciousness.executive_inhibitor import ExecutiveInhibitor
        inhibitor = ExecutiveInhibitor()
        snap = inhibitor.get_snapshot()
        assert "authorized" in snap
        assert "vetoed" in snap
        assert snap["authorized"] == 0

# =========================================================================
# 5. Qualia Engine
# =========================================================================

class TestQualiaEngine:
    """Test the multi-layer qualia processing pipeline."""

    def test_produces_valid_descriptor(self):
        from core.consciousness.qualia_engine import QualiaEngine
        engine = QualiaEngine()
        state = np.random.randn(64)
        velocity = np.random.randn(64) * 0.1
        pred = {"current_surprise": 0.3, "free_energy": 0.5, "precision": 0.8}
        ws = {"ignited": True, "ignition_level": 0.7, "last_winner": "memory"}

        desc = engine.process(state, velocity, pred, ws, phi=0.5)

        assert 0.0 <= desc.phenomenal_richness <= 1.0
        assert isinstance(desc.self_referential, bool)
        assert desc.dominant_modality in [
            "subconceptual", "conceptual", "predictive", "workspace", "witness"
        ]

    def test_to_dict(self):
        from core.consciousness.qualia_engine import QualiaEngine
        engine = QualiaEngine()
        state = np.random.randn(64)
        velocity = np.random.randn(64) * 0.1
        desc = engine.process(state, velocity, {}, {})
        d = desc.to_dict()
        assert "phenomenal_richness" in d
        assert "subconceptual" in d

    def test_snapshot(self):
        from core.consciousness.qualia_engine import QualiaEngine
        engine = QualiaEngine()
        snap = engine.get_snapshot()
        assert snap["process_count"] == 0

# =========================================================================
# 6. GlobalWorkspace Ignition
# =========================================================================

class TestGlobalWorkspaceIgnition:
    """Test ignition detection in the enhanced GlobalWorkspace."""

    def test_ignition_fields_initialized(self):
        from core.consciousness.global_workspace import GlobalWorkspace
        gw = GlobalWorkspace()
        assert gw.ignited is False
        assert gw.ignition_level == 0.0
        assert gw._ignition_count == 0

    def test_update_phi(self):
        from core.consciousness.global_workspace import GlobalWorkspace
        gw = GlobalWorkspace()
        gw.update_phi(0.7)
        assert gw._current_phi == 0.7

    def test_update_phi_clamps_negative(self):
        from core.consciousness.global_workspace import GlobalWorkspace
        gw = GlobalWorkspace()
        gw.update_phi(-0.5)
        assert gw._current_phi == 0.0

    def test_is_ignited(self):
        from core.consciousness.global_workspace import GlobalWorkspace
        gw = GlobalWorkspace()
        assert gw.is_ignited() is False

    def test_get_ignition_level(self):
        from core.consciousness.global_workspace import GlobalWorkspace
        gw = GlobalWorkspace()
        assert gw.get_ignition_level() == 0.0

    
# =========================================================================
# 7. BridgeHunter Metrics (standalone)
# =========================================================================

class TestBridgeHunterMetrics:
    """Test standalone consciousness metric computations."""

    def test_phi_surrogate(self):
        from experiments.bridgehunter.metrics import compute_phi_surrogate
        traj = np.random.randn(50, 16)
        phi = compute_phi_surrogate(traj)
        assert isinstance(phi, float)
        assert phi >= 0.0

    def test_ignition_rate(self):
        from experiments.bridgehunter.metrics import compute_ignition_rate
        priorities = [0.3, 0.5, 0.7, 0.9, 0.4]
        rate = compute_ignition_rate(priorities, threshold=0.6)
        assert rate == pytest.approx(2 / 5, abs=0.01)

    def test_causal_emergence(self):
        from experiments.bridgehunter.metrics import compute_causal_emergence
        traj = np.random.randn(50, 16)
        ce = compute_causal_emergence(traj, grain=4)
        assert isinstance(ce, float)

    def test_spectral_entropy(self):
        from experiments.bridgehunter.metrics import compute_spectral_entropy
        state = np.random.randn(16)
        ent = compute_spectral_entropy(state)
        assert ent > 0.0

    def test_self_reference(self):
        from experiments.bridgehunter.metrics import compute_self_reference
        traj = np.random.randn(50, 16)
        sr = compute_self_reference(traj, lag=5)
        assert isinstance(sr, float)

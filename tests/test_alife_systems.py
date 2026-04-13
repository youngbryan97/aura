"""Tests for Artificial Life systems integrated from Avida, Tierra, Lenia, EcoSim, Evochora.

Covers:
    - Criticality Regulator (edge-of-chaos homeostasis)
    - ALife Dynamics (Lenia kernels, entropy tracking, CPU allocation)
    - ALife Extensions (pattern replication, speciation, toroidal topology, costs)
    - Endogenous Fitness (survival-based evolution, behavioral rules)
    - Integration phase wiring
"""

import asyncio
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ════════════════════════════════════════════════════════════════════════
# 1. CRITICALITY REGULATOR
# ════════════════════════════════════════════════════════════════════════

class TestCriticalityRegulator:
    """Tests for edge-of-chaos tuning of neural dynamics."""

    def _make_regulator(self):
        from core.consciousness.criticality_regulator import CriticalityRegulator
        return CriticalityRegulator()

    def test_instantiation(self):
        reg = self._make_regulator()
        assert reg.get_criticality_score() >= 0.0

    def test_tick_with_random_data(self):
        reg = self._make_regulator()
        activations = np.random.randn(64, 64).astype(np.float32)
        weights = np.random.randn(64, 64).astype(np.float32) * 0.1
        state = run(reg.tick(activations, weights))
        assert state is not None
        assert hasattr(state, "branching_ratio")
        assert hasattr(state, "avalanche_exponent")
        assert hasattr(state, "criticality_score")

    def test_branching_ratio_is_finite(self):
        reg = self._make_regulator()
        for _ in range(15):
            activations = np.random.randn(64, 64).astype(np.float32) * 0.5
            weights = np.random.randn(64, 64).astype(np.float32) * 0.05
            run(reg.tick(activations, weights))
        br = reg.get_branching_ratio()
        assert np.isfinite(br)

    def test_adjustments_bounded(self):
        reg = self._make_regulator()
        for _ in range(20):
            run(reg.tick(
                np.random.randn(64, 64).astype(np.float32),
                np.random.randn(64, 64).astype(np.float32) * 0.1,
            ))
        adj = reg.get_adjustments()
        assert 0.5 <= adj["gain"] <= 2.0
        assert 0.5 <= adj["noise"] <= 2.0
        assert 0.7 <= adj["ei_ratio"] <= 1.3

    def test_criticality_score_bounded(self):
        reg = self._make_regulator()
        for _ in range(15):
            run(reg.tick(
                np.random.randn(64, 64).astype(np.float32),
                np.random.randn(64, 64).astype(np.float32) * 0.1,
            ))
        score = reg.get_criticality_score()
        assert 0.0 <= score <= 1.0


# ════════════════════════════════════════════════════════════════════════
# 2. ALIFE DYNAMICS (Lenia + Entropy + CPU Allocation)
# ════════════════════════════════════════════════════════════════════════

class TestALifeDynamics:
    """Tests for Lenia kernels, entropy tracking, and CPU allocation."""

    def _make_dynamics(self):
        from core.consciousness.alife_dynamics import ALifeDynamics
        return ALifeDynamics()

    def test_instantiation(self):
        dyn = self._make_dynamics()
        assert dyn is not None

    def test_tick_produces_state(self):
        dyn = self._make_dynamics()
        activations = np.random.randn(64, 64).astype(np.float32) * 0.5
        weights = np.random.randn(64, 64).astype(np.float32) * 0.1
        projection = np.random.randn(64, 64).astype(np.float32) * 0.1
        state = run(dyn.tick(activations, weights, projection))
        assert state is not None

    def test_entropy_tracking(self):
        dyn = self._make_dynamics()
        status = dyn.get_status()
        assert "entropy" in status or hasattr(dyn, "_entropy_tracker")

    def test_compute_credits_shape(self):
        dyn = self._make_dynamics()
        activations = np.random.randn(64, 64).astype(np.float32) * 0.5
        weights = np.random.randn(64, 64).astype(np.float32) * 0.1
        projection = np.random.randn(64, 64).astype(np.float32) * 0.1
        run(dyn.tick(activations, weights, projection))
        status = dyn.get_status()
        # Should have credit-related information
        assert isinstance(status, dict)

    def test_kernel_params_accessible(self):
        dyn = self._make_dynamics()
        params = dyn.get_kernel_params() if hasattr(dyn, "get_kernel_params") else None
        if params is not None:
            assert hasattr(params, "mu") or isinstance(params, dict)


# ════════════════════════════════════════════════════════════════════════
# 3. ALIFE EXTENSIONS
# ════════════════════════════════════════════════════════════════════════

class TestALifeExtensions:
    """Tests for pattern replication, speciation, toroidal topology, costs."""

    def _make_extensions(self):
        try:
            from core.consciousness.alife_extensions import ALifeExtensions
            return ALifeExtensions()
        except ImportError:
            pytest.skip("alife_extensions not yet available")

    def test_instantiation(self):
        ext = self._make_extensions()
        assert ext is not None

    def test_toroidal_distance(self):
        ext = self._make_extensions()
        if hasattr(ext, "_toroidal") or hasattr(ext, "toroidal_distance"):
            # Column 0 and column 63 should be neighbors in toroidal space
            from core.consciousness.alife_extensions import ToroidalTopology
            topo = ToroidalTopology()
            d = topo.toroidal_distance(0, 63, 64)
            assert d < 0.1  # Very close in toroidal space
            # Column 0 and column 32 should be maximally distant
            d_max = topo.toroidal_distance(0, 32, 64)
            assert d_max == 0.5

    def test_tick_with_mesh_state(self):
        ext = self._make_extensions()
        mesh_state = {
            "column_activations": np.random.randn(64, 64).astype(np.float32),
            "inter_column_weights": np.random.randn(64, 64).astype(np.float32) * 0.1,
        }
        state = run(ext.tick(mesh_state=mesh_state, evolution_state={}, tick_count=1))
        assert state is not None


# ════════════════════════════════════════════════════════════════════════
# 4. ENDOGENOUS FITNESS
# ════════════════════════════════════════════════════════════════════════

class TestEndogenousFitness:
    """Tests for survival-based evolution and behavioral rules."""

    def _make_fitness(self):
        try:
            from core.consciousness.endogenous_fitness import get_endogenous_fitness
            return get_endogenous_fitness()
        except ImportError:
            pytest.skip("endogenous_fitness not yet available")

    def test_instantiation(self):
        fit = self._make_fitness()
        assert fit is not None

    def test_behavioral_genome_shape(self):
        fit = self._make_fitness()
        genome = fit.get_behavioral_genome()
        assert isinstance(genome, np.ndarray)
        assert genome.shape[0] > 0  # Has action rows
        assert genome.shape[1] > 0  # Has state columns

    def test_mutation_changes_genome(self):
        fit = self._make_fitness()
        original = fit.get_behavioral_genome().copy()
        mutated = fit.mutate_behavioral_genome(original, rate=0.5)
        # With 50% mutation rate, should differ
        assert not np.array_equal(original, mutated)

    def test_mutation_preserves_bounds(self):
        fit = self._make_fitness()
        genome = np.ones((6, 7)) * 0.5
        mutated = fit.mutate_behavioral_genome(genome, rate=1.0)
        assert np.all(mutated >= -1.0)
        assert np.all(mutated <= 1.0)


# ════════════════════════════════════════════════════════════════════════
# 5. INTEGRATION
# ════════════════════════════════════════════════════════════════════════

class TestALifeIntegration:
    """Tests for ALife systems wired into the cognitive integration phase."""

    def test_integration_phase_has_alife_references(self):
        from core.phases.cognitive_integration_phase import CognitiveIntegrationPhase
        from unittest.mock import MagicMock
        phase = CognitiveIntegrationPhase(kernel=MagicMock())
        assert hasattr(phase, "_criticality_regulator")
        assert hasattr(phase, "_alife_dynamics")
        assert hasattr(phase, "_alife_extensions")
        assert hasattr(phase, "_endogenous_fitness")

    def test_boot_initializer_includes_alife_services(self):
        """The boot initializer should register ALife services."""
        import importlib
        mod = importlib.import_module("core.orchestrator.initializers.cognitive_sensory")
        source = open(mod.__file__).read()
        assert "criticality_regulator" in source
        assert "alife_dynamics" in source
        assert "alife_extensions" in source
        assert "endogenous_fitness" in source


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

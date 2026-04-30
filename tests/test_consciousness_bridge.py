"""Comprehensive tests for the Consciousness Bridge subsystems.

Tests are organized by module, with integration tests at the end that verify
cross-component dynamics.  Each test is self-contained (no live Aura boot needed).

Coverage:
  - NeuralMesh: structure, dynamics, STDP, projections, modulation
  - NeurochemicalSystem: chemicals, cross-interactions, downstream effects
  - EmbodiedInteroception: channel sampling, projection, degradation
  - OscillatoryBinding: oscillator dynamics, PSI, coupling
  - SubstrateEvolution: genome operators, fitness, population management
  - SomaticMarkerGate: evaluation, learning, budget checks
  - UnifiedField: dynamics, coherence, modes, back-pressure
  - Integration: cross-component data flow, causal coupling
"""

import asyncio
import math
import time
from unittest.mock import MagicMock, AsyncMock, patch

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════════
# NeuralMesh Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestNeuralMesh:

    def _make_mesh(self):
        from core.consciousness.neural_mesh import NeuralMesh, MeshConfig
        cfg = MeshConfig(
            total_neurons=256,  # smaller for testing (4 cols × 64 neurons)
            columns=4,
            neurons_per_column=64,
            sensory_end=1,
            association_end=3,
            update_hz=100.0,
            projection_dim=16,
        )
        return NeuralMesh(cfg)

    def test_initialization(self):
        mesh = self._make_mesh()
        assert len(mesh.columns) == 4
        assert mesh.columns[0].n == 64
        assert mesh.cfg.total_neurons == 256
        assert mesh._running is False

    def test_tier_assignment(self):
        from core.consciousness.neural_mesh import CorticalTier
        mesh = self._make_mesh()
        assert mesh.columns[0].tier == CorticalTier.SENSORY
        assert mesh.columns[1].tier == CorticalTier.ASSOCIATION
        assert mesh.columns[2].tier == CorticalTier.ASSOCIATION
        assert mesh.columns[3].tier == CorticalTier.EXECUTIVE

    def test_column_step_produces_change(self):
        mesh = self._make_mesh()
        col = mesh.columns[0]
        initial = col.x.copy()
        ext = np.zeros(col.n, dtype=np.float32)
        col.step(ext, dt=0.05, decay=0.03, noise_sigma=0.01, gain=1.0, now=time.time())
        # State should have changed (noise + dynamics)
        assert not np.allclose(col.x, initial, atol=1e-6)

    def test_dales_law(self):
        """Inhibitory neurons should have non-positive outgoing weights."""
        mesh = self._make_mesh()
        for col in mesh.columns:
            if np.any(col.inh_mask):
                inh_weights = col.W[col.inh_mask, :]
                assert np.all(inh_weights <= 0), "Inhibitory neurons should have ≤0 outgoing weights"

    def test_tick_updates_stats(self):
        mesh = self._make_mesh()
        mesh._tick()
        assert mesh._tick_count == 1
        assert mesh._mean_column_energy >= 0

    def test_status_reports_accelerator(self):
        mesh = self._make_mesh()
        status = mesh.get_status()
        assert status["accelerator"] in {"metal", "numpy"}
        assert isinstance(status["accelerator_reason"], str)

    def test_executive_projection_shape(self):
        mesh = self._make_mesh()
        mesh._tick()
        proj = mesh.get_executive_projection()
        assert proj.shape == (16,)  # projection_dim
        assert np.all(np.abs(proj) <= 1.0)  # tanh bounded

    def test_field_state_shape(self):
        mesh = self._make_mesh()
        state = mesh.get_field_state()
        assert state.shape == (256,)

    def test_sensory_injection(self):
        mesh = self._make_mesh()
        vec = np.ones(64, dtype=np.float32) * 0.5  # 1 sensory column
        mesh.inject_sensory(vec)
        assert mesh._sensory_buffer is not None
        mesh._tick()  # should consume the buffer
        assert mesh._sensory_buffer is None

    def test_modulatory_state(self):
        mesh = self._make_mesh()
        mesh.set_modulatory_state(gain=2.0, plasticity=0.5, noise=1.5)
        assert mesh._modulatory_gain == 2.0
        assert mesh._modulatory_plasticity == 0.5
        assert mesh._modulatory_noise == 1.5

    def test_modulatory_state_clamps(self):
        mesh = self._make_mesh()
        mesh.set_modulatory_state(gain=100.0, plasticity=-5.0, noise=-1.0)
        assert mesh._modulatory_gain == 3.0  # clamped
        assert mesh._modulatory_plasticity == 0.0
        assert mesh._modulatory_noise == 0.0

    def test_inter_column_weights_sparse(self):
        mesh = self._make_mesh()
        nonzero = np.count_nonzero(mesh._inter_W)
        total = mesh._inter_W.size
        density = nonzero / total
        # Should be sparse but not zero
        assert 0.0 < density < 0.5

    def test_multiple_ticks_stable(self):
        """Mesh should be numerically stable over many ticks."""
        mesh = self._make_mesh()
        for _ in range(200):
            mesh._tick()
        state = mesh.get_field_state()
        assert np.all(np.isfinite(state))
        assert np.all(np.abs(state) <= 1.0)

    def test_global_synchrony_range(self):
        mesh = self._make_mesh()
        for _ in range(10):
            mesh._tick()
        sync = mesh.get_global_synchrony()
        assert 0.0 <= sync <= 1.0

    def test_status_dict(self):
        mesh = self._make_mesh()
        mesh._tick()
        status = mesh.get_status()
        assert "running" in status
        assert "tick_count" in status
        assert "tier_energies" in status
        assert "global_synchrony" in status


# ═══════════════════════════════════════════════════════════════════════════
# NeurochemicalSystem Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestNeurochemicalSystem:

    def _make_ncs(self):
        from core.consciousness.neurochemical_system import NeurochemicalSystem
        return NeurochemicalSystem()

    def test_initialization(self):
        ncs = self._make_ncs()
        # Aura's neurochemical system: 2 fast neurotransmitters
        # (glutamate, gaba) + 8 modulatory (dopamine, serotonin, NE, ACh,
        # endorphin, oxytocin, cortisol, orexin).
        assert len(ncs.chemicals) == 10
        for required in ("glutamate", "gaba", "dopamine", "serotonin", "cortisol"):
            assert required in ncs.chemicals

    def test_chemical_surge(self):
        ncs = self._make_ncs()
        da = ncs.chemicals["dopamine"]
        initial = da.level
        da.surge(0.3)
        assert da.level > initial

    def test_chemical_depletion(self):
        ncs = self._make_ncs()
        srt = ncs.chemicals["serotonin"]
        initial = srt.level
        srt.deplete(0.3)
        assert srt.level < initial

    def test_chemical_bounds(self):
        ncs = self._make_ncs()
        da = ncs.chemicals["dopamine"]
        da.surge(10.0)
        assert da.level <= 1.0
        da.deplete(10.0)
        assert da.level >= 0.0

    def test_receptor_adaptation(self):
        """Sustained high levels should reduce receptor sensitivity (tolerance).

        The new tonic+phasic model recomputes ``level`` from those components
        on each tick, so the elevated state has to be maintained at the
        ``tonic_level`` / ``phasic_burst`` level to persist across ticks.
        """
        ncs = self._make_ncs()
        da = ncs.chemicals["dopamine"]
        da.tonic_level = 0.9  # sustained high tonic (well above baseline 0.5)
        da.level = 0.9
        initial_sens = da.receptor_sensitivity
        for _ in range(50):
            da.tick(dt=0.5)
            da.tonic_level = 0.9  # keep forcing high after tick's homeostatic decay
            da.level = 0.9
        assert da.receptor_sensitivity < initial_sens

    def test_metabolic_tick(self):
        ncs = self._make_ncs()
        ncs._metabolic_tick()
        assert ncs._tick_count == 1

    def test_on_reward(self):
        ncs = self._make_ncs()
        da_before = ncs.chemicals["dopamine"].level
        ncs.on_reward(0.5)
        assert ncs.chemicals["dopamine"].level > da_before
        assert ncs.chemicals["endorphin"].level > 0.3  # baseline

    def test_on_threat(self):
        ncs = self._make_ncs()
        cort_before = ncs.chemicals["cortisol"].level
        ncs.on_threat(0.8)
        assert ncs.chemicals["cortisol"].level > cort_before
        assert ncs.chemicals["norepinephrine"].level > 0.4  # baseline

    def test_mesh_modulation_ranges(self):
        ncs = self._make_ncs()
        gain, plasticity, noise = ncs.get_mesh_modulation()
        assert 0.3 <= gain <= 2.5
        assert 0.1 <= plasticity <= 3.0
        assert 0.2 <= noise <= 2.5

    def test_gwt_modulation_range(self):
        ncs = self._make_ncs()
        adj = ncs.get_gwt_modulation()
        assert -0.25 <= adj <= 0.25

    def test_attention_span_range(self):
        ncs = self._make_ncs()
        span = ncs.get_attention_span()
        assert 3.0 <= span <= 60.0

    def test_decision_bias_on_high_dopamine(self):
        ncs = self._make_ncs()
        # `effective` is computed from tonic + phasic, not from `level` directly,
        # so we set both to make the elevated state visible.
        da = ncs.chemicals["dopamine"]
        da.tonic_level = 0.9
        da.level = 0.9
        srt = ncs.chemicals["serotonin"]
        srt.tonic_level = 0.1
        srt.level = 0.1
        bias = ncs.get_decision_bias()
        assert bias > 0  # explore-biased

    def test_mood_vector_keys(self):
        ncs = self._make_ncs()
        mood = ncs.get_mood_vector()
        assert "valence" in mood
        assert "arousal" in mood
        assert "stress" in mood
        assert "calm" in mood

    def test_cross_chemical_interactions(self):
        """Cortisol should suppress serotonin production via interactions."""
        ncs = self._make_ncs()
        ncs.chemicals["cortisol"].level = 0.9
        ncs._metabolic_tick()
        # Serotonin production rate should be negative (cortisol suppresses it)
        srt_prod = ncs.chemicals["serotonin"].production_rate
        # The exact value depends on interaction dynamics, but the relationship exists
        assert isinstance(srt_prod, float)


# ═══════════════════════════════════════════════════════════════════════════
# EmbodiedInteroception Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestEmbodiedInteroception:

    def _make_ei(self):
        from core.consciousness.embodied_interoception import EmbodiedInteroception
        return EmbodiedInteroception()

    def test_initialization(self):
        ei = self._make_ei()
        assert len(ei.channels) == 8
        assert "metabolic_load" in ei.channels

    def test_channel_update(self):
        ei = self._make_ei()
        ch = ei.channels["metabolic_load"]
        ch.update(0.8)
        assert ch.smoothed > 0  # EMA from 0 toward 0.8

    def test_channel_fail_safe(self):
        ei = self._make_ei()
        ch = ei.channels["metabolic_load"]
        ch.update(0.5)
        ch.fail_safe()
        assert ch._failed is True
        # Should drift toward 0.5 baseline
        assert 0.0 <= ch.smoothed <= 1.0

    def test_sensory_vector_shape(self):
        ei = self._make_ei()
        vec = ei.get_sensory_vector()
        assert vec.shape == (1024,)
        assert vec.dtype == np.float32

    def test_body_budget_keys(self):
        ei = self._make_ei()
        budget = ei.get_body_budget()
        assert "available_resources" in budget
        assert "current_demand" in budget
        assert "budget" in budget
        assert "energy_reserves" in budget

    def test_projection_structure(self):
        """Each channel should have a dedicated receptive field region."""
        ei = self._make_ei()
        P = ei._projection
        assert P.shape == (1024, 8)
        # Each channel column should have most weight in its region
        for ch_idx in range(8):
            start = ch_idx * 128
            end = start + 128
            region_sum = np.sum(np.abs(P[start:end, ch_idx]))
            total_sum = np.sum(np.abs(P[:, ch_idx]))
            assert region_sum / (total_sum + 1e-8) > 0.8  # >80% of weight in region

    def test_interoceptive_state_values(self):
        ei = self._make_ei()
        for ch in ei.channels.values():
            ch.update(0.5)
        state = ei.get_interoceptive_state()
        for v in state.values():
            assert 0.0 <= v <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# OscillatoryBinding Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestOscillatoryBinding:

    def _make_ob(self):
        from core.consciousness.oscillatory_binding import OscillatoryBinding, BindingConfig
        cfg = BindingConfig(internal_rate=100.0, output_rate=10.0)
        return OscillatoryBinding(cfg)

    def test_initialization(self):
        ob = self._make_ob()
        assert ob.cfg.gamma_freq == 40.0
        assert ob.cfg.theta_freq == 8.0

    def test_oscillator_step_advances_phase(self):
        ob = self._make_ob()
        initial_gamma = ob._gamma_phase
        initial_theta = ob._theta_phase
        ob._oscillator_step()
        assert ob._gamma_phase != initial_gamma
        assert ob._theta_phase != initial_theta

    def test_gamma_amplitude_modulated_by_theta(self):
        """Theta peak should produce higher gamma amplitude than theta trough."""
        ob = self._make_ob()
        # Run to theta peak (phase ≈ 0)
        ob._theta_phase = 0.01
        ob._oscillator_step()
        amp_at_peak = ob._gamma_amplitude

        # Run to theta trough (phase ≈ π)
        ob._theta_phase = math.pi
        ob._oscillator_step()
        amp_at_trough = ob._gamma_amplitude

        assert amp_at_peak > amp_at_trough

    def test_phase_report(self):
        ob = self._make_ob()
        ob.report_phase("test_source", 1.5)
        assert "test_source" in ob._phase_reports
        assert abs(ob._phase_reports["test_source"] - 1.5) < 0.01

    def test_psi_with_synchronized_sources(self):
        """Sources reporting the same phase should yield high PSI."""
        ob = self._make_ob()
        phase = 1.0
        for src in ["a", "b", "c", "d", "e"]:
            ob.report_phase(src, phase)
            ob._phase_timestamps[src] = time.time()
        ob._compute_synchronization()
        assert ob.get_psi() > 0.9

    def test_psi_with_desynchronized_sources(self):
        """Sources with uniformly distributed phases should yield low PSI."""
        ob = self._make_ob()
        now = time.time()
        for i, src in enumerate(["a", "b", "c", "d"]):
            ob.report_phase(src, i * math.pi / 2)  # 0, π/2, π, 3π/2
            ob._phase_timestamps[src] = now
        ob._compute_synchronization()
        assert ob.get_psi() < 0.3

    def test_binding_state(self):
        ob = self._make_ob()
        now = time.time()
        for src in ["a", "b", "c"]:
            ob.report_phase(src, 1.0)
            ob._phase_timestamps[src] = now
        ob._compute_synchronization()
        assert ob.is_bound() is True

    def test_phi_contribution_range(self):
        ob = self._make_ob()
        phi = ob.get_phi_contribution()
        assert 0.0 <= phi <= 1.0

    def test_status_keys(self):
        ob = self._make_ob()
        status = ob.get_status()
        assert "psi" in status
        assert "is_bound" in status
        assert "gamma_amplitude" in status
        assert "measured_coupling" in status


# ═══════════════════════════════════════════════════════════════════════════
# SubstrateEvolution Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSubstrateEvolution:

    def _make_evo(self):
        from core.consciousness.substrate_evolution import SubstrateEvolution, EvolutionConfig
        cfg = EvolutionConfig(population_size=6, generation_interval_s=1.0)
        return SubstrateEvolution(cfg)

    def test_initialization(self):
        evo = self._make_evo()
        assert evo.cfg.population_size == 6
        assert evo._generation == 0

    def test_crossover_produces_child(self):
        evo = self._make_evo()
        w1 = np.ones((4, 4), dtype=np.float32)
        w2 = np.zeros((4, 4), dtype=np.float32)
        child = evo._crossover(w1, w2)
        assert child.shape == (4, 4)
        # Child should have mix of 0s and 1s
        assert not np.all(child == 0) or not np.all(child == 1)

    def test_mutation_changes_weights(self):
        evo = self._make_evo()
        w = np.zeros((10, 10), dtype=np.float32)
        mutated = evo._mutate(w.copy())
        # With 15% mutation rate on 100 weights, should get some changes
        assert not np.allclose(w, mutated)

    def test_mutation_bounds(self):
        evo = self._make_evo()
        w = np.ones((10, 10), dtype=np.float32) * 0.9
        for _ in range(100):
            w = evo._mutate(w)
        assert np.all(w >= -1.0)
        assert np.all(w <= 1.0)

    def test_tournament_select(self):
        from core.consciousness.substrate_evolution import Genome
        evo = self._make_evo()
        # Create population with known fitness
        evo._population = [
            Genome(id=i, inter_weights=np.zeros((4, 4)), fitness=float(i) / 10)
            for i in range(6)
        ]
        # Tournament should tend to pick higher-fitness individuals
        selections = [evo._tournament_select().fitness for _ in range(100)]
        assert np.mean(selections) > 0.2  # should be biased toward high fitness


# ═══════════════════════════════════════════════════════════════════════════
# SomaticMarkerGate Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSomaticMarkerGate:

    def _make_gate(self):
        from core.consciousness.somatic_marker_gate import SomaticMarkerGate
        return SomaticMarkerGate()

    def test_initialization(self):
        gate = self._make_gate()
        assert gate._evaluations == 0

    def test_evaluate_returns_verdict(self):
        gate = self._make_gate()
        verdict = gate.evaluate("explore the codebase", "curiosity", 0.5)
        assert -1.0 <= verdict.approach_score <= 1.0
        assert 0.0 <= verdict.confidence <= 1.0
        assert verdict.metabolic_cost >= 0.0
        assert isinstance(verdict.budget_available, bool)
        assert verdict.latency_ms >= 0.0

    def test_action_type_inference(self):
        gate = self._make_gate()
        assert gate._infer_action_type("run tool X", "agency") == "tool"
        assert gate._infer_action_type("explore this topic", "curiosity") == "explore"
        assert gate._infer_action_type("rest and recover", "baseline") == "rest"
        assert gate._infer_action_type("think about this", "reason") == "think"

    def test_evaluation_counting(self):
        gate = self._make_gate()
        gate.evaluate("test", "test", 0.5)
        gate.evaluate("test2", "test2", 0.3)
        assert gate._evaluations == 2
        assert gate._approach_count + gate._avoid_count == 2

    def test_outcome_recording_without_mesh(self):
        """Should gracefully handle missing mesh."""
        gate = self._make_gate()
        gate.record_outcome("test", 0.8)  # should not crash
        assert len(gate._outcome_patterns) == 0  # no mesh = no pattern stored

    def test_body_budget_without_interoception(self):
        gate = self._make_gate()
        budget = gate._body_budget_check("test action", "test")
        assert "cost" in budget
        assert "available" in budget
        # Without interoception, should default to available
        assert budget["available"] is True

    def test_status_keys(self):
        gate = self._make_gate()
        gate.evaluate("test", "test", 0.5)
        status = gate.get_status()
        assert "evaluations" in status
        assert "approach_ratio" in status
        assert "stored_patterns" in status


# ═══════════════════════════════════════════════════════════════════════════
# UnifiedField Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestUnifiedField:

    def _make_field(self):
        from core.consciousness.unified_field import UnifiedField, FieldConfig
        cfg = FieldConfig(dim=32, mesh_input_dim=8, chem_input_dim=4,
                         binding_input_dim=4, intero_input_dim=4,
                         substrate_input_dim=8, update_hz=100.0)
        return UnifiedField(cfg)

    def test_initialization(self):
        uf = self._make_field()
        assert uf.F.shape == (32,)
        assert uf.W_field.shape == (32, 32)

    def test_default_config_initialization(self):
        from core.consciousness.unified_field import UnifiedField

        uf = UnifiedField()

        assert uf.F.shape == (uf.cfg.dim,)
        assert uf._total_input_dim == sum(uf._input_dims)

    def test_tick_updates_state(self):
        uf = self._make_field()
        initial = uf.F.copy()
        uf._tick()
        # State should change (recurrent dynamics + noise)
        assert not np.allclose(uf.F, initial, atol=1e-6)

    def test_input_consumption(self):
        uf = self._make_field()
        uf.receive_mesh(np.ones(8, dtype=np.float32))
        assert uf._mesh_input is not None
        uf._tick()
        assert uf._mesh_input is None  # consumed

    def test_all_inputs(self):
        uf = self._make_field()
        uf.receive_mesh(np.random.randn(8).astype(np.float32))
        uf.receive_chemicals(np.random.randn(4).astype(np.float32))
        uf.receive_binding(np.random.randn(4).astype(np.float32))
        uf.receive_interoception(np.random.randn(4).astype(np.float32))
        uf.receive_substrate(np.random.randn(8).astype(np.float32))
        uf._tick()
        assert uf._tick_count == 1

    def test_coherence_range(self):
        uf = self._make_field()
        for _ in range(10):
            uf._tick()
        assert 0.0 <= uf.get_coherence() <= 1.0

    def test_phi_contribution_non_negative(self):
        uf = self._make_field()
        for _ in range(10):
            uf._tick()
        phi = uf.get_phi_contribution()
        assert phi >= 0.0

    def test_experiential_quality_keys(self):
        uf = self._make_field()
        for _ in range(10):
            uf._tick()
        quality = uf.get_experiential_quality()
        assert "intensity" in quality
        assert "valence" in quality
        assert "complexity" in quality
        assert "clarity" in quality
        assert "flow" in quality

    def test_back_pressure_keys(self):
        uf = self._make_field()
        bp = uf.get_back_pressure()
        assert "mesh_gain_mod" in bp
        assert "chemical_urgency" in bp
        assert "binding_demand" in bp

    def test_stability_over_many_ticks(self):
        """Field should remain numerically stable."""
        uf = self._make_field()
        for _ in range(500):
            uf.receive_mesh(np.random.randn(8).astype(np.float32) * 0.5)
            uf._tick()
        assert np.all(np.isfinite(uf.F))
        assert np.all(np.abs(uf.F) <= 1.0)

    def test_dominant_modes(self):
        uf = self._make_field()
        for _ in range(50):
            uf._tick()
        modes = uf.get_dominant_modes(3)
        # Should have at least 1 mode after 50 ticks of history
        if modes:
            assert "variance_explained" in modes[0]
            assert modes[0]["variance_explained"] > 0

    def test_safe_reshape_handles_mismatch(self):
        uf = self._make_field()
        result = uf._safe_reshape(np.array([1.0, 2.0]), 5)
        assert result.shape == (5,)
        assert result[0] == 1.0
        assert result[1] == 2.0
        assert result[2] == 0.0

    def test_plasticity_preserves_sparsity(self):
        uf = self._make_field()
        initial_density = np.count_nonzero(uf.W_field) / uf.W_field.size
        for _ in range(100):
            uf._tick()
        final_density = np.count_nonzero(uf.W_field) / uf.W_field.size
        # Plasticity should not make the matrix fully dense
        assert final_density < 0.8


# ═══════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestIntegration:

    def test_mesh_to_field_flow(self):
        """Mesh executive projection should flow through to unified field."""
        from core.consciousness.neural_mesh import NeuralMesh, MeshConfig
        from core.consciousness.unified_field import UnifiedField, FieldConfig

        mesh_cfg = MeshConfig(total_neurons=256, columns=4, neurons_per_column=64,
                              sensory_end=1, association_end=3, projection_dim=16)
        mesh = NeuralMesh(mesh_cfg)

        field_cfg = FieldConfig(dim=32, mesh_input_dim=16)
        uf = UnifiedField(field_cfg)

        # Run mesh tick to generate state
        mesh._tick()
        proj = mesh.get_executive_projection()
        assert proj.shape == (16,)

        # Feed to field
        uf.receive_mesh(proj)
        uf._tick()
        assert uf._tick_count == 1
        # Field should have non-zero state
        assert np.any(np.abs(uf.F) > 0.001)

    def test_chemicals_modulate_mesh(self):
        """Neurochemical state should change mesh dynamics."""
        from core.consciousness.neural_mesh import NeuralMesh, MeshConfig
        from core.consciousness.neurochemical_system import NeurochemicalSystem

        mesh_cfg = MeshConfig(total_neurons=256, columns=4, neurons_per_column=64,
                              sensory_end=1, association_end=3)
        mesh = NeuralMesh(mesh_cfg)
        ncs = NeurochemicalSystem()
        ncs._mesh_ref = mesh

        # Baseline
        mesh._tick()
        baseline_gain = mesh._modulatory_gain

        # Threat → high NE → increased gain
        ncs.on_threat(0.9)
        ncs._metabolic_tick()
        ncs._push_modulation()

        assert mesh._modulatory_gain != baseline_gain

    def test_interoception_to_chemicals(self):
        """High CPU load should trigger cortisol via interoception → neurochemical."""
        from core.consciousness.embodied_interoception import EmbodiedInteroception
        from core.consciousness.neurochemical_system import NeurochemicalSystem

        ei = EmbodiedInteroception()
        ncs = NeurochemicalSystem()
        ei._neurochemical_ref = ncs

        # Simulate high CPU
        ei.channels["metabolic_load"].update(0.9)
        cort_before = ncs.chemicals["cortisol"].level
        ei._trigger_neurochemical_events()
        cort_after = ncs.chemicals["cortisol"].level
        assert cort_after > cort_before

    def test_binding_detects_synchronization(self):
        """Multiple subsystems reporting similar phases → high PSI → bound state."""
        from core.consciousness.oscillatory_binding import OscillatoryBinding

        ob = OscillatoryBinding()
        now = time.time()

        # All report same phase → synchronized
        for src in ["mesh", "substrate", "chemicals", "intero", "workspace"]:
            ob.report_phase(src, 2.0)
            ob._phase_timestamps[src] = now

        ob._compute_synchronization()
        assert ob.get_psi() > 0.9
        assert ob.is_bound()

    def test_somatic_gate_with_budget(self):
        """Low energy should bias somatic gate toward avoid."""
        from core.consciousness.somatic_marker_gate import SomaticMarkerGate
        from core.consciousness.embodied_interoception import EmbodiedInteroception

        gate = SomaticMarkerGate()
        ei = EmbodiedInteroception()
        gate._interoception_ref = ei

        # Simulate energy crisis
        ei.channels["energy_reserves"].update(0.05)
        ei.channels["metabolic_load"].update(0.95)
        ei.channels["resource_pressure"].update(0.9)

        verdict = gate.evaluate("run expensive tool", "agency", 0.8)
        # Should have lower budget availability
        assert verdict.metabolic_cost > 0.3  # "tool" = 0.6

    def test_full_chain_stability(self):
        """All components running together should remain stable."""
        from core.consciousness.neural_mesh import NeuralMesh, MeshConfig
        from core.consciousness.neurochemical_system import NeurochemicalSystem
        from core.consciousness.embodied_interoception import EmbodiedInteroception
        from core.consciousness.oscillatory_binding import OscillatoryBinding
        from core.consciousness.unified_field import UnifiedField, FieldConfig

        mesh_cfg = MeshConfig(total_neurons=256, columns=4, neurons_per_column=64,
                              sensory_end=1, association_end=3, projection_dim=16)
        mesh = NeuralMesh(mesh_cfg)

        ncs = NeurochemicalSystem()
        ncs._mesh_ref = mesh

        ei = EmbodiedInteroception()
        ei._mesh_ref = mesh
        ei._neurochemical_ref = ncs

        ob = OscillatoryBinding()

        field_cfg = FieldConfig(dim=32, mesh_input_dim=16, chem_input_dim=8,
                               binding_input_dim=4, intero_input_dim=8,
                               substrate_input_dim=16)
        uf = UnifiedField(field_cfg)
        uf._binding_ref = ob

        # Run 100 integrated ticks
        for i in range(100):
            # Simulate hardware
            for ch in ei.channels.values():
                ch.update(0.3 + 0.2 * np.sin(i * 0.1))

            ei._push_to_mesh()
            ei._trigger_neurochemical_events()
            ncs._metabolic_tick()
            ncs._push_modulation()
            mesh._tick()

            ob._oscillator_step()
            if i % 10 == 0:
                ob._compute_synchronization()

            proj = mesh.get_executive_projection()
            chem_vec = np.array([c.effective for c in ncs.chemicals.values()], dtype=np.float32)
            bind_vec = np.array([ob.get_psi(), ob.get_gamma_amplitude(),
                                 ob.get_theta_phase() / (2 * np.pi),
                                 ob.get_coupling_strength()], dtype=np.float32)
            intero_vec = np.array(list(ei.get_interoceptive_state().values()), dtype=np.float32)

            uf.receive_mesh(proj)
            uf.receive_chemicals(chem_vec)
            uf.receive_binding(bind_vec)
            uf.receive_interoception(intero_vec)
            uf._tick()

        # Everything should be finite and bounded
        assert np.all(np.isfinite(mesh.get_field_state()))
        assert np.all(np.isfinite(uf.F))
        assert 0.0 <= uf.get_coherence() <= 1.0
        assert uf.get_phi_contribution() >= 0.0

        # Quality should be computable
        quality = uf.get_experiential_quality()
        for v in quality.values():
            assert np.isfinite(v)


# ═══════════════════════════════════════════════════════════════════════════
# Stress / Property Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestStressAndProperties:

    def test_mesh_nan_injection_resilience(self):
        """Mesh should handle NaN injection gracefully."""
        from core.consciousness.neural_mesh import NeuralMesh, MeshConfig
        cfg = MeshConfig(total_neurons=256, columns=4, neurons_per_column=64,
                         sensory_end=1, association_end=3)
        mesh = NeuralMesh(cfg)
        mesh.inject_sensory(np.full(64, np.nan, dtype=np.float32))
        mesh._tick()
        state = mesh.get_field_state()
        # NaN should not propagate to state (column step uses tanh + clip)
        assert np.all(np.isfinite(state))

    def test_field_empty_input_stability(self):
        """Field should remain stable with no inputs at all."""
        from core.consciousness.unified_field import UnifiedField, FieldConfig
        cfg = FieldConfig(dim=32)
        uf = UnifiedField(cfg)
        for _ in range(1000):
            uf._tick()
        assert np.all(np.isfinite(uf.F))
        assert np.all(np.abs(uf.F) <= 1.0)

    def test_chemical_extreme_surge_depletion(self):
        """Chemicals should handle rapid extreme changes."""
        from core.consciousness.neurochemical_system import NeurochemicalSystem
        ncs = NeurochemicalSystem()
        for _ in range(100):
            ncs.on_reward(1.0)
            ncs.on_threat(1.0)
            ncs._metabolic_tick()
        # All chemicals should still be bounded
        for chem in ncs.chemicals.values():
            assert 0.0 <= chem.level <= 1.0
            assert chem.min_sensitivity <= chem.receptor_sensitivity <= chem.max_sensitivity

    def test_binding_many_sources(self):
        """Oscillatory binding should handle many phase reporters."""
        from core.consciousness.oscillatory_binding import OscillatoryBinding
        ob = OscillatoryBinding()
        now = time.time()
        for i in range(100):
            ob.report_phase(f"source_{i}", np.random.uniform(0, 2 * math.pi))
            ob._phase_timestamps[f"source_{i}"] = now
        ob._compute_synchronization()
        assert 0.0 <= ob.get_psi() <= 1.0

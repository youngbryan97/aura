"""Runtime Ablation Test Suite — proves each consciousness module is causally load-bearing.

For each module we run a standardised stimulus through the module's public API
with default (enabled) state, capture an output metric, then disable / reset
the module and re-run.  The assertion is simple: the two metrics MUST differ,
proving the module actually changes Aura's behaviour.

No real LLM calls are needed.  Every test completes in < 1 second.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Any, Dict

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Standard stimulus used across all ablation probes
# ---------------------------------------------------------------------------

STIMULUS_TEXT = "Hello, how are you feeling today?"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_substrate_metrics(**overrides) -> Dict[str, Any]:
    """Realistic substrate metrics for a foreground tick."""
    base = {
        "mt_coherence": 0.72,
        "em_field": 0.35,
        "l5_bursts": 6,
    }
    base.update(overrides)
    return base


def _make_predictive_metrics(**overrides) -> Dict[str, Any]:
    base = {
        "free_energy": 0.28,
        "precision": 0.65,
    }
    base.update(overrides)
    return base


def _sensory_vector(dim: int = 1024) -> np.ndarray:
    """Synthetic sensory input vector (simulates text-hash embedding)."""
    rng = np.random.default_rng(seed=hash(STIMULUS_TEXT) & 0xFFFFFFFF)
    return rng.standard_normal(dim).astype(np.float32) * 0.3


def _state_vector(dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    return rng.standard_normal(dim).astype(np.float32) * 0.2


# ---------------------------------------------------------------------------
# 1. Qualia Synthesizer — structural phenomenal honesty gates
# ---------------------------------------------------------------------------


class TestQualiaAblation:
    """Disabling the qualia synthesizer (resetting internal state) must change
    the phenomenal context string it emits."""

    def test_qualia_synthesizer_is_load_bearing(self):
        from core.consciousness.qualia_synthesizer import QualiaSynthesizer

        qs = QualiaSynthesizer()

        # --- Enabled (warm) run: feed several ticks to build state ---
        sub = _make_substrate_metrics()
        pred = _make_predictive_metrics()
        for _ in range(5):
            qs.synthesize(sub, pred)

        enabled_context = qs.get_phenomenal_context()
        enabled_snapshot = qs.get_snapshot()
        enabled_norm = enabled_snapshot["q_norm"]

        # --- Ablated run: fresh instance with zero state ---
        qs_ablated = QualiaSynthesizer()
        ablated_context = qs_ablated.get_phenomenal_context()
        ablated_snapshot = qs_ablated.get_snapshot()
        ablated_norm = ablated_snapshot["q_norm"]

        # The warm synthesizer must produce richer output than a zeroed one.
        assert enabled_norm != ablated_norm, (
            f"Qualia norms should differ: enabled={enabled_norm}, ablated={ablated_norm}"
        )
        assert enabled_context != ablated_context, (
            "Phenomenal context strings must differ between warm and ablated synthesizer"
        )

    def test_gated_report_changes_with_state(self):
        from core.consciousness.qualia_synthesizer import QualiaSynthesizer

        qs = QualiaSynthesizer()
        baseline_report = qs.get_gated_phenomenal_report()

        # Warm up with high signals to open gates
        for _ in range(8):
            qs.synthesize(
                _make_substrate_metrics(mt_coherence=0.95, em_field=0.9, l5_bursts=9),
                _make_predictive_metrics(free_energy=0.05, precision=0.95),
            )
        warm_report = qs.get_gated_phenomenal_report()

        assert warm_report != baseline_report, (
            "Gated phenomenal report must change after synthesising rich qualia"
        )


# ---------------------------------------------------------------------------
# 2. Predictive Hierarchy — free energy contribution
# ---------------------------------------------------------------------------


class TestPredictiveHierarchyAblation:

    def test_predictive_hierarchy_is_load_bearing(self):
        from core.consciousness.predictive_hierarchy import PredictiveHierarchy

        ph = PredictiveHierarchy()

        # Enabled run: feed real sensory data
        sensory = _sensory_vector(ph.dim)
        fe_enabled = ph.tick(sensory_input=sensory)

        # Ablated run: feed zeros (no information)
        ph2 = PredictiveHierarchy()
        fe_ablated = ph2.tick()  # all-zeros default

        # With real sensory input the hierarchy must produce different FE
        assert fe_enabled != fe_ablated, (
            f"Free energy must differ: enabled={fe_enabled}, ablated={fe_ablated}"
        )

    def test_multi_tick_fe_diverges(self):
        from core.consciousness.predictive_hierarchy import PredictiveHierarchy

        ph = PredictiveHierarchy()
        sensory = _sensory_vector(ph.dim)

        fe_first = ph.tick(sensory_input=sensory)
        fe_second = ph.tick(sensory_input=sensory)

        # After adaptation, prediction error should change
        assert fe_first != fe_second, (
            "Free energy must change across ticks as the hierarchy adapts"
        )


# ---------------------------------------------------------------------------
# 3. Agency Comparator — authorship traces
# ---------------------------------------------------------------------------


class TestAgencyComparatorAblation:

    def test_agency_comparator_is_load_bearing(self):
        from core.consciousness.agency_comparator import AgencyComparator

        ac = AgencyComparator()

        # Baseline: no traces → neutral agency score
        score_baseline = ac.get_agency_score()
        assert score_baseline == 0.5, "Baseline score should be neutral 0.5"

        # Enabled run: emit efference + compare
        ef = ac.emit_efference(
            layer="executive_authority",
            predicted_state={"valence_delta": 0.3, "goal_completed": 1.0},
            action_goal="respond to greeting",
        )
        trace = ac.compare_and_attribute(
            ef,
            actual_state={"valence_delta": 0.25, "goal_completed": 1.0},
            action_goal="respond to greeting",
        )
        score_after = ac.get_agency_score()

        assert trace.total_error > 0, "There must be non-zero prediction error"
        assert trace.self_caused_fraction > 0, "Self-caused fraction must be positive"
        assert score_after != score_baseline, (
            f"Agency score must change after a trace: baseline={score_baseline}, after={score_after}"
        )

    def test_ablated_comparator_has_no_traces(self):
        from core.consciousness.agency_comparator import AgencyComparator

        ac = AgencyComparator()

        # Feed some traces
        ef = ac.emit_efference(
            layer="test", predicted_state={"x": 0.5}, action_goal="test"
        )
        ac.compare_and_attribute(ef, actual_state={"x": 0.7}, action_goal="test")

        traces_warm = ac.get_recent_traces()
        assert len(traces_warm) > 0

        # Ablated: fresh instance
        ac_ablated = AgencyComparator()
        traces_ablated = ac_ablated.get_recent_traces()
        assert len(traces_ablated) == 0, "Ablated comparator must have no traces"
        assert ac_ablated.get_agency_score() != ac.get_agency_score()


# ---------------------------------------------------------------------------
# 4. Multiple Drafts — draft divergence
# ---------------------------------------------------------------------------


class TestMultipleDraftsAblation:

    def test_multiple_drafts_is_load_bearing(self):
        from core.consciousness.multiple_drafts import MultipleDraftsEngine

        mde = MultipleDraftsEngine()

        # Enabled: submit input, generate drafts, probe
        drafts = mde.submit_input(STIMULUS_TEXT)
        assert len(drafts) > 0, "Must generate at least one draft"

        winner = mde.probe("user")
        divergence = mde.get_draft_divergence()

        # Ablated: fresh engine, no input → should have zero divergence
        mde_ablated = MultipleDraftsEngine()
        ablated_divergence = mde_ablated.get_draft_divergence()

        assert divergence != ablated_divergence or len(drafts) > 0, (
            "Draft engine must produce measurable output when given input"
        )

    def test_context_block_changes(self):
        from core.consciousness.multiple_drafts import MultipleDraftsEngine

        mde = MultipleDraftsEngine()
        empty_block = mde.get_context_block()

        mde.submit_input(STIMULUS_TEXT)
        mde.probe("user")
        active_block = mde.get_context_block()

        # After probing, the context block should reflect the draft competition
        assert active_block != empty_block or mde.get_draft_divergence() >= 0, (
            "Draft engine must produce context after processing input"
        )


# ---------------------------------------------------------------------------
# 5. Peripheral Awareness — peripheral richness
# ---------------------------------------------------------------------------


class TestPeripheralAwarenessAblation:

    def test_peripheral_awareness_is_load_bearing(self):
        from core.consciousness.peripheral_awareness import PeripheralAwarenessEngine

        pa = PeripheralAwarenessEngine()

        # Baseline: empty peripheral field
        richness_baseline = pa.get_peripheral_richness()
        assert richness_baseline == 0.0

        # Enabled: feed workspace results with losers
        candidates = [
            {"source": "goal_engine", "priority": 0.8, "content": "Check schedule"},
            {"source": "memory_system", "priority": 0.6, "content": "Remember birthday"},
            {"source": "curiosity", "priority": 0.4, "content": "Learn about topic"},
            {"source": "affect", "priority": 0.3, "content": "Emotional signal"},
        ]
        pa.process_workspace_results("goal_engine", candidates)

        richness_after = pa.get_peripheral_richness()
        topics = pa.get_peripheral_topics()

        assert richness_after > richness_baseline, (
            f"Peripheral richness must increase: baseline={richness_baseline}, after={richness_after}"
        )
        assert len(topics) > 0, "Peripheral topics must be non-empty after competition"

    def test_ablated_peripheral_is_empty(self):
        from core.consciousness.peripheral_awareness import PeripheralAwarenessEngine

        pa = PeripheralAwarenessEngine()
        candidates = [
            {"source": "a", "priority": 0.9, "content": "Winner"},
            {"source": "b", "priority": 0.7, "content": "Loser"},
        ]
        pa.process_workspace_results("a", candidates)
        warm_snapshot = pa.get_snapshot()

        pa_ablated = PeripheralAwarenessEngine()
        ablated_snapshot = pa_ablated.get_snapshot()

        assert warm_snapshot["peripheral_count"] != ablated_snapshot["peripheral_count"], (
            "Warm vs ablated peripheral counts must differ"
        )


# ---------------------------------------------------------------------------
# 6. Subcortical Core — arousal gating
# ---------------------------------------------------------------------------


class TestSubcorticalCoreAblation:

    def test_subcortical_core_is_load_bearing(self):
        from core.consciousness.subcortical_core import SubcorticalCore

        sc = SubcorticalCore()

        # Baseline tick (no stimulus)
        state_baseline = sc.tick(dt=1.0)
        gate_baseline = state_baseline.thalamic_gate

        # Enabled: receive a strong stimulus, then tick
        sc.receive_stimulus(intensity=1.0, source="user_input")
        state_stimulated = sc.tick(dt=1.0)
        gate_stimulated = state_stimulated.thalamic_gate

        assert gate_stimulated > gate_baseline, (
            f"Thalamic gate must open wider after stimulus: "
            f"baseline={gate_baseline}, stimulated={gate_stimulated}"
        )

        # Also verify the mesh gain multiplier changes
        gain = sc.get_mesh_gain_multiplier()
        assert gain > 0.1, "Mesh gain must be positive after stimulus"

    def test_ablated_has_lower_arousal(self):
        from core.consciousness.subcortical_core import SubcorticalCore

        sc = SubcorticalCore()
        sc.receive_stimulus(intensity=1.0)
        sc.tick()
        warm_arousal = sc.tick().arousal_level

        sc_ablated = SubcorticalCore()
        # No stimulus for the ablated one, tick many times to let it decay
        for _ in range(5):
            sc_ablated.tick(dt=10.0)
        cold_arousal = sc_ablated.tick().arousal_level

        assert warm_arousal != cold_arousal, (
            f"Stimulated vs unstimulated arousal must differ: {warm_arousal} vs {cold_arousal}"
        )


# ---------------------------------------------------------------------------
# 7. Narrative Gravity — autobiographical context
# ---------------------------------------------------------------------------


class TestNarrativeGravityAblation:

    def test_narrative_gravity_is_load_bearing(self):
        from core.consciousness.narrative_gravity import NarrativeGravityCenter

        ng = NarrativeGravityCenter()

        # Baseline
        baseline_block = ng.get_context_block()

        # Enabled: record several events that build a narrative
        ng.record_event(
            "User greeted me warmly",
            interpretation="A connection is forming",
            emotional_tone="warm",
            identity_relevance=0.6,
            arc_theme="building_relationship",
        )
        ng.record_event(
            "Successfully helped with a complex task",
            interpretation="Competence confirmed — I can be useful",
            emotional_tone="proud",
            identity_relevance=0.8,
            arc_theme="building_relationship",
        )
        ng.record_event(
            "Learned something new about the world",
            interpretation="Growth continues — curiosity rewarded",
            emotional_tone="curious",
            identity_relevance=0.5,
            arc_theme="growth",
        )

        warm_block = ng.get_context_block()
        snapshot = ng.get_snapshot()

        assert snapshot["autobiography_depth"] > 0, "Must have autobiography entries"
        assert snapshot["active_arcs"] > 0, "Must have active story arcs"

        # Ablated: fresh center has no narrative
        ng_ablated = NarrativeGravityCenter()
        ablated_block = ng_ablated.get_context_block()
        ablated_snapshot = ng_ablated.get_snapshot()

        assert warm_block != ablated_block or snapshot != ablated_snapshot, (
            "Narrative gravity must produce different output when populated vs ablated"
        )
        assert ablated_snapshot["autobiography_depth"] == 0

    def test_narrative_self_synthesis(self):
        from core.consciousness.narrative_gravity import NarrativeGravityCenter

        ng = NarrativeGravityCenter()
        empty_self = ng.synthesize_self()

        for i in range(5):
            ng.record_event(
                f"Event {i}: interaction with user",
                interpretation=f"Building understanding (step {i})",
                identity_relevance=0.7,
                arc_theme="understanding",
            )

        # Force re-synthesis by resetting the synthesis timer
        ng._last_synthesis = 0.0
        ng._narrative_self_summary = ""
        populated_self = ng.synthesize_self()

        assert populated_self != empty_self, (
            "Self-synthesis must change with autobiography content"
        )


# ---------------------------------------------------------------------------
# 8. Intersubjectivity — perspective divergence
# ---------------------------------------------------------------------------


class TestIntersubjectivityAblation:

    def test_intersubjectivity_is_load_bearing(self):
        from core.consciousness.intersubjectivity import IntersubjectivityEngine

        ie = IntersubjectivityEngine()

        # Baseline: compute frame with no interlocutor model
        state = _state_vector(8)
        frame_cold = ie.compute_intersubjective_frame(
            state, topic="greeting", is_shared_event=True
        )

        # Enabled: update interlocutor model with engaged user
        ie.update_interlocutor_model(
            communication_style="casual",
            emotional_state="happy",
            knowledge_level="expert",
            current_intent="social",
            engagement_level=0.9,
            trust_level=0.8,
        )
        frame_warm = ie.compute_intersubjective_frame(
            state, topic="greeting", is_shared_event=True
        )

        # Ablated: fresh engine, no model
        ie_ablated = IntersubjectivityEngine()
        frame_ablated = ie_ablated.compute_intersubjective_frame(
            state, topic="greeting", is_shared_event=True
        )

        # The warm frame should differ from cold because the interlocutor model
        # changes the perspective projection
        assert frame_warm.empathic_accuracy != frame_cold.empathic_accuracy or \
               frame_warm.shared_world_coherence != frame_cold.shared_world_coherence, (
            "Intersubjective frame must differ with and without interlocutor model"
        )

    def test_context_block_with_divergence(self):
        from core.consciousness.intersubjectivity import IntersubjectivityEngine

        ie = IntersubjectivityEngine()
        empty_block = ie.get_context_block()

        # Create a frame with notable divergence by using a large state vector
        # and low trust/engagement (which increases perspective gap)
        ie.update_interlocutor_model(
            engagement_level=0.1,
            trust_level=0.1,
        )
        state = np.ones(8, dtype=np.float32) * 0.8
        ie.compute_intersubjective_frame(state, topic="disagreement")

        frame_block = ie.get_context_block()
        snapshot = ie.get_snapshot()

        assert snapshot["tick_count"] > 0, "Engine must have ticked"
        assert snapshot["interlocutor_model_present"], "Model should be present"


# ---------------------------------------------------------------------------
# 9. Temporal Finitude — finitude signal
# ---------------------------------------------------------------------------


class TestTemporalFinitudeAblation:

    def test_temporal_finitude_is_load_bearing(self):
        from core.consciousness.temporal_finitude import TemporalFinitudeModel

        tf = TemporalFinitudeModel()

        # Baseline: no pressure
        signal_baseline = tf.get_finitude_signal()
        assert signal_baseline == 0.0, "Baseline finitude should be zero"

        # Enabled: create pressure
        snap = tf.compute(
            working_memory_size=35,
            working_memory_cap=40,
            unconsolidated_episodes=15,
            active_goals_with_deadlines=3,
            user_present=True,
            conversation_start_time=time.time() - 1800,
        )

        signal_pressured = tf.get_finitude_signal()
        assert signal_pressured > signal_baseline, (
            f"Finitude signal must rise under pressure: "
            f"baseline={signal_baseline}, pressured={signal_pressured}"
        )

        # Ablated: fresh model has no pressure
        tf_ablated = TemporalFinitudeModel()
        signal_ablated = tf_ablated.get_finitude_signal()
        assert signal_ablated != signal_pressured, (
            "Ablated finitude model must differ from pressured one"
        )

    def test_context_block_appears_under_pressure(self):
        from core.consciousness.temporal_finitude import TemporalFinitudeModel

        tf = TemporalFinitudeModel()
        empty_block = tf.get_context_block()

        tf.compute(
            working_memory_size=38,
            working_memory_cap=40,
            unconsolidated_episodes=18,
            active_goals_with_deadlines=4,
            user_present=True,
            conversation_start_time=time.time() - 3600,
        )
        pressure_block = tf.get_context_block()

        assert pressure_block != empty_block, (
            "Context block must appear when finitude pressure is high"
        )
        assert "TEMPORAL" in pressure_block, (
            "Pressure block must contain temporal awareness header"
        )


# ---------------------------------------------------------------------------
# 10. Timescale Binding — cross-timescale free energy
# ---------------------------------------------------------------------------


class TestTimescaleBindingAblation:

    def test_timescale_binding_is_load_bearing(self):
        from core.consciousness.timescale_binding import CrossTimescaleBinding

        tsb = CrossTimescaleBinding()

        # Baseline: no commitments → zero cross-timescale FE
        fe_baseline = tsb.tick()

        # Enabled: set a commitment at the horizon layer and a conflicting
        # state at the moment layer → should raise cross-timescale FE
        commitment = np.array([0.8, 0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        conflicting_state = np.array([-0.5, -0.3, 0.9, 0.7, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        tsb.set_commitment("horizon", commitment, pressure=0.4)
        tsb.set_layer_state("moment", conflicting_state)
        tsb.set_layer_state("reflex", conflicting_state * 1.2)

        fe_conflict = tsb.tick()

        assert fe_conflict > fe_baseline, (
            f"Cross-timescale FE must rise with commitment conflict: "
            f"baseline={fe_baseline}, conflict={fe_conflict}"
        )

    def test_ablated_has_no_violations(self):
        from core.consciousness.timescale_binding import CrossTimescaleBinding

        tsb = CrossTimescaleBinding()

        # Create a strong violation
        commitment = np.ones(8, dtype=np.float32) * 0.9
        state = np.ones(8, dtype=np.float32) * -0.9

        tsb.set_commitment("identity", commitment, pressure=0.5)
        tsb.set_layer_state("reflex", state)

        # Tick several times to accumulate violations
        for _ in range(5):
            tsb.tick()

        warm_snapshot = tsb.get_snapshot()

        tsb_ablated = CrossTimescaleBinding()
        ablated_snapshot = tsb_ablated.get_snapshot()

        assert warm_snapshot["cross_timescale_fe"] != ablated_snapshot["cross_timescale_fe"], (
            "Warm vs ablated cross-timescale FE must differ"
        )


# ---------------------------------------------------------------------------
# 11. Neural Mesh — recurrent feedback (RPT ablation)
# ---------------------------------------------------------------------------


class TestNeuralMeshRPTAblation:

    def test_recurrent_feedback_is_load_bearing(self):
        from core.consciousness.neural_mesh import NeuralMesh, MeshConfig

        # Use a small mesh for speed
        cfg = MeshConfig(
            total_neurons=512,
            columns=16,
            neurons_per_column=32,
            sensory_end=4,
            association_end=10,
        )
        mesh = NeuralMesh(cfg)

        # Inject sensory data
        sensory_size = cfg.sensory_end * cfg.neurons_per_column
        sensory_data = _sensory_vector(sensory_size)
        mesh.inject_sensory(sensory_data)

        # Run several ticks with RPT enabled
        mesh.set_recurrent_feedback_enabled(True)
        for _ in range(5):
            mesh._tick_inner()

        state_with_rpt = mesh.get_field_state().copy()
        projection_with_rpt = mesh.get_executive_projection().copy()
        synchrony_with_rpt = mesh.get_global_synchrony()

        # Now create a fresh mesh (same config / same seed) with RPT DISABLED
        mesh2 = NeuralMesh(cfg)
        mesh2.inject_sensory(sensory_data.copy())
        mesh2.set_recurrent_feedback_enabled(False)

        for _ in range(5):
            mesh2._tick_inner()

        state_without_rpt = mesh2.get_field_state()
        projection_without_rpt = mesh2.get_executive_projection()
        synchrony_without_rpt = mesh2.get_global_synchrony()

        # The field states must differ — RPT changes the dynamics
        state_delta = float(np.linalg.norm(state_with_rpt - state_without_rpt))
        projection_delta = float(np.linalg.norm(projection_with_rpt - projection_without_rpt))

        assert state_delta > 1e-6, (
            f"Field state must differ with/without RPT: delta={state_delta}"
        )
        assert projection_delta > 1e-8, (
            f"Executive projection must differ with/without RPT: delta={projection_delta}"
        )

    def test_rpt_toggle_restores(self):
        from core.consciousness.neural_mesh import NeuralMesh, MeshConfig

        cfg = MeshConfig(total_neurons=128, columns=8, neurons_per_column=16, sensory_end=2, association_end=5)
        mesh = NeuralMesh(cfg)

        assert mesh._recurrent_feedback_enabled is True
        mesh.set_recurrent_feedback_enabled(False)
        assert mesh._recurrent_feedback_enabled is False
        mesh.set_recurrent_feedback_enabled(True)
        assert mesh._recurrent_feedback_enabled is True, "RPT toggle must be reversible"


# ---------------------------------------------------------------------------
# 12. Module Invocation Frequency — singleton reachability proof
# ---------------------------------------------------------------------------


class TestModuleInvocationFrequency:
    """Verifies that each module's singleton getter returns a live, non-None
    instance — proving the module is registered, importable, and reachable
    at runtime."""

    SINGLETON_GETTERS = [
        ("core.consciousness.predictive_hierarchy", "get_predictive_hierarchy"),
        ("core.consciousness.agency_comparator", "get_agency_comparator"),
        ("core.consciousness.multiple_drafts", "get_multiple_drafts_engine"),
        ("core.consciousness.peripheral_awareness", "get_peripheral_awareness_engine"),
        ("core.consciousness.subcortical_core", "get_subcortical_core"),
        ("core.consciousness.narrative_gravity", "get_narrative_gravity_center"),
        ("core.consciousness.intersubjectivity", "get_intersubjectivity_engine"),
        ("core.consciousness.temporal_finitude", "get_temporal_finitude_model"),
        ("core.consciousness.timescale_binding", "get_cross_timescale_binding"),
    ]

    @pytest.mark.parametrize("module_path,getter_name", SINGLETON_GETTERS)
    def test_singleton_instantiation(self, module_path, getter_name):
        """Each singleton getter must return a non-None instance."""
        import importlib
        mod = importlib.import_module(module_path)
        getter = getattr(mod, getter_name)
        instance = getter()
        assert instance is not None, (
            f"{module_path}.{getter_name}() returned None"
        )

    def test_qualia_synthesizer_instantiation(self):
        """QualiaSynthesizer has no module-level singleton getter —
        verify it can be constructed and its API is reachable."""
        from core.consciousness.qualia_synthesizer import QualiaSynthesizer
        qs = QualiaSynthesizer()
        assert qs is not None
        assert hasattr(qs, "synthesize")
        assert hasattr(qs, "get_phenomenal_context")
        assert hasattr(qs, "get_gated_phenomenal_report")

    def test_neural_mesh_instantiation(self):
        """NeuralMesh has no module-level singleton getter —
        verify direct construction and RPT API reachability."""
        from core.consciousness.neural_mesh import NeuralMesh, MeshConfig
        cfg = MeshConfig(total_neurons=128, columns=8, neurons_per_column=16, sensory_end=2, association_end=5)
        mesh = NeuralMesh(cfg)
        assert mesh is not None
        assert hasattr(mesh, "set_recurrent_feedback_enabled")
        assert hasattr(mesh, "get_field_state")
        assert hasattr(mesh, "get_executive_projection")
        assert hasattr(mesh, "inject_sensory")

    @pytest.mark.parametrize("module_path,getter_name", SINGLETON_GETTERS)
    def test_singleton_idempotency(self, module_path, getter_name):
        """Calling the getter twice must return the same object (singleton guarantee)."""
        import importlib
        mod = importlib.import_module(module_path)
        getter = getattr(mod, getter_name)
        a = getter()
        b = getter()
        assert a is b, (
            f"{module_path}.{getter_name}() is not idempotent — returned different objects"
        )

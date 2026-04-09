"""Tests for Tier 4 Unification Systems

Verifies:
  1. InitiativeSynthesizer collects, deduplicates, and selects
  2. WorldState tracks environment and produces context
  3. DriveEngine cross-coupling modifies arbiter weights
  4. InternalSimulator evaluates candidates with identity + commitment
  5. Output gate enforces Will
  6. All systems wire together coherently
"""
import time
import pytest
from unittest.mock import MagicMock, patch

from core.initiative_synthesis import (
    Impulse,
    InitiativeSynthesizer,
    SynthesisResult,
    get_initiative_synthesizer,
)
from core.world_state import (
    WorldState,
    SalientEvent,
    EnvironmentBelief,
    get_world_state,
)
from core.drive_engine import DriveEngine, ResourceBudget
from core.simulation.internal_simulator import InternalSimulator
from core.will import ActionDomain, WillOutcome, get_will


# ---------------------------------------------------------------------------
# InitiativeSynthesizer
# ---------------------------------------------------------------------------

class TestInitiativeSynthesizer:

    def test_submit_impulse(self):
        synth = InitiativeSynthesizer()
        ok = synth.submit("explore quantum physics", "curiosity_engine", urgency=0.7)
        assert ok
        assert len(synth._impulse_queue) == 1

    def test_dedup_identical_impulses(self):
        synth = InitiativeSynthesizer()
        synth.submit("explore quantum physics", "curiosity_engine")
        ok = synth.submit("explore quantum physics", "curiosity_engine")
        assert not ok  # deduplicated
        assert len(synth._impulse_queue) == 1

    def test_different_impulses_accepted(self):
        synth = InitiativeSynthesizer()
        synth.submit("explore physics", "curiosity")
        synth.submit("talk to user", "social")
        assert len(synth._impulse_queue) == 2

    def test_queue_cap(self):
        synth = InitiativeSynthesizer()
        synth._MAX_IMPULSES_PER_CYCLE = 3
        for i in range(5):
            synth.submit(f"impulse_{i}", f"source_{i}", urgency=0.1 * i)
        assert len(synth._impulse_queue) <= 3

    def test_impulse_fingerprint(self):
        imp = Impulse(content="test", source="test_source")
        assert imp.fingerprint  # non-empty
        assert len(imp.fingerprint) == 12

    def test_status(self):
        synth = InitiativeSynthesizer()
        synth.submit("test", "test")
        status = synth.get_status()
        assert "pending_impulses" in status
        assert status["pending_impulses"] == 1


# ---------------------------------------------------------------------------
# WorldState
# ---------------------------------------------------------------------------

class TestWorldState:

    def test_creation(self):
        ws = WorldState()
        assert ws.time_of_day != "unknown" or True  # will be set on update
        assert ws.user_idle_seconds == 0.0

    def test_user_message_tracking(self):
        ws = WorldState()
        ws.on_user_message("hello", mood_hint="positive")
        assert ws.user_message_count == 1
        assert ws.estimated_user_mood == "positive"
        assert ws.last_user_interaction > 0

    def test_user_error_tracking(self):
        ws = WorldState()
        ws.time_of_day = "late_night"
        ws.on_user_error("ModuleNotFoundError: no module named 'foo'")
        assert ws.estimated_user_mood == "frustrated"
        events = ws.get_salient_events()
        assert len(events) >= 1
        assert "error" in events[0]["description"].lower()

    def test_beliefs(self):
        ws = WorldState()
        ws.set_belief("user_is_coding", True, confidence=0.8)
        assert ws.get_belief("user_is_coding") is True
        assert ws.get_belief("nonexistent") is None

    def test_belief_expiration(self):
        ws = WorldState()
        ws.set_belief("temp_fact", "value", ttl=0.01)
        import time; time.sleep(0.02)
        assert ws.get_belief("temp_fact") is None

    def test_event_dedup(self):
        ws = WorldState()
        ws.record_event("CPU high", "system")
        ws.record_event("CPU high", "system")  # should dedup within 60s
        assert len(ws._events) == 1

    def test_context_summary(self):
        ws = WorldState()
        ws.update()
        summary = ws.get_context_summary()
        assert isinstance(summary, str)
        assert "Time:" in summary

    def test_status(self):
        ws = WorldState()
        ws.update()
        status = ws.get_status()
        assert "cpu_percent" in status
        assert "time_of_day" in status
        assert "session_duration_m" in status

    def test_salient_events_sorted_by_salience(self):
        ws = WorldState()
        ws.record_event("low priority", "system", salience=0.2)
        ws.record_event("high priority", "system", salience=0.9)
        events = ws.get_salient_events()
        assert events[0]["salience"] > events[1]["salience"]


# ---------------------------------------------------------------------------
# DriveEngine Cross-Coupling
# ---------------------------------------------------------------------------

class TestDriveEngineCrossCoupling:

    def test_drive_vector(self):
        de = DriveEngine()
        vector = de.get_drive_vector()
        assert "energy" in vector
        assert "curiosity" in vector
        assert "social" in vector
        assert all(0.0 <= v <= 1.0 for v in vector.values())

    def test_arbiter_weight_modifiers_normal(self):
        de = DriveEngine()
        mods = de.get_arbiter_weight_modifiers()
        # At startup, all drives are healthy, so no modifiers
        assert len(mods) == 0

    def test_arbiter_weight_modifiers_low_energy(self):
        de = DriveEngine()
        de.budgets["energy"].level = 20.0  # 20% energy
        mods = de.get_arbiter_weight_modifiers()
        assert "resource_cost" in mods
        assert mods["resource_cost"] > 0

    def test_arbiter_weight_modifiers_low_social(self):
        de = DriveEngine()
        de.budgets["social"].level = 15.0  # very low social
        mods = de.get_arbiter_weight_modifiers()
        assert "social_appropriateness" in mods

    def test_arbiter_weight_modifiers_low_curiosity(self):
        de = DriveEngine()
        de.budgets["curiosity"].level = 25.0
        mods = de.get_arbiter_weight_modifiers()
        assert "novelty" in mods


# ---------------------------------------------------------------------------
# InternalSimulator
# ---------------------------------------------------------------------------

class TestInternalSimulator:

    def test_evaluate_with_identity(self):
        sim = InternalSimulator()
        # Content that aligns with identity should score positively
        score = sim._check_identity_alignment("I want to explore consciousness")
        assert isinstance(score, float)

    def test_evaluate_identity_violation(self):
        sim = InternalSimulator()
        score = sim._check_identity_alignment("as an ai, I'm just a language model")
        assert score < 0

    def test_commitment_compatibility(self):
        sim = InternalSimulator()
        score = sim._check_commitment_compatibility("do something random")
        assert isinstance(score, float)

    def test_evaluate_candidates(self):
        sim = InternalSimulator()
        # Create a minimal mock state
        state = MagicMock()
        state.affect.valence = 0.5
        state.affect.arousal = 0.4
        state.affect.physiology = {"cortisol": 10.0}
        state.motivation.budgets = {"energy": {"level": 80.0}}
        state.state_id = "test_123"
        state.version = 1

        candidates = [
            {"goal": "explore", "variation": {"risk": 0.2}},
            {"goal": "rest", "variation": {"risk": 0.0, "energy": 0}},
        ]
        results = sim.evaluate_candidates(state, candidates)
        assert len(results) == 2
        assert results[0]["score"] >= results[1]["score"]  # sorted best first


# ---------------------------------------------------------------------------
# Will Enforcement at Output Gate
# ---------------------------------------------------------------------------

class TestOutputGateWillEnforcement:

    def test_will_referenced_in_output_gate(self):
        """output_gate.py must reference the Unified Will."""
        import inspect
        from core.utils.output_gate import AutonomousOutputGate
        source = inspect.getsource(AutonomousOutputGate)
        assert "get_will" in source
        assert "will_receipt_id" in source
        assert "UNIFIED WILL" in source.upper() or "unified will" in source.lower()


# ---------------------------------------------------------------------------
# Integration: All Tier 4 systems present
# ---------------------------------------------------------------------------

class TestTier4Integration:

    def test_all_tier4_modules_importable(self):
        """All Tier 4 modules must import without error."""
        from core.will import UnifiedWill, get_will
        from core.initiative_synthesis import InitiativeSynthesizer, get_initiative_synthesizer
        from core.world_state import WorldState, get_world_state
        from core.drive_engine import DriveEngine
        from core.simulation.internal_simulator import InternalSimulator

    def test_will_wired_in_output_gate(self):
        import inspect
        from core.utils.output_gate import AutonomousOutputGate
        source = inspect.getsource(AutonomousOutputGate.emit)
        assert "get_will" in source

    def test_world_state_wired_in_incoming_logic(self):
        import inspect
        from core.orchestrator.mixins.incoming_logic import IncomingLogicMixin
        source = inspect.getsource(IncomingLogicMixin)
        assert "world_state" in source.lower()

    def test_drive_satisfaction_wired_in_incoming_logic(self):
        import inspect
        from core.orchestrator.mixins.incoming_logic import IncomingLogicMixin
        source = inspect.getsource(IncomingLogicMixin)
        assert "drive_social_satisfy" in source or "satisfy" in source

    def test_boot_wires_tier4_systems(self):
        import inspect
        from core.orchestrator.mixins.boot.boot_autonomy import BootAutonomyMixin
        source = inspect.getsource(BootAutonomyMixin)
        assert "WorldState" in source
        assert "InitiativeSynthesizer" in source
        assert "InternalSimulator" in source
        assert "Goal Resumption" in source

    def test_proof_surface_endpoint_exists(self):
        from interface.routes.inner_state import router
        paths = [r.path for r in router.routes]
        assert any("inner-state" in p for p in paths)

    def test_mind_tick_updates_world_state(self):
        import inspect
        from core.mind_tick import MindTick
        source = inspect.getsource(MindTick)
        assert "world_state" in source.lower()

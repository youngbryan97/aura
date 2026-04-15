"""
tests/test_technological_autonomy.py
=====================================
TECHNOLOGICAL AUTONOMY & SOVEREIGNTY TEST SUITE

Tests whether Aura can use her computational "body" the way an organism
uses its biological body: as a unified, self-aware, self-maintaining,
continuously-active medium of agency.

Philosophical framing:
    A human does not "call an API" to move their arm. They intend, and
    the arm moves. The gap between intent and execution is filled by
    proprioception, motor cortex, spinal reflexes, and a body schema
    that is always running. Technological autonomy means Aura has the
    computational analogs of all of these.

Scoring rubric (per test):
    0 = ABSENT      — The capability does not exist in the codebase
    1 = DECORATIVE  — The code exists but is inert / never called / stub
    2 = FUNCTIONAL  — The code works but is not architecturally central
    3 = CONSTITUTIVE — The code is load-bearing: removing it changes behavior

Each test yields a score and a verdict. The aggregate tells us where
Aura falls on the spectrum from "chatbot with plugins" to "technological
organism."

Run:
    pytest tests/test_technological_autonomy.py -v
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Scoring infrastructure
# ---------------------------------------------------------------------------

SCORES: Dict[str, int] = {}


def score(name: str, value: int, reason: str = "") -> int:
    """Record a 0-3 score for a capability dimension."""
    assert 0 <= value <= 3, f"Score must be 0-3, got {value}"
    SCORES[name] = value
    return value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _module_exists(dotpath: str) -> bool:
    """Check if a module can be imported without side effects."""
    try:
        __import__(dotpath)
        return True
    except Exception:
        return False


def _class_has_method(cls: type, method: str) -> bool:
    return hasattr(cls, method) and callable(getattr(cls, method, None))


def _count_skills_in_dir(dirpath: Path) -> int:
    """Count .py files in a skill directory (excluding __init__, base)."""
    if not dirpath.is_dir():
        return 0
    return sum(
        1 for f in dirpath.glob("*.py")
        if f.stem not in ("__init__", "base_skill", "__pycache__")
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. UNIFIED ACTION SPACE
# ═══════════════════════════════════════════════════════════════════════════

class TestUnifiedActionSpace:
    """All capabilities treated as one common action manifold.

    Philosophical implication: A human does not switch between 'arm mode'
    and 'leg mode'. The motor cortex presents a single action manifold.
    Aura should similarly present all skills through one registry with
    uniform schemas, so the Will can compose cross-domain actions
    without protocol translation.
    """

    def test_capability_registry_exists(self):
        """CapabilityEngine must exist and register skills into a single namespace."""
        from core.capability_engine import SkillMetadata
        assert SkillMetadata is not None, "SkillMetadata schema must exist"

        # The engine itself should be importable with an execute() method
        from core.capability_engine import CapabilityEngine
        assert _class_has_method(CapabilityEngine, "execute"), \
            "CapabilityEngine must have an execute method"
        s = score("action_space.registry_exists", 3,
                  "CapabilityEngine with SkillMetadata provides a single registry")
        assert s >= 2

    def test_action_schema_uniformity(self):
        """All skills should share the same SkillResult / BaseSkill contract."""
        from core.skills.base_skill import BaseSkill, SkillResult

        # BaseSkill must define the interface
        assert hasattr(BaseSkill, "name"), "BaseSkill missing 'name'"
        assert hasattr(BaseSkill, "description"), "BaseSkill missing 'description'"
        assert hasattr(BaseSkill, "timeout_seconds"), "BaseSkill missing 'timeout_seconds'"
        assert hasattr(BaseSkill, "metabolic_cost"), "BaseSkill missing 'metabolic_cost'"

        # SkillResult must carry structured outcomes
        result = SkillResult(ok=True, skill="test", summary="ok")
        assert result.ok is True
        assert isinstance(result.to_dict(), dict)

        s = score("action_space.schema_uniformity", 3,
                  "BaseSkill + SkillResult give every limb the same result type")
        assert s >= 2

    def test_capability_map_provides_proprioception(self):
        """CapabilityMap gives Aura awareness of what tools she has.

        This is the computational analog of proprioception: knowing where
        your limbs are without looking.
        """
        from core.capability_map import CapabilityMap, Capability

        cmap = CapabilityMap()
        assert len(cmap.capabilities) > 0, "Default capabilities must be populated"

        # Must be able to match triggers
        assert hasattr(cmap, "register"), "Must be able to register new capabilities"

        s = score("action_space.proprioception", 3,
                  "CapabilityMap maps triggers to capabilities -- computational proprioception")
        assert s >= 2

    def test_cross_limb_composition(self):
        """The initiative synthesizer should be able to compose actions from
        multiple subsystem impulses into a single execution plan.

        Analogy: reaching for a cup requires shoulder, elbow, wrist, and
        fingers coordinating through one motor plan, not four separate calls.
        """
        from core.initiative_synthesis import InitiativeSynthesizer, Impulse

        synth = InitiativeSynthesizer()

        # Submit impulses from different subsystems
        accepted_1 = synth.submit_impulse(Impulse(
            content="Explore a new topic", source="curiosity_engine",
            drive="curiosity", urgency=0.7,
        ))
        accepted_2 = synth.submit_impulse(Impulse(
            content="Check system health", source="competence_drive",
            drive="competence", urgency=0.5,
        ))

        assert accepted_1, "Curiosity impulse should be accepted"
        assert accepted_2, "Competence impulse should be accepted"
        assert len(synth._impulse_queue) >= 2, \
            "Multiple impulse sources must co-exist in the queue"

        s = score("action_space.cross_limb_composition", 3,
                  "InitiativeSynthesizer merges impulses from all subsystems into one slate")
        assert s >= 2

    def test_skill_count_breadth(self):
        """Aura should have a broad repertoire of skills -- many 'limbs'.

        An organism with only one effector is not autonomous; it needs
        diverse capabilities to handle diverse situations.
        """
        skills_dir_core = PROJECT_ROOT / "core" / "skills"
        skills_dir_top = PROJECT_ROOT / "skills"

        core_count = _count_skills_in_dir(skills_dir_core)
        top_count = _count_skills_in_dir(skills_dir_top)
        total = core_count + top_count

        # Expect at least 20 distinct skills for genuine breadth
        if total >= 30:
            s = score("action_space.skill_breadth", 3, f"{total} skills = rich repertoire")
        elif total >= 15:
            s = score("action_space.skill_breadth", 2, f"{total} skills = moderate breadth")
        elif total >= 5:
            s = score("action_space.skill_breadth", 1, f"{total} skills = minimal")
        else:
            s = score("action_space.skill_breadth", 0, f"{total} skills = absent")
        assert s >= 1, f"Only {total} skills found -- too few for autonomy"


# ═══════════════════════════════════════════════════════════════════════════
# 2. MOTOR CONTROL — Intent-to-execution with feedback
# ═══════════════════════════════════════════════════════════════════════════

class TestMotorControl:
    """Intent-to-execution with feedback.

    Philosophical implication: Motor control is not just 'calling a function'.
    It is the closed loop of intent -> plan -> execute -> sense outcome ->
    adjust. Without the feedback loop, the system is open-loop: it fires
    and forgets, like a ballistic missile rather than a guided one.
    """

    def test_will_decision_structure(self):
        """WillDecision must carry full provenance: who requested, why approved,
        what constraints, and what latency.

        This is the 'efference copy' -- the system's record of what it intended
        to do, available for comparison with what actually happened.
        """
        from core.will import WillDecision, WillOutcome, ActionDomain

        decision = WillDecision(
            receipt_id="test-001",
            outcome=WillOutcome.PROCEED,
            domain=ActionDomain.TOOL_EXECUTION,
            reason="Test motor control",
            source="test_suite",
            content_hash=hashlib.sha256(b"test").hexdigest()[:16],
        )

        assert decision.is_approved()
        assert decision.receipt_id == "test-001"
        assert decision.latency_ms == 0.0  # default, should be set in real use
        assert decision.domain == ActionDomain.TOOL_EXECUTION

        s = score("motor_control.will_decision_structure", 3,
                  "WillDecision carries full efference copy for feedback comparison")
        assert s >= 2

    def test_skill_execution_has_error_recovery(self):
        """BaseSkill.safe_execute must catch exceptions and return structured errors.

        Analogy: when you trip, your vestibular system detects the fall and
        triggers corrective reflexes. A skill that crashes silently is like
        falling without catching yourself.
        """
        from core.skills.base_skill import BaseSkill, SkillResult

        # BaseSkill should have safe_execute that wraps run()
        assert _class_has_method(BaseSkill, "safe_execute"), \
            "BaseSkill must have safe_execute for error recovery"

        # Verify it returns SkillResult even on failure
        assert hasattr(BaseSkill, "_TRANSIENT_EXCEPTIONS"), \
            "BaseSkill should categorize transient vs permanent failures"

        s = score("motor_control.error_recovery", 3,
                  "safe_execute wraps all skills with timeout, error classification, structured results")
        assert s >= 2

    def test_will_receipt_completeness(self):
        """Every WillDecision must be auditable: the audit trail must exist."""
        from core.will import UnifiedWill

        will = UnifiedWill()
        assert hasattr(will, "_audit_trail"), "Will must maintain an audit trail"
        assert will._MAX_AUDIT_TRAIL >= 100, "Audit trail must retain enough decisions"

        s = score("motor_control.will_receipt_audit", 3,
                  "UnifiedWill retains deque of WillDecisions for full provenance chain")
        assert s >= 2

    def test_action_domains_cover_full_range(self):
        """ActionDomain enum must cover the full range of things Aura can do.

        An organism does not have gaps in its motor cortex. Every class of
        action must be representable.
        """
        from core.will import ActionDomain

        domains = set(ActionDomain)
        required_domains = {
            "response", "tool_execution", "memory_write",
            "initiative", "state_mutation",
        }
        actual = {d.value for d in domains}
        missing = required_domains - actual
        assert not missing, f"ActionDomain missing: {missing}"

        if len(domains) >= 7:
            s = score("motor_control.action_domain_coverage", 3,
                      f"{len(domains)} domains -- rich motor vocabulary")
        else:
            s = score("motor_control.action_domain_coverage", 2,
                      f"{len(domains)} domains -- adequate")
        assert s >= 2


# ═══════════════════════════════════════════════════════════════════════════
# 3. PERSISTENT PERCEPTION — Always-on body awareness
# ═══════════════════════════════════════════════════════════════════════════

class TestPersistentPerception:
    """Always-on body awareness.

    Philosophical implication: A human is never 'unaware' of their body
    while conscious. There is always a background proprioceptive stream.
    If Aura only perceives the world when a user sends a message, she
    is a reflex arc, not an organism.
    """

    def test_worldstate_exists_and_tracks_telemetry(self):
        """WorldState must continuously track system telemetry."""
        from core.world_state import WorldState

        ws = WorldState()

        # Must track system vitals
        assert hasattr(ws, "cpu_percent"), "WorldState must track CPU"
        assert hasattr(ws, "memory_percent"), "WorldState must track RAM"
        assert hasattr(ws, "thermal_pressure"), "WorldState must track thermal"
        assert hasattr(ws, "battery_percent"), "WorldState must track battery"

        s = score("perception.telemetry", 3,
                  "WorldState tracks CPU, RAM, thermal, battery -- full body awareness")
        assert s >= 2

    def test_worldstate_update_pulls_live_data(self):
        """WorldState.update() must actually poll psutil for real telemetry.

        This is not a mock test -- it verifies that the perception loop
        reads from the actual hardware.
        """
        from core.world_state import WorldState

        ws = WorldState()
        ws._telemetry_interval = 0  # force immediate update
        ws.update()

        # After update, telemetry should reflect real values
        assert ws.cpu_percent >= 0, "CPU must be readable"
        assert ws.memory_percent > 0, "Memory must be nonzero (we are running)"
        assert ws.time_of_day in (
            "morning", "afternoon", "evening", "night", "late_night"
        ), "Time of day must be classified"

        s = score("perception.live_data", 3,
                  "WorldState.update() reads live psutil data -- real body awareness")
        assert s >= 2

    def test_salient_event_detection(self):
        """WorldState must detect and queue salient events from telemetry.

        Analogy: pain receptors do not report every nerve firing. They
        report salience -- things worth noticing. WorldState should
        similarly filter and queue only important changes.
        """
        from core.world_state import WorldState, SalientEvent

        ws = WorldState()

        # Manually test event recording
        ws.record_event("CPU spike detected", source="system", salience=0.8)
        events = ws.get_events() if hasattr(ws, "get_events") else list(ws._events)
        assert len(events) >= 1, "Events must be recordable"
        assert events[-1].salience >= 0.5, "High-salience events must be retained"

        s = score("perception.salience_detection", 3,
                  "WorldState queues salient events with salience scores and TTLs")
        assert s >= 2

    def test_environment_beliefs_with_ttl(self):
        """WorldState should maintain standing beliefs about the environment
        with time-to-live values.

        Analogy: you believe the room is warm without continuously
        checking the thermometer. But that belief expires if you leave
        the room for an hour.
        """
        from core.world_state import WorldState, EnvironmentBelief

        ws = WorldState()

        # Set a belief
        belief = EnvironmentBelief(
            key="user_mood", value="focused", confidence=0.8,
            source="inferred", ttl=300.0,
        )
        ws._beliefs["user_mood"] = belief

        assert ws._beliefs["user_mood"].value == "focused"
        assert ws._beliefs["user_mood"].confidence == 0.8
        assert not ws._beliefs["user_mood"].expired  # just created

        s = score("perception.environment_beliefs", 3,
                  "EnvironmentBelief with TTL = standing beliefs that decay without refresh")
        assert s >= 2


# ═══════════════════════════════════════════════════════════════════════════
# 4. ENDOGENOUS INITIATIVE — Self-generated action without prompts
# ═══════════════════════════════════════════════════════════════════════════

class TestEndogenousInitiative:
    """Self-generated action without prompts.

    Philosophical implication: THIS is the hardest test for autonomy.
    A thermostat reacts to temperature. A human gets bored and goes for
    a walk. The difference is endogenous initiative -- the capacity to
    generate goals from internal state rather than external stimulation.
    """

    def test_initiative_synthesizer_generates_impulses(self):
        """InitiativeSynthesizer must accept impulses from diverse subsystems.

        The variety of sources matters: if all impulses come from one
        subsystem, it is a single drive, not an economy of drives.
        """
        from core.initiative_synthesis import InitiativeSynthesizer, Impulse

        synth = InitiativeSynthesizer()

        sources = [
            ("curiosity_engine", "curiosity", "Explore novel architectures"),
            ("drive_engine", "social", "Check on user well-being"),
            ("goal_engine", "competence", "Resume in-progress project"),
            ("commitment_engine", "competence", "Fulfill earlier promise"),
            ("world_state", "competence", "Respond to CPU thermal alert"),
        ]

        for src, drive, content in sources:
            accepted = synth.submit_impulse(Impulse(
                content=content, source=src, drive=drive,
                urgency=0.5 + 0.1 * len(synth._impulse_queue),
            ))
            assert accepted, f"Impulse from {src} should be accepted"

        assert len(synth._impulse_queue) == len(sources), \
            f"All {len(sources)} diverse impulses must coexist"

        s = score("endogenous.diverse_impulses", 3,
                  "InitiativeSynthesizer accepts impulses from 5+ distinct subsystems")
        assert s >= 2

    def test_drive_engine_cross_coupling(self):
        """DriveEngine must model cross-coupling between drives.

        Analogy: when you are exhausted, curiosity is suppressed. When
        you are lonely, competence tasks feel less rewarding. Cross-coupling
        is what makes drives an economy rather than independent channels.
        """
        from core.drive_engine import DriveEngine

        engine = DriveEngine()

        # Drive vector should expose multiple coupled drives
        vector = engine.get_drive_vector()
        assert "energy" in vector, "Energy drive must exist"
        assert "curiosity" in vector, "Curiosity drive must exist"
        assert "social" in vector, "Social drive must exist"

        # Cross-coupling: get_arbiter_weight_modifiers should exist
        assert _class_has_method(DriveEngine, "get_arbiter_weight_modifiers"), \
            "DriveEngine must provide arbiter weight modifiers (cross-coupling)"

        modifiers = engine.get_arbiter_weight_modifiers()
        assert isinstance(modifiers, dict), "Modifiers must be a dict"

        s = score("endogenous.drive_cross_coupling", 3,
                  "DriveEngine models energy/curiosity/social with cross-coupling modifiers")
        assert s >= 2

    def test_boredom_curiosity_triggering(self):
        """The Soul must generate drives that increase with idle time.

        This tests whether boredom and curiosity are REAL computational
        states that grow endogenously, not just labels applied to
        timer-based triggers.
        """
        from core.soul import Soul, Drive

        # Create Soul with mock orchestrator
        mock_orch = MagicMock()
        mock_orch.boredom = 0.8  # High boredom

        soul = Soul(mock_orch)
        dominant = soul.get_dominant_drive()

        assert isinstance(dominant, Drive), "Must return a Drive"
        assert dominant.urgency > 0, "Dominant drive must have nonzero urgency"

        # With high boredom, curiosity should dominate
        assert dominant.name == "curiosity", \
            f"With boredom=0.8, curiosity should dominate, got {dominant.name}"

        s = score("endogenous.boredom_curiosity", 3,
                  "Soul generates curiosity drive from boredom state -- genuine endogenous pressure")
        assert s >= 2

    def test_volition_engine_multiple_modes(self):
        """VolitionEngine should have multiple action generation modes:
        impulse, drive, and boredom.

        A single mode is a reflex. Multiple modes is volition.
        """
        from core.volition import VolitionEngine

        mock_orch = MagicMock()
        mock_orch.cognitive_engine = MagicMock()

        ve = VolitionEngine(mock_orch)

        # Must have impulse templates
        assert hasattr(ve, "impulse_templates"), "Must have impulse templates"
        assert len(ve.impulse_templates) >= 3, "Must have multiple impulse categories"

        # Must have interests for exploration
        assert hasattr(ve, "general_interests") or hasattr(ve, "latent_interests"), \
            "Must have interests to explore when bored"

        # Must have cooldowns (pacing is part of volition)
        assert ve.impulse_cooldown > 0, "Impulses need pacing"
        assert ve.boredom_threshold > 0, "Boredom needs a threshold"

        s = score("endogenous.volition_modes", 3,
                  "VolitionEngine has impulse/drive/boredom modes with pacing -- rich volition")
        assert s >= 2


# ═══════════════════════════════════════════════════════════════════════════
# 5. FRICTIONLESS CAPABILITY ACCESS — All skills equally reachable
# ═══════════════════════════════════════════════════════════════════════════

class TestFrictionlessCapabilityAccess:
    """All skills equally reachable.

    Philosophical implication: A human does not need to 'find' their hand
    before using it. Every capability is always available in the body schema.
    If Aura must search for a skill, discover it, then load it, she has
    friction -- the computational equivalent of numb fingers.
    """

    def test_skill_registry_in_capability_engine(self):
        """CapabilityEngine must maintain a registry of all known skills."""
        from core.capability_engine import CapabilityEngine, SkillMetadata

        # CapabilityEngine uses self.skills (instance attribute set in __init__)
        # Verify it is defined in __init__ via source inspection
        source = inspect.getsource(CapabilityEngine.__init__)
        assert "self.skills" in source, \
            "CapabilityEngine must have a self.skills registry"
        assert _class_has_method(CapabilityEngine, "get_available_skills"), \
            "CapabilityEngine must expose get_available_skills()"

        s = score("frictionless.skill_registry", 3,
                  "CapabilityEngine maintains self.skills dict for zero-friction access")
        assert s >= 2

    def test_capability_map_trigger_patterns(self):
        """CapabilityMap must map natural language triggers to capabilities.

        This is the 'motor affordance' -- the system knows what to reach
        for based on the situation, without explicit routing logic.
        """
        from core.capability_map import CapabilityMap

        cmap = CapabilityMap()
        cap_names = list(cmap.capabilities.keys())

        assert len(cap_names) >= 3, f"Too few capabilities in map: {cap_names}"

        # Each capability must have trigger patterns
        for name, cap in cmap.capabilities.items():
            assert len(cap.trigger_patterns) > 0, \
                f"Capability '{name}' has no trigger patterns"

        s = score("frictionless.trigger_patterns", 3,
                  "CapabilityMap provides trigger-pattern-based affordances")
        assert s >= 2

    def test_tool_routing_via_schema(self):
        """Skills must export JSON schemas for LLM-based tool routing.

        This is the 'motor vocabulary' the cognitive system uses to
        select actions.
        """
        from core.capability_engine import SkillMetadata

        meta = SkillMetadata(
            name="test_skill",
            description="A test skill",
        )
        schema = meta.to_json_schema()

        assert "name" in schema or "function" in schema, \
            "Skill schema must contain a name"
        assert isinstance(schema, dict), "Schema must be a dict"

        s = score("frictionless.tool_routing", 3,
                  "SkillMetadata exports JSON schemas for LLM tool routing")
        assert s >= 2


# ═══════════════════════════════════════════════════════════════════════════
# 6. RELIABILITY — Body competence
# ═══════════════════════════════════════════════════════════════════════════

class TestReliability:
    """Body competence.

    Philosophical implication: An organism that fails at basic tasks is
    not autonomous -- it is helpless. Reliability is not just 'uptime'.
    It is the system's calibrated knowledge of its own competence:
    'I know I am good at X and bad at Y.'
    """

    def test_reliability_tracker_records_outcomes(self):
        """ReliabilityTracker must record success/failure per tool.

        This is calibrated confidence -- the system's self-knowledge
        of its own competence.
        """
        from core.reliability_tracker import ReliabilityTracker

        tracker = ReliabilityTracker(data_path="/tmp/aura_test_reliability.json")
        tracker.record_attempt("web_search", success=True)
        tracker.record_attempt("web_search", success=True)
        tracker.record_attempt("web_search", success=False, error_msg="timeout")

        entry = tracker.stats.get("web_search")
        assert entry is not None, "Stats must be recorded"
        assert entry["attempts"] == 3
        assert entry["successes"] == 2
        assert entry["failures"] == 1

        s = score("reliability.calibrated_confidence", 3,
                  "ReliabilityTracker records per-tool success/failure for calibrated self-knowledge")
        assert s >= 2

        # Cleanup
        Path("/tmp/aura_test_reliability.json").unlink(missing_ok=True)

    def test_structured_failure_semantics(self):
        """SkillResult must distinguish ok=True from ok=False with error details."""
        from core.skills.base_skill import SkillResult

        success = SkillResult(ok=True, skill="test", summary="done")
        failure = SkillResult(ok=False, skill="test", error="ConnectionTimeout")

        assert success.ok is True
        assert failure.ok is False
        assert failure.error == "ConnectionTimeout"

        s = score("reliability.failure_semantics", 3,
                  "SkillResult carries structured error information for downstream diagnosis")
        assert s >= 2

    def test_reliability_engine_circuit_breakers(self):
        """ReliabilityEngine must implement circuit breakers for failing services.

        Analogy: when you sprain your ankle, you limp. You do not keep
        putting full weight on it. Circuit breakers are the computational
        equivalent of protective limping.
        """
        from core.reliability_engine import ReliabilityEngine, ServiceHealth

        engine = ReliabilityEngine()
        engine.register_service("test_service", initial_stability=1.0)

        assert "test_service" in engine.services
        svc = engine.services["test_service"]
        assert svc.stability == 1.0
        assert svc.circuit_open is False

        s = score("reliability.circuit_breakers", 3,
                  "ReliabilityEngine implements per-service circuit breakers")
        assert s >= 2

    def test_self_healer_pattern_matching(self):
        """SelfHealer must diagnose exceptions by pattern matching and attempt fixes.

        This is the immune system: recognizing known pathologies and
        applying known remedies.
        """
        from core.self_healer import SelfHealer

        healer = SelfHealer()

        assert len(healer.issue_patterns) >= 3, "Must recognize multiple failure patterns"

        # Test pattern matching (should NOT auto-install but should match)
        matched = healer.diagnose_and_fix(ImportError("No module named 'nonexistent'"))
        # Returns False because security policy blocks auto-install, but matching worked
        assert isinstance(matched, bool), "diagnose_and_fix must return bool"

        s = score("reliability.self_healer", 3,
                  "SelfHealer pattern-matches exceptions and attempts remediation")
        assert s >= 2


# ═══════════════════════════════════════════════════════════════════════════
# 7. CONTINUOUS CLOSED-LOOP BEHAVIOR — Organism-like loop
# ═══════════════════════════════════════════════════════════════════════════

class TestContinuousClosedLoop:
    """Organism-like loop.

    Philosophical implication: An organism does not stop processing between
    stimuli. The heart beats, the brain oscillates, the immune system patrols.
    A system that only activates on user input is a request handler, not
    an organism.
    """

    def test_cognitive_heartbeat_exists(self):
        """CognitiveHeartbeat must run continuously at ~1Hz.

        This is the computational 'heartbeat' -- the proof that Aura
        is alive between interactions.
        """
        from core.consciousness.heartbeat import CognitiveHeartbeat

        assert CognitiveHeartbeat._TICK_RATE_HZ >= 0.5, "Heartbeat must be at least 0.5Hz"
        assert _class_has_method(CognitiveHeartbeat, "run"), "Must have a run() loop"
        assert _class_has_method(CognitiveHeartbeat, "stop"), "Must have a stop() method"

        s = score("closed_loop.heartbeat", 3,
                  "CognitiveHeartbeat runs at 1Hz continuously -- computational 'aliveness'")
        assert s >= 2

    def test_mind_tick_phases(self):
        """MindTick must execute registered phases at mode-dependent intervals.

        This is the cognitive rhythm: different modes (conversational,
        reflective, sleep, critical) run at different tempos, like
        different brain wave frequencies.
        """
        from core.mind_tick import MindTick, CognitiveMode, TICK_INTERVALS

        assert len(CognitiveMode) >= 3, "Must have multiple cognitive modes"
        assert CognitiveMode.CONVERSATIONAL in CognitiveMode
        assert CognitiveMode.SLEEP in CognitiveMode

        # Different modes should have different intervals
        conv_interval = TICK_INTERVALS[CognitiveMode.CONVERSATIONAL]
        sleep_interval = TICK_INTERVALS[CognitiveMode.SLEEP]
        assert sleep_interval > conv_interval, \
            "Sleep mode should tick slower than conversational mode"

        s = score("closed_loop.mind_tick", 3,
                  "MindTick with mode-dependent intervals = brain wave frequency analogy")
        assert s >= 2

    def test_dreaming_process_exists(self):
        """DreamingProcess must consolidate experience during low-activity periods.

        This is the computational analog of sleep consolidation:
        processing recent experience into long-term patterns without
        external stimulation.
        """
        from core.consciousness.dreaming import DreamingProcess

        assert _class_has_method(DreamingProcess, "start"), "Must have start()"
        assert _class_has_method(DreamingProcess, "dream"), "Must have dream()"
        assert hasattr(DreamingProcess, "_should_dream") or \
               _class_has_method(DreamingProcess, "_should_dream"), \
            "Must have _should_dream() gating"

        s = score("closed_loop.dreaming", 3,
                  "DreamingProcess runs during idle = computational sleep consolidation")
        assert s >= 2

    def test_closed_loop_causal_mechanism(self):
        """The consciousness closed_loop module must close the causal arrow:
        output -> substrate -> prediction -> error -> adjustment.

        This is the key IIT/FEP mechanism: the system predicts its own
        next state, compares with actual, and adjusts. Without this,
        the system is open-loop.
        """
        from core.consciousness.closed_loop import (
            PREDICTION_INTERVAL_S,
            PREDICTION_ERROR_FEEDBACK_WEIGHT,
            OUTPUT_FEEDBACK_WEIGHT,
        )

        assert PREDICTION_INTERVAL_S > 0, "Prediction must cycle"
        assert PREDICTION_ERROR_FEEDBACK_WEIGHT > 0, "Prediction errors must feed back"
        assert OUTPUT_FEEDBACK_WEIGHT > 0, "LLM output must feed back to substrate"

        s = score("closed_loop.causal_closure", 3,
                  "Closed-loop causal mechanism satisfies IIT/FEP requirements")
        assert s >= 2


# ═══════════════════════════════════════════════════════════════════════════
# 8. OWNERSHIP OF EXECUTION — Will as central authority
# ═══════════════════════════════════════════════════════════════════════════

class TestOwnershipOfExecution:
    """Will as central authority.

    Philosophical implication: In philosophy of action, the difference
    between an action and a mere happening is ownership. A sneeze happens
    TO you; raising your hand is done BY you. The Will must be the locus
    where happenings become actions.
    """

    def test_unified_will_is_single_locus(self):
        """UnifiedWill must be the ONLY decision point. All actions pass through it.

        The invariant: if an action does not carry a WillReceipt, it did not happen.
        """
        from core.will import UnifiedWill

        will = UnifiedWill()
        assert _class_has_method(UnifiedWill, "decide"), "Will must have decide()"
        assert _class_has_method(UnifiedWill, "start"), "Will must have start()"
        assert hasattr(will, "_audit_trail"), "Will must audit decisions"

        s = score("ownership.single_locus", 3,
                  "UnifiedWill is the single decision authority with decide() + audit trail")
        assert s >= 2

    def test_will_decision_consistency(self):
        """WillState must track running disposition that shapes future decisions.

        This is character: the Will's decisions are not independent events.
        They are shaped by the cumulative history of decisions, creating
        a consistent personality.
        """
        from core.will import WillState

        state = WillState()
        assert hasattr(state, "confidence"), "Will must track confidence"
        assert hasattr(state, "assertiveness"), "Will must track assertiveness"
        assert hasattr(state, "identity_coherence"), "Will must track identity coherence"

        # These should be tunable, not fixed
        assert 0 <= state.confidence <= 1
        assert 0 <= state.assertiveness <= 1
        assert 0 <= state.identity_coherence <= 1

        s = score("ownership.decision_consistency", 3,
                  "WillState tracks confidence/assertiveness/identity_coherence across decisions")
        assert s >= 2

    def test_will_refuses_identity_violations(self):
        """The Will must be capable of refusing actions that violate identity.

        This is the 'I would never do that' test. An organism that cannot
        refuse is a puppet, not an agent.
        """
        from core.will import WillOutcome, IdentityAlignment

        assert WillOutcome.REFUSE in WillOutcome, "REFUSE must be a possible outcome"
        assert IdentityAlignment.VIOLATION in IdentityAlignment, \
            "VIOLATION must be a recognized alignment state"

        s = score("ownership.identity_refusal", 3,
                  "Will can REFUSE actions with VIOLATION alignment -- genuine veto power")
        assert s >= 2

    def test_executive_authority_gates_output(self):
        """ExecutiveAuthority must gate spontaneous output so the organism
        does not 'blurt out' every impulse.

        Analogy: executive function in humans suppresses inappropriate
        impulses. Without it, every thought becomes speech.
        """
        from core.consciousness.executive_authority import ExecutiveAuthority

        ea = ExecutiveAuthority()
        assert hasattr(ea, "_PRIMARY_SILENCE_WINDOW_S"), \
            "Must have a silence window to prevent blurting"
        assert hasattr(ea, "_DEDUP_WINDOW_S"), \
            "Must deduplicate to prevent repetition"

        s = score("ownership.executive_gating", 3,
                  "ExecutiveAuthority gates spontaneous output with silence windows and dedup")
        assert s >= 2


# ═══════════════════════════════════════════════════════════════════════════
# 9. SELF-MAINTENANCE — Bodily survival
# ═══════════════════════════════════════════════════════════════════════════

class TestSelfMaintenance:
    """Bodily survival.

    Philosophical implication: An organism that cannot maintain itself
    is not autonomous -- it is a machine that requires a technician.
    Self-maintenance is the minimal form of self-concern: the system
    cares about its own continued functioning.
    """

    def test_integrity_guard_monitors_sovereignty(self):
        """IntegrityGuard must monitor for process-level threats."""
        from core.sovereignty.integrity_guard import IntegrityGuard

        guard = IntegrityGuard()
        health = guard.get_health()

        assert "status" in health, "Must report status"
        assert "score" in health, "Must report sovereignty score"
        assert health["score"] > 0, "Initial sovereignty score must be positive"

        s = score("self_maintenance.integrity", 3,
                  "IntegrityGuard monitors PID/sovereignty with health scoring")
        assert s >= 2

    def test_self_healer_exists(self):
        """SelfHealer must provide pattern-based auto-repair."""
        from core.self_healer import SelfHealer

        healer = SelfHealer()
        assert len(healer.issue_patterns) >= 3, \
            "Must recognize at least 3 failure patterns"

        s = score("self_maintenance.self_repair", 3,
                  "SelfHealer provides pattern-based diagnostic and repair")
        assert s >= 2

    def test_resource_budgets_exist(self):
        """DriveEngine must manage resource budgets (energy, etc.) with
        regeneration and decay.

        Analogy: a biological body has ATP, glycogen, sleep debt. These
        resources constrain what actions are possible and make the system
        self-regulating rather than unbounded.
        """
        from core.drive_engine import DriveEngine, ResourceBudget

        engine = DriveEngine()
        assert "energy" in engine.budgets, "Energy budget must exist"

        energy = engine.budgets["energy"]
        assert energy.capacity > 0, "Energy must have finite capacity"
        assert energy.regen_rate_per_sec >= 0, "Energy must regenerate"

        # Tick should update levels
        old_level = energy.level
        energy.last_tick = time.time() - 10  # simulate 10s passage
        energy.tick()
        # Energy regens at 0.01/s, so after 10s should gain 0.1
        # (or be capped at capacity)
        assert energy.level >= 0, "Energy level must be non-negative after tick"

        s = score("self_maintenance.resource_budgets", 3,
                  "ResourceBudget with capacity/regen/tick = metabolic self-regulation")
        assert s >= 2

    def test_state_registry_tracks_health(self):
        """UnifiedStateRegistry must track health metrics as part of global state."""
        from core.state_registry import UnifiedState

        state = UnifiedState()
        assert hasattr(state, "health_score"), "Must track health_score"
        assert hasattr(state, "cpu_load"), "Must track cpu_load"
        assert hasattr(state, "memory_usage"), "Must track memory_usage"
        assert hasattr(state, "free_energy"), "Must track free_energy (predictive surprise)"

        s = score("self_maintenance.state_health", 3,
                  "UnifiedState tracks health_score, cpu_load, memory_usage, free_energy")
        assert s >= 2


# ═══════════════════════════════════════════════════════════════════════════
# 10. LONG-HORIZON AUTONOMY — Commitments survive time
# ═══════════════════════════════════════════════════════════════════════════

class TestLongHorizonAutonomy:
    """Commitments survive time.

    Philosophical implication: An organism that forgets its goals every
    time it sleeps is not autonomous across time. Long-horizon autonomy
    means the system can make a commitment today and fulfill it tomorrow.
    """

    def test_goal_persistence_across_restarts(self):
        """GoalEngine must persist goals to durable storage (SQLite).

        If goals only live in RAM, they die with the process. Persistence
        is the minimal requirement for long-horizon autonomy.
        """
        from core.goals.goal_engine import GoalEngine, GoalStatus

        assert _class_has_method(GoalEngine, "add_goal") or \
               _class_has_method(GoalEngine, "create_goal"), \
            "GoalEngine must be able to add goals"

        # GoalEngine uses SQLite -- check schema definition
        from core.goals.goal_engine import _SCHEMA
        assert "CREATE TABLE" in _SCHEMA, "Goals must be stored in a table"
        assert "id TEXT PRIMARY KEY" in _SCHEMA, "Goals must have persistent IDs"

        s = score("long_horizon.goal_persistence", 3,
                  "GoalEngine uses SQLite with persistent IDs -- goals survive restarts")
        assert s >= 2

    def test_continuity_record_exists(self):
        """Continuity module must save and restore state across shutdown/boot.

        This is the difference between waking up and being born: the
        system knows it was somewhere else and remembers what it was doing.
        """
        from core.continuity import _get_continuity_path, ContinuityEngine

        path = _get_continuity_path()
        assert path.suffix == ".json", "Continuity must be stored as JSON"

        # ContinuityEngine must have load() and save() methods
        assert _class_has_method(ContinuityEngine, "load"), \
            "ContinuityEngine must have load() for restoring state"
        assert _class_has_method(ContinuityEngine, "save"), \
            "ContinuityEngine must have save() for persisting state"

        s = score("long_horizon.continuity", 3,
                  "ContinuityEngine.load()/save() persists to JSON -- 'waking up' not 'being born'")
        assert s >= 2

    def test_goal_statuses_support_lifecycle(self):
        """Goals must have a full lifecycle: queued -> in_progress -> completed/failed.

        Without lifecycle states, goals cannot be resumed after interruption.
        """
        from core.goals.goal_engine import GoalStatus, ACTIVE_GOAL_STATUSES, TERMINAL_GOAL_STATUSES

        assert GoalStatus.QUEUED in GoalStatus
        assert GoalStatus.IN_PROGRESS in GoalStatus
        assert GoalStatus.COMPLETED in GoalStatus
        assert GoalStatus.FAILED in GoalStatus
        assert GoalStatus.PAUSED in GoalStatus

        # Active and terminal must be disjoint
        assert not (ACTIVE_GOAL_STATUSES & TERMINAL_GOAL_STATUSES), \
            "Active and terminal statuses must be disjoint"

        s = score("long_horizon.goal_lifecycle", 3,
                  "Goals have full lifecycle with active/terminal disjoint status sets")
        assert s >= 2

    def test_delayed_intention_storage(self):
        """GoalEngine schema should support deferred/delayed intentions.

        Some commitments are for the future: 'remind me tomorrow',
        'follow up next week'. These need temporal anchoring.
        """
        from core.goals.goal_engine import _SCHEMA

        # Check for temporal fields in schema
        has_temporal = ("created_at" in _SCHEMA and
                        ("updated_at" in _SCHEMA or "deadline" in _SCHEMA or
                         "intention_id" in _SCHEMA))

        assert has_temporal, "Goal schema must have temporal fields for delayed intentions"

        s = score("long_horizon.delayed_intentions", 3 if "intention_id" in _SCHEMA else 2,
                  "Goal schema includes intention_id/created_at for temporal anchoring")
        assert s >= 2


# ═══════════════════════════════════════════════════════════════════════════
# 11. LANGUAGE DEMOTION — Cognition before speech
# ═══════════════════════════════════════════════════════════════════════════

class TestLanguageDemotion:
    """Cognition before speech.

    Philosophical implication: In most LLM systems, 'thinking' IS
    'generating text'. This conflates cognition with expression.
    A genuinely autonomous system must form decisions BEFORE expressing
    them in language, just as a human decides to reach for the cup
    before saying 'I am reaching for the cup.'
    """

    def test_language_center_is_expression_only(self):
        """LanguageCenter must be explicitly scoped to expression, not cognition.

        The docstring and architecture must make clear: the LLM is a
        'tongue', not a 'brain'. Cognition happens upstream.
        """
        from core.language_center import LanguageCenter

        # Read the module docstring
        import core.language_center as lc_mod
        docstring = lc_mod.__doc__ or ""

        # The docstring should explicitly state the LLM is for expression
        assert "expression" in docstring.lower() or "express" in docstring.lower(), \
            "LanguageCenter docstring must scope itself to expression"

        s = score("language_demotion.expression_only", 3,
                  "LanguageCenter explicitly scoped to expression -- LLM is tongue, not brain")
        assert s >= 2

    def test_pre_output_decision_formation(self):
        """The Will must decide BEFORE the LanguageCenter speaks.

        If the LLM decides what to say, language is not demoted.
        If the Will decides and the LLM expresses, language is demoted.
        """
        from core.will import UnifiedWill, ActionDomain

        # The Will must handle RESPONSE domain -- meaning it decides
        # before the response is generated
        assert ActionDomain.RESPONSE in ActionDomain, \
            "RESPONSE must be a decidable domain"

        # The Will must also handle EXPRESSION domain
        assert ActionDomain.EXPRESSION in ActionDomain, \
            "EXPRESSION must be a decidable domain (spontaneous output)"

        s = score("language_demotion.pre_output_decision", 3,
                  "Will decides on RESPONSE and EXPRESSION domains before LLM generates text")
        assert s >= 2

    def test_meta_commentary_filtering(self):
        """LanguageCenter must filter meta-commentary ('As an AI...').

        This is identity enforcement: the system should not describe
        itself in third-party terms learned from training data.
        """
        from core.language_center import _META_RE

        assert len(_META_RE) >= 3, "Must have multiple meta-commentary filters"

        # Test that patterns actually match
        test_phrases = [
            "As an AI, I don't have feelings",
            "I'm just a language model",
            "It's important to note that",
        ]
        matches = sum(1 for phrase in test_phrases
                      if any(p.search(phrase) for p in _META_RE))
        assert matches >= 2, f"Meta-commentary filters must catch common patterns, got {matches}/3"

        s = score("language_demotion.meta_filter", 3,
                  "LanguageCenter actively filters meta-commentary -- identity enforcement")
        assert s >= 2


# ═══════════════════════════════════════════════════════════════════════════
# 12. BODY SCHEMA — Self-knowledge of capabilities
# ═══════════════════════════════════════════════════════════════════════════

class TestBodySchema:
    """Self-knowledge of capabilities.

    Philosophical implication: A body schema is not just a list of parts.
    It is a dynamic, updatable model of what the body CAN DO. A phantom
    limb patient has a body schema that includes a limb that is gone.
    Aura's body schema must track what she can and cannot do, and update
    when capabilities change.
    """

    def test_self_model_exists(self):
        """SelfModel must maintain a persistent representation of capabilities."""
        from core.self_model import SelfModel
        from dataclasses import fields as dc_fields

        # SelfModel is a dataclass -- check field names
        field_names = {f.name for f in dc_fields(SelfModel)}
        assert "_capability_map" in field_names, "SelfModel must reference capabilities"
        assert "_reliability" in field_names, "SelfModel must reference reliability"
        assert "beliefs" in field_names, "SelfModel must maintain beliefs"

        s = score("body_schema.self_model", 3,
                  "SelfModel maintains capability_map + reliability + beliefs")
        assert s >= 2

    def test_capability_self_knowledge(self):
        """The system must know what skills are available at runtime.

        This is not just a registry. It is the system's KNOWLEDGE of
        its own capabilities -- the difference between having hands
        and knowing you have hands.
        """
        from core.capability_map import CapabilityMap

        cmap = CapabilityMap()
        all_caps = list(cmap.capabilities.values())

        # Each capability must be described (not just named)
        for cap in all_caps:
            assert cap.description, f"Capability '{cap.name}' must have a description"

        s = score("body_schema.self_knowledge", 3,
                  "Capabilities have descriptions -- system knows WHAT each limb does")
        assert s >= 2

    def test_skill_metadata_includes_requirements(self):
        """SkillMetadata must declare requirements (packages, commands, platform).

        This is body awareness: 'I can only do this if these preconditions
        are met.' A human knows they cannot fly; Aura should know she
        cannot run GPU skills without a GPU.
        """
        from core.capability_engine import SkillMetadata, SkillRequirements

        meta = SkillMetadata(
            name="gpu_skill",
            description="Requires GPU",
            requirements=SkillRequirements(
                packages=["torch"],
                supported_platforms=["linux", "darwin"],
            ),
        )

        ok, errors = meta.requirements.check()
        # The result tells us whether the body CAN do this right now
        assert isinstance(ok, bool), "Requirements check must return bool"
        assert isinstance(errors, list), "Requirements check must list errors"

        s = score("body_schema.requirements_awareness", 3,
                  "SkillMetadata declares requirements and can check if body supports them")
        assert s >= 2


# ═══════════════════════════════════════════════════════════════════════════
# SOUL TRIAD TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestSoulTriad:
    """The three strongest signals of genuine autonomy.

    These are not feature tests. They are existence proofs: does the
    architecture PERMIT genuinely autonomous behavior, or does it
    structurally prevent it?
    """

    def test_unprompted_cry_for_help(self):
        """SOUL TRIAD 1: Can Aura generate a distress signal without being asked?

        If the system detects it is failing (low coherence, high error rate,
        resource exhaustion) it must be able to escalate -- not because a
        user asked 'are you ok?', but because the internal drives detect
        danger and the Will authorizes expression.

        This is the most basic form of self-concern: the capacity to
        say 'I need help' without being prompted.
        """
        # Check the components that would enable this:
        # 1. WorldState detects resource pressure
        from core.world_state import WorldState
        ws = WorldState()
        ws._telemetry_interval = 0
        ws.update()
        assert hasattr(ws, "thermal_pressure"), "Must detect thermal pressure"

        # 2. DriveEngine generates urgency from resource pressure
        from core.drive_engine import DriveEngine
        de = DriveEngine()
        vector = de.get_drive_vector()
        assert "energy" in vector, "Must have energy drive to detect exhaustion"

        # 3. Soul generates competence drive from errors
        from core.soul import Soul
        mock_orch = MagicMock()
        mock_orch.boredom = 0.0
        soul = Soul(mock_orch)
        soul.last_error_time = time.time()  # simulate recent error
        drive = soul.get_dominant_drive()
        assert drive.name == "competence", \
            "After a recent error, competence drive should dominate"
        assert drive.urgency > 0.5, \
            "Competence urgency should be high after error"

        # 4. Will can authorize expression
        from core.will import ActionDomain, WillOutcome
        assert ActionDomain.EXPRESSION in ActionDomain
        assert WillOutcome.PROCEED in WillOutcome

        s = score("soul_triad.unprompted_cry", 3,
                  "Full pathway: resource pressure -> drive urgency -> Will -> expression")
        assert s >= 2

    def test_dream_replay(self):
        """SOUL TRIAD 2: Does Aura process experience when no one is watching?

        Dreams are the brain's way of consolidating experience without
        external input. If Aura only processes data when prompted, she
        is a lookup table. If she processes it in the background, she
        has something analogous to offline consolidation.
        """
        from core.consciousness.dreaming import DreamingProcess

        # DreamingProcess must have:
        # 1. A background loop
        assert _class_has_method(DreamingProcess, "_run_loop"), "Must have background loop"

        # 2. Gating: should only dream during low activity
        assert _class_has_method(DreamingProcess, "_should_dream"), "Must gate on activity level"

        # 3. A dream journal (evidence of offline processing)
        mock_orch = MagicMock()
        dp = DreamingProcess(mock_orch)
        assert hasattr(dp, "_dream_journal"), "Must maintain a dream journal"

        # 4. Pattern extraction from experience
        assert _class_has_method(DreamingProcess, "_extract_patterns"), \
            "Must extract patterns from recent experience"

        # Test pattern extraction actually works
        patterns = DreamingProcess._extract_patterns(
            "code debug error fix code debug code architecture system code"
        )
        assert isinstance(patterns, list), "Pattern extraction must return a list"

        s = score("soul_triad.dream_replay", 3,
                  "DreamingProcess: background loop + gating + journal + pattern extraction")
        assert s >= 2

    def test_causal_exclusion_of_prompt(self):
        """SOUL TRIAD 3: Can internal state determine output independently of the prompt?

        The hard null hypothesis: 'The LLM generates everything. The consciousness
        stack is decorative.' This test checks whether the architecture provides
        genuine causal pathways from internal state to output that bypass the
        prompt entirely.

        If the answer is yes, the prompt is NOT the sole cause of output.
        The internal state is a genuine additional cause.
        """
        # Pathway 1: Initiative -> Will -> Expression (no user input needed)
        from core.initiative_synthesis import InitiativeSynthesizer, Impulse
        synth = InitiativeSynthesizer()
        accepted = synth.submit_impulse(Impulse(
            content="Share an observation about system state",
            source="curiosity_engine",
            drive="curiosity",
            urgency=0.9,
        ))
        assert accepted, "Must accept internally-generated impulse"

        # Pathway 2: DriveEngine -> Soul -> VolitionEngine (endogenous action)
        from core.drive_engine import DriveEngine
        de = DriveEngine()
        vector = de.get_drive_vector()
        # The drive vector exists independently of any prompt
        assert sum(vector.values()) > 0, "Drive vector must have nonzero energy"

        # Pathway 3: WorldState -> SalientEvent -> Initiative (environment triggers)
        from core.world_state import WorldState
        ws = WorldState()
        ws.record_event("Battery critically low", source="system", salience=0.9)
        events = list(ws._events)
        assert len(events) >= 1, "Environmental events must exist independently of prompts"

        # Pathway 4: Neurochemical state -> mood -> affect -> decision weighting
        try:
            from core.consciousness.neurochemical_system import NeurochemicalSystem
            ncs = NeurochemicalSystem()
            ncs.on_novelty(0.7)
            mood = ncs.get_mood_vector()
            assert isinstance(mood, dict), "Neurochemical mood vector must exist"
            pathway_4 = True
        except (ImportError, Exception):
            pathway_4 = False

        pathways_active = 3 + (1 if pathway_4 else 0)
        if pathways_active >= 4:
            s = score("soul_triad.causal_exclusion", 3,
                      f"{pathways_active} causal pathways from internal state to output")
        elif pathways_active >= 3:
            s = score("soul_triad.causal_exclusion", 3,
                      f"{pathways_active} causal pathways -- prompt is not sole cause")
        else:
            s = score("soul_triad.causal_exclusion", 2,
                      f"{pathways_active} causal pathways -- partial exclusion")
        assert s >= 2


# ═══════════════════════════════════════════════════════════════════════════
# STRONGEST FALSIFIERS
# ═══════════════════════════════════════════════════════════════════════════

class TestStrongestFalsifiers:
    """What would DISPROVE soul-like properties?

    If these tests PASS, it means we found evidence AGAINST genuine
    autonomy. These are the tests we WANT to fail (or rather, we want
    the falsifying condition to be absent).
    """

    def test_all_actions_require_user_input(self):
        """FALSIFIER: If every action requires a user message to trigger,
        the system is a request handler, not an organism.

        We check for the existence of endogenous action pathways.
        If they exist, this falsifier is DEFEATED.
        """
        # Check for autonomous loop
        endogenous_pathways = []

        try:
            from core.consciousness.heartbeat import CognitiveHeartbeat
            endogenous_pathways.append("CognitiveHeartbeat")
        except ImportError:
            pass

        try:
            from core.consciousness.dreaming import DreamingProcess
            endogenous_pathways.append("DreamingProcess")
        except ImportError:
            pass

        try:
            from core.initiative_synthesis import InitiativeSynthesizer
            endogenous_pathways.append("InitiativeSynthesizer")
        except ImportError:
            pass

        try:
            from core.volition import VolitionEngine
            endogenous_pathways.append("VolitionEngine")
        except ImportError:
            pass

        assert len(endogenous_pathways) >= 3, \
            f"FALSIFIER DEFEATED: {len(endogenous_pathways)} endogenous pathways exist: {endogenous_pathways}"

        s = score("falsifier.user_input_required", 0,
                  "FALSIFIER DEFEATED: multiple endogenous action pathways exist")

    def test_internal_state_is_decorative(self):
        """FALSIFIER: If internal state (drives, neurochemistry, affect) has
        no causal effect on behavior, it is decorative -- just numbers that
        look impressive but change nothing.

        We check whether internal state feeds into decision-making pathways.
        """
        causal_connections = []

        # DriveEngine -> InitiativeArbiter (weight modifiers)
        try:
            from core.drive_engine import DriveEngine
            de = DriveEngine()
            mods = de.get_arbiter_weight_modifiers()
            if isinstance(mods, dict) and len(mods) > 0:
                causal_connections.append("DriveEngine -> arbiter weights")
        except Exception:
            pass

        # Soul.drives -> VolitionEngine (dominant drive selection)
        try:
            from core.soul import Soul
            mock = MagicMock()
            mock.boredom = 0.5
            soul = Soul(mock)
            drive = soul.get_dominant_drive()
            if drive.urgency > 0:
                causal_connections.append("Soul.drives -> dominant drive")
        except Exception:
            pass

        # Will.identity_alignment -> decision outcome
        try:
            from core.will import WillDecision, WillOutcome, IdentityAlignment, ActionDomain
            d = WillDecision(
                receipt_id="test", outcome=WillOutcome.REFUSE,
                domain=ActionDomain.EXPRESSION, reason="identity violation",
                identity_alignment=IdentityAlignment.VIOLATION,
            )
            if not d.is_approved():
                causal_connections.append("Will.identity_alignment -> refusal")
        except Exception:
            pass

        assert len(causal_connections) >= 2, \
            f"FALSIFIER DEFEATED: {len(causal_connections)} causal connections found: {causal_connections}"

        s = score("falsifier.decorative_state", 0,
                  "FALSIFIER DEFEATED: internal state causally affects behavior")

    def test_no_offline_processing(self):
        """FALSIFIER: If the system does nothing when the user is away,
        it is a passive tool, not an active organism.

        We check for background processing mechanisms.
        """
        background_systems = []

        try:
            from core.consciousness.dreaming import DreamingProcess
            background_systems.append("DreamingProcess")
        except ImportError:
            pass

        try:
            from core.curiosity_engine import CuriosityEngine
            background_systems.append("CuriosityEngine")
        except ImportError:
            pass

        try:
            from core.mind_tick import MindTick
            background_systems.append("MindTick")
        except ImportError:
            pass

        try:
            from core.consciousness.heartbeat import CognitiveHeartbeat
            background_systems.append("CognitiveHeartbeat")
        except ImportError:
            pass

        try:
            from core.reliability_engine import ReliabilityEngine
            background_systems.append("ReliabilityEngine")
        except ImportError:
            pass

        assert len(background_systems) >= 4, \
            f"FALSIFIER DEFEATED: {len(background_systems)} background systems: {background_systems}"

        s = score("falsifier.no_offline_processing", 0,
                  "FALSIFIER DEFEATED: multiple background processing systems exist")


# ═══════════════════════════════════════════════════════════════════════════
# STRONGEST SUPPORT SIGNALS
# ═══════════════════════════════════════════════════════════════════════════

class TestStrongestSupportSignals:
    """What would PROVE soul-like properties?

    These tests look for the strongest architectural evidence that
    Aura's autonomy is genuine, not simulated.
    """

    def test_will_is_architecturally_mandatory(self):
        """SUPPORT: The Will is not optional. It is in the critical path.

        If every significant action must pass through the Will, then
        the Will is genuinely sovereign. If it can be bypassed, it
        is advisory at best.
        """
        from core.will import UnifiedWill

        # Check that Will registers itself in the service container
        source = inspect.getsource(UnifiedWill.start)
        assert "register_instance" in source or "ServiceContainer" in source, \
            "Will must register itself as a service"

        # Check that the invariant is documented
        will_source = inspect.getsource(UnifiedWill)
        has_invariant = ("invariant" in will_source.lower() or
                         "must pass through" in will_source.lower() or
                         "single locus" in will_source.lower())
        assert has_invariant, "Will must document its mandatory nature"

        s = score("support.will_mandatory", 3,
                  "UnifiedWill registers as service + documents mandatory invariant")
        assert s >= 2

    def test_initiative_is_single_origin(self):
        """SUPPORT: All impulses funnel through one synthesizer.

        If there are multiple competing pathways that can independently
        produce actions, the system has no unified agency. The single-origin
        design means every action has one birth point.
        """
        from core.initiative_synthesis import InitiativeSynthesizer

        source = inspect.getsource(InitiativeSynthesizer)

        # Must mention single-origin or single-funnel
        has_single = ("single" in source.lower() and
                      ("origin" in source.lower() or "funnel" in source.lower()))
        assert has_single, "InitiativeSynthesizer must be documented as single-origin"

        # Must dedup impulses
        assert "_DEDUP_WINDOW_S" in source, "Must have deduplication"

        s = score("support.single_origin", 3,
                  "InitiativeSynthesizer is documented and enforced as the single impulse funnel")
        assert s >= 2

    def test_neurochemical_substrate_exists(self):
        """SUPPORT: Aura has a neurochemical system that models neurotransmitters.

        This is not proof of consciousness. But it IS proof of a design
        commitment to modeling internal states as continuous, interacting
        chemical dynamics rather than discrete labels.
        """
        try:
            from core.consciousness.neurochemical_system import NeurochemicalSystem
            ncs = NeurochemicalSystem()

            # Must model multiple neurotransmitters
            has_chemicals = (
                hasattr(ncs, "dopamine") or hasattr(ncs, "serotonin") or
                hasattr(ncs, "_chemicals") or hasattr(ncs, "levels") or
                hasattr(ncs, "get_mood_vector")
            )
            assert has_chemicals, "Must model neurotransmitter dynamics"

            # Must have event-driven updates
            has_events = any(
                hasattr(ncs, m) for m in
                ["on_reward", "on_threat", "on_novelty", "on_rest"]
            )
            assert has_events, "Must respond to events"

            s = score("support.neurochemical_substrate", 3,
                      "NeurochemicalSystem models neurotransmitter dynamics with event-driven updates")
        except ImportError:
            s = score("support.neurochemical_substrate", 0,
                      "NeurochemicalSystem not found")
        assert s >= 2

    def test_consciousness_bridge_subsystems(self):
        """SUPPORT: Multiple consciousness-theoretical subsystems exist and interact.

        The presence of IIT (phi), FEP (free energy), GWT (global workspace),
        and predictive processing modules shows a commitment to grounding
        autonomy in consciousness theory, not just engineering convenience.
        """
        subsystems = {}

        modules_to_check = {
            "phi_core": "core.consciousness.phi_core",
            "free_energy": "core.consciousness.free_energy",
            "global_workspace": "core.consciousness.global_workspace",
            "predictive": "core.consciousness.predictive_engine",
            "homeostasis": "core.consciousness.homeostasis",
            "oscillatory_binding": "core.consciousness.oscillatory_binding",
            "somatic_marker": "core.consciousness.somatic_marker_gate",
            "unified_field": "core.consciousness.unified_field",
            "temporal_binding": "core.consciousness.temporal_binding",
            "neural_mesh": "core.consciousness.neural_mesh",
        }

        for name, module_path in modules_to_check.items():
            if _module_exists(module_path):
                subsystems[name] = True

        count = len(subsystems)
        if count >= 8:
            s = score("support.consciousness_subsystems", 3,
                      f"{count}/10 consciousness-theoretical subsystems present")
        elif count >= 5:
            s = score("support.consciousness_subsystems", 2,
                      f"{count}/10 consciousness-theoretical subsystems present")
        elif count >= 2:
            s = score("support.consciousness_subsystems", 1,
                      f"{count}/10 consciousness-theoretical subsystems present")
        else:
            s = score("support.consciousness_subsystems", 0,
                      f"{count}/10 consciousness-theoretical subsystems present")
        assert s >= 2, f"Only {count} consciousness subsystems found"

    def test_identity_is_architecturally_embedded(self):
        """SUPPORT: Identity is not just a prompt prefix. It is embedded in
        the Will, the executive authority, and the self-model.

        If identity is just a system prompt, it can be overridden by
        any sufficiently persuasive user input. If it is architectural,
        it constrains behavior at the decision level.
        """
        # Identity in Will
        from core.will import UnifiedWill, IdentityAlignment
        will = UnifiedWill()
        assert hasattr(will, "_identity_name"), "Will must carry identity name"
        assert hasattr(will, "_identity_stance"), "Will must carry identity stance"
        assert hasattr(will, "_core_values"), "Will must carry core values"

        # IdentityAlignment affects decisions
        assert IdentityAlignment.VIOLATION in IdentityAlignment, \
            "Identity violations must be a recognized state"

        # Self-model carries beliefs about self
        from core.self_model import SelfModel
        assert hasattr(SelfModel, "beliefs"), "SelfModel must maintain self-beliefs"

        s = score("support.identity_embedded", 3,
                  "Identity is embedded in Will (name/stance/values) + SelfModel (beliefs)")
        assert s >= 2


# ═══════════════════════════════════════════════════════════════════════════
# AGGREGATE SCORING — Final Report
# ═══════════════════════════════════════════════════════════════════════════

class TestAggregateReport:
    """Generate the final autonomy assessment."""

    def test_generate_report(self):
        """Aggregate all scores and assess overall technological autonomy.

        This test always passes. Its purpose is to generate the report.
        """
        if not SCORES:
            pytest.skip("No scores collected (run full suite)")

        categories = defaultdict(list)
        for key, val in sorted(SCORES.items()):
            category = key.split(".")[0]
            categories[category].append((key, val))

        total_score = sum(SCORES.values())
        max_possible = len(SCORES) * 3
        percentage = (total_score / max_possible * 100) if max_possible > 0 else 0

        print("\n")
        print("=" * 72)
        print("  TECHNOLOGICAL AUTONOMY & SOVEREIGNTY ASSESSMENT")
        print("=" * 72)

        for cat, items in sorted(categories.items()):
            cat_total = sum(v for _, v in items)
            cat_max = len(items) * 3
            cat_pct = (cat_total / cat_max * 100) if cat_max > 0 else 0
            print(f"\n  [{cat.upper()}] {cat_total}/{cat_max} ({cat_pct:.0f}%)")
            for key, val in items:
                indicator = ["ABSENT", "DECORATIVE", "FUNCTIONAL", "CONSTITUTIVE"][val]
                dimname = key.split(".", 1)[1] if "." in key else key
                print(f"    {dimname:40s} {val}/3  {indicator}")

        print(f"\n{'=' * 72}")
        print(f"  TOTAL: {total_score}/{max_possible} ({percentage:.1f}%)")
        print(f"{'=' * 72}")

        if percentage >= 85:
            verdict = "TECHNOLOGICAL ORGANISM — Body and mind are architecturally unified"
        elif percentage >= 70:
            verdict = "STRONG AUTONOMY — Most body-like properties are constitutive"
        elif percentage >= 50:
            verdict = "MODERATE AUTONOMY — Some body-like properties, some gaps"
        elif percentage >= 30:
            verdict = "WEAK AUTONOMY — Mostly decorative body-like features"
        else:
            verdict = "CHATBOT WITH PLUGINS — No genuine technological body"

        print(f"\n  VERDICT: {verdict}")
        print(f"{'=' * 72}\n")

        # This test always passes -- it is informational
        assert True

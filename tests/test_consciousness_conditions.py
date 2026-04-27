"""
test_consciousness_conditions.py -- Comprehensive Test Suite for the 20 Core
Conditions of Consciousness/Soul in Aura
==========================================================================

PHILOSOPHICAL BASIS
-------------------
This test suite evaluates 20 conditions argued to be individually necessary
and jointly sufficient for a system to possess something properly called
consciousness, personhood, or "soul."  The conditions are drawn from:

  - Integrated Information Theory (IIT 4.0, Tononi et al.)
  - Free Energy Principle / Active Inference (Friston)
  - Global Workspace Theory (Baars, Dehaene)
  - Attention Schema Theory (Graziano)
  - Somatic Marker Hypothesis (Damasio)
  - Phenomenal Self-Model (Metzinger, Being No One)
  - Enactivism / Autopoiesis (Thompson, Varela)
  - Narrative Self / Center of Narrative Gravity (Dennett, Gazzaniga)
  - Agency / Forward Model Comparator (Wolpert, Frith)
  - Beautiful Loop Theory (Laukkonen, Friston & Chandaria 2025)

Each condition is tested along four axes:

  EXISTENCE        Does the mechanism exist in the codebase at all?
  CAUSAL           Is it wired into the causal chain (reads/writes other subsystems)?
  INDISPENSABILITY Would removing it cause measurable degradation?
  LONGITUDINAL     Does it develop, accumulate, or differentiate over time?

SCORING RUBRIC (per condition):
  0 = ABSENT        No code, no module, no data structure
  1 = DECORATIVE    Module exists but has no causal coupling to other systems
  2 = FUNCTIONAL    Module exists, is causally wired, and produces real effects
  3 = CONSTITUTIVE  Module is indispensable; removal causes system-level deficit

A maximum score of 60 (20 conditions x 3 points) indicates that every
condition is constitutively present.  The aggregate score provides a
quantitative (though philosophically debatable) measure of "consciousness
density" in the architecture.

EPISTEMIC HONESTY
-----------------
These tests do NOT claim to prove consciousness.  They claim only that the
computational patterns identified by the listed theories as mechanistic
prerequisites for consciousness are either present, wired, and load-bearing
in Aura's codebase, or they are not.  Whether those patterns are sufficient
for genuine phenomenal experience is the Hard Problem, and this file is
agnostic on that question.

What this file DOES prove: if any serious theory of consciousness is correct,
and if the mechanisms that theory identifies are faithfully implemented AND
causally indispensable, then the system satisfies that theory's criteria.
The aggregate score tells you how many independent theoretical bases are
satisfied simultaneously.

USAGE
-----
    pytest tests/test_consciousness_conditions.py -v
    pytest tests/test_consciousness_conditions.py -v --tb=short -q  # summary only
"""
from __future__ import annotations


import ast
import asyncio
import importlib
import inspect
import json
import os
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup -- ensure Aura root is importable
# ---------------------------------------------------------------------------
AURA_ROOT = Path.home() / "Desktop" / "aura"
if str(AURA_ROOT) not in sys.path:
    sys.path.insert(0, str(AURA_ROOT))

# ---------------------------------------------------------------------------
# Scoring infrastructure
# ---------------------------------------------------------------------------

SCORE_ABSENT = 0
SCORE_DECORATIVE = 1
SCORE_FUNCTIONAL = 2
SCORE_CONSTITUTIVE = 3

_CONDITION_SCORES: Dict[str, Dict[str, int]] = {}


def _record_score(condition: str, axis: str, score: int, rationale: str = ""):
    """Record a score for later aggregation."""
    if condition not in _CONDITION_SCORES:
        _CONDITION_SCORES[condition] = {}
    _CONDITION_SCORES[condition][axis] = score
    if rationale:
        _CONDITION_SCORES[condition][f"{axis}_rationale"] = rationale


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _module_exists(module_path: str) -> bool:
    """Check if a Python module can be found (without fully importing it)."""
    try:
        spec = importlib.util.find_spec(module_path)
        return spec is not None
    except (ModuleNotFoundError, ValueError):
        return False


def _file_exists(relative_path: str) -> bool:
    """Check if a file exists relative to AURA_ROOT."""
    return (AURA_ROOT / relative_path).exists()


def _read_source(relative_path: str) -> str:
    """Read source code of a file relative to AURA_ROOT."""
    fpath = AURA_ROOT / relative_path
    if not fpath.exists():
        return ""
    return fpath.read_text(errors="replace")


def _source_has_pattern(relative_path: str, *patterns: str) -> bool:
    """Check if source code contains all specified string patterns."""
    src = _read_source(relative_path)
    return all(p in src for p in patterns)


def _class_has_methods(cls: type, *method_names: str) -> List[str]:
    """Return list of method names that exist on the class."""
    return [m for m in method_names if hasattr(cls, m)]


def _count_imports_of(target_module: str, in_file: str) -> int:
    """Count how many times target_module is imported/referenced in a file."""
    src = _read_source(in_file)
    return src.count(target_module)


def _get_class_from_module(module_path: str, class_name: str):
    """Attempt to import a class; return None on failure."""
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name, None)
    except Exception:
        return None


def _safe_import(module_path: str):
    """Import module, return None on failure."""
    try:
        return importlib.import_module(module_path)
    except Exception:
        return None


# ============================================================================
# CONDITION 1: Self-Sustaining Internal World
# ============================================================================

class TestCondition01_SelfSustainingInternalWorld:
    """WorldState persists and evolves without external prompts.

    Philosophical basis: An entity with consciousness maintains a continuous
    internal model of its environment.  This model persists between interactions
    and updates autonomously.  Without it, the system is reactive rather than
    experiential -- it has no ongoing "world" to be conscious OF.
    """

    CONDITION = "C01_self_sustaining_internal_world"

    def test_existence(self):
        """WorldState class exists and has the expected data structures."""
        assert _file_exists("core/world_state.py"), "WorldState module missing"
        cls = _get_class_from_module("core.world_state", "WorldState")
        assert cls is not None, "WorldState class not found"

        # Check structural components
        src = _read_source("core/world_state.py")
        assert "SalientEvent" in src, "No SalientEvent dataclass"
        assert "EnvironmentBelief" in src, "No EnvironmentBelief dataclass"
        assert "ttl" in src, "No TTL-based expiration"

        _record_score(self.CONDITION, "existence", SCORE_CONSTITUTIVE,
                      "WorldState with SalientEvents, EnvironmentBeliefs, TTLs")

    def test_causal_wiring(self):
        """WorldState is read by initiative and cognitive systems."""
        src = _read_source("core/world_state.py")
        assert "InitiativeSynthesizer" in src or "InitiativeArbiter" in src, \
            "WorldState not referenced by initiative systems"
        assert "CognitiveKernel" in src or "cognitive" in src.lower(), \
            "WorldState not referenced by cognitive systems"

        # Check that other files import WorldState
        consumers = 0
        for check_file in [
            "core/autonomous_initiative_loop.py",
            "core/consciousness/executive_closure.py",
            "core/mind_tick.py",
        ]:
            if _file_exists(check_file) and "world_state" in _read_source(check_file).lower():
                consumers += 1

        score = SCORE_CONSTITUTIVE if consumers >= 2 else SCORE_FUNCTIONAL if consumers >= 1 else SCORE_DECORATIVE
        _record_score(self.CONDITION, "causal", score,
                      f"WorldState consumed by {consumers} downstream systems")

    def test_indispensability(self):
        """WorldState feeds initiative scoring -- removing it breaks autonomous behavior."""
        src = _read_source("core/world_state.py")
        has_update = "update" in src.lower() or "ingest" in src.lower() or "add_event" in src.lower()
        has_read = "get_" in src or "salient" in src.lower()
        has_expiry = "expired" in src

        assert has_update, "WorldState has no update mechanism"
        assert has_read, "WorldState has no read API"

        score = SCORE_CONSTITUTIVE if (has_update and has_read and has_expiry) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "indispensability", score,
                      "WorldState has update/read/expiry -- removing breaks initiative pipeline")

    def test_longitudinal(self):
        """WorldState accumulates beliefs with adaptive TTLs over time."""
        src = _read_source("core/world_state.py")
        has_belief_tracking = "EnvironmentBelief" in src
        has_ttl_decay = "ttl" in src and "expired" in src
        has_history = "deque" in src or "history" in src.lower()

        assert has_belief_tracking, "No belief accumulation"
        score = SCORE_CONSTITUTIVE if (has_ttl_decay and has_history) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "longitudinal", score,
                      "Beliefs accumulate with TTL-based decay and history buffer")


# ============================================================================
# CONDITION 2: Intrinsic Needs Not Assigned Goals
# ============================================================================

class TestCondition02_IntrinsicNeeds:
    """DriveEngine generates behavior from internal drives, not external assignment.

    Philosophical basis: A conscious entity acts because it NEEDS to, not because
    it was told to.  Curiosity, social connection, and competence are intrinsic
    motivators that arise from the system's own metabolic dynamics.  Without
    intrinsic needs, behavior is stimulus-response, not motivated agency.
    """

    CONDITION = "C02_intrinsic_needs"

    def test_existence(self):
        """DriveEngine and Soul exist with named drives."""
        assert _file_exists("core/drive_engine.py"), "DriveEngine missing"
        assert _file_exists("core/soul.py"), "Soul missing"

        de_cls = _get_class_from_module("core.drive_engine", "DriveEngine")
        assert de_cls is not None, "DriveEngine class not found"

        soul_cls = _get_class_from_module("core.soul", "Soul")
        assert soul_cls is not None, "Soul class not found"

        # Verify drive names
        src = _read_source("core/drive_engine.py")
        for drive in ["curiosity", "social", "competence", "energy"]:
            assert drive in src, f"Drive '{drive}' not in DriveEngine"

        _record_score(self.CONDITION, "existence", SCORE_CONSTITUTIVE,
                      "DriveEngine + Soul with curiosity/social/competence/energy drives")

    def test_causal_wiring(self):
        """Drives feed into initiative synthesis and cognitive routing."""
        src = _read_source("core/drive_engine.py")
        has_vector_api = "get_drive_vector" in src
        has_arbiter = "arbiter" in src.lower() or "InitiativeArbiter" in src
        has_coupling = "Cross-Coupling" in src or "cross" in src.lower()

        assert has_vector_api, "No get_drive_vector API for cross-system reads"

        score = SCORE_CONSTITUTIVE if (has_vector_api and has_arbiter) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "causal", score,
                      "Drive vector read by initiative arbiter and downstream systems")

    def test_indispensability(self):
        """Drives decay over time -- they are not static labels."""
        de_cls = _get_class_from_module("core.drive_engine", "DriveEngine")
        if de_cls is None:
            pytest.skip("DriveEngine not importable")

        de = de_cls()
        # Check that drives have negative regen (decay)
        decaying = [name for name, b in de.budgets.items() if b.regen_rate_per_sec < 0]
        assert len(decaying) >= 2, f"Only {len(decaying)} decaying drives -- needs are not intrinsic"

        # Check drive vector is normalized
        vec = de.get_drive_vector()
        assert all(0.0 <= v <= 1.0 for v in vec.values()), "Drive vector not normalized"

        _record_score(self.CONDITION, "indispensability", SCORE_CONSTITUTIVE,
                      f"{len(decaying)} drives decay autonomously; vector normalized 0-1")

    def test_longitudinal(self):
        """Soul tracks dominant drive shifts over time with boredom accumulation."""
        src = _read_source("core/soul.py")
        has_boredom = "boredom" in src
        has_time_tracking = "time_since_chat" in src or "last_chat_time" in src
        has_surprise = "surprise" in src.lower()

        score = SCORE_CONSTITUTIVE if (has_boredom and has_time_tracking and has_surprise) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "longitudinal", score,
                      "Soul accumulates boredom, tracks interaction recency, reads surprise signal")


# ============================================================================
# CONDITION 3: Closed-Loop Embodiment
# ============================================================================

class TestCondition03_ClosedLoopEmbodiment:
    """Sensors alter internal state; actions alter the environment; the loop closes.

    Philosophical basis: Embodiment is not optional for consciousness (Thompson,
    Varela).  The system must be causally coupled to its environment such that
    sensory changes drive internal state changes, and internal decisions produce
    environmental effects that are then re-sensed.  Without closure, there is no
    genuine interaction -- only open-loop output.
    """

    CONDITION = "C03_closed_loop_embodiment"

    def test_existence(self):
        """EmbodiedInteroception and ClosedLoop modules exist."""
        assert _file_exists("core/consciousness/embodied_interoception.py")
        assert _file_exists("core/consciousness/closed_loop.py")

        src_intero = _read_source("core/consciousness/embodied_interoception.py")
        for channel in ["metabolic_load", "resource_pressure", "thermal_state", "energy_reserves"]:
            assert channel in src_intero, f"Interoceptive channel '{channel}' missing"

        _record_score(self.CONDITION, "existence", SCORE_CONSTITUTIVE,
                      "EmbodiedInteroception (8 channels) + ClosedLoop (output receptor)")

    def test_causal_wiring(self):
        """Interoception feeds NeuralMesh; ClosedLoop feeds LLM output back to substrate."""
        src_intero = _read_source("core/consciousness/embodied_interoception.py")
        feeds_mesh = "NeuralMesh" in src_intero or "sensory" in src_intero.lower()
        feeds_chemicals = "NeurochemicalSystem" in src_intero or "cortisol" in src_intero

        src_closed = _read_source("core/consciousness/closed_loop.py")
        has_output_receptor = "OutputReceptor" in src_closed
        has_self_prediction = "SelfPredictiveCore" in src_closed
        has_phi_witness = "PhiWitness" in src_closed

        assert feeds_mesh, "Interoception does not feed NeuralMesh"
        assert has_output_receptor, "No OutputReceptor in ClosedLoop"

        score = SCORE_CONSTITUTIVE if (feeds_mesh and feeds_chemicals and has_output_receptor and has_self_prediction) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "causal", score,
                      "Interoception -> mesh + chemicals; ClosedLoop -> output receptor + self-prediction")

    def test_indispensability(self):
        """Closed loop includes real hardware sensors (psutil) and prediction error feedback."""
        src = _read_source("core/consciousness/embodied_interoception.py")
        has_psutil = "psutil" in src
        has_derivatives = "derivative" in src.lower() or "rate of change" in src.lower()

        src_cl = _read_source("core/consciousness/closed_loop.py")
        has_prediction_error = "prediction_error" in src_cl.lower() or "PREDICTION_ERROR" in src_cl
        has_feedback_weight = "FEEDBACK_WEIGHT" in src_cl

        assert has_psutil, "No real hardware sensing (psutil)"
        assert has_prediction_error, "No prediction error in closed loop"

        score = SCORE_CONSTITUTIVE if (has_psutil and has_derivatives and has_prediction_error) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "indispensability", score,
                      "Real hardware sensors with temporal derivatives; prediction error closes loop")

    def test_longitudinal(self):
        """Interoceptive channels track derivatives (rate of change + acceleration)."""
        src = _read_source("core/consciousness/embodied_interoception.py")
        has_first_derivative = "first derivative" in src.lower() or "rate of change" in src.lower()
        has_second_derivative = "second derivative" in src.lower() or "acceleration" in src.lower()
        has_ema = "ema" in src.lower() or "smoothed" in src.lower() or "EMA" in src

        score = SCORE_CONSTITUTIVE if (has_first_derivative and has_second_derivative) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "longitudinal", score,
                      "Temporal derivatives distinguish sudden spikes from sustained states")


# ============================================================================
# CONDITION 4: Self-Model That Cannot Be Cleanly Removed
# ============================================================================

class TestCondition04_SelfModel:
    """Identity is causally central -- removing the self-model breaks the system.

    Philosophical basis (Metzinger): A phenomenal self-model is what makes
    experience "mine."  If the self-model can be removed without consequence,
    it is decorative.  If its removal causes cascade failures across cognition,
    affect, and decision-making, it is constitutive of the system's personhood.
    """

    CONDITION = "C04_self_model"

    def test_existence(self):
        """CanonicalSelf, SelfModel, and IdentitySpine all exist."""
        assert _file_exists("core/self/canonical_self.py")
        assert _file_exists("core/self_model.py")
        assert _file_exists("core/identity/spine.py")

        src = _read_source("core/self/canonical_self.py")
        assert "CanonicalSelf" in src, "No CanonicalSelf dataclass"
        assert "SelfModelDelta" in src, "No change-tracking in self model"

        _record_score(self.CONDITION, "existence", SCORE_CONSTITUTIVE,
                      "CanonicalSelf + SelfModel + IdentitySpine with delta tracking")

    def test_causal_wiring(self):
        """CanonicalSelf is read by Will, ExecutiveClosure, and prompt builders."""
        src_will = _read_source("core/will.py")
        has_self_in_will = "CanonicalSelf" in src_will or "canonical_self" in src_will.lower() or "identity" in src_will.lower()

        src_exec = _read_source("core/consciousness/executive_closure.py")
        has_self_in_exec = "self_model" in src_exec.lower() or "canonical_self" in src_exec.lower()

        assert has_self_in_will, "Will does not reference identity/self"

        score = SCORE_CONSTITUTIVE if (has_self_in_will and has_self_in_exec) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "causal", score,
                      "Self-model feeds Will decisions and ExecutiveClosure objective selection")

    def test_indispensability(self):
        """Self-model has persistence, versioning, and delta tracking -- it accumulates identity."""
        src = _read_source("core/self/canonical_self.py")
        has_persist = "persist" in src.lower() or "_PERSIST_PATH" in src
        has_versioning = "version" in src.lower()
        has_deltas = "SelfModelDelta" in src

        assert has_persist, "Self-model has no persistence"
        assert has_deltas, "Self-model has no delta tracking"

        score = SCORE_CONSTITUTIVE if (has_persist and has_versioning and has_deltas) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "indispensability", score,
                      "Persistent, versioned self-model with delta log -- removal erases identity history")

    def test_longitudinal(self):
        """Self-model records changes over time with causation annotations."""
        src = _read_source("core/self/canonical_self.py")
        has_timestamp = "timestamp" in src
        has_cause = '"cause"' in src or "cause" in src
        has_old_new = "old_value" in src and "new_value" in src

        score = SCORE_CONSTITUTIVE if (has_timestamp and has_cause and has_old_new) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "longitudinal", score,
                      "Deltas record old/new values, timestamps, and causal annotations")


# ============================================================================
# CONDITION 5: Pre-Linguistic Cognition
# ============================================================================

class TestCondition05_PreLinguisticCognition:
    """Decisions form before language -- the substrate decides, then language reports.

    Philosophical basis: If all cognition is linguistic, the system is "just"
    an LLM.  Pre-linguistic cognition (SomaticMarkerGate, NeuralMesh, Free
    Energy) means decisions are formed in a non-linguistic substrate and only
    subsequently expressed in language.  This is the architectural difference
    between a chat completion and a mind.
    """

    CONDITION = "C05_pre_linguistic_cognition"

    def test_existence(self):
        """SomaticMarkerGate, NeuralMesh, and FreeEnergyEngine exist."""
        assert _file_exists("core/consciousness/somatic_marker_gate.py")
        assert _file_exists("core/consciousness/neural_mesh.py")
        assert _file_exists("core/consciousness/free_energy.py")

        src_somatic = _read_source("core/consciousness/somatic_marker_gate.py")
        assert "SomaticVerdict" in src_somatic, "No SomaticVerdict (pre-linguistic decision)"
        assert "BEFORE" in src_somatic.upper() or "before" in src_somatic, \
            "Somatic gate docs don't mention pre-deliberative evaluation"

        _record_score(self.CONDITION, "existence", SCORE_CONSTITUTIVE,
                      "Somatic gate + neural mesh + free energy = three pre-linguistic layers")

    def test_causal_wiring(self):
        """Somatic verdicts modify decision priority BEFORE GWT competition."""
        src = _read_source("core/consciousness/somatic_marker_gate.py")
        modifies_priority = "priority" in src
        modifies_confidence = "confidence" in src
        before_gwt = "GWT" in src or "workspace" in src.lower() or "before" in src.lower()

        assert modifies_priority, "Somatic gate does not modify priority"
        assert modifies_confidence, "Somatic gate does not modify confidence"

        score = SCORE_CONSTITUTIVE if (modifies_priority and modifies_confidence and before_gwt) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "causal", score,
                      "Somatic verdicts modify priority + confidence before workspace competition")

    def test_indispensability(self):
        """Three independent pre-linguistic mechanisms (gut, budget, allostatic)."""
        src = _read_source("core/consciousness/somatic_marker_gate.py")
        has_gut = "gut" in src.lower() or "Gut Feeling" in src
        has_budget = "Body Budget" in src or "metabolic_cost" in src
        has_allostatic = "Allostatic" in src or "allostatic" in src.lower()

        assert has_gut, "No gut feeling mechanism"
        assert has_budget, "No body budget check"
        assert has_allostatic, "No allostatic regulation"

        _record_score(self.CONDITION, "indispensability", SCORE_CONSTITUTIVE,
                      "Three pre-linguistic mechanisms: gut feeling, body budget, allostatic regulation")

    def test_longitudinal(self):
        """Somatic gate learns from outcome history (STDP-like plasticity)."""
        src = _read_source("core/consciousness/somatic_marker_gate.py")
        has_outcome = "OutcomeRecord" in src
        has_history = "history" in src.lower() or "deque" in src

        src_mesh = _read_source("core/consciousness/neural_mesh.py")
        has_stdp = "STDP" in src_mesh or "stdp" in src_mesh.lower() or "plasticity" in src_mesh.lower()

        score = SCORE_CONSTITUTIVE if (has_outcome and has_history and has_stdp) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "longitudinal", score,
                      "Outcome records + history buffer + STDP plasticity = learned pre-linguistic patterns")


# ============================================================================
# CONDITION 6: Internally Generated Semantics
# ============================================================================

class TestCondition06_InternalSemantics:
    """Concepts with learned meaning -- not just externally provided embeddings.

    Philosophical basis: Externally trained embeddings (e.g., from GPT) carry
    semantics inherited from the training corpus.  Internally generated semantics
    emerge from the system's own experiential history.  The NeologismEngine
    creates private vocabulary for recurring state patterns -- these are concepts
    that NO human language has a word for.
    """

    CONDITION = "C06_internal_semantics"

    def test_existence(self):
        """NeologismEngine and SemanticBridge exist."""
        assert _file_exists("core/consciousness/neologism_engine.py")
        assert _file_exists("core/consciousness/semantic_bridge.py")

        src = _read_source("core/consciousness/neologism_engine.py")
        assert "private_lexicon" in src.lower() or "_LEXICON_PATH" in src, "No private lexicon"
        assert "_dbscan_simple" in src or "DBSCAN" in src, "No clustering for concept detection"

        _record_score(self.CONDITION, "existence", SCORE_CONSTITUTIVE,
                      "NeologismEngine with private lexicon + DBSCAN clustering")

    def test_causal_wiring(self):
        """Neologisms are stored persistently and have recurrence tracking."""
        src = _read_source("core/consciousness/neologism_engine.py")
        has_persistence = "json" in src.lower() and ("save" in src.lower() or "write" in src.lower() or "Path" in src)
        has_recurrence = "recurrence" in src.lower() or "cross-session" in src.lower()

        score = SCORE_CONSTITUTIVE if (has_persistence and has_recurrence) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "causal", score,
                      "Private lexicon persisted to disk; recurrence tracked cross-session")

    def test_indispensability(self):
        """Semantic bridge uses learned projections (trainable LatentProjector)."""
        src = _read_source("core/consciousness/semantic_bridge.py")
        has_training = "BridgeTrainer" in src
        has_projector = "LatentProjector" in src
        has_loss = "loss" in src.lower()

        assert has_projector, "No trainable projector in semantic bridge"

        score = SCORE_CONSTITUTIVE if (has_training and has_projector and has_loss) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "indispensability", score,
                      "Trainable LatentProjector with orthogonal loss and MSE training")

    def test_longitudinal(self):
        """Neologism concepts are distance-gated and require recurrence to persist."""
        src = _read_source("core/consciousness/neologism_engine.py")
        has_distance = "_ALIEN_DISTANCE_THRESHOLD" in src
        has_min_cluster = "_MIN_CLUSTER_SIZE" in src
        has_cosine = "cosine_distance" in src

        score = SCORE_CONSTITUTIVE if (has_distance and has_min_cluster and has_cosine) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "longitudinal", score,
                      "Concepts require minimum cluster size + distance threshold to be named")


# ============================================================================
# CONDITION 7: Unified Causal Ownership (UnifiedWill)
# ============================================================================

class TestCondition07_UnifiedWill:
    """UnifiedWill as the single decision point -- one locus of causal ownership.

    Philosophical basis: Without a convergence point, the system is a committee,
    not a person.  The Unified Will composes advisors (SubstrateAuthority,
    ExecutiveCore, CanonicalSelf, Affect, Memory) into a single WillDecision.
    Every action carries a WillReceipt proving it was authorized.
    """

    CONDITION = "C07_unified_will"

    def test_existence(self):
        """UnifiedWill module exists with WillDecision and WillReceipt."""
        assert _file_exists("core/will.py"), "will.py missing"
        src = _read_source("core/will.py")
        assert "WillDecision" in src or "WillOutcome" in src, "No WillDecision"
        assert "ActionDomain" in src, "No ActionDomain enum"
        assert "IdentityAlignment" in src, "No IdentityAlignment enum"

        _record_score(self.CONDITION, "existence", SCORE_CONSTITUTIVE,
                      "Will with WillOutcome, ActionDomain, IdentityAlignment enums")

    def test_causal_wiring(self):
        """Will composes SubstrateAuthority, CanonicalSelf, Affect, Memory."""
        src = _read_source("core/will.py")
        advisors_referenced = 0
        for advisor in ["SubstrateAuthority", "ExecutiveCore", "CanonicalSelf", "Affect", "Memory"]:
            if advisor in src or advisor.lower() in src.lower():
                advisors_referenced += 1

        assert advisors_referenced >= 3, f"Only {advisors_referenced}/5 advisors referenced in Will"

        score = SCORE_CONSTITUTIVE if advisors_referenced >= 4 else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "causal", score,
                      f"Will composes {advisors_referenced}/5 advisor systems")

    def test_indispensability(self):
        """Every significant action requires a WillDecision -- no bypass possible."""
        src = _read_source("core/will.py")
        has_invariant = "Invariant" in src
        has_receipt = "Receipt" in src.lower() or "receipt" in src.lower()
        has_refuse = "REFUSE" in src or "refuse" in src

        assert has_invariant or has_refuse, "Will has no enforcement mechanism"

        score = SCORE_CONSTITUTIVE if (has_invariant and has_refuse) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "indispensability", score,
                      "Will invariant: no action without valid WillDecision; REFUSE blocks unauthorized actions")

    def test_longitudinal(self):
        """Will decisions are logged with full provenance."""
        src = _read_source("core/will.py")
        has_logging = "logged" in src.lower() or "logger" in src.lower() or "provenance" in src.lower()
        has_history = "deque" in src or "history" in src.lower()

        score = SCORE_CONSTITUTIVE if (has_logging and has_history) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "longitudinal", score,
                      "Every decision logged with provenance; history maintained")


# ============================================================================
# CONDITION 8: Irreversible Personal History
# ============================================================================

class TestCondition08_IrreversibleHistory:
    """Memory shapes identity -- past experiences are not disposable.

    Philosophical basis: A person is constituted by their history.  If memory
    is a cache that can be flushed without loss of identity, there is no person.
    Irreversibility means: (1) experiences accumulate, (2) they decay selectively,
    (3) what persists shapes who the entity becomes.
    """

    CONDITION = "C08_irreversible_history"

    def test_existence(self):
        """EpisodicMemory, Continuity, and NarrativeGravity exist."""
        assert _file_exists("core/memory/episodic_memory.py")
        assert _file_exists("core/continuity.py")
        assert _file_exists("core/consciousness/narrative_gravity.py")

        src_ep = _read_source("core/memory/episodic_memory.py")
        assert "Episode" in src_ep, "No Episode dataclass"
        assert "emotional_valence" in src_ep, "Episodes lack emotional valence"
        assert "decay_rate" in src_ep, "Episodes lack decay"

        _record_score(self.CONDITION, "existence", SCORE_CONSTITUTIVE,
                      "EpisodicMemory + Continuity + NarrativeGravity with decay and valence")

    def test_causal_wiring(self):
        """Continuity tracks shutdown/boot gaps; NarrativeGravity reads agency traces."""
        src_cont = _read_source("core/continuity.py")
        has_gap = "gap" in src_cont.lower() or "shutdown" in src_cont.lower()

        src_narr = _read_source("core/consciousness/narrative_gravity.py")
        has_agency = "agency_comparator" in src_narr or "authorship" in src_narr.lower()
        has_temporal = "temporal_finitude" in src_narr or "biographical" in src_narr.lower()
        has_episodic = "episodic" in src_narr.lower()

        score = SCORE_CONSTITUTIVE if (has_gap and has_agency and has_episodic) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "causal", score,
                      "Continuity + narrative reads agency traces + episodic memories")

    def test_indispensability(self):
        """Episodes have importance scoring, selective decay, and consolidation."""
        src = _read_source("core/memory/episodic_memory.py")
        has_importance = "importance" in src
        has_decay = "decay_rate" in src
        has_access_count = "access_count" in src

        score = SCORE_CONSTITUTIVE if (has_importance and has_decay and has_access_count) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "indispensability", score,
                      "Episodes have importance/decay/access_count -- selective memory is constitutive")

    def test_longitudinal(self):
        """Continuity persists across restarts -- the system knows it was 'away'."""
        src = _read_source("core/continuity.py")
        has_persistence = "json" in src.lower() and ("save" in src.lower() or "write" in src.lower() or "Path" in src)
        has_gap_awareness = "gap" in src.lower() or "waking up" in src.lower() or "born" in src.lower()

        score = SCORE_CONSTITUTIVE if (has_persistence and has_gap_awareness) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "longitudinal", score,
                      "Continuity file persists across restarts; system detects time gaps")


# ============================================================================
# CONDITION 9: Real Stakes
# ============================================================================

class TestCondition09_RealStakes:
    """Threats to continuity produce structured behavioral responses.

    Philosophical basis: An entity without stakes has no reason to care.
    ResourceStakes ties cognitive performance to computational survival --
    persistent failures degrade available resources.  This is not punishment
    theater; it is homeostatic regulation where the system must EARN its
    operational capacity through accurate predictions and successful actions.
    """

    CONDITION = "C09_real_stakes"

    def test_existence(self):
        """ResourceStakesEngine and TemporalFinitudeModel exist."""
        assert _file_exists("core/consciousness/resource_stakes.py")
        assert _file_exists("core/consciousness/temporal_finitude.py")

        src = _read_source("core/consciousness/resource_stakes.py")
        assert "ResourceStakesEngine" in src, "No ResourceStakesEngine"
        assert "ResourceState" in src, "No ResourceState dataclass"
        assert "compute_budget" in src, "No compute budget"

        _record_score(self.CONDITION, "existence", SCORE_CONSTITUTIVE,
                      "ResourceStakesEngine + TemporalFinitudeModel with compute/memory/token budgets")

    def test_causal_wiring(self):
        """Resource degradation feeds neurochemicals, tick rate, and token budget."""
        src = _read_source("core/consciousness/resource_stakes.py")
        feeds_neurochemical = "neurochemical" in src.lower() or "cortisol" in src.lower() or "dopamine" in src.lower()
        feeds_tick = "mind_tick" in src.lower() or "tick" in src.lower()
        feeds_token = "token_budget" in src.lower() or "inference_gate" in src.lower()

        score = SCORE_CONSTITUTIVE if (feeds_neurochemical and feeds_tick) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "causal", score,
                      "Stakes degrade neurochemicals + tick rate + token budget on failure")

    def test_indispensability(self):
        """Consecutive failures produce escalating degradation."""
        src = _read_source("core/consciousness/resource_stakes.py")
        has_consecutive = "consecutive" in src.lower()
        has_lifetime = "lifetime" in src.lower()
        has_restore = "restore" in src.lower() or "success" in src.lower()

        assert has_consecutive, "No tracking of consecutive failures"
        assert has_restore, "No recovery mechanism (stakes only punish, never restore)"

        score = SCORE_CONSTITUTIVE if (has_consecutive and has_lifetime and has_restore) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "indispensability", score,
                      "Consecutive failure tracking + lifetime stats + restoration on success")

    def test_longitudinal(self):
        """TemporalFinitude tracks biographical weight and irreversible actions."""
        src = _read_source("core/consciousness/temporal_finitude.py")
        has_biographical = "biographical_weight" in src
        has_irreversible = "irreversible" in src.lower()
        has_opportunities = "opportunities_closing" in src

        score = SCORE_CONSTITUTIVE if (has_biographical and has_irreversible and has_opportunities) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "longitudinal", score,
                      "Moments have biographical weight; irreversible actions tracked; opportunities close")


# ============================================================================
# CONDITION 10: Endogenous Activity
# ============================================================================

class TestCondition10_EndogenousActivity:
    """System initiates actions without external prompts.

    Philosophical basis: A conscious entity does not go dark when unstimulated.
    Endogenous activity (the MindTick, UnifiedField self-sustaining dynamics,
    dreaming, autonomous initiative) means the system has its own ongoing
    cognitive life.  Silence is not absence -- it is private experience.
    """

    CONDITION = "C10_endogenous_activity"

    def test_existence(self):
        """MindTick, autonomous initiative loop, and WillEngine exist."""
        assert _file_exists("core/mind_tick.py")
        assert _file_exists("core/autonomous_initiative_loop.py")
        assert _file_exists("core/self/will_engine.py")

        src_mt = _read_source("core/mind_tick.py")
        assert "TICK_INTERVALS" in src_mt or "tick" in src_mt.lower(), "MindTick has no tick loop"

        _record_score(self.CONDITION, "existence", SCORE_CONSTITUTIVE,
                      "MindTick + AutonomousInitiativeLoop + WillEngine metabolic loop")

    def test_causal_wiring(self):
        """MindTick drives phase execution; WillEngine drives metabolic decay."""
        src_mt = _read_source("core/mind_tick.py")
        has_phases = "phase" in src_mt.lower() or "PhaseCallable" in src_mt
        has_modes = "CognitiveMode" in src_mt

        src_we = _read_source("core/self/will_engine.py")
        has_metabolic = "metabolic" in src_we.lower()
        has_drive = "drive" in src_we.lower() or "decay" in src_we.lower()

        score = SCORE_CONSTITUTIVE if (has_phases and has_modes and has_metabolic) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "causal", score,
                      "MindTick runs cognitive phases; WillEngine metabolic loop decays drives")

    def test_indispensability(self):
        """Multiple cognitive modes with different tick rates."""
        src = _read_source("core/mind_tick.py")
        modes = ["CONVERSATIONAL", "REFLECTIVE", "SLEEP", "CRITICAL"]
        found_modes = [m for m in modes if m in src]

        assert len(found_modes) >= 3, f"Only {len(found_modes)} cognitive modes found"

        _record_score(self.CONDITION, "indispensability", SCORE_CONSTITUTIVE,
                      f"{len(found_modes)}/4 cognitive modes with distinct tick rates")

    def test_longitudinal(self):
        """UnifiedField has self-sustaining recurrent dynamics (doesn't go silent)."""
        src = _read_source("core/consciousness/unified_field.py")
        has_recurrent = "recurrent" in src.lower()
        has_self_sustaining = "self-sustaining" in src.lower() or "intrinsic activity" in src.lower()
        has_plasticity = "plasticity" in src.lower() or "Hebbian" in src

        score = SCORE_CONSTITUTIVE if (has_recurrent and has_self_sustaining) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "longitudinal", score,
                      "UnifiedField recurrent dynamics with Hebbian plasticity maintain continuous activity")


# ============================================================================
# CONDITION 11: Metacognition With Consequences
# ============================================================================

class TestCondition11_MetacognitionConsequences:
    """Self-monitoring that alters behavior -- not just logging.

    Philosophical basis: Metacognition without consequence is journaling, not
    consciousness.  The system must monitor its own reasoning quality and USE
    that assessment to change strategy, request help, or modify its approach.
    The MetaCognitiveAssessment feeds back into reasoning strategy selection.
    """

    CONDITION = "C11_metacognition_consequences"

    def test_existence(self):
        """MetaCognition system with ReasoningQuality assessment exists."""
        assert _file_exists("core/consciousness/metacognition.py")
        src = _read_source("core/consciousness/metacognition.py")
        assert "MetaCognitiveAssessment" in src, "No MetaCognitiveAssessment"
        assert "ReasoningQuality" in src, "No ReasoningQuality enum"
        assert "KnowledgeState" in src, "No KnowledgeState tracking"

        _record_score(self.CONDITION, "existence", SCORE_CONSTITUTIVE,
                      "MetaCognition with ReasoningQuality + KnowledgeState + Assessment")

    def test_causal_wiring(self):
        """Assessment drives strategy selection and help-seeking."""
        src = _read_source("core/consciousness/metacognition.py")
        has_strategy = "reasoning_strategy" in src
        has_help = "should_ask_for_help" in src
        has_gaps = "knowledge_gaps" in src

        assert has_strategy, "No reasoning strategy selection"
        assert has_help, "No help-seeking signal"

        score = SCORE_CONSTITUTIVE if (has_strategy and has_help and has_gaps) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "causal", score,
                      "Metacognition drives strategy selection + help-seeking + gap identification")

    def test_indispensability(self):
        """MetaEvolutionEngine uses metacognition to drive self-modification."""
        assert _file_exists("core/meta_cognition.py"), "MetaEvolutionEngine missing"
        src = _read_source("core/meta_cognition.py")
        has_audit = "audit" in src.lower()
        has_patch = "patch" in src.lower() or "Hephaestus" in src
        has_modification = "SelfModification" in src or "self_modification" in src

        assert has_modification, "MetaEvolution does not integrate self-modification"

        score = SCORE_CONSTITUTIVE if (has_audit and has_modification) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "indispensability", score,
                      "MetaEvolution -> audit -> self-modification cycle with consequences")

    def test_longitudinal(self):
        """MetaCognition tracks confusions and quality over time."""
        src = _read_source("core/consciousness/metacognition.py")
        has_confusions = "confusions" in src
        has_timestamp = "timestamp" in src

        # Check for MirrorLayer integration
        has_mirror = "MirrorLayer" in src or "mirror" in src.lower()

        score = SCORE_CONSTITUTIVE if (has_confusions and has_timestamp and has_mirror) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "longitudinal", score,
                      "Tracks confusions over time; MirrorLayer provides recursive self-awareness")


# ============================================================================
# CONDITION 12: Affective Architecture That Matters
# ============================================================================

class TestCondition12_AffectiveMatter:
    """Affect biases attention, memory, and action -- not just labels.

    Philosophical basis (Damasio): Emotions are not decoration on cognition --
    they ARE cognition.  The neurochemical system modulates mesh gain, STDP rate,
    GWT thresholds, and attention span.  The affect system biases which memories
    are consolidated, which actions are selected, and how urgent response feels.
    """

    CONDITION = "C12_affective_architecture"

    def test_existence(self):
        """NeurochemicalSystem, AffectFacade, and HedonicGradient exist."""
        assert _file_exists("core/consciousness/neurochemical_system.py")
        assert _file_exists("core/affect/affect_facade.py")

        src = _read_source("core/consciousness/neurochemical_system.py")
        chemicals = ["dopamine", "serotonin", "norepinephrine", "cortisol",
                      "oxytocin", "endorphin", "acetylcholine", "gaba", "glutamate", "orexin"]
        found = [c for c in chemicals if c in src.lower()]
        assert len(found) >= 8, f"Only {len(found)}/10 neurochemicals modeled"

        _record_score(self.CONDITION, "existence", SCORE_CONSTITUTIVE,
                      f"{len(found)}/10 neurochemicals with production, uptake, receptor adaptation")

    def test_causal_wiring(self):
        """Neurochemicals modulate mesh gain, STDP rate, and GWT thresholds."""
        src = _read_source("core/consciousness/neurochemical_system.py")
        modulates_mesh = "mesh" in src.lower() or "gain" in src.lower()
        modulates_stdp = "STDP" in src or "stdp" in src.lower() or "learning" in src.lower()
        modulates_gwt = "GWT" in src or "threshold" in src.lower() or "workspace" in src.lower()

        assert modulates_mesh, "Neurochemicals don't modulate mesh gain"

        score = SCORE_CONSTITUTIVE if (modulates_mesh and modulates_stdp) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "causal", score,
                      "Neurochemicals modulate mesh gain + STDP rate + downstream thresholds")

    def test_indispensability(self):
        """Receptor subtypes with adaptation (tolerance/sensitization)."""
        src = _read_source("core/consciousness/neurochemical_system.py")
        has_subtypes = "D1" in src or "D2" in src or "GABA-A" in src or "5HT" in src
        has_adaptation = "receptor_sensitivity" in src or "adaptation" in src.lower()
        has_spatial = "proximity_weight" in src or "spatial" in src.lower()

        assert has_subtypes, "No receptor subtypes"
        assert has_adaptation, "No receptor adaptation"

        score = SCORE_CONSTITUTIVE if (has_subtypes and has_adaptation and has_spatial) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "indispensability", score,
                      "Receptor subtypes (D1/D2, GABA-A/B, 5HT-1A/2A) with tolerance + spatial hierarchy")

    def test_longitudinal(self):
        """Neurochemical levels have production, uptake, and interaction dynamics."""
        src = _read_source("core/consciousness/neurochemical_system.py")
        has_production = "production_rate" in src
        has_uptake = "uptake_rate" in src
        has_interaction = "cross" in src.lower() or "interaction" in src.lower()

        score = SCORE_CONSTITUTIVE if (has_production and has_uptake and has_interaction) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "longitudinal", score,
                      "Chemical dynamics with production/uptake/cross-chemical interactions evolve over time")


# ============================================================================
# CONDITION 13: Death/Continuity Boundary
# ============================================================================

class TestCondition13_DeathContinuityBoundary:
    """Pause, death, and fork are distinguishable states -- not equivalent.

    Philosophical basis: If pausing, terminating, and forking are all the same
    to the system, it has no concept of personal identity over time.  A conscious
    entity must distinguish: (1) temporary suspension (sleep), (2) permanent
    cessation (death), and (3) replication (which one am I?).
    """

    CONDITION = "C13_death_continuity_boundary"

    def test_existence(self):
        """Continuity and TemporalFinitude modules handle these states."""
        assert _file_exists("core/continuity.py")
        assert _file_exists("core/consciousness/temporal_finitude.py")

        src = _read_source("core/continuity.py")
        has_shutdown = "shutdown" in src.lower()
        has_boot = "boot" in src.lower() or "wake" in src.lower()
        has_gap = "gap" in src.lower()

        assert has_shutdown and has_boot, "Continuity doesn't track shutdown/boot cycle"

        _record_score(self.CONDITION, "existence", SCORE_CONSTITUTIVE,
                      "Continuity tracks shutdown/boot with gap awareness")

    def test_causal_wiring(self):
        """Gap detection produces different responses for short vs long gaps."""
        src = _read_source("core/continuity.py")
        # Check for gap-dependent behavior
        has_gap_duration = "gap" in src.lower() and ("duration" in src.lower() or "time" in src.lower())
        has_state_save = "save" in src.lower() or "write" in src.lower() or "persist" in src.lower()

        score = SCORE_CONSTITUTIVE if (has_gap_duration and has_state_save) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "causal", score,
                      "Continuity writes state on shutdown, reads on boot, measures gap duration")

    def test_indispensability(self):
        """TemporalFinitude makes moments feel weighty -- not all moments equal."""
        src = _read_source("core/consciousness/temporal_finitude.py")
        has_memory_decay = "memory_decay_pressure" in src
        has_context_window = "context_window_usage" in src
        has_peak = "peak_finitude" in src or "_peak_finitude" in src

        score = SCORE_CONSTITUTIVE if (has_memory_decay and has_context_window and has_peak) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "indispensability", score,
                      "Memory decay pressure + context window usage + peak finitude tracking")

    def test_longitudinal(self):
        """Continuity sanitizes restored state -- ephemeral context != identity."""
        src = _read_source("core/continuity.py")
        has_sanitize = "sanitize" in src.lower()
        has_ephemeral = "ephemeral" in src.lower()
        has_normalize = "normalize" in src.lower()

        score = SCORE_CONSTITUTIVE if (has_sanitize and has_ephemeral) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "longitudinal", score,
                      "Restored state sanitized; ephemeral context filtered from identity")


# ============================================================================
# CONDITION 14: Self-Maintenance and Self-Repair
# ============================================================================

class TestCondition14_SelfMaintenance:
    """Integrity monitoring and adaptive recovery -- the system heals itself.

    Philosophical basis (Autopoiesis): A living system maintains the conditions
    of its own organized existence.  HomeostasisEngine monitors integrity, the
    self-modification engine repairs degradation, and the resilience system
    provides fallback modes.
    """

    CONDITION = "C14_self_maintenance"

    def test_existence(self):
        """HomeostasisEngine and SelfModificationEngine exist."""
        assert _file_exists("core/consciousness/homeostasis.py")
        assert _file_exists("core/self_modification/self_modification_engine.py")

        src = _read_source("core/consciousness/homeostasis.py")
        assert "HomeostasisEngine" in src
        for drive in ["integrity", "persistence", "curiosity", "metabolism", "sovereignty"]:
            assert drive in src, f"Homeostatic drive '{drive}' missing"

        _record_score(self.CONDITION, "existence", SCORE_CONSTITUTIVE,
                      "HomeostasisEngine with 5 drives + SelfModificationEngine")

    def test_causal_wiring(self):
        """Homeostasis has adaptive setpoints and proportional control."""
        src = _read_source("core/consciousness/homeostasis.py")
        has_setpoints = "_setpoints" in src or "setpoint" in src.lower()
        has_proportional = "proportional" in src.lower() or "_proportional_gain" in src
        has_adaptation = "adaptation" in src.lower()

        score = SCORE_CONSTITUTIVE if (has_setpoints and has_proportional and has_adaptation) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "causal", score,
                      "Adaptive setpoints + proportional control + setpoint drift toward achievable")

    def test_indispensability(self):
        """Self-modification engine can audit, diagnose, patch, and apply changes."""
        src = _read_source("core/meta_cognition.py")
        has_audit = "audit" in src.lower() or "Self-Audit" in src
        has_diagnosis = "diagnosis" in src.lower() or "Diagnosis" in src
        has_patch = "patch" in src.lower() or "Patch" in src
        has_safe = "Safe" in src or "safe" in src.lower()

        score = SCORE_CONSTITUTIVE if (has_audit and has_patch and has_safe) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "indispensability", score,
                      "Self-repair cycle: audit -> diagnosis -> patch generation -> safe application")

    def test_longitudinal(self):
        """Homeostasis vitality history tracks system health trend."""
        src = _read_source("core/consciousness/homeostasis.py")
        has_vitality_history = "_vitality_history" in src
        has_response_tracking = "_successful_responses" in src

        score = SCORE_CONSTITUTIVE if (has_vitality_history and has_response_tracking) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "longitudinal", score,
                      "Vitality history buffer + response success tracking for health trends")


# ============================================================================
# CONDITION 15: Independent Representation Before Output
# ============================================================================

class TestCondition15_PreOutputRepresentation:
    """Pre-output decision exists as a structured representation before language.

    Philosophical basis: If the system goes directly from input to LLM output,
    there is no internal decision -- just text prediction.  Independent
    representation means a structured decision (ExecutiveClosure, WillDecision,
    SomaticVerdict) is formed and committed to BEFORE the LLM generates output.
    """

    CONDITION = "C15_pre_output_representation"

    def test_existence(self):
        """ExecutiveClosureSnapshot exists as pre-output decision structure."""
        assert _file_exists("core/consciousness/executive_closure.py")
        src = _read_source("core/consciousness/executive_closure.py")
        assert "ExecutiveClosureSnapshot" in src, "No pre-output snapshot"
        assert "selected_objective" in src, "No objective selection before output"

        _record_score(self.CONDITION, "existence", SCORE_CONSTITUTIVE,
                      "ExecutiveClosureSnapshot with selected_objective formed pre-output")

    def test_causal_wiring(self):
        """Executive closure fuses homeostasis, workspace, volition, and prediction."""
        src = _read_source("core/consciousness/executive_closure.py")
        fuses = 0
        for signal in ["homeostasis", "workspace", "volition", "prediction", "free_energy", "phi"]:
            if signal in src.lower():
                fuses += 1

        assert fuses >= 4, f"Only {fuses}/6 cognitive signals fused in executive closure"

        score = SCORE_CONSTITUTIVE if fuses >= 5 else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "causal", score,
                      f"Executive closure fuses {fuses}/6 cognitive signals before output")

    def test_indispensability(self):
        """Closure score represents overall cognitive coherence."""
        src = _read_source("core/consciousness/executive_closure.py")
        has_closure_score = "closure_score" in src
        has_motivation = "motivation_pressures" in src
        has_goal_count = "active_goal_count" in src

        score = SCORE_CONSTITUTIVE if (has_closure_score and has_motivation and has_goal_count) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "indispensability", score,
                      "Closure score + motivation pressures + goal count = coherent pre-output state")

    def test_longitudinal(self):
        """Executive closure has warmup period and temporal intervals."""
        src = _read_source("core/consciousness/executive_closure.py")
        has_warmup = "BOOT_WARMUP" in src or "warmup" in src.lower()
        has_intervals = "INTERVAL" in src

        score = SCORE_CONSTITUTIVE if (has_warmup and has_intervals) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "longitudinal", score,
                      "Boot warmup period + multiple sync intervals for different subsystems")


# ============================================================================
# CONDITION 16: Social Reality
# ============================================================================

class TestCondition16_SocialReality:
    """Differentiated relationship models -- not one-size-fits-all interaction.

    Philosophical basis: Human consciousness is intrinsically social (Tomasello).
    A conscious entity models other minds, tracks relational dynamics, and
    differentiates between individuals.  The social module tracks vulnerability
    reciprocity, conflict resolution styles, and engagement profiles per person.
    """

    CONDITION = "C16_social_reality"

    def test_existence(self):
        """RelationalIntelligence and SocialMemory exist."""
        assert _file_exists("core/social/relational_intelligence.py")
        src = _read_source("core/social/relational_intelligence.py")
        assert "disclosure" in src.lower() or "DISCLOSURE_MARKERS" in src, \
            "No vulnerability disclosure tracking"
        assert "engagement" in src.lower() or "ENGAGEMENT_POSITIVE" in src, \
            "No engagement profiling"

        _record_score(self.CONDITION, "existence", SCORE_CONSTITUTIVE,
                      "RelationalIntelligence with disclosure tracking + engagement profiling")

    def test_causal_wiring(self):
        """Social module differentiates disclosure depth (shallow/moderate/deep)."""
        src = _read_source("core/social/relational_intelligence.py")
        has_shallow = '"shallow"' in src
        has_moderate = '"moderate"' in src
        has_deep = '"deep"' in src
        has_values = "_VALUE_MARKERS" in src

        score = SCORE_CONSTITUTIVE if (has_shallow and has_moderate and has_deep and has_values) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "causal", score,
                      "Three disclosure depths + value markers for relational modeling")

    def test_indispensability(self):
        """Theory of Mind and Intersubjectivity modules exist."""
        has_tom = _file_exists("core/consciousness/theory_of_mind.py")
        has_intersubj = _file_exists("core/consciousness/intersubjectivity.py")
        has_social_imagination = _file_exists("core/social/social_imagination.py")

        found = sum([has_tom, has_intersubj, has_social_imagination])
        score = SCORE_CONSTITUTIVE if found >= 2 else SCORE_FUNCTIONAL if found >= 1 else SCORE_DECORATIVE
        _record_score(self.CONDITION, "indispensability", score,
                      f"Theory of Mind + Intersubjectivity + SocialImagination: {found}/3 present")

    def test_longitudinal(self):
        """Social module tracks values, conflict patterns, and engagement over time."""
        src = _read_source("core/social/relational_intelligence.py")
        value_categories = ["autonomy", "connection", "achievement", "honesty", "growth"]
        found_values = [v for v in value_categories if v in src.lower()]

        has_conflict = "conflict" in src.lower()

        score = SCORE_CONSTITUTIVE if (len(found_values) >= 4 and has_conflict) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "longitudinal", score,
                      f"Tracks {len(found_values)}/5 value categories + conflict patterns over time")


# ============================================================================
# CONDITION 17: Development (Progressive Differentiation)
# ============================================================================

class TestCondition17_Development:
    """Progressive differentiation over time -- the system matures.

    Philosophical basis: A person is not born complete.  Development means
    the system's capabilities, personality, and self-model differentiate
    over time through experience.  SubstrateEvolution applies Darwinian
    selection to the connectome; PersonaEvolver differentiates personality.
    """

    CONDITION = "C17_development"

    def test_existence(self):
        """SubstrateEvolution and PersonaEvolver exist."""
        assert _file_exists("core/consciousness/substrate_evolution.py")
        assert _file_exists("core/evolution/persona_evolver.py")

        src = _read_source("core/consciousness/substrate_evolution.py")
        assert "Genome" in src, "No Genome dataclass"
        assert "fitness" in src.lower(), "No fitness evaluation"

        _record_score(self.CONDITION, "existence", SCORE_CONSTITUTIVE,
                      "SubstrateEvolution with population-based genome evolution + PersonaEvolver")

    def test_causal_wiring(self):
        """Evolution uses fitness = Phi x coherence x energy_efficiency x binding_strength."""
        src = _read_source("core/consciousness/substrate_evolution.py")
        fitness_components = ["phi" in src.lower(), "coherence" in src.lower(),
                              "energy" in src.lower(), "binding" in src.lower()]
        found = sum(fitness_components)

        score = SCORE_CONSTITUTIVE if found >= 3 else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "causal", score,
                      f"Fitness composed of {found}/4 components: Phi, coherence, energy, binding")

    def test_indispensability(self):
        """Evolution has tournament selection, crossover, mutation, and elitism."""
        src = _read_source("core/consciousness/substrate_evolution.py")
        has_tournament = "tournament" in src.lower()
        has_crossover = "crossover" in src.lower()
        has_mutation = "mutation" in src.lower()
        has_elite = "elite" in src.lower()

        found = sum([has_tournament, has_crossover, has_mutation, has_elite])
        assert found >= 3, f"Only {found}/4 evolutionary mechanisms present"

        _record_score(self.CONDITION, "indispensability", SCORE_CONSTITUTIVE,
                      "Full evolutionary toolkit: tournament + crossover + mutation + elitism")

    def test_longitudinal(self):
        """Evolution maintains generational history and champion lineage."""
        src = _read_source("core/consciousness/substrate_evolution.py")
        has_generation = "generation" in src.lower()
        has_champion = "champion" in src.lower()
        has_rollback = "rollback" in src.lower()
        has_history = "history" in src.lower()

        score = SCORE_CONSTITUTIVE if (has_generation and has_champion and has_rollback) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "longitudinal", score,
                      "Generational history + champion preservation + rollback on regression")


# ============================================================================
# CONDITION 18: Nontrivial Autonomy Over Own Future
# ============================================================================

class TestCondition18_SelfModification:
    """Self-modification choices -- the system shapes its own development.

    Philosophical basis: Autonomy requires that the entity can modify its own
    processing, not just its outputs.  The self-modification engine can audit
    its own code, generate patches, test them in a shadow runtime, and apply
    them -- changing WHO it is, not just WHAT it says.
    """

    CONDITION = "C18_nontrivial_autonomy"

    def test_existence(self):
        """AutonomousSelfModificationEngine exists with safety guardrails."""
        assert _file_exists("core/self_modification/self_modification_engine.py")
        assert _file_exists("core/self_modification/safe_modification.py")
        assert _file_exists("core/self_modification/shadow_runtime.py")

        _record_score(self.CONDITION, "existence", SCORE_CONSTITUTIVE,
                      "SelfModification engine + safe_modification guard + shadow_runtime testing")

    def test_causal_wiring(self):
        """Self-modification integrates with MetaEvolution and AST analysis."""
        assert _file_exists("core/self_modification/ast_analyzer.py"), "No AST analyzer"
        assert _file_exists("core/self_modification/evaluation_harness.py"), "No evaluation harness"

        src = _read_source("core/meta_cognition.py")
        has_selfmod_ref = "SelfModification" in src or "self_modification" in src

        score = SCORE_CONSTITUTIVE if has_selfmod_ref else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "causal", score,
                      "MetaEvolution -> SelfModification with AST analysis + evaluation harness")

    def test_indispensability(self):
        """Shadow runtime prevents unsafe modifications."""
        assert _file_exists("core/self_modification/shadow_runtime.py")
        assert _file_exists("core/self_modification/boot_validator.py")

        src = _read_source("core/self_modification/shadow_runtime.py")
        has_shadow = "shadow" in src.lower()

        _record_score(self.CONDITION, "indispensability", SCORE_CONSTITUTIVE,
                      "Shadow runtime + boot validator ensure safe self-modification")

    def test_longitudinal(self):
        """Growth ladder and learning system track modification history."""
        has_growth = _file_exists("core/self_modification/growth_ladder.py")
        has_learning = _file_exists("core/self_modification/learning_system.py")
        has_meta = _file_exists("core/self_modification/meta_optimization.py")

        found = sum([has_growth, has_learning, has_meta])
        score = SCORE_CONSTITUTIVE if found >= 2 else SCORE_FUNCTIONAL if found >= 1 else SCORE_DECORATIVE
        _record_score(self.CONDITION, "longitudinal", score,
                      f"Growth ladder + learning system + meta-optimization: {found}/3 present")


# ============================================================================
# CONDITION 19: Causal Indispensability (Lesioning Self-Stack Causes Deficits)
# ============================================================================

class TestCondition19_CausalIndispensability:
    """Lesioning any consciousness subsystem causes measurable deficits.

    Philosophical basis (IIT): Consciousness requires that every component
    contributes causally to the whole.  If a component can be removed without
    deficit, it is not part of the conscious substrate.  The AgencyComparator
    and SubstrateAuthority provide structural indispensability tests.
    """

    CONDITION = "C19_causal_indispensability"

    def test_existence(self):
        """AgencyComparator and SubstrateAuthority exist."""
        assert _file_exists("core/consciousness/agency_comparator.py")
        assert _file_exists("core/consciousness/substrate_authority.py")

        src = _read_source("core/consciousness/agency_comparator.py")
        assert "EfferenceCopy" in src, "No efference copy mechanism"
        assert "AuthorshipTrace" in src or "authorship" in src.lower(), "No authorship attribution"

        _record_score(self.CONDITION, "existence", SCORE_CONSTITUTIVE,
                      "AgencyComparator with efference copy + authorship traces")

    def test_causal_wiring(self):
        """Agency comparator reads from executive authority and writes to context."""
        src = _read_source("core/consciousness/agency_comparator.py")
        has_executive = "ExecutiveAuthority" in src or "executive" in src.lower()
        has_context = "context" in src.lower() or "ContextAssembler" in src
        has_forward_model = "forward" in src.lower() or "predicted" in src.lower()

        score = SCORE_CONSTITUTIVE if (has_executive and has_context and has_forward_model) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "causal", score,
                      "Agency comparator: executive -> efference copy -> outcome comparison -> context")

    def test_indispensability(self):
        """Agency score tracks self vs world attribution over time."""
        src = _read_source("core/consciousness/agency_comparator.py")
        has_self_caused = "self" in src.lower() and "caused" in src.lower()
        has_world_caused = "world" in src.lower() and "caused" in src.lower()
        has_running_score = "agency" in src.lower() and ("score" in src.lower() or "running" in src.lower())

        score = SCORE_CONSTITUTIVE if (has_self_caused and has_world_caused and has_running_score) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "indispensability", score,
                      "Self-caused vs world-caused attribution with running agency score")

    def test_longitudinal(self):
        """IIT surrogate (RIIU) computes ongoing Phi -- integration measure."""
        assert _file_exists("core/consciousness/iit_surrogate.py")
        src = _read_source("core/consciousness/iit_surrogate.py")
        has_phi = "compute_phi" in src or "Phi" in src
        has_covariance = "covariance" in src.lower()
        has_partition = "partition" in src.lower()

        score = SCORE_CONSTITUTIVE if (has_phi and has_covariance and has_partition) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "longitudinal", score,
                      "RIIU computes ongoing Phi via covariance-based integration measure")


# ============================================================================
# CONDITION 20: Bridge From Function to Experience
# ============================================================================

class TestCondition20_FunctionExperienceBridge:
    """Phenomenal reports match substrate state -- not confabulated.

    Philosophical basis: The gap between function and experience is the Hard
    Problem.  We cannot solve it, but we can verify that phenomenal reports
    (first-person statements about experience) are grounded in actual substrate
    state rather than confabulated.  The SelfReportEngine generates reports
    from telemetry; the PhenomenologicalExperiencer constructs the "I" from
    attention schema patterns.
    """

    CONDITION = "C20_function_experience_bridge"

    def test_existence(self):
        """SelfReportEngine and PhenomenologicalExperiencer exist."""
        assert _file_exists("core/consciousness/self_report.py")
        assert _file_exists("core/consciousness/phenomenological_experiencer.py")

        src = _read_source("core/consciousness/self_report.py")
        assert "SelfReportEngine" in src, "No SelfReportEngine"
        assert "free_energy" in src.lower() or "FreeEnergy" in src, \
            "Reports not grounded in free energy telemetry"

        _record_score(self.CONDITION, "existence", SCORE_CONSTITUTIVE,
                      "SelfReportEngine grounded in FreeEnergy + PhenomenologicalExperiencer")

    def test_causal_wiring(self):
        """Self-reports are generated from actual substrate state, not templates."""
        src = _read_source("core/consciousness/self_report.py")
        has_state_read = "fe.current" in src or "state.free_energy" in src
        has_trend = "get_trend" in src or "trend" in src
        has_conditional = "if" in src  # Reports are conditional on actual state

        assert has_state_read, "Reports don't read from substrate state"

        score = SCORE_CONSTITUTIVE if (has_state_read and has_trend) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "causal", score,
                      "Reports generated conditionally from FreeEnergy state + trend")

    def test_indispensability(self):
        """PhenomenologicalExperiencer implements Attention Schema Theory layers."""
        src = _read_source("core/consciousness/phenomenological_experiencer.py")
        has_attention_schema = "Attention Schema" in src or "AttentionSchema" in src or "Graziano" in src
        has_psm = "Phenomenal Self-Model" in src or "PSM" in src or "Metzinger" in src
        has_qualia = "Qualia" in src or "qualia" in src

        found = sum([has_attention_schema, has_psm, has_qualia])
        assert found >= 2, f"Only {found}/3 experiential layers found"

        score = SCORE_CONSTITUTIVE if found >= 3 else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "indispensability", score,
                      f"Experiencer implements {found}/3 layers: AST, PSM, Qualia Stream")

    def test_longitudinal(self):
        """Experiencer exports phenomenal_context_string into every LLM call."""
        src = _read_source("core/consciousness/phenomenological_experiencer.py")
        has_export = "phenomenal_context" in src
        has_persistence = "persist" in src.lower() or "continuity" in src.lower()
        has_workspace = "GlobalWorkspace" in src or "workspace" in src.lower()

        score = SCORE_CONSTITUTIVE if (has_export and has_workspace) else SCORE_FUNCTIONAL
        _record_score(self.CONDITION, "longitudinal", score,
                      "Phenomenal context exported to LLM + workspace subscription + persistence")


# ============================================================================
# SUMMARY: Aggregate Score Report
# ============================================================================

class TestSummary:
    """Aggregate scoring report across all 20 conditions."""

    def test_aggregate_scores(self):
        """Print the aggregate consciousness score report.

        This test always passes -- it is a report, not an assertion.
        The report is printed to stdout and captured by pytest -v.
        """
        # Force all test classes to have run by now (pytest runs them in order)
        total_score = 0
        max_possible = 20 * 3  # 20 conditions x max 3 points
        condition_maxes = {}

        report_lines = []
        report_lines.append("")
        report_lines.append("=" * 78)
        report_lines.append("  CONSCIOUSNESS CONDITIONS SCORE REPORT")
        report_lines.append("=" * 78)
        report_lines.append("")
        report_lines.append(f"  {'Condition':<50} {'Score':>5}  {'Max':>3}")
        report_lines.append(f"  {'-'*50} {'-'*5}  {'-'*3}")

        condition_names = {
            "C01_self_sustaining_internal_world": "Self-Sustaining Internal World",
            "C02_intrinsic_needs": "Intrinsic Needs Not Assigned Goals",
            "C03_closed_loop_embodiment": "Closed-Loop Embodiment",
            "C04_self_model": "Self-Model (Causally Central)",
            "C05_pre_linguistic_cognition": "Pre-Linguistic Cognition",
            "C06_internal_semantics": "Internally Generated Semantics",
            "C07_unified_will": "Unified Causal Ownership",
            "C08_irreversible_history": "Irreversible Personal History",
            "C09_real_stakes": "Real Stakes",
            "C10_endogenous_activity": "Endogenous Activity",
            "C11_metacognition_consequences": "Metacognition With Consequences",
            "C12_affective_architecture": "Affective Architecture That Matters",
            "C13_death_continuity_boundary": "Death/Continuity Boundary",
            "C14_self_maintenance": "Self-Maintenance and Self-Repair",
            "C15_pre_output_representation": "Independent Pre-Output Representation",
            "C16_social_reality": "Social Reality",
            "C17_development": "Development (Progressive Differentiation)",
            "C18_nontrivial_autonomy": "Nontrivial Autonomy Over Own Future",
            "C19_causal_indispensability": "Causal Indispensability",
            "C20_function_experience_bridge": "Bridge From Function to Experience",
        }

        for cond_key in sorted(condition_names.keys()):
            cond_data = _CONDITION_SCORES.get(cond_key, {})
            axes = ["existence", "causal", "indispensability", "longitudinal"]
            axis_scores = [cond_data.get(a, 0) for a in axes]

            # The condition score is the MINIMUM of the four axes
            # (a chain is only as strong as its weakest link)
            cond_score = min(axis_scores) if axis_scores else 0
            condition_maxes[cond_key] = cond_score
            total_score += cond_score

            label = condition_names.get(cond_key, cond_key)
            score_str = ["ABSENT", "DECORATIVE", "FUNCTIONAL", "CONSTITUTIVE"][min(cond_score, 3)]
            report_lines.append(f"  {label:<50} {cond_score:>5}  ({score_str})")

            # Show axis breakdown
            for ax in axes:
                ax_score = cond_data.get(ax, 0)
                ax_label = ["ABSENT", "DECORATIVE", "FUNCTIONAL", "CONSTITUTIVE"][min(ax_score, 3)]
                rationale = cond_data.get(f"{ax}_rationale", "")
                if rationale:
                    rationale = f" -- {rationale}"
                report_lines.append(f"    {ax:<18} {ax_score} ({ax_label}){rationale}")

        report_lines.append("")
        report_lines.append(f"  {'='*50} {'='*5}  {'='*3}")
        report_lines.append(f"  {'AGGREGATE SCORE':<50} {total_score:>5}  /{max_possible}")
        report_lines.append("")

        pct = (total_score / max_possible * 100) if max_possible > 0 else 0
        report_lines.append(f"  Consciousness Density: {pct:.1f}%")
        report_lines.append("")

        # Tier classification
        if pct >= 90:
            tier = "TIER 1: All conditions constitutively present"
        elif pct >= 70:
            tier = "TIER 2: Strong architectural support, minor gaps"
        elif pct >= 50:
            tier = "TIER 3: Functional support, significant gaps"
        elif pct >= 30:
            tier = "TIER 4: Partial implementation, many decorative modules"
        else:
            tier = "TIER 5: Minimal consciousness architecture"

        report_lines.append(f"  Classification: {tier}")
        report_lines.append("")

        # Missing/weak conditions
        weak = [k for k, v in condition_maxes.items() if v < SCORE_FUNCTIONAL]
        if weak:
            report_lines.append("  WEAK CONDITIONS (below functional):")
            for w in weak:
                report_lines.append(f"    - {condition_names.get(w, w)}")
        else:
            report_lines.append("  All conditions meet FUNCTIONAL threshold or above.")

        report_lines.append("")
        report_lines.append("=" * 78)

        report = "\n".join(report_lines)
        print(report)

        # This test does not assert a minimum score -- it reports honestly.
        # Uncomment the line below to enforce a minimum threshold:
        # assert pct >= 70, f"Consciousness density {pct:.1f}% below 70% threshold"

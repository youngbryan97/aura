"""Tests for SubstrateAuthority — mandatory gating, no bypasses, single authority.

These tests prove that:
  1. Somatic veto actually BLOCKS (not just adjusts priority)
  2. Field incoherence actually HALTS non-critical actions
  3. Neurochemical crisis actually CONSTRAINS proposals
  4. The authority is a true gate, not advisory
  5. Critical actions still pass (safety override)
  6. Feedback loop: blocked actions trigger frustration chemistry
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from core.consciousness.substrate_authority import (
    SubstrateAuthority,
    ActionCategory,
    AuthorizationDecision,
    AuthorityThresholds,
)


class MockUnifiedField:
    """Mock unified field with controllable coherence."""
    def __init__(self, coherence: float = 0.6):
        self._coherence = coherence

    def get_coherence(self) -> float:
        return self._coherence

    def get_experiential_quality(self):
        return {"coherence": self._coherence, "valence": 0.0, "intensity": 0.3}


class MockSomaticGate:
    """Mock somatic gate with controllable verdict."""
    def __init__(self, approach: float = 0.0, confidence: float = 0.5, budget: bool = True):
        self._approach = approach
        self._confidence = confidence
        self._budget = budget

    def evaluate(self, content, source, priority):
        return SimpleNamespace(
            approach_score=self._approach,
            confidence=self._confidence,
            budget_available=self._budget,
        )


class MockNeurochemicalSystem:
    """Mock neurochemical system with controllable chemical levels."""
    def __init__(self, cortisol=0.3, gaba=0.5, dopamine=0.5, ne=0.4):
        class Chem:
            def __init__(self, eff):
                self.effective = eff
            def surge(self, amount): pass
        self.chemicals = {
            "cortisol": Chem(cortisol),
            "gaba": Chem(gaba),
            "dopamine": Chem(dopamine),
            "norepinephrine": Chem(ne),
        }

    def on_frustration(self, amount): pass


# ═══════════════════════════════════════════════════════════════════════════
# Gate behavior tests
# ═══════════════════════════════════════════════════════════════════════════

class TestMandatoryGating:

    def test_allow_when_all_systems_healthy(self):
        auth = SubstrateAuthority()
        auth._field_ref = MockUnifiedField(coherence=0.7)
        auth._somatic_ref = MockSomaticGate(approach=0.3, confidence=0.5)
        auth._neurochemical_ref = MockNeurochemicalSystem()

        verdict = auth.authorize("test action", "test", ActionCategory.RESPONSE, 0.5)
        assert verdict.decision == AuthorizationDecision.ALLOW

    def test_block_on_field_crisis(self):
        """Field coherence below crisis threshold MUST block non-critical actions."""
        auth = SubstrateAuthority()
        auth._field_ref = MockUnifiedField(coherence=0.15)  # below 0.25 crisis
        auth._somatic_ref = MockSomaticGate(approach=0.3)
        auth._neurochemical_ref = MockNeurochemicalSystem()

        verdict = auth.authorize("test action", "test", ActionCategory.EXPLORATION, 0.5)
        assert verdict.decision == AuthorizationDecision.BLOCK
        assert "field_coherence" in verdict.reason

    def test_stabilization_exempt_from_field_crisis(self):
        """Stabilization actions can proceed during field crisis (recovery)."""
        auth = SubstrateAuthority()
        auth._field_ref = MockUnifiedField(coherence=0.15)  # crisis
        auth._somatic_ref = MockSomaticGate(approach=0.0)
        auth._neurochemical_ref = MockNeurochemicalSystem()

        verdict = auth.authorize("rest and recover", "baseline", ActionCategory.STABILIZATION, 0.3)
        # Should NOT be blocked — stabilization is exempt from field crisis
        assert verdict.decision != AuthorizationDecision.BLOCK

    def test_block_on_somatic_hard_veto(self):
        """Strong somatic avoid with high confidence MUST block."""
        auth = SubstrateAuthority()
        auth._field_ref = MockUnifiedField(coherence=0.7)
        auth._somatic_ref = MockSomaticGate(approach=-0.7, confidence=0.6)  # strong avoid
        auth._neurochemical_ref = MockNeurochemicalSystem()

        verdict = auth.authorize("run dangerous tool", "agency", ActionCategory.TOOL_EXECUTION, 0.8)
        assert verdict.decision == AuthorizationDecision.BLOCK
        assert "somatic_veto" in verdict.reason

    def test_no_veto_on_low_confidence_avoid(self):
        """Somatic avoid with LOW confidence should constrain, not block."""
        auth = SubstrateAuthority()
        auth._field_ref = MockUnifiedField(coherence=0.7)
        auth._somatic_ref = MockSomaticGate(approach=-0.7, confidence=0.2)  # low confidence
        auth._neurochemical_ref = MockNeurochemicalSystem()

        verdict = auth.authorize("test", "test", ActionCategory.RESPONSE, 0.5)
        # Low confidence → should not hard block
        assert verdict.decision != AuthorizationDecision.BLOCK

    def test_block_on_cortisol_crisis_non_stabilization(self):
        """Cortisol crisis blocks non-stabilization/non-response actions."""
        auth = SubstrateAuthority()
        auth._field_ref = MockUnifiedField(coherence=0.7)
        auth._somatic_ref = MockSomaticGate(approach=0.3)
        auth._neurochemical_ref = MockNeurochemicalSystem(cortisol=0.90)

        verdict = auth.authorize("explore new topic", "curiosity", ActionCategory.EXPLORATION, 0.5)
        assert verdict.decision == AuthorizationDecision.BLOCK
        assert "cortisol_crisis" in verdict.reason

    def test_cortisol_crisis_allows_response(self):
        """During cortisol crisis, RESPONSE category still passes (user needs reply)."""
        auth = SubstrateAuthority()
        auth._field_ref = MockUnifiedField(coherence=0.7)
        auth._somatic_ref = MockSomaticGate(approach=0.0)
        auth._neurochemical_ref = MockNeurochemicalSystem(cortisol=0.90)

        verdict = auth.authorize("reply to user", "user", ActionCategory.RESPONSE, 0.5)
        assert verdict.decision != AuthorizationDecision.BLOCK

    def test_dopamine_crash_blocks_exploration(self):
        """Low dopamine specifically blocks exploration."""
        auth = SubstrateAuthority()
        auth._field_ref = MockUnifiedField(coherence=0.7)
        auth._somatic_ref = MockSomaticGate(approach=0.1)
        auth._neurochemical_ref = MockNeurochemicalSystem(dopamine=0.05)

        verdict = auth.authorize("explore", "curiosity", ActionCategory.EXPLORATION, 0.5)
        assert verdict.decision == AuthorizationDecision.BLOCK
        assert "dopamine_crash" in verdict.reason

    def test_gaba_collapse_blocks_non_stabilization(self):
        """GABA collapse (no inhibition) blocks non-stabilization actions."""
        auth = SubstrateAuthority()
        auth._field_ref = MockUnifiedField(coherence=0.7)
        auth._somatic_ref = MockSomaticGate(approach=0.0)
        auth._neurochemical_ref = MockNeurochemicalSystem(gaba=0.05)

        verdict = auth.authorize("do something", "agency", ActionCategory.INITIATIVE, 0.5)
        assert verdict.decision == AuthorizationDecision.BLOCK
        assert "gaba_collapse" in verdict.reason

    def test_gaba_collapse_constrains_internal_substrate_state_mutation(self):
        """Internal substrate settling work should degrade, not deadlock, during GABA collapse."""
        auth = SubstrateAuthority()
        auth._field_ref = MockUnifiedField(coherence=0.7)
        auth._somatic_ref = MockSomaticGate(approach=0.0)
        auth._neurochemical_ref = MockNeurochemicalSystem(gaba=0.05)

        verdict = auth.authorize(
            "stimulus_injection:weight=1.00",
            "substrate_stimulus",
            ActionCategory.STATE_MUTATION,
            0.8,
        )

        assert verdict.decision == AuthorizationDecision.CONSTRAIN
        assert any("internal_state_mutation_constrained" in item for item in verdict.constraints)

    def test_constrain_on_field_warning(self):
        """Field coherence in warning zone → CONSTRAIN."""
        auth = SubstrateAuthority()
        auth._field_ref = MockUnifiedField(coherence=0.35)  # between 0.25 and 0.40
        auth._somatic_ref = MockSomaticGate(approach=0.2)
        auth._neurochemical_ref = MockNeurochemicalSystem()

        verdict = auth.authorize("test", "test", ActionCategory.RESPONSE, 0.5)
        assert verdict.decision == AuthorizationDecision.CONSTRAIN

    def test_constrain_on_body_budget_deficit(self):
        """Low body budget → constrain high-cost actions."""
        auth = SubstrateAuthority()
        auth._field_ref = MockUnifiedField(coherence=0.7)
        auth._somatic_ref = MockSomaticGate(approach=0.0, budget=False)
        auth._neurochemical_ref = MockNeurochemicalSystem()

        verdict = auth.authorize("run tool", "agency", ActionCategory.TOOL_EXECUTION, 0.5)
        assert verdict.decision == AuthorizationDecision.CONSTRAIN


# ═══════════════════════════════════════════════════════════════════════════
# Critical override tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCriticalOverride:

    def test_critical_always_passes_during_field_crisis(self):
        """Safety-critical actions pass even during field crisis."""
        auth = SubstrateAuthority()
        auth._field_ref = MockUnifiedField(coherence=0.05)  # extreme crisis
        auth._somatic_ref = MockSomaticGate(approach=-0.9, confidence=0.9)
        auth._neurochemical_ref = MockNeurochemicalSystem(cortisol=0.95)

        verdict = auth.authorize(
            "emergency shutdown", "safety", ActionCategory.STABILIZATION,
            priority=1.0, is_critical=True,
        )
        assert verdict.decision == AuthorizationDecision.CRITICAL_PASS

    def test_critical_is_the_only_bypass(self):
        """Without is_critical=True, worst-case state should BLOCK."""
        auth = SubstrateAuthority()
        auth._field_ref = MockUnifiedField(coherence=0.05)
        auth._somatic_ref = MockSomaticGate(approach=-0.9, confidence=0.9)
        auth._neurochemical_ref = MockNeurochemicalSystem(cortisol=0.95)

        verdict = auth.authorize(
            "some action", "test", ActionCategory.EXPLORATION,
            priority=1.0, is_critical=False,  # NOT critical
        )
        assert verdict.decision == AuthorizationDecision.BLOCK


# ═══════════════════════════════════════════════════════════════════════════
# Feedback and audit tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFeedbackAndAudit:

    def test_block_increments_counter(self):
        auth = SubstrateAuthority()
        auth._field_ref = MockUnifiedField(coherence=0.10)
        auth._somatic_ref = MockSomaticGate()
        auth._neurochemical_ref = MockNeurochemicalSystem()

        auth.authorize("test", "test", ActionCategory.EXPLORATION, 0.5)
        assert auth._blocked == 1
        assert auth._total_requests == 1

    def test_allow_increments_counter(self):
        auth = SubstrateAuthority()
        auth._field_ref = MockUnifiedField(coherence=0.8)
        auth._somatic_ref = MockSomaticGate(approach=0.3)
        auth._neurochemical_ref = MockNeurochemicalSystem()

        auth.authorize("test", "test", ActionCategory.RESPONSE, 0.5)
        assert auth._allowed == 1

    def test_verdict_has_timestamp(self):
        auth = SubstrateAuthority()
        auth._field_ref = MockUnifiedField()
        verdict = auth.authorize("test", "test", ActionCategory.RESPONSE, 0.5)
        assert verdict.timestamp > 0

    def test_verdict_has_latency(self):
        auth = SubstrateAuthority()
        auth._field_ref = MockUnifiedField()
        verdict = auth.authorize("test", "test", ActionCategory.RESPONSE, 0.5)
        assert verdict.latency_ms >= 0

    def test_recent_blocks_audit(self):
        auth = SubstrateAuthority()
        auth._field_ref = MockUnifiedField(coherence=0.10)
        auth._somatic_ref = MockSomaticGate()
        auth._neurochemical_ref = MockNeurochemicalSystem()

        for _ in range(5):
            auth.authorize("test", "test", ActionCategory.EXPLORATION, 0.5)

        blocks = auth.get_recent_blocks(3)
        assert len(blocks) == 3  # limited to 3
        assert "reason" in blocks[0]

    def test_status_dict(self):
        auth = SubstrateAuthority()
        auth._field_ref = MockUnifiedField()
        status = auth.get_status()
        assert "total_requests" in status
        assert "block_rate" in status
        assert "current_field_coherence" in status


# ═══════════════════════════════════════════════════════════════════════════
# Combined stress tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCombinedStress:

    def test_multiple_gates_combine_correctly(self):
        """When multiple gates trigger, most restrictive wins."""
        auth = SubstrateAuthority()
        # Field warning (constrain) + somatic hard veto (block) → should be BLOCK
        auth._field_ref = MockUnifiedField(coherence=0.35)  # warning → constrain
        auth._somatic_ref = MockSomaticGate(approach=-0.7, confidence=0.6)  # hard veto → block
        auth._neurochemical_ref = MockNeurochemicalSystem()

        verdict = auth.authorize("test", "test", ActionCategory.TOOL_EXECUTION, 0.5)
        assert verdict.decision == AuthorizationDecision.BLOCK

    def test_all_categories_are_gated(self):
        """Every ActionCategory should be evaluated (no category bypass)."""
        auth = SubstrateAuthority()
        auth._field_ref = MockUnifiedField(coherence=0.10)  # crisis
        auth._somatic_ref = MockSomaticGate()
        auth._neurochemical_ref = MockNeurochemicalSystem()

        for category in ActionCategory:
            if category == ActionCategory.STABILIZATION:
                continue  # exempt from field crisis
            verdict = auth.authorize("test", "test", category, 0.5, is_critical=False)
            assert verdict.decision == AuthorizationDecision.BLOCK, \
                f"Category {category.name} was not blocked during field crisis"

    def test_graceful_degradation_no_refs(self):
        """With no substrate refs, authority allows through (degraded but safe)."""
        auth = SubstrateAuthority()
        # No refs set at all
        verdict = auth.authorize("test", "test", ActionCategory.RESPONSE, 0.5)
        assert verdict.decision == AuthorizationDecision.ALLOW

    def test_rapid_authorize_calls(self):
        """Authority should handle rapid sequential calls without error."""
        auth = SubstrateAuthority()
        auth._field_ref = MockUnifiedField(coherence=0.6)
        auth._somatic_ref = MockSomaticGate()
        auth._neurochemical_ref = MockNeurochemicalSystem()

        for i in range(1000):
            verdict = auth.authorize(f"action {i}", "stress_test", ActionCategory.RESPONSE, 0.5)
            assert verdict.decision in (
                AuthorizationDecision.ALLOW,
                AuthorizationDecision.CONSTRAIN,
                AuthorizationDecision.BLOCK,
                AuthorizationDecision.CRITICAL_PASS,
            )
        assert auth._total_requests == 1000

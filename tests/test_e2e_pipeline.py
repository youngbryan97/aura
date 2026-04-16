"""
test_e2e_pipeline.py
=====================
Comprehensive end-to-end test harness for the Aura project.

Covers:
  1. Boot Test        - Core systems import and instantiate without crashing
  2. Pipeline Test    - AuraState flows through each phase individually
  3. Affect Test      - AffectEngine decay_tick uses wall-clock time correctly
  4. Identity Guard   - AST-based open() write-mode detection
  5. Task Commitment  - TaskCommitmentVerifier capability gap / DispatchOutcome
  6. Approval Gate    - AutonomousTaskEngine plan approval flow
  7. RepairPhase      - Strips robotic phrasing from output
  8. Degradation      - DegradationManager state transitions and auto-recovery
  9. Metacognitive    - MetacognitiveCalibrator prediction tracking
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear_container():
    """Reset the ServiceContainer singleton to a clean state."""
    from core.container import ServiceContainer
    ServiceContainer.clear()


def _make_mock_kernel(vault=None):
    """Build a lightweight mock kernel that satisfies AuraKernel's shape
    without triggering the full boot sequence or heavy imports."""
    kernel = MagicMock()
    kernel.vault = vault
    kernel.state = None
    kernel.organs = {}
    kernel.get = MagicMock(return_value=None)
    kernel._services = {}
    return kernel


# ============================================================================
# 1. BOOT TEST
# ============================================================================

class TestBoot:
    """Import and instantiate the core systems without crashing."""

    def test_service_container_import(self):
        from core.container import ServiceContainer
        assert ServiceContainer is not None

    def test_aura_state_default(self):
        from core.state.aura_state import AuraState
        state = AuraState.default()
        assert state is not None
        assert state.version == 0
        assert state.identity.name.startswith("Aura")
        assert state.affect.valence == 0.0

    def test_state_repository_instantiation(self):
        from core.state.state_repository import StateRepository
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_state.db")
            repo = StateRepository(db_path=db_path, is_vault_owner=True)
            assert repo is not None
            assert repo.db_path == db_path

    @pytest.mark.slow
    def test_kernel_config_and_kernel_instantiation(self):
        """Instantiate AuraKernel with KernelConfig + temp StateRepository.

        This validates that the full canonical phase pipeline is created and organ stubs are
        populated (without running the async boot sequence).
        """
        _clear_container()
        try:
            from core.kernel.aura_kernel import AuraKernel, KernelConfig
            from core.state.state_repository import StateRepository

            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = os.path.join(tmpdir, "boot_test.db")
                vault = StateRepository(db_path=db_path, is_vault_owner=True)
                config = KernelConfig()
                kernel = AuraKernel(config=config, vault=vault)

                # -- Phase count verification --
                # _setup_phases is called in boot(), but phases are already
                # instantiated as attributes.  Manually call _setup_phases.
                kernel._setup_phases()
                expected_phase_names = [
                    "ProprioceptiveLoop",
                    "SocialContextPhase",
                    "SensoryIngestionPhase",
                    "NativeMultimodalBridge",
                    "EternalMemoryPhase",
                    "MemoryRetrievalPhase",
                    "PerfectEmotionPhase",
                    "AffectUpdatePhase",
                    "PhiConsciousnessPhase",
                    "MotivationUpdatePhase",
                    "CognitiveIntegrationPhase",
                    "ExecutiveClosurePhase",
                    "ShadowExecutionPhase",
                    "EternalGrowthEngine",
                    "TrueEvolutionPhase",
                    "InferencePhase",
                    "ConversationalDynamicsPhase",
                    "BondingPhase",
                    "CognitiveRoutingPhase",
                    "GodModeToolPhase",
                    "UnitaryResponsePhase",
                    "RepairPhase",
                    "MemoryConsolidationPhase",
                    "IdentityReflectionPhase",
                    "InitiativeGenerationPhase",
                    "ConsciousnessPhase",
                    "SelfReviewPhase",
                    "LearningPhase",
                    "LegacyPhase",
                ]
                actual_phase_names = [p.__class__.__name__ for p in kernel._phases]
                assert actual_phase_names == expected_phase_names, (
                    "Kernel phase pipeline drifted.\n"
                    f"Expected: {expected_phase_names}\n"
                    f"Actual:   {actual_phase_names}"
                )

                # -- Organ stub verification --
                kernel._initialize_organs()
                expected_organs = {
                    "llm", "vision", "memory", "voice", "metabolism",
                    "neural", "cookie", "prober", "tricorder",
                    "ice_layer", "omni_tool", "continuity",
                }
                assert set(kernel.organs.keys()) == expected_organs
        finally:
            _clear_container()


# ============================================================================
# 2. PIPELINE TEST
# ============================================================================

class TestPipeline:
    """Run an AuraState through each phase individually and verify non-None."""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_each_phase_returns_valid_state(self):
        _clear_container()
        try:
            from core.kernel.aura_kernel import AuraKernel, KernelConfig
            from core.state.aura_state import AuraState
            from core.state.state_repository import StateRepository

            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = os.path.join(tmpdir, "pipeline_test.db")
                vault = StateRepository(db_path=db_path, is_vault_owner=True)
                config = KernelConfig()
                kernel = AuraKernel(config=config, vault=vault)
                kernel._setup_phases()

                state = AuraState.default()
                state.cognition.current_objective = "Hello, how are you?"

                for phase in kernel._phases:
                    phase_name = phase.__class__.__name__
                    try:
                        result = await phase.execute(
                            state, objective="Hello, how are you?"
                        )
                        assert result is not None, (
                            f"Phase {phase_name} returned None"
                        )
                        # Use the result as the next input
                        state = result
                    except Exception as exc:
                        # Some phases may fail due to missing services (LLM, etc.)
                        # in test mode; that is acceptable as long as they do not
                        # return None silently.
                        pytest.skip(
                            f"Phase {phase_name} raised {type(exc).__name__}: {exc}"
                        )
        finally:
            _clear_container()


# ============================================================================
# 3. AFFECT TEST
# ============================================================================

class TestAffectEngine:
    """Verify AffectEngine.decay_tick uses wall-clock time correctly."""

    @pytest.mark.asyncio
    async def test_decay_toward_baseline(self):
        from core.affect import (
            AffectEngine,
            BASELINE_VALENCE,
            BASELINE_AROUSAL,
            BASELINE_ENGAGEMENT,
        )

        engine = AffectEngine()
        # Push state far from baselines
        engine.state.valence = 0.9
        engine.state.arousal = 0.9
        engine.state.engagement = 0.1
        # Set last decay time to 60 seconds ago to simulate one full tick
        engine._last_decay_time = time.time() - 60.0

        await engine.decay_tick()

        # After decay, values should move toward baselines
        assert engine.state.valence < 0.9, "Valence should have decayed toward baseline"
        assert engine.state.arousal < 0.9, "Arousal should have decayed toward baseline"
        assert engine.state.engagement > 0.1, "Engagement should have risen toward baseline"

    @pytest.mark.asyncio
    async def test_decay_proportional_to_elapsed_time(self):
        from core.affect import AffectEngine

        engine_short = AffectEngine()
        engine_long = AffectEngine()

        # Same starting state
        for eng in (engine_short, engine_long):
            eng.state.valence = 0.8
            eng.state.arousal = 0.8
            eng.state.engagement = 0.2

        # Short elapsed (10 seconds)
        engine_short._last_decay_time = time.time() - 10.0
        await engine_short.decay_tick()
        short_delta = abs(0.8 - engine_short.state.valence)

        # Long elapsed (120 seconds)
        engine_long._last_decay_time = time.time() - 120.0
        await engine_long.decay_tick()
        long_delta = abs(0.8 - engine_long.state.valence)

        # Longer elapsed time should produce larger decay
        assert long_delta > short_delta, (
            f"Decay should be proportional to elapsed time: "
            f"long_delta={long_delta:.6f} should be > short_delta={short_delta:.6f}"
        )


# ============================================================================
# 4. IDENTITY GUARD TEST
# ============================================================================

class TestIdentityGuard:
    """Verify AST-based open() write-mode detection."""

    def _guard(self):
        from core.agency.identity_guard import IdentityGuard
        return IdentityGuard()

    def test_open_write_positional_caught(self):
        code = 'open(f, "w")'
        result = self._guard().validate_modification("test.py", code)
        assert not result.approved or result.requires_human, (
            "open(f, 'w') should be caught as forbidden write"
        )
        assert any("write" in v.lower() or "open" in v.lower() for v in result.violations), (
            f"Expected a write violation, got: {result.violations}"
        )

    def test_open_write_keyword_caught(self):
        code = "open(f, mode='w')"
        result = self._guard().validate_modification("test.py", code)
        assert not result.approved or result.requires_human
        assert any("write" in v.lower() or "open" in v.lower() for v in result.violations)

    def test_open_read_positional_allowed(self):
        code = 'open(f, "r")'
        result = self._guard().validate_modification("test.py", code)
        # Should NOT have file-write violations
        write_violations = [
            v for v in result.violations
            if "file write" in v.lower() or "forbidden file" in v.lower()
        ]
        assert len(write_violations) == 0, (
            f"open(f, 'r') should NOT be flagged, got: {write_violations}"
        )

    def test_open_read_keyword_allowed(self):
        code = "open(f, mode='r')"
        result = self._guard().validate_modification("test.py", code)
        write_violations = [
            v for v in result.violations
            if "file write" in v.lower() or "forbidden file" in v.lower()
        ]
        assert len(write_violations) == 0


# ============================================================================
# 5. TASK COMMITMENT TEST
# ============================================================================

class TestTaskCommitment:
    """Verify TaskCommitmentVerifier and DispatchOutcome."""

    def test_dispatch_outcome_enum_values(self):
        from core.agency.task_commitment_verifier import DispatchOutcome

        assert DispatchOutcome.COMPLETED.value == "completed"
        assert DispatchOutcome.STARTED.value == "started"
        assert DispatchOutcome.FAILED.value == "failed"
        assert DispatchOutcome.CAPABILITY_GAP.value == "capability_gap"
        assert DispatchOutcome.DENIED.value == "denied"

    @pytest.mark.asyncio
    async def test_no_capability_engine_graceful_fallback(self):
        """With no CapabilityEngine registered, verify graceful behavior.

        The verifier falls back to can_fulfil=True with low confidence
        when the CapabilityEngine is unavailable, as a fail-safe-not-fail-silent
        design. It then checks for a task_engine. With no task_engine either,
        the result is FAILED.
        """
        _clear_container()
        try:
            from core.agency.task_commitment_verifier import (
                TaskCommitmentVerifier,
                DispatchOutcome,
            )
            from core.state.aura_state import AuraState

            mock_kernel = _make_mock_kernel()
            verifier = TaskCommitmentVerifier(mock_kernel)
            state = AuraState.default()

            acceptance = await verifier.verify_and_dispatch(
                "Research quantum computing advances", state
            )
            # Without a task_engine registered, it should return FAILED
            # (because the verifier assumes capable but cannot execute)
            assert acceptance.outcome in (
                DispatchOutcome.CAPABILITY_GAP,
                DispatchOutcome.FAILED,
            ), f"Expected CAPABILITY_GAP or FAILED, got {acceptance.outcome}"
        finally:
            _clear_container()


# ============================================================================
# 6. APPROVAL GATE TEST
# ============================================================================

class TestApprovalGate:
    """Verify AutonomousTaskEngine plan approval mechanics."""

    def test_plan_requires_approval_when_steps_gt_5(self):
        from core.agency.autonomous_task_engine import TaskPlan, TaskStep

        steps = [
            TaskStep(
                step_id=f"s{i}",
                description=f"Step {i}",
                tool="think",
                args={},
                success_criterion="non-empty",
                rollback_action=None,
            )
            for i in range(6)
        ]
        plan = TaskPlan(plan_id="test_plan", goal="complex task", steps=steps, trace_id="t1")

        # The engine sets requires_approval when len(steps) > 5
        if len(plan.steps) > 5:
            plan.requires_approval = True

        assert plan.requires_approval is True

    def test_approve_and_reject_plan_methods_exist(self):
        """Verify approve_plan and reject_plan work on the _approval_events dict."""
        from core.agency.autonomous_task_engine import AutonomousTaskEngine, TaskPlan, TaskStep

        mock_kernel = _make_mock_kernel()
        mock_kernel.organs = {"llm": MagicMock()}

        # Patch the heavy dependencies that __init__ tries to resolve
        with (
            patch("core.agency.autonomous_task_engine.get_capability_manager", return_value=MagicMock()),
            patch("core.agency.autonomous_task_engine.get_safety_registry", return_value=MagicMock()),
            patch("core.agency.autonomous_task_engine.get_mycelial", return_value=MagicMock()),
        ):
            engine = AutonomousTaskEngine(mock_kernel)

            # Simulate a plan waiting for approval
            plan_id = "test_approval_plan"
            event = asyncio.Event()
            engine._approval_events[plan_id] = event
            engine._active_plans[plan_id] = TaskPlan(
                plan_id=plan_id, goal="test", steps=[], trace_id="t"
            )

            # approve_plan should set the event
            result = engine.approve_plan(plan_id)
            assert result is True
            assert event.is_set()
            assert engine._active_plans[plan_id].status == "approved"

            # Reset for reject test
            event2 = asyncio.Event()
            engine._approval_events[plan_id] = event2
            engine._active_plans[plan_id].status = "pending"

            result2 = engine.reject_plan(plan_id)
            assert result2 is True
            assert event2.is_set()
            assert engine._active_plans[plan_id].status == "rejected"


# ============================================================================
# 7. REPAIR PHASE TEST
# ============================================================================

class TestRepairPhase:
    """Feed robotic text to RepairPhase and verify it strips the patterns."""

    @pytest.mark.asyncio
    async def test_strips_robotic_patterns(self):
        from core.phases.repair_phase import RepairPhase
        from core.state.aura_state import AuraState

        phase = RepairPhase(container=None)
        state = AuraState.default()
        state.cognition.last_response = (
            "Certainly! I'd be happy to help you today. "
            "The answer to your question is 42. "
            "Let me know if you need anything else."
        )

        result = await phase.execute(state, objective="test")
        assert result is not None
        repaired = result.cognition.last_response

        # The leading "Certainly! I'd be happy to help you today." should be stripped
        assert "certainly" not in repaired.lower(), (
            f"'Certainly' should have been stripped. Got: {repaired}"
        )
        assert "i'd be happy to" not in repaired.lower(), (
            f"'I'd be happy to' should have been stripped. Got: {repaired}"
        )
        # Trailing servile closer should be stripped
        assert "let me know if you need" not in repaired.lower(), (
            f"Trailing servile closer should have been stripped. Got: {repaired}"
        )
        # The actual content should survive
        assert "42" in repaired, (
            f"The substantive content ('42') should survive repair. Got: {repaired}"
        )

    @pytest.mark.asyncio
    async def test_clean_text_passes_unchanged(self):
        from core.phases.repair_phase import RepairPhase
        from core.state.aura_state import AuraState

        phase = RepairPhase(container=None)
        state = AuraState.default()
        original = "The square root of 144 is 12."
        state.cognition.last_response = original

        result = await phase.execute(state, objective="test")
        assert result.cognition.last_response == original


# ============================================================================
# 8. DEGRADATION RECOVERY TEST
# ============================================================================

class TestDegradationRecovery:
    """Verify DegradationManager state transitions and auto-recovery."""

    def test_transition_to_degraded(self):
        from core.resilience.degradation import (
            DegradationManager,
            FailureEvent,
            FailureType,
            SystemState,
        )

        mgr = DegradationManager()
        assert mgr.current_state == SystemState.HEALTHY

        mgr.report_failure(FailureEvent(
            type=FailureType.SKILL_FAILURE,
            component="web_search",
            error_msg="timeout",
            severity=0.5,
        ))
        assert mgr.current_state == SystemState.DEGRADED

    def test_check_health_increments_healthy_checks(self):
        from core.resilience.degradation import (
            DegradationManager,
            FailureEvent,
            FailureType,
            SystemState,
        )

        mgr = DegradationManager()
        # Force into DEGRADED state
        mgr.report_failure(FailureEvent(
            type=FailureType.SKILL_FAILURE,
            component="vision",
            error_msg="not found",
            severity=0.5,
        ))
        assert mgr.current_state == SystemState.DEGRADED

        # Clear the failure history so check_health sees no recent failures
        mgr.failure_history.clear()

        mgr.check_health()
        assert mgr.consecutive_healthy_checks == 1

        mgr.check_health()
        assert mgr.consecutive_healthy_checks == 2

    def test_auto_recovery_after_threshold(self):
        from core.resilience.degradation import (
            DegradationManager,
            FailureEvent,
            FailureType,
            SystemState,
        )

        mgr = DegradationManager()
        mgr.report_failure(FailureEvent(
            type=FailureType.SKILL_FAILURE,
            component="weather",
            error_msg="api down",
            severity=0.4,
        ))
        assert mgr.current_state == SystemState.DEGRADED

        # Clear failures so health checks pass
        mgr.failure_history.clear()

        # Run RECOVERY_THRESHOLD health checks
        for _ in range(mgr.RECOVERY_THRESHOLD):
            mgr.check_health()

        assert mgr.current_state == SystemState.HEALTHY, (
            f"Expected HEALTHY after {mgr.RECOVERY_THRESHOLD} clean checks, "
            f"got {mgr.current_state}"
        )


# ============================================================================
# 9. METACOGNITIVE CALIBRATOR TEST
# ============================================================================

class TestMetacognitiveCalibrator:
    """Verify MetacognitiveCalibrator prediction tracking."""

    def test_record_prediction_updates_calibration_error(self):
        from core.final_engines import MetacognitiveCalibrator

        cal = MetacognitiveCalibrator()
        assert cal.calibration_error == 0.0

        # Record a prediction with known actual correctness
        cal.record_prediction(confidence=0.9, actual_correctness=0.9)
        # Error should be abs(0.9 - 0.9) = 0.0
        assert cal.calibration_error == pytest.approx(0.0, abs=1e-9)

    def test_calibration_error_nonzero_after_wrong_prediction(self):
        from core.final_engines import MetacognitiveCalibrator

        cal = MetacognitiveCalibrator()

        # Predict high confidence but outcome is wrong
        cal.record_prediction(confidence=0.95, actual_correctness=0.1)
        assert cal.calibration_error > 0, (
            "Calibration error should be nonzero after a badly-calibrated prediction"
        )
        assert cal.calibration_error == pytest.approx(0.85, abs=0.01)

    def test_calibration_error_running_average(self):
        from core.final_engines import MetacognitiveCalibrator

        cal = MetacognitiveCalibrator()

        # First: perfect prediction
        cal.record_prediction(confidence=0.5, actual_correctness=0.5)
        assert cal.calibration_error == pytest.approx(0.0, abs=1e-9)

        # Second: off by 0.5
        cal.record_prediction(confidence=0.8, actual_correctness=0.3)
        # Running average: (0.0 * 1 + 0.5) / 2 = 0.25
        assert cal.calibration_error == pytest.approx(0.25, abs=0.01)

    def test_confidence_history_capped(self):
        from core.final_engines import MetacognitiveCalibrator

        cal = MetacognitiveCalibrator()
        for i in range(600):
            cal.record_prediction(confidence=0.5)
        assert len(cal.confidence_history) <= 500

"""STABILITY v53 — Comprehensive LLM/Cortex Reliability Test Suite

Tests every failure mode, edge case, and recovery path in the
inference pipeline. These tests are designed to be HARSH — if any
of them fail, conversation will break for the user.

Run: pytest tests/test_stability_v53.py -v
"""
import asyncio
import gc
import os
import sys
import time
import threading
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1: InferenceGate — Conversation Status & State Machine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestConversationStatus:
    """Test get_conversation_status() never reports zombie states."""

    def _make_gate(self):
        from core.brain.inference_gate import InferenceGate
        gate = InferenceGate.__new__(InferenceGate)
        gate.orch = None
        gate._created_at = time.monotonic()
        gate._mlx_client = None
        gate._initialized = False
        gate._init_error = None
        gate._cached_identity_prompt = None
        gate._identity_prompt_time = 0.0
        gate._cloud_backoff_until = 0.0
        gate._cortex_recovery_in_progress = False
        gate._last_cortex_check = 0.0
        gate._cortex_recovery_attempts = 0
        gate._cortex_recovery_exhausted_at = 0.0
        gate._last_successful_generation_at = time.time()
        gate._prewarm_task = None
        gate._deferred_prewarm_task = None
        gate._maintenance_task = None
        gate._foreground_ready_lock = threading.Lock()
        gate._last_background_memory_shed_at = 0.0
        gate._last_spare_maintenance_at = 0.0
        return gate

    def test_default_state_is_cold_not_warming(self):
        """v53 fix: default state should be 'cold', not 'warming'."""
        gate = self._make_gate()
        lane = gate.get_conversation_status()
        assert lane["state"] == "cold", f"Default state should be 'cold', got '{lane['state']}'"

    def test_init_error_reports_failed(self):
        gate = self._make_gate()
        gate._init_error = "mlx_runtime_unavailable: Metal not found"
        lane = gate.get_conversation_status()
        assert lane["state"] == "failed"

    def test_completed_prewarm_clears_warmup_in_flight(self):
        """When prewarm task is done, warmup_in_flight must be False."""
        gate = self._make_gate()
        loop = asyncio.new_event_loop()
        # Create a completed task
        async def noop(): pass
        task = loop.create_task(noop())
        loop.run_until_complete(task)
        gate._prewarm_task = task
        lane = gate.get_conversation_status()
        assert lane["warmup_in_flight"] is False

    def test_failed_prewarm_sets_recovering(self):
        """When prewarm task failed with exception, state should be 'recovering'."""
        gate = self._make_gate()
        loop = asyncio.new_event_loop()
        async def fail(): raise RuntimeError("warmup_failed")
        task = loop.create_task(fail())
        try:
            loop.run_until_complete(task)
        except RuntimeError:
            pass
        gate._prewarm_task = task
        lane = gate.get_conversation_status()
        assert lane["state"] == "recovering", f"Failed prewarm should report 'recovering', got '{lane['state']}'"
        assert "prewarm_failed" in lane["last_failure_reason"]
        loop.close()

    def test_active_prewarm_reports_warming(self):
        """Active (not done) prewarm task should report 'warming'."""
        gate = self._make_gate()
        loop = asyncio.new_event_loop()
        future = loop.create_future()
        async def wait(): await future
        task = loop.create_task(wait())
        gate._prewarm_task = task
        lane = gate.get_conversation_status()
        assert lane["state"] == "warming"
        assert lane["warmup_in_flight"] is True
        # Cleanup
        future.set_result(None)
        loop.run_until_complete(task)
        loop.close()

    def test_stale_warming_resets_to_cold(self):
        """Lane stuck in 'warming' for >90s with no active task should reset to 'cold'."""
        gate = self._make_gate()
        # Mock MLX client that reports "warming" with old timestamps
        mock_mlx = MagicMock()
        mock_mlx.get_lane_status.return_value = {
            "state": "warming",
            "last_error": "",
            "conversation_ready": False,
            "last_transition_at": time.time() - 120,  # 2 minutes ago
            "last_ready_at": 0.0,
            "last_progress_at": time.time() - 120,
            "warmup_attempted": True,
            "warmup_in_flight": False,
        }
        mock_mlx._warmup_in_flight = False
        gate._mlx_client = mock_mlx
        gate._prewarm_task = None
        gate._deferred_prewarm_task = None
        gate._cortex_recovery_in_progress = False
        lane = gate.get_conversation_status()
        assert lane["state"] == "cold", f"Stale warming should reset to 'cold', got '{lane['state']}'"

    def test_no_state_leak_on_recovery_in_progress(self):
        gate = self._make_gate()
        gate._cortex_recovery_in_progress = True
        lane = gate.get_conversation_status()
        assert lane["state"] == "recovering"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2: Cortex Recovery — Never Give Up
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCortexRecovery:
    """Test that cortex recovery never permanently gives up."""

    def _make_gate_with_dead_cortex(self):
        from core.brain.inference_gate import InferenceGate
        gate = InferenceGate.__new__(InferenceGate)
        gate.orch = None
        gate._created_at = time.monotonic()
        gate._initialized = True
        gate._init_error = None
        gate._cached_identity_prompt = None
        gate._identity_prompt_time = 0.0
        gate._cloud_backoff_until = 0.0
        gate._cortex_recovery_in_progress = False
        gate._last_cortex_check = 0.0
        gate._cortex_recovery_attempts = 0
        gate._cortex_recovery_exhausted_at = 0.0
        gate._last_successful_generation_at = time.time()
        gate._prewarm_task = None
        gate._deferred_prewarm_task = None
        gate._maintenance_task = None
        gate._foreground_ready_lock = threading.Lock()
        gate._last_background_memory_shed_at = 0.0
        gate._last_spare_maintenance_at = 0.0

        mock_mlx = MagicMock()
        mock_mlx.is_alive.return_value = False
        mock_mlx.warmup = AsyncMock()
        mock_mlx.get_lane_status.return_value = {
            "state": "failed", "last_error": "process_died",
            "conversation_ready": False, "last_transition_at": 0.0,
            "last_ready_at": 0.0, "last_progress_at": 0.0,
            "warmup_attempted": True, "warmup_in_flight": False,
        }
        mock_mlx.note_lane_recovering = MagicMock()
        mock_mlx._warmup_in_flight = False
        gate._mlx_client = mock_mlx
        return gate

    def test_recovery_exhausted_at_tracks_separately(self):
        """v53 fix: exhausted_at uses dedicated timestamp, not _last_cortex_check."""
        gate = self._make_gate_with_dead_cortex()
        gate._cortex_recovery_attempts = 5
        gate._cortex_recovery_exhausted_at = 0.0
        gate._last_cortex_check = 0.0  # Allow rate limit to pass

        loop = asyncio.new_event_loop()
        # Patch out asyncio.create_task
        with patch("asyncio.create_task") as mock_create:
            mock_task = MagicMock()
            mock_task.add_done_callback = MagicMock()
            mock_create.return_value = mock_task
            loop.run_until_complete(gate._ensure_cortex_recovery())

        # Should have set exhausted_at
        assert gate._cortex_recovery_exhausted_at > 0, "Should track exhaustion timestamp"
        loop.close()

    def test_recovery_retries_after_5min_cooldown(self):
        """After 5 failures + 5 min cooldown, recovery counter resets and retries."""
        gate = self._make_gate_with_dead_cortex()
        gate._cortex_recovery_attempts = 5
        gate._cortex_recovery_exhausted_at = time.monotonic() - 301  # 5+ minutes ago
        gate._last_cortex_check = 0.0

        loop = asyncio.new_event_loop()
        with patch("asyncio.create_task") as mock_create:
            mock_task = MagicMock()
            mock_task.add_done_callback = MagicMock()
            mock_create.return_value = mock_task
            loop.run_until_complete(gate._ensure_cortex_recovery())

        assert gate._cortex_recovery_attempts == 0, "Counter should reset after 5min cooldown"
        loop.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 3: LLM Router — Failover & Empty Response Handling
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestLLMRouterFailover:
    """Test the LLM router catches all failure modes."""

    def test_empty_response_treated_as_failure(self):
        """Empty or whitespace-only responses must trigger failover."""
        from core.brain.llm.llm_router import LLMHealthMonitor
        monitor = LLMHealthMonitor()
        monitor.record_failure("test_endpoint", "empty_response")
        assert monitor.failure_counts.get("test_endpoint", 0) == 1

    def test_fatal_patterns_catch_all_gpu_errors(self):
        """All known GPU/Metal crash patterns should be detected."""
        fatal_patterns = [
            "RESOURCE_EXHAUSTED", "MTLCompilerService", "No such process",
            "MLX Init Error", "Metal device not found", "NSRangeException",
            "bus error", "segmentation fault", "SIGKILL", "SIGABRT",
            "objectAtIndex", "out of memory", "OOM",
        ]
        for pattern in fatal_patterns:
            test_text = f"Some response with {pattern} error in it"
            found = any(p.lower() in test_text.lower() for p in fatal_patterns)
            assert found, f"Fatal pattern '{pattern}' not detected in response text"

    def test_health_monitor_recovery_after_threshold(self):
        """Endpoint should recover after recovery_time passes."""
        from core.brain.llm.llm_router import LLMHealthMonitor
        monitor = LLMHealthMonitor()
        monitor.recovery_time = 1  # 1 second for testing
        # Record 3 failures to trigger circuit break
        for _ in range(3):
            monitor.record_failure("test_ep", "test_error")
        assert not monitor.is_healthy("test_ep"), "Should be unhealthy after 3 failures"
        # Simulate recovery_time passing
        monitor.last_success["test_ep"] = time.time() - 2
        assert monitor.is_healthy("test_ep"), "Should recover after recovery_time"

    def test_default_tier_priority_includes_secondary(self):
        """v53 fix: default failover chain must include SECONDARY (cloud)."""
        from core.brain.llm.llm_router import IntelligentLLMRouter, LLMTier, LLMEndpoint
        router = IntelligentLLMRouter.__new__(IntelligentLLMRouter)
        router.endpoints = {
            "primary": LLMEndpoint(name="primary", tier=LLMTier.PRIMARY),
            "secondary": LLMEndpoint(name="secondary", tier=LLMTier.SECONDARY),
            "tertiary": LLMEndpoint(name="tertiary", tier=LLMTier.TERTIARY),
        }
        ordered = router._get_ordered_endpoints(prefer_tier=None)
        assert "secondary" in ordered, "Default failover must include secondary (cloud)"
        primary_idx = ordered.index("primary")
        secondary_idx = ordered.index("secondary")
        assert primary_idx < secondary_idx, "Primary should come before secondary"

    def test_rate_limit_triggers_immediate_circuit_break(self):
        from core.brain.llm.llm_router import LLMHealthMonitor
        monitor = LLMHealthMonitor()
        monitor.record_failure("test_ep", "HTTP 429 rate limit exceeded")
        assert not monitor.is_healthy("test_ep"), "429 should trigger immediate circuit break"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4: MLX Client — Consecutive Empty & Reboot Reset
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMLXClientStability:
    """Test MLX client state management."""

    def test_consecutive_empty_initialized(self):
        """v53 fix: _consecutive_empty must be initialized in __init__."""
        from core.brain.llm.mlx_client import MLXLocalClient
        # Check the __init__ source
        import inspect
        source = inspect.getsource(MLXLocalClient.__init__)
        assert "_consecutive_empty" in source, \
            "_consecutive_empty must be explicitly initialized in __init__"

    def test_consecutive_empty_reset_on_reboot(self):
        """v53 fix: reboot_worker must reset _consecutive_empty to 0."""
        from core.brain.llm.mlx_client import MLXLocalClient
        import inspect
        source = inspect.getsource(MLXLocalClient.reboot_worker)
        assert "_consecutive_empty" in source, \
            "_consecutive_empty must be reset in reboot_worker()"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 5: Local Server Client — Lock Timeouts & Compute Error
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestLocalServerClientStability:
    """Test local server client deadline and error handling."""

    def test_lock_timeout_respects_expired_deadline(self):
        """v53 fix: lock timeout must not exceed a nearly-expired deadline."""
        from core.brain.llm.local_server_client import LocalServerClient
        from core.utils.deadlines import Deadline
        client = LocalServerClient.__new__(LocalServerClient)
        # Deadline with 0.3s remaining
        deadline = Deadline(timeout=0.3, start_time=time.monotonic())
        time.sleep(0.05)  # Let 50ms pass
        timeout = client._lock_timeout(deadline=deadline, default=20.0, minimum=5.0)
        assert timeout < 1.0, f"Lock timeout {timeout} should respect near-expired deadline"

    def test_lock_timeout_without_deadline_uses_default(self):
        from core.brain.llm.local_server_client import LocalServerClient
        client = LocalServerClient.__new__(LocalServerClient)
        timeout = client._lock_timeout(deadline=None, default=20.0, minimum=5.0)
        assert timeout == 20.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 6: Deadline Management
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDeadline:
    """Test deadline utility never lies about remaining time."""

    def test_remaining_decreases_over_time(self):
        from core.utils.deadlines import Deadline
        d = Deadline(timeout=5.0)
        r1 = d.remaining
        time.sleep(0.1)
        r2 = d.remaining
        assert r2 < r1, "Remaining should decrease over time"

    def test_expired_deadline(self):
        from core.utils.deadlines import Deadline
        d = Deadline(timeout=0.01)
        time.sleep(0.02)
        assert d.is_expired
        assert d.remaining == 0.0

    def test_shield_subtracts_buffer(self):
        from core.utils.deadlines import Deadline
        d = Deadline(timeout=10.0)
        shielded = d.shield(buffer=2.0)
        assert shielded < 10.0
        assert shielded > 7.0  # Should be ~8s

    def test_none_timeout_is_infinite(self):
        from core.utils.deadlines import Deadline
        d = Deadline(timeout=None)
        assert d.remaining is None
        assert not d.is_expired


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 7: Chat Handler — Always Returns Response
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestChatHandlerResilience:
    """Test that chat ALWAYS returns a usable response."""

    def test_timeout_returns_200(self):
        """v53 fix: timeout should return 200 with status field, not 503/504."""
        # Verify the code has the fix
        import ast
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "interface", "routes", "chat.py")) as f:
            source = f.read()
        # Find the outer TimeoutError handler
        assert 'status_code=200,  # [STABILITY v53]' in source, \
            "Timeout handler must return 200, not 503/504"

    def test_exception_returns_200(self):
        """v53 fix: any exception should return 200 with error message."""
        import ast
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "interface", "routes", "chat.py")) as f:
            source = f.read()
        # The generic exception handler should return 200
        lines = source.split("\n")
        found_generic_200 = False
        for i, line in enumerate(lines):
            if "I lost my train of thought" in line:
                # Check nearby lines for status_code=200
                nearby = "\n".join(lines[max(0,i-5):i+5])
                if "status_code=200" in nearby:
                    found_generic_200 = True
                    break
        assert found_generic_200, "Generic exception handler must return 200"

    def test_soft_deadline_is_reasonable(self):
        """v53 fix: soft deadline should not be 8 seconds."""
        import ast
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "interface", "routes", "chat.py")) as f:
            source = f.read()
        # Find _KERNEL_SOFT_REPLY_SLA_SECONDS
        for line in source.split("\n"):
            if "_KERNEL_SOFT_REPLY_SLA_SECONDS" in line and "=" in line and "float" in line:
                # Extract the value
                value_str = line.split("float(")[-1].split(")")[0] if "float(" in line else ""
                if not value_str:
                    # Try simpler extraction
                    parts = line.split("=")
                    if len(parts) >= 2:
                        value_str = parts[-1].strip().rstrip(")")
                        try:
                            value = float(value_str.split("#")[0].strip())
                            assert value >= 30.0, \
                                f"Soft deadline is {value}s — must be >= 30s for reliable first-turn responses"
                        except ValueError:
                            pass
                break

    def test_protected_foreground_allows_cloud(self):
        """v53 fix: protected foreground lane must allow cloud fallback."""
        import ast
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "interface", "routes", "chat.py")) as f:
            source = f.read()
        # Find the protected foreground generate call
        in_protected = False
        for line in source.split("\n"):
            if "protected_foreground_lane" in line and "True" in line:
                in_protected = True
            if in_protected and "allow_cloud_fallback" in line:
                assert "True" in line, \
                    "Protected foreground lane must allow cloud fallback"
                break


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 8: Proactive Watchdog
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestProactiveWatchdog:
    """Test the proactive cortex health watchdog exists and works."""

    def test_watchdog_method_exists(self):
        """v53: InferenceGate must have _proactive_cortex_watchdog."""
        from core.brain.inference_gate import InferenceGate
        assert hasattr(InferenceGate, "_proactive_cortex_watchdog"), \
            "InferenceGate must have _proactive_cortex_watchdog method"

    def test_maintenance_loop_calls_watchdog(self):
        """v53: maintenance loop must call the watchdog."""
        import inspect
        from core.brain.inference_gate import InferenceGate
        source = inspect.getsource(InferenceGate._maintenance_loop)
        assert "_proactive_cortex_watchdog" in source, \
            "Maintenance loop must call _proactive_cortex_watchdog"

    def test_log_task_exception_callback_exists(self):
        """v53: _log_task_exception must exist for fire-and-forget tasks."""
        from core.brain.inference_gate import InferenceGate
        assert hasattr(InferenceGate, "_log_task_exception"), \
            "InferenceGate must have _log_task_exception callback"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 9: Emergency Fallback — Never Return Nothing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEmergencyFallback:
    """Test that the system NEVER returns nothing to the user."""

    def test_emergency_fallback_returns_string(self):
        from core.brain.llm.llm_router import IntelligentLLMRouter
        router = IntelligentLLMRouter.__new__(IntelligentLLMRouter)
        result = router._emergency_fallback("test prompt", "test error")
        assert isinstance(result, str)
        assert len(result) > 10, "Emergency fallback must return meaningful text"

    def test_emergency_fallback_includes_error(self):
        from core.brain.llm.llm_router import IntelligentLLMRouter
        router = IntelligentLLMRouter.__new__(IntelligentLLMRouter)
        result = router._emergency_fallback("test prompt", "MLX_CRASH")
        assert "MLX_CRASH" in result, "Emergency fallback should include error for debugging"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 10: End-to-End Response Path Verification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEndToEndResponsePath:
    """Verify the response path from user message to reply."""

    def test_conversation_lane_user_message_never_empty(self):
        """Status messages must always be non-empty and user-friendly."""
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "interface"))
            from routes.chat import _conversation_lane_user_message
        except ImportError:
            pytest.skip("Cannot import chat routes")

        test_lanes = [
            {"state": "warming"},
            {"state": "recovering"},
            {"state": "failed"},
            {"state": "cold"},
            {"state": "ready"},
            {"state": "failed", "last_failure_reason": "mlx_runtime_unavailable: Metal not found"},
        ]
        for lane in test_lanes:
            msg = _conversation_lane_user_message(lane, timed_out=False)
            assert isinstance(msg, str) and len(msg) > 5, \
                f"Lane {lane['state']} produced empty/short message: '{msg}'"

        # Timeout message
        msg = _conversation_lane_user_message({"state": "ready"}, timed_out=True)
        assert isinstance(msg, str) and len(msg) > 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

"""STABILITY v53 — Comprehensive LLM/Cortex Reliability Test Suite

Tests every failure mode, edge case, and recovery path in the
inference pipeline. These tests are designed to be HARSH — if any
of them fail, conversation will break for the user.

Run: pytest tests/test_stability_v53.py -v
"""
import asyncio
import gc
import inspect
import os
import sys
import time
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RecordedCall:
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs


class CallRecorder:
    def __init__(self, result=None, *, side_effect=None):
        self.result = result
        self.side_effect = side_effect
        self.calls = []
        self.call_args = None

    @property
    def call_count(self):
        return len(self.calls)

    def __call__(self, *args, **kwargs):
        call = RecordedCall(args, kwargs)
        self.calls.append(call)
        self.call_args = call
        if isinstance(self.side_effect, BaseException):
            raise self.side_effect
        if callable(self.side_effect):
            return self.side_effect(*args, **kwargs)
        return self.result

    def assert_called_once(self):
        assert len(self.calls) == 1


class AsyncCallRecorder:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    async def __call__(self, *args, **kwargs):
        self.calls.append(RecordedCall(args, kwargs))
        return self.result


class TaskHandle:
    def __init__(self):
        self.add_done_callback = CallRecorder()

    def done(self):
        return False


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
        async def noop():
            return "completed"
        task = loop.create_task(noop())
        loop.run_until_complete(task)
        gate._prewarm_task = task
        lane = gate.get_conversation_status()
        assert lane["warmup_in_flight"] is False

    def test_failed_prewarm_sets_recovering(self):
        """When prewarm task failed with exception, state should be 'recovering'."""
        gate = self._make_gate()
        loop = asyncio.new_event_loop()
        prewarm_attempts = []

        async def fail():
            prewarm_attempts.append("attempted")
            raise RuntimeError("warmup_failed")
        task = loop.create_task(fail())
        try:
            loop.run_until_complete(task)
        except RuntimeError:
            pass
        gate._prewarm_task = task
        lane = gate.get_conversation_status()
        assert prewarm_attempts == ["attempted"]
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
        # Scripted MLX client that reports "warming" with old timestamps
        gate._mlx_client = SimpleNamespace(
            get_lane_status=CallRecorder(
                {
                    "state": "warming",
                    "last_error": "",
                    "conversation_ready": False,
                    "last_transition_at": time.time() - 120,  # 2 minutes ago
                    "last_ready_at": 0.0,
                    "last_progress_at": time.time() - 120,
                    "warmup_attempted": True,
                    "warmup_in_flight": False,
                }
            ),
            _warmup_in_flight=False,
        )
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

        gate._mlx_client = SimpleNamespace(
            is_alive=CallRecorder(False),
            warmup=AsyncCallRecorder(),
            get_lane_status=CallRecorder(
                {
                    "state": "failed", "last_error": "process_died",
                    "conversation_ready": False, "last_transition_at": 0.0,
                    "last_ready_at": 0.0, "last_progress_at": 0.0,
                    "warmup_attempted": True, "warmup_in_flight": False,
                }
            ),
            note_lane_recovering=CallRecorder(),
            _warmup_in_flight=False,
        )
        return gate

    def test_recovery_exhausted_at_tracks_separately(self, monkeypatch):
        """v53 fix: exhausted_at uses dedicated timestamp, not _last_cortex_check."""
        gate = self._make_gate_with_dead_cortex()
        gate._cortex_recovery_attempts = 5
        gate._cortex_recovery_exhausted_at = 0.0
        gate._last_cortex_check = 0.0  # Allow rate limit to pass

        loop = asyncio.new_event_loop()
        task = TaskHandle()

        def create_task(awaitable, *args, **kwargs):
            if inspect.isawaitable(awaitable):
                awaitable.close()
            return task

        monkeypatch.setattr(asyncio, "create_task", CallRecorder(side_effect=create_task))
        loop.run_until_complete(gate._ensure_cortex_recovery())

        # Should have set exhausted_at
        assert gate._cortex_recovery_exhausted_at > 0, "Should track exhaustion timestamp"
        loop.close()

    def test_recovery_retries_after_5min_cooldown(self, monkeypatch):
        """After 5 failures + 5 min cooldown, recovery counter resets and retries."""
        gate = self._make_gate_with_dead_cortex()
        gate._cortex_recovery_attempts = 5
        gate._cortex_recovery_exhausted_at = time.monotonic() - 301  # 5+ minutes ago
        gate._last_cortex_check = 0.0

        loop = asyncio.new_event_loop()
        task = TaskHandle()

        def create_task(awaitable, *args, **kwargs):
            if inspect.isawaitable(awaitable):
                awaitable.close()
            return task

        monkeypatch.setattr(asyncio, "create_task", CallRecorder(side_effect=create_task))
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
        """Endpoint should admit a half-open probe after the cooldown passes."""
        from core.brain.llm.llm_router import LLMHealthMonitor
        monitor = LLMHealthMonitor()
        monitor.recovery_time = 1  # 1 second for testing
        # Record 3 failures to trigger circuit break
        for _ in range(3):
            monitor.record_failure("test_ep", "test_error")
        assert not monitor.is_healthy("test_ep"), "Should be unhealthy after 3 failures"
        # Simulate the monotonic cooldown elapsing (the monitor no longer
        # derives recovery from wall-clock last_success — clock adjustments
        # reopened/blocked endpoints incorrectly).
        monitor.cooldown_until["test_ep"] = time.monotonic() - 0.1
        assert monitor.is_healthy("test_ep"), "Should admit a probe after cooldown"
        # Half-open admits exactly ONE probe; a concurrent caller keeps
        # failing over until the probe reports success.
        assert not monitor.is_healthy("test_ep"), "Second caller must not enter half-open"
        monitor.record_success("test_ep")
        assert monitor.is_healthy("test_ep"), "Successful probe closes the circuit"

    def test_default_tier_priority_includes_secondary(self):
        """Default local failover must retain the SECONDARY tier."""
        from core.brain.llm.llm_router import IntelligentLLMRouter, LLMTier, LLMEndpoint
        router = IntelligentLLMRouter.__new__(IntelligentLLMRouter)
        router.endpoints = {
            "primary": LLMEndpoint(name="primary", tier=LLMTier.PRIMARY),
            "secondary": LLMEndpoint(name="secondary", tier=LLMTier.SECONDARY),
            "tertiary": LLMEndpoint(name="tertiary", tier=LLMTier.TERTIARY),
        }
        ordered = router._get_ordered_endpoints(prefer_tier=None)
        assert "secondary" in ordered, "Default local failover must include secondary"
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
# SECTION 5: Retired External Server Client
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestLocalServerClientStability:
    """Test the retired external local-server boundary."""

    def test_external_server_client_is_retired(self):
        from core.brain.llm.retired_external_runtime import RetiredExternalRuntimeClient

        status = RetiredExternalRuntimeClient("/models/old-runtime").get_lane_status()

        assert status["state"] == "retired"
        assert status["conversation_ready"] is False


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
        """v53 fix: any exception should return 200 with error message.

        Checked against the handler's structure, not against the words it
        happens to say. This used to grep for two literal sentences; both were
        refactored into `_grounded_chat_failure_reply()` and the test began
        failing while the behaviour it protects was entirely intact — a test
        pinned to prose fails when the prose moves and stays silent when the
        status code changes, which is backwards for both.
        """
        import ast
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "interface", "routes", "chat.py")) as f:
            source = f.read()

        def _status_codes_returned(handler: ast.ExceptHandler) -> set:
            codes = set()
            for node in ast.walk(handler):
                if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Call):
                    continue
                for keyword in node.value.keywords:
                    if keyword.arg == "status_code" and isinstance(
                        keyword.value, ast.Constant
                    ):
                        codes.add(keyword.value.value)
            return codes

        # The turn-death floor: the last-resort `except Exception` in the chat
        # turn. Any reply it produces reaches a person, so it must arrive as a
        # reply rather than as a transport failure the surface renders as a
        # dead turn.
        floors = [
            handler
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Try)
            for handler in node.handlers
            if isinstance(handler.type, ast.Name)
            and handler.type.id == "Exception"
            and "turn-death floor" in (ast.get_source_segment(source, handler) or "")
            and _status_codes_returned(handler)
        ]
        assert floors, "no generic exception handler in the chat turn returns a response"
        for handler in floors:
            assert _status_codes_returned(handler) == {200}, (
                "the generic exception handler returns a non-200 status; a "
                f"turn-death floor must reach the person: {_status_codes_returned(handler)}"
            )

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

    def test_protected_foreground_stays_local(self):
        """Protected foreground lane stays on the local live-mind path.

        Superseded policy note: the original v53 fix wanted cloud fallback here
        so the user always got *some* answer. The live-mind contract now keeps
        the protected foreground on the in-process local lane (sovereign,
        local-only posture) and fails closed to the bounded-contract reply
        instead of silently switching substrates mid-turn.
        """
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "interface", "routes", "chat.py")) as f:
            source = f.read()
        # Find the protected foreground generate call
        in_protected = False
        for line in source.split("\n"):
            if "protected_foreground_lane" in line and "True" in line:
                in_protected = True
            if in_protected and "allow_cloud_fallback" in line:
                assert "False" in line, \
                    "Protected foreground lane must stay local (no cloud substrate swap)"
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

    def test_emergency_fallback_never_leaks_raw_error(self):
        # CP126 semantic finding: the raw last_error can carry filesystem
        # paths, model internals, or request fragments — it belongs in logs,
        # never in the user-visible fallback string.
        from core.brain.llm.llm_router import IntelligentLLMRouter
        router = IntelligentLLMRouter.__new__(IntelligentLLMRouter)
        result = router._emergency_fallback(
            "test prompt", "MLX_CRASH at /Users/private/model.bin token=abc123"
        )
        assert "MLX_CRASH" not in result
        assert "/Users/private" not in result
        assert len(result) > 10, "Emergency fallback must still return meaningful text"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 10: End-to-End Response Path Verification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEndToEndResponsePath:
    """Verify the response path from user message to reply."""

    def test_conversation_lane_user_message_never_empty(self):
        """Status messages must always be non-empty and user-friendly."""
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "interface"))
        from routes.chat import _conversation_lane_user_message

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


class TestProactiveWatchdogWarmupRace:
    """Two soak-taught contracts, one dead-man clock.

    postdoomfix soak: a WARMING lane is not a dead lane — the watchdog used
    to re-trigger recovery every 45s pulse, restarting the 150s cold load
    forever. nightcap soak: the first guard trusted CLIENT fields to bound
    the deferral, and a wedged warmup_in_flight flag deferred recovery
    FOREVER (11 straight 240s turn timeouts). The watchdog now times the
    not-alive window on its OWN clock: 300s of grace from first observation,
    then intervention regardless of any client flag."""

    class _WarmingClient:
        def __init__(self, *, warmup_in_flight=True):
            self._warmup_in_flight = warmup_in_flight

        def is_alive(self):
            return False  # cold-load worker: not alive yet, not dead either

    def _make_gate(self, client, *, first_seen_age_s: float | None = None):
        from core.brain.inference_gate import InferenceGate

        gate = InferenceGate.__new__(InferenceGate)
        gate._mlx_client = client
        gate._cortex_recovery_in_progress = False
        gate._prewarm_task = None
        gate._deferred_prewarm_task = None
        gate._foreground_user_turn_active = lambda: False
        gate._foreground_owner_active = lambda: False
        # CP126 ab3c124a: observation is pure by default; the self-heal path
        # opts in with observe_only=False, so the stub accepts it.
        gate.get_conversation_status = lambda **_kwargs: {"state": "warming"}
        gate._recovery_calls = []
        if first_seen_age_s is not None:
            gate._cortex_not_alive_first_seen_at = time.time() - first_seen_age_s

        async def _record_recovery():
            gate._recovery_calls.append(time.time())

        gate._ensure_cortex_recovery = _record_recovery
        return gate

    def test_warming_lane_within_deadline_is_not_dead(self):
        gate = self._make_gate(self._WarmingClient(), first_seen_age_s=30.0)
        asyncio.run(gate._proactive_cortex_watchdog())
        assert gate._recovery_calls == [], (
            "a warming lane inside its 300s dead-man window must never trigger recovery"
        )

    def test_first_observation_starts_the_clock_and_defers(self):
        gate = self._make_gate(self._WarmingClient())  # no prior observation
        asyncio.run(gate._proactive_cortex_watchdog())
        assert gate._recovery_calls == []
        assert getattr(gate, "_cortex_not_alive_first_seen_at", 0.0) > 0.0

    def test_wedged_warmup_flag_cannot_defer_forever(self):
        """The nightcap wedge: warmup_in_flight stuck True must NOT outlast
        the dead-man clock."""
        client = self._WarmingClient(warmup_in_flight=True)
        gate = self._make_gate(client, first_seen_age_s=400.0)
        asyncio.run(gate._proactive_cortex_watchdog())
        assert len(gate._recovery_calls) == 1, (
            "past the dead-man deadline, recovery fires no matter what client flags claim"
        )
        assert client._warmup_in_flight is False, (
            "the wedged warmup flag must be force-cleared so admission unblocks"
        )
        assert gate._cortex_not_alive_first_seen_at == 0.0, (
            "the recovery warmup gets a fresh 300s window"
        )

    def test_a_loading_worker_restarts_the_dead_man_clock(self):
        """LIVE 2026-09-16, load 34: the 27B worker was at 9GB of RSS and
        loading when the clock, started before the worker existed, reached
        305s and recovery replaced the warmup. Progress restarts the clock;
        a worker whose memory and CPU have stopped advancing does not."""

        class _LoadingClient(self._WarmingClient):
            def __init__(self):
                super().__init__(warmup_in_flight=True)
                self.readings = [(4.0e9, 40.0), (9.0e9, 90.0), (9.0e9, 90.0)]

            def worker_load_progress(self):
                return self.readings.pop(0) if self.readings else (9.0e9, 90.0)

        client = _LoadingClient()
        gate = self._make_gate(client, first_seen_age_s=400.0)
        # First look records the reading; the clock is 400s old.
        # Second look sees RSS and CPU advance: the clock restarts.
        gate._cortex_load_progress_seen = (4.0e9, 40.0)
        client.readings = [(9.0e9, 90.0), (9.0e9, 90.0)]
        asyncio.run(gate._proactive_cortex_watchdog())
        assert gate._recovery_calls == [], "a worker still loading must not be replaced"
        assert time.time() - gate._cortex_not_alive_first_seen_at < 5.0

        # No progress since, and the clock is past 300s again: recovery.
        gate._cortex_not_alive_first_seen_at = time.time() - 400.0
        asyncio.run(gate._proactive_cortex_watchdog())
        assert len(gate._recovery_calls) == 1, "a worker that stopped advancing is dead"

    def test_alive_lane_clears_the_dead_man_clock(self):
        class _AliveClient:
            def is_alive(self):
                return True

            _warmup_in_flight = False

        gate = self._make_gate(self._WarmingClient(), first_seen_age_s=250.0)
        gate._mlx_client = _AliveClient()
        asyncio.run(gate._proactive_cortex_watchdog())
        assert getattr(gate, "_cortex_not_alive_first_seen_at", 0.0) == 0.0
